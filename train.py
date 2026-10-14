"""Train the action expert on the dashboard's HDF5 episodes.

    python train.py --data /path/to/dataset --out runs/va1
    python train.py --data /path/to/dataset --out runs/smoke --backbone dummy \
        --steps 20 --batch 2 --device cpu          # checks the code, no download

Episodes are split by episode, never by frame: neighbouring frames are near
copies, so a frame split would grade the model on what it trained on. The
held-out episodes are the newest of both kinds, left-arm and right-arm
grasps, and every batch is half of each (BalancedBatchSampler; --no-balance
for a plain shuffle). Every --eval-every steps the held-out episodes give the
diffusion loss and, from sampled chunks, the error in radians (arm joints)
and millimetres (grippers), also apart for the working arm and the one held
still.

TensorBoard, unless --no-tensorboard: `tensorboard --logdir <out>/tb` shows
the training loss, gradient norm and learning rate, the held-out errors 1, 10
and Ta steps ahead, and at each evaluation one held-out chunk drawn against
what the right arm did.

A checkpoint holds the expert only (the backbone is frozen and reloaded by
name), the qpos stats the actions are normalised with, and the settings.
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import (ARM_JOINT_COLUMNS, ARM_NAMES, JOINT_NAMES, BalancedBatchSampler, Collate,
                  EpisodeDataset, compute_stats, episode_arms, list_episodes, split_episodes)
from model import DummyBackbone, QwenBackbone, QwenInputs, VAPolicy

ARM_JOINTS = [i for i in range(16) if i not in (7, 15)]
GRIPPERS = [7, 15]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", required=True, help="folder of episode_*.hdf5 (searched recursively)")
    p.add_argument("--out", required=True, help="run folder: checkpoints, stats, log")
    p.add_argument("--backbone", default="Qwen/Qwen3-VL-2B-Instruct",
                   help="Qwen3-VL model id or local path; 'dummy' for a code check")
    p.add_argument("--blocks", type=int, default=0,
                   help="expert blocks; 0 = half the backbone's layers")
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--chunk", type=int, default=50, help="Ta: steps predicted (50 = 1 s)")
    p.add_argument("--diffusion-steps", type=int, default=100)
    p.add_argument("--sample-steps", type=int, default=10, help="DDIM steps when evaluating")
    p.add_argument("--batch", type=int, default=8, help="samples per forward pass")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="forward passes per optimizer step: the batch the optimizer "
                        "sees is batch x grad-accum, the memory that of one batch")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-episodes", type=int, default=2,
                   help="held out: the newest N, split between left- and right-arm grasps")
    p.add_argument("--no-balance", action="store_true",
                   help="shuffle plainly instead of half left-arm, half right-arm batches")
    p.add_argument("--min-std-joint", type=float, default=1e-2,
                   help="floor of a joint's std (rad) when normalising")
    p.add_argument("--min-std-gripper", type=float, default=2e-3,
                   help="floor of a gripper's std (m) when normalising")
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--no-tensorboard", action="store_true", help="only log.jsonl")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-bf16", action="store_true", help="train the expert in fp32")
    p.add_argument("--resume", default="", help="a checkpoint .pt to continue from")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def to_device(batch, device):
    out = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            out[key] = to_device(value, device)
        else:
            out[key] = value.to(device, non_blocking=True)
    return out


def loop(loader):
    while True:
        yield from loader


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def horizons(chunk):
    return sorted({1, 10, chunk} & set(range(1, chunk + 1)))


@torch.no_grad()
def evaluate(policy, loader, args, std, device, autocast):
    """Held-out metrics, and one chunk (prediction, truth, state) to draw."""
    policy.eval()
    losses, joint_err, grip_err, working_err, idle_err = [], [], [], [], []
    by_kind = {name: [] for name in ARM_NAMES}
    ahead = {h: [] for h in horizons(args.chunk)}
    example = None
    for i, batch in enumerate(loader):
        if i >= args.eval_batches:
            break
        batch = to_device(batch, device)
        with autocast():
            losses.append(policy.loss(batch).item())
            pred = policy.sample(batch, steps=args.sample_steps)
        # Back to radians / metres: only the std matters for an error.
        err = (pred.float() - batch["actions"]).abs() * std
        keep = ~batch["is_pad"]
        joint_err.append(err[..., ARM_JOINTS][keep].mean().item())
        grip_err.append(err[..., GRIPPERS][keep].mean().item())
        for h in ahead:
            rows = keep[:, h - 1]
            if rows.any():
                ahead[h].append(err[rows, h - 1][:, ARM_JOINTS].mean().item())
        for b, arm in enumerate(batch["arm"].tolist()):
            rows = err[b][keep[b]]
            working_err.append(rows[:, ARM_JOINT_COLUMNS[arm]].mean().item())
            idle_err.append(rows[:, ARM_JOINT_COLUMNS[1 - arm]].mean().item())
            by_kind[ARM_NAMES[arm]].append(rows[:, ARM_JOINTS].mean().item())
        if example is None:
            example = (pred[0].float(), batch["actions"][0], batch["state"][0],
                       int(batch["arm"][0]))
    policy.train()
    metrics = {"val_loss": float(np.mean(losses)),
               "val_joint_rad": float(np.mean(joint_err)),
               "val_gripper_mm": float(np.mean(grip_err)) * 1000}
    for h, values in ahead.items():
        if values:
            metrics[f"val_joint_rad_{h}_ahead"] = float(np.mean(values))
    # The arm doing the grasp, and the one that should stay where it is.
    metrics["val_working_arm_rad"] = float(np.mean(working_err))
    metrics["val_idle_arm_rad"] = float(np.mean(idle_err))
    for name, values in by_kind.items():
        if values:
            metrics[f"val_joint_rad_{name}_grasps"] = float(np.mean(values))
    return metrics, example


def chunk_figure(example, mean, std):
    """The working arm's 7 joints and gripper over one chunk: dashed is what
    the arm did, solid the prediction, the dot where it started. None
    without matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    *tensors, arm = example
    pred, truth, state = (x.cpu().numpy() * std + mean for x in tensors)
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.5), sharex=True)
    for ax, j in zip(axes.flat, range(8 * arm, 8 * arm + 8)):
        ax.plot(truth[:, j], "--", color="0.4", label="recorded")
        ax.plot(pred[:, j], color="tab:blue", label="predicted")
        ax.plot(-1, state[j], "o", color="tab:orange", label="qpos[t]")
        unit = "m" if j in (7, 15) else "rad"
        ax.set_title(f"{JOINT_NAMES[j]} ({unit})", fontsize=10)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("steps ahead (20 ms)")
    fig.suptitle(f"a {ARM_NAMES[arm]}-arm grasp: the working arm", fontsize=11)
    fig.tight_layout()
    return fig


def save(path, policy, optimizer, step, args, stats, metrics):
    checkpoint = {"expert": policy.expert.state_dict(), "optimizer": optimizer.state_dict(),
                  "step": step, "args": vars(args), "stats": stats,
                  "highway_layers": policy.backbone.layers, "metrics": metrics}
    if args.backbone == "dummy":
        # Random, so kept: infer.py needs the same one. Qwen reloads by name.
        checkpoint["dummy_backbone"] = policy.backbone.state_dict()
    torch.save(checkpoint, path)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    print(f"checking the episodes under {args.data}", flush=True)
    episodes = list_episodes(args.data)
    arms = episode_arms(episodes)
    train_pairs, val_pairs = split_episodes(episodes, arms, args.val_episodes)
    train_eps, train_arms = [p for p, _ in train_pairs], [a for _, a in train_pairs]
    val_eps, val_arms = [p for p, _ in val_pairs], [a for _, a in val_pairs]
    # Over both kinds together, so each arm's std is its real range of motion.
    stats = compute_stats(train_eps, args.min_std_joint, args.min_std_gripper)
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(args.out, "split.json"), "w") as f:
        json.dump({kind: [{"episode": os.path.basename(p), "arm": ARM_NAMES[a]}
                          for p, a in pairs]
                   for kind, pairs in (("train", train_pairs), ("val", val_pairs))},
                  f, indent=2)
    print(f"{len(train_eps)} train / {len(val_eps)} val episodes; held out: " +
          ", ".join(f"{os.path.basename(p)} ({ARM_NAMES[a]})" for p, a in val_pairs))

    # The backbone first: it fixes the expert's depth (half its layers).
    if args.backbone == "dummy":
        backbone = DummyBackbone(args.blocks or 4)
        prepare = None
    else:
        probe_layers = args.blocks
        if not probe_layers:
            from transformers import AutoConfig
            layers = AutoConfig.from_pretrained(args.backbone).text_config.num_hidden_layers
            probe_layers = layers // 2
        # Silent otherwise: the download's progress bar rarely survives a pipe.
        print(f"loading {args.backbone} (the first run downloads it, ~4.5 GB for "
              f"the 2B, into ~/.cache/huggingface)...", flush=True)
        started = time.time()
        backbone = QwenBackbone(args.backbone, probe_layers)
        prepare = QwenInputs(args.backbone)
        print(f"loaded in {time.time() - started:.0f} s; moving to {device}", flush=True)
    policy = VAPolicy(backbone, args.chunk, args.dim, args.heads,
                      args.diffusion_steps).to(device)
    print(f"backbone {args.backbone}: highways from layers {backbone.layers}")
    trainable = sum(p.numel() for p in policy.expert.parameters())
    print(f"expert: {len(policy.expert.blocks)} blocks, {trainable / 1e6:.1f} M parameters")

    print("indexing the frames...", flush=True)
    train_set = EpisodeDataset(train_eps, stats, args.chunk, arms=train_arms)
    val_set = EpisodeDataset(val_eps, stats, args.chunk, arms=val_arms)
    collate = Collate(prepare)
    common = dict(num_workers=args.workers, collate_fn=collate,
                  pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    if args.no_balance:
        train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                                  drop_last=True, **common)
    else:
        train_loader = DataLoader(train_set, batch_sampler=BalancedBatchSampler(
            train_set.groups(), args.batch, args.seed), **common)
    # Balanced as well, so the few evaluation batches grade both kinds.
    val_loader = DataLoader(val_set, batch_sampler=BalancedBatchSampler(
        val_set.groups(), args.batch, args.seed + 1), **common)
    print(f"{len(train_set)} train / {len(val_set)} val frames; "
          f"{args.batch} x {args.grad_accum} samples a step, "
          f"{len(train_set) / (args.batch * args.grad_accum):.0f} steps an epoch")

    optimizer = torch.optim.AdamW(policy.expert.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    start = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        policy.expert.load_state_dict(checkpoint["expert"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start = checkpoint["step"]
        print(f"resumed from {args.resume} at step {start}")

    use_bf16 = not args.no_bf16 and device.type == "cuda"

    def autocast():
        return torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16)

    std = torch.tensor(stats["std"], device=device)
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    writer = None
    if not args.no_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            print("no tensorboard (pip install tensorboard): logging to log.jsonl only")
        else:
            # purge_step: a resumed run drops what the first one logged past it.
            writer = SummaryWriter(os.path.join(args.out, "tb"), purge_step=start or None)
            writer.add_text("args", "\n".join(f"    {k}: {v}" for k, v in vars(args).items()))
    best = float("inf")
    policy.train()
    batches = loop(train_loader)
    print(f"training from step {start}; a line every {args.log_every} steps, "
          f"an evaluation every {args.eval_every}", flush=True)
    started, seen = time.time(), 0
    for step in range(start, args.steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, args)
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(args.grad_accum):
            batch = to_device(next(batches), device)
            with autocast():
                loss = policy.loss(batch) / args.grad_accum
            loss.backward()
            total += loss.item()
        grad = torch.nn.utils.clip_grad_norm_(policy.expert.parameters(), args.grad_clip)
        optimizer.step()
        seen += 1

        done = step + 1
        if done % args.log_every == 0 or done == args.steps:
            rate = seen / (time.time() - started)
            record = {"step": done, "loss": total, "grad": grad.item(),
                      "lr": optimizer.param_groups[0]["lr"], "steps_per_s": rate}
            if device.type == "cuda":
                # The peak since the start: what --batch costs, eval included.
                record["gpu_mem_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
            print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}"
                           for k, v in record.items()), flush=True)
            log.write(json.dumps(record) + "\n")
            log.flush()
            if writer is not None:
                for key, value in record.items():
                    if key != "step":
                        writer.add_scalar(f"train/{key}", value, done)
        if done % args.eval_every == 0 or done == args.steps:
            metrics, example = evaluate(policy, val_loader, args, std, device, autocast)
            print("eval " + " ".join(f"{k} {v:.4g}" for k, v in metrics.items()), flush=True)
            log.write(json.dumps({"step": done, **metrics}) + "\n")
            log.flush()
            if writer is not None:
                for key, value in metrics.items():
                    writer.add_scalar(f"val/{key.removeprefix('val_')}", value, done)
                figure = chunk_figure(example, np.asarray(stats["mean"]),
                                      np.asarray(stats["std"]))
                if figure is not None:
                    writer.add_figure("val/working_arm_chunk", figure, done)
                writer.flush()
            if metrics["val_joint_rad"] < best:
                best = metrics["val_joint_rad"]
                save(os.path.join(args.out, "best.pt"), policy, optimizer, done, args,
                     stats, metrics)
        if done % args.save_every == 0 or done == args.steps:
            save(os.path.join(args.out, "last.pt"), policy, optimizer, done, args, stats, {})
    log.close()
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()

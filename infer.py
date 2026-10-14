"""Run a trained action expert: one observation in, the next Ta joint targets out.

In code:

    from infer import VAInference
    policy = VAInference("runs/va1/best.pt")
    chunk = policy.predict(rgb, depth_mm, qpos)
    # rgb (2, 480, 640, 3) uint8 and depth_mm (2, 480, 640) uint16, chest
    # then front; qpos (16,) -> chunk (Ta, 16): radians, grippers in metres,
    # in the order left_j1..j7, left_gripper, right_j1..j7, right_gripper.

On recorded episodes (a file, or a folder searched recursively), to see how
far the chunks land from what the arm did:

    python infer.py --checkpoint runs/va1/best.pt --episode ep.hdf5 --every 25
    python infer.py --checkpoint runs/va1/best.pt --episode /data/val \
        --tensorboard runs/va1/tb_eval --out runs/va1/eval

--tensorboard writes, per episode (a TensorBoard run each, "all" their mean),
the joint and gripper error 1, 10 and Ta steps ahead at the checkpoint's
training step, so evaluations of successive checkpoints line up, the error of
the working arm and of the one held still, and a plot of the working arm's
chunks over the whole episode. --out keeps the chunks.

The working arm is the one whose joints travel further in the recording, as
in training. Run a chunk to its end before asking for the next: averaging
overlapping chunks (temporal ensembling) can blend a left-arm chunk with a
right-arm one when the object sits between them.

The chunk is absolute joint targets for the steps after the observation:
row 0 is t+1, 20 ms later at 50 Hz. A controller sends the rows one per
control step, or blends overlapping chunks (temporal ensembling) when it
predicts more often than every Ta steps.
"""

import argparse
import os
import time

import h5py
import numpy as np
import torch

from data import (ARM_JOINT_COLUMNS, ARM_NAMES, CAMERAS, IMAGE_HW, JOINT_NAMES, NUM_JOINTS,
                  depth_input, list_episodes, working_arm)
from model import DummyBackbone, QwenBackbone, QwenInputs, VAPolicy

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
ARM_JOINTS = [i for i in range(NUM_JOINTS) if i not in (7, 15)]
GRIPPERS = [7, 15]


class VAInference:
    def __init__(self, checkpoint, backbone=None, device="cuda", sample_steps=10,
                 dtype="bf16"):
        """`backbone` overrides the name saved in the checkpoint, e.g. a local
        copy of the same Qwen3-VL. fp16 for GPUs without bf16 (before Ampere)."""
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        args = saved["args"]
        self.device = torch.device(device)
        self.dtype = DTYPES[dtype] if self.device.type == "cuda" else torch.float32
        self.sample_steps = sample_steps
        name = backbone or args["backbone"]
        layers = saved["highway_layers"]
        if name == "dummy":
            model = DummyBackbone(len(layers))
            model.load_state_dict(saved["dummy_backbone"])
            self.prepare = None
        else:
            model = QwenBackbone(name, len(layers), dtype=self.dtype)
            self.prepare = QwenInputs(name)
        if model.layers != layers:
            raise ValueError(f"{name} gives highway layers {model.layers}, the "
                             f"checkpoint was trained on {layers}")
        self.policy = VAPolicy(model, args["chunk"], args["dim"], args["heads"],
                               args["diffusion_steps"])
        self.policy.expert.load_state_dict(saved["expert"])
        self.policy.to(self.device).eval()
        self.mean = np.asarray(saved["stats"]["mean"], dtype=np.float32)
        self.std = np.asarray(saved["stats"]["std"], dtype=np.float32)
        self.depth_range_m = tuple(args.get("depth_range_m", (0.2, 1.2)))
        self.chunk = args["chunk"]
        self.step = int(saved.get("step", 0))

    def _batch(self, rgb, depth_mm, qpos):
        rgb = np.asarray(rgb)
        depth_mm = np.asarray(depth_mm)
        cams = len(CAMERAS)
        if rgb.shape[-4:] != (cams, *IMAGE_HW, 3) or rgb.dtype != np.uint8:
            raise ValueError(f"rgb must be uint8 ({cams}, {IMAGE_HW[0]}, {IMAGE_HW[1]}, 3), "
                             f"chest then front; got {rgb.dtype} {rgb.shape}")
        if depth_mm.shape[-3:] != (cams, *IMAGE_HW):
            raise ValueError(f"depth must be ({cams}, {IMAGE_HW[0]}, {IMAGE_HW[1]}) in mm; "
                             f"got {depth_mm.shape}")
        if rgb.ndim == 4:                       # one observation
            rgb, depth_mm, qpos = rgb[None], depth_mm[None], np.asarray(qpos)[None]
        state = (np.asarray(qpos, dtype=np.float32) - self.mean) / self.std
        batch = {"rgb": torch.from_numpy(np.ascontiguousarray(rgb)),
                 "depth": torch.from_numpy(depth_input(depth_mm, self.depth_range_m)),
                 "state": torch.from_numpy(state.astype(np.float32))}
        if self.prepare is not None:
            batch["backbone"] = self.prepare(batch["rgb"])
        return _to(batch, self.device)

    @torch.no_grad()
    def predict(self, rgb, depth_mm, qpos):
        """-> (Ta, 16) for one observation, (B, Ta, 16) for a batch of them."""
        single = np.asarray(rgb).ndim == 4
        batch = self._batch(rgb, depth_mm, qpos)
        with torch.autocast(self.device.type, dtype=self.dtype,
                            enabled=self.dtype != torch.float32):
            chunk = self.policy.sample(batch, steps=self.sample_steps)
        chunk = chunk.float().cpu().numpy() * self.std + self.mean
        return chunk[0] if single else chunk


def _to(batch, device):
    return {k: _to(v, device) if isinstance(v, dict) else v.to(device)
            for k, v in batch.items()}


def run_episode(policy, path, every):
    """Predict every `every` steps of a recorded episode and compare each
    chunk with the qpos the arm actually reached.
    -> (metrics, steps, chunks (N, Ta, 16), qpos)."""
    with h5py.File(path, "r") as f:
        qpos = f["observations/qpos"][:]
        images = f["observations/images"]
        length = len(qpos)
        steps = list(range(0, length - 1, every))
        chunks, times = [], []
        for t in steps:
            rgb = np.stack([images[f"{c}_rgb"][t] for c in CAMERAS])
            depth = np.stack([images[f"{c}_depth"][t] for c in CAMERAS])
            started = time.perf_counter()
            chunks.append(policy.predict(rgb, depth, qpos[t]))
            times.append(time.perf_counter() - started)

    arm, travel = working_arm(qpos)
    metrics = {"ms_per_chunk": float(np.median(times) * 1000)}
    print(f"{path}: a {ARM_NAMES[arm]}-arm grasp (travel left {travel[0]:.2f}, right "
          f"{travel[1]:.2f} rad); {len(steps)} chunks, {metrics['ms_per_chunk']:.0f} ms each "
          f"(median, the first {times[0] * 1000:.0f} ms)")
    for h in sorted({1, 10, policy.chunk} & set(range(1, policy.chunk + 1))):
        joint, grip = [], []
        for t, chunk in zip(steps, chunks):
            if t + h < length:
                err = np.abs(chunk[h - 1] - qpos[t + h])
                joint.append(err[ARM_JOINTS].mean())
                grip.append(err[GRIPPERS].mean())
        metrics[f"joint_rad_{h}_ahead"] = float(np.mean(joint))
        metrics[f"gripper_mm_{h}_ahead"] = float(np.mean(grip) * 1000)
        print(f"  {h:>3} steps ahead ({h * 20} ms): joints {np.mean(joint):.4f} rad, "
              f"grippers {np.mean(grip) * 1000:.2f} mm")
    # Over every row of every chunk: the grasping arm, and the one to keep still.
    working, idle = [], []
    for t, chunk in zip(steps, chunks):
        rows = min(len(chunk), length - 1 - t)
        err = np.abs(chunk[:rows] - qpos[t + 1:t + 1 + rows])
        working.append(err[:, ARM_JOINT_COLUMNS[arm]].mean())
        idle.append(err[:, ARM_JOINT_COLUMNS[1 - arm]].mean())
    metrics["working_arm_rad"] = float(np.mean(working))
    metrics["idle_arm_rad"] = float(np.mean(idle))
    print(f"  whole chunks: working ({ARM_NAMES[arm]}) arm {metrics['working_arm_rad']:.4f} "
          f"rad, idle arm {metrics['idle_arm_rad']:.4f} rad")
    return metrics, np.asarray(steps), np.stack(chunks), qpos, arm


def episode_figure(steps, chunks, qpos, title, arm):
    """The working arm over the whole episode: the recording dashed, each
    predicted chunk drawn from the step it was predicted at."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharex=True)
    for ax, j in zip(axes.flat, range(8 * arm, 8 * arm + 8)):
        ax.plot(qpos[:, j], "--", color="0.4", linewidth=1)
        for t, chunk in zip(steps, chunks):
            ax.plot(np.arange(t + 1, t + 1 + len(chunk)), chunk[:, j],
                    color="tab:blue", linewidth=1, alpha=0.8)
        ax.set_title(f"{JOINT_NAMES[j]} ({'m' if j in (7, 15) else 'rad'})", fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("step (20 ms)")
    fig.suptitle(f"{title}, {ARM_NAMES[arm]} arm: recorded (dashed), predicted chunks (blue)",
                 fontsize=11)
    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episode", required=True,
                   help="an episode_*.hdf5, or a folder of them (searched recursively)")
    p.add_argument("--every", type=int, default=25, help="predict every N steps")
    p.add_argument("--backbone", default=None, help="override the saved backbone name/path")
    p.add_argument("--sample-steps", type=int, default=10, help="DDIM steps")
    p.add_argument("--dtype", choices=DTYPES, default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tensorboard", default="", help="write the evaluation to this log dir")
    p.add_argument("--out", default="", help="save each episode's chunks to <out>/<name>.npz")
    args = p.parse_args()

    episodes = (list_episodes(args.episode) if os.path.isdir(args.episode)
                else [args.episode])
    policy = VAInference(args.checkpoint, args.backbone, args.device, args.sample_steps,
                         args.dtype)
    print(f"{args.checkpoint}: trained {policy.step} steps; {len(episodes)} episode(s)")
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
    if args.out:
        os.makedirs(args.out, exist_ok=True)

    every_metrics = []
    for path in episodes:
        name = os.path.splitext(os.path.basename(path))[0]
        metrics, steps, chunks, qpos, arm = run_episode(policy, path, args.every)
        every_metrics.append(metrics)
        if args.out:
            np.savez(os.path.join(args.out, f"{name}.npz"), steps=steps, chunks=chunks,
                     qpos=qpos)
        if args.tensorboard:
            with SummaryWriter(os.path.join(args.tensorboard, name)) as writer:
                for key, value in metrics.items():
                    writer.add_scalar(f"eval/{key}", value, policy.step)
                writer.add_figure("eval/working_arm_episode",
                                  episode_figure(steps, chunks, qpos, name, arm), policy.step)
    mean = {key: float(np.mean([m[key] for m in every_metrics])) for key in every_metrics[0]}
    if len(episodes) > 1:
        print("mean over episodes: " + ", ".join(f"{k} {v:.4g}" for k, v in mean.items()))
    if args.tensorboard:
        with SummaryWriter(os.path.join(args.tensorboard, "all")) as writer:
            for key, value in mean.items():
                writer.add_scalar(f"eval/{key}", value, policy.step)
            writer.add_text("eval/checkpoint", f"{args.checkpoint} (step {policy.step}), "
                            f"{len(episodes)} episodes, every {args.every} steps, "
                            f"{args.sample_steps} DDIM steps")
        print(f"tensorboard --logdir {args.tensorboard}")


if __name__ == "__main__":
    main()

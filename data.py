"""The dashboard's HDF5 episodes as training samples for the action expert.

One sample is one step t of one episode:

    rgb      (2, 480, 640, 3) uint8    chest, then front
    depth    (2, 2, 480, 640) float32  per camera: depth in 0..1 over
                                       depth_range_m, and a valid mask (the
                                       sensor writes 0 where it measured
                                       nothing; those pixels are 0 in both)
    state    (16,)  float32            qpos[t], normalised
    actions  (Ta, 16) float32          qpos[t+1 : t+1+Ta], normalised; the
                                       last row repeats past the episode end
    is_pad   (Ta,)  bool               those repeated rows, left out of the loss

The label is the measured qpos ahead, not the file's `action`: the recorder
fills `action` from qpos[t] itself, so it holds where the arm already is.

The 16 values are left_j1..j7, left_gripper, right_j1..j7, right_gripper
(the file's joint_names_json); the model splits them into 2 arms x 8.

Two kinds of episode are trained together: a grasp with the right arm (the
left one held still) and one with the left. Nothing in a file says which,
so the arm whose joints travel further is taken as the working one
(working_arm), each sample carries it as `arm` (0 left, 1 right) for the
sampler and the metrics, and the model never sees it: on the robot there is
no such input. BalancedBatchSampler fills every batch half and half.
"""

import glob
import json
import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

CAMERAS = ("chest", "front")
NUM_JOINTS = 16
NUM_ARMS = 2
ARM_DIM = 8
IMAGE_HW = (480, 640)
ARM_NAMES = ("left", "right")
ARM_JOINT_COLUMNS = (list(range(0, 7)), list(range(8, 15)))     # the 7 joints, no gripper
GRIPPER_COLUMNS = (7, 15)
JOINT_NAMES = [f"{side}_{name}" for side in ("left", "right")
               for name in [f"j{i}" for i in range(1, 8)] + ["gripper"]]


def list_episodes(root):
    """Every whole episode under `root`, sorted by name (= by time)."""
    paths = sorted(glob.glob(os.path.join(root, "**", "*.hdf5"), recursive=True))
    episodes = []
    for path in paths:
        if path.endswith(".partial.hdf5"):
            continue
        with h5py.File(path, "r") as f:
            if not bool(f.attrs.get("complete", True)):
                print(f"skip {path}: not complete")
                continue
            names = json.loads(f.attrs.get("joint_names_json", "[]"))
            if names and names != JOINT_NAMES:
                raise ValueError(f"{path}: joints are {names}, expected {JOINT_NAMES}")
            for camera in CAMERAS:
                for kind in ("rgb", "depth"):
                    key = f"observations/images/{camera}_{kind}"
                    if key not in f:
                        raise ValueError(f"{path}: no {key}")
                    if tuple(f[key].shape[1:3]) != IMAGE_HW:
                        raise ValueError(f"{path}: {key} is {f[key].shape[1:3]}, "
                                         f"expected {IMAGE_HW}")
            virtual = json.loads(f.attrs.get("virtual_cameras_json", "[]"))
            if virtual:
                print(f"warning {path}: {virtual} recorded black frames")
        episodes.append(path)
    if not episodes:
        raise FileNotFoundError(f"no episodes under {root}")
    return episodes


def working_arm(qpos):
    """(arm, travel): the arm whose 7 joints travel further over the episode
    (0 left, 1 right), and each arm's travel in radians (sum of per-joint
    ranges)."""
    travel = [float(np.ptp(qpos[:, cols], axis=0).sum()) for cols in ARM_JOINT_COLUMNS]
    return int(np.argmax(travel)), travel


def episode_arms(paths, ambiguous_ratio=3.0):
    """The working arm of each episode, with a report: an episode where the
    other arm also travelled more than 1/ambiguous_ratio as far is listed,
    since it may be a two-handed one or a mislabelled grasp."""
    arms = []
    for path in paths:
        with h5py.File(path, "r") as f:
            arm, travel = working_arm(f["observations/qpos"][:])
        arms.append(arm)
        other = travel[1 - arm]
        if other > 0 and travel[arm] / other < ambiguous_ratio:
            print(f"  check {os.path.basename(path)}: {ARM_NAMES[arm]} arm taken as working, "
                  f"but left travels {travel[0]:.2f} rad, right {travel[1]:.2f} rad")
    counts = {name: arms.count(i) for i, name in enumerate(ARM_NAMES)}
    print(f"episodes by working arm: {counts['left']} left, {counts['right']} right")
    return arms


def split_episodes(paths, arms, val_count):
    """Hold out `val_count` episodes, the newest of each kind, split as
    evenly as the kinds allow, so the held-out set has both."""
    kinds = sorted(set(arms))
    per_kind = {kind: [p for p, a in zip(paths, arms) if a == kind] for kind in kinds}
    take = {kind: 0 for kind in kinds}
    for i in range(val_count):
        order = sorted(kinds, key=lambda k: (take[k], k))
        kind = next((k for k in order if take[k] < len(per_kind[k]) - 1), None)
        if kind is None:
            raise ValueError(f"too few episodes to hold {val_count} out and keep some to train")
        take[kind] += 1
    val = {p for kind in kinds for p in per_kind[kind][len(per_kind[kind]) - take[kind]:]}
    train = [(p, a) for p, a in zip(paths, arms) if p not in val]
    held = [(p, a) for p, a in zip(paths, arms) if p in val]
    return train, held


def compute_stats(paths, min_std_joint=1e-2, min_std_gripper=2e-3):
    """Per-joint mean/std of qpos over the episodes, both kinds together.

    The std is floored: an arm that hardly moves would otherwise blow sensor
    noise up into large values. The floors are in the joints' units: 0.01
    rad, and 2 mm for a gripper, whose whole stroke is under 2 cm.
    """
    qpos = np.concatenate([h5py.File(p, "r")["observations/qpos"][:] for p in paths])
    floor = np.full(NUM_JOINTS, min_std_joint, dtype=np.float64)
    floor[list(GRIPPER_COLUMNS)] = min_std_gripper
    return {"mean": qpos.mean(0).tolist(),
            "std": np.maximum(qpos.std(0), floor).tolist(),
            "joint_names": JOINT_NAMES}


def depth_input(depth_mm, depth_range_m):
    """uint16 mm [..., H, W] -> float32 [..., 2, H, W]: normalised depth, valid."""
    low, high = depth_range_m
    metres = depth_mm.astype(np.float32) / 1000.0
    valid = metres > 0
    norm = (np.clip(metres, low, high) - low) / (high - low)
    norm[~valid] = 0.0
    return np.stack([norm, valid.astype(np.float32)], axis=-3)


def jitter_colour(rgb, rng, strength=0.2):
    """(H, W, 3) uint8 with its saturation, contrast and brightness each
    scaled by a factor in 1 ± strength.

    The three are linear, so they fold into one pass: saturation s gives
    gray + s (x - gray), whose mean is still the frame's mean m; contrast c
    gives c (y - m) + m and brightness b scales that, so the result is
    b c s x + b c (1 - s) gray + b (1 - c) m.
    """
    s, c, b = rng.uniform(1 - strength, 1 + strength, size=3)
    x = rgb.astype(np.float32)
    gray = (x[..., :1] + x[..., 1:2] + x[..., 2:]) / 3
    m = float(gray.mean())
    gray *= b * c * (1 - s)
    gray += b * (1 - c) * m
    x *= b * c * s
    x += gray
    np.clip(x, 0, 255, out=x)
    return x.astype(np.uint8)


def shift(image, dy, dx, rows_axis, mode):
    """`image` moved dy px down and dx right, its size kept: rows and columns
    are axes rows_axis and rows_axis + 1, and the uncovered edge is filled
    as np.pad's `mode` fills it."""
    s = max(abs(dy), abs(dx))
    pad = [(0, 0)] * image.ndim
    pad[rows_axis] = pad[rows_axis + 1] = (s, s)
    padded = np.pad(image, pad, mode=mode)
    h, w = image.shape[rows_axis], image.shape[rows_axis + 1]
    window = [slice(None)] * image.ndim
    window[rows_axis] = slice(s - dy, s - dy + h)
    window[rows_axis + 1] = slice(s - dx, s - dx + w)
    return padded[tuple(window)]


class EpisodeDataset(Dataset):
    """`augment`: each camera's frame colour-jittered and shifted up to
    `shift_px` each way, its depth by the same amount so a depth token stays
    on its image token's cell. For training only."""

    def __init__(self, paths, stats, chunk=50, depth_range_m=(0.2, 1.2), arms=None,
                 augment=False, shift_px=16):
        self.paths = list(paths)
        self.arms = list(arms) if arms is not None else [
            working_arm(h5py.File(p, "r")["observations/qpos"][:])[0] for p in self.paths]
        self.chunk = int(chunk)
        self.depth_range_m = tuple(depth_range_m)
        self.mean = np.asarray(stats["mean"], dtype=np.float32)
        self.std = np.asarray(stats["std"], dtype=np.float32)
        self.index = []
        for i, path in enumerate(self.paths):
            with h5py.File(path, "r") as f:
                length = f["observations/qpos"].shape[0]
            # The last step has no future to learn.
            self.index.extend((i, t) for t in range(length - 1))
        self.augment = augment
        self.shift_px = int(shift_px)
        self._files = {}
        self._pid = None
        self._rng = None

    def __len__(self):
        return len(self.index)

    def groups(self):
        """{arm: [sample indices]} of the episodes of each kind."""
        out = {}
        for item, (i, _) in enumerate(self.index):
            out.setdefault(self.arms[i], []).append(item)
        return out

    def _file(self, i):
        # h5py handles do not survive a fork: each worker opens its own. Its
        # draws too, from its torch seed, which differs per worker and run.
        if self._pid != os.getpid():
            self._files, self._pid = {}, os.getpid()
            self._rng = np.random.default_rng(torch.initial_seed())
        if i not in self._files:
            self._files[i] = h5py.File(self.paths[i], "r")
        return self._files[i]

    def normalise(self, qpos):
        return (qpos - self.mean) / self.std

    def __getitem__(self, item):
        i, t = self.index[item]
        f = self._file(i)
        qpos = f["observations/qpos"]
        future = qpos[t + 1:t + 1 + self.chunk]
        is_pad = np.zeros(self.chunk, dtype=bool)
        if len(future) < self.chunk:
            is_pad[len(future):] = True
            future = np.concatenate(
                [future, np.repeat(future[-1:], self.chunk - len(future), axis=0)])
        images = f["observations/images"]
        rgb = [images[f"{c}_rgb"][t] for c in CAMERAS]
        depth = [depth_input(images[f"{c}_depth"][t], self.depth_range_m) for c in CAMERAS]
        if self.augment:
            for c in range(len(CAMERAS)):
                dy, dx = (int(v) for v in self._rng.integers(
                    -self.shift_px, self.shift_px + 1, size=2))
                rgb[c] = shift(jitter_colour(rgb[c], self._rng), dy, dx, 0, "edge")
                # 0 is "not measured", in both the depth and its valid mask.
                depth[c] = shift(depth[c], dy, dx, 1, "constant")
        rgb, depth = np.stack(rgb), np.stack(depth)
        return {
            "rgb": torch.from_numpy(rgb),
            "depth": torch.from_numpy(depth),
            "state": torch.from_numpy(self.normalise(qpos[t]).astype(np.float32)),
            "actions": torch.from_numpy(self.normalise(future).astype(np.float32)),
            "is_pad": torch.from_numpy(is_pad),
            "arm": torch.tensor(self.arms[i]),
        }


class BalancedBatchSampler:
    """Batches half from each kind of episode (left-arm grasps, right-arm
    grasps), whatever their counts: drawn at random within each kind, the
    smaller kind's frames coming round more often. One pass is as many
    batches as the frames fill. With one kind only, a plain shuffle."""

    def __init__(self, groups, batch_size, seed=0):
        self.groups = [list(g) for g in groups.values() if g]
        self.batch_size = batch_size
        self.total = sum(len(g) for g in self.groups)
        self.rng = random.Random(seed)

    def __len__(self):
        return max(1, self.total // self.batch_size)

    def _stream(self, group):
        while True:
            order = group[:]
            self.rng.shuffle(order)
            yield from order

    def __iter__(self):
        streams = [self._stream(g) for g in self.groups]
        shares = [self.batch_size // len(streams)] * len(streams)
        for i in range(self.batch_size - sum(shares)):
            shares[i] += 1
        for _ in range(len(self)):
            batch = [next(s) for s, n in zip(streams, shares) for _ in range(n)]
            self.rng.shuffle(batch)
            yield batch


class Collate:
    """Default collation, plus the backbone's own inputs built from the RGB
    frames here, in the loader's workers, rather than on the training loop."""

    def __init__(self, prepare=None):
        self.prepare = prepare

    def __call__(self, samples):
        batch = {key: torch.stack([s[key] for s in samples]) for key in samples[0]}
        if self.prepare is not None:
            batch["backbone"] = self.prepare(batch["rgb"])
        return batch

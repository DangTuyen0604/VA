# openarm_va: the action expert on OpenArm episodes

Trains the action expert (VA) of the diagram on the episodes the dashboard
records (`dataset/<day>/episode_*.hdf5`): a frozen Qwen3-VL over the chest
and front RGB frames, a depth encoder on the same 32 px grid, and a
diffusion transformer that predicts the next `Ta` joint targets of both
arms, `(B, 2, Ta, 8)`.

Runs on a GPU machine, not the robot. Copy this folder and the episodes
there.

```bash
pip install -r requirements.txt          # torch for the machine's CUDA first

# Check the data and the code, no download, a minute on a CPU:
python train.py --data /data/openarm --out runs/smoke --backbone dummy \
    --steps 20 --batch 2 --device cpu --workers 0

# Train (downloads Qwen/Qwen3-VL-2B-Instruct, ~4.5 GB, on the first run):
python train.py --data /data/openarm --out runs/va1
```

`runs/va1/` then holds `best.pt` (lowest joint error on the held-out
episodes), `last.pt`, `stats.json`, `split.json` (which episodes trained,
which were held out, and each one's working arm), `log.jsonl` and `tb/`.
`--resume runs/va1/last.pt` continues a run.

## Left-arm and right-arm grasps

The episodes are of two kinds: a grasp with the right arm, the left held
still, and one with the left. They train together, into one model that
always predicts both arms; it has to tell from the images which arm should
move. Nothing in a file says which kind it is, so the arm whose joints
travel further is taken as the working one. The start of a run prints the
count of each and lists any episode where the other arm also moved a good
deal (`check ...`): look at those.

- Every batch is half left-arm, half right-arm grasps, however uneven the
  counts (`--no-balance` for a plain shuffle).
- `--val-episodes 2` holds out the newest of each kind; an even number keeps
  the two kinds level.
- The mean/std are taken over both kinds, so each arm's std is its real
  range of motion, floored at `--min-std-joint 0.01` rad and
  `--min-std-gripper 0.002` m.
- On the robot, run each chunk to its end before asking for the next.
  Averaging overlapping chunks can blend a left-arm chunk with a right-arm
  one when the object sits between the two.

## TensorBoard

```bash
tensorboard --logdir runs/va1/tb          # or runs/ to compare runs
```

| Tag | Every | What |
|---|---|---|
| `train/loss`, `train/grad`, `train/lr`, `train/steps_per_s`, `train/gpu_mem_gb` | `--log-every` | diffusion loss, gradient norm before clipping, peak GPU memory |
| `val/loss` | `--eval-every` | the same loss on the held-out episodes |
| `val/joint_rad`, `val/gripper_mm` | `--eval-every` | sampled chunks against the recorded qpos, over the whole chunk |
| `val/joint_rad_{1,10,50}_ahead` | `--eval-every` | the joint error at 20 ms, 200 ms and 1 s ahead |
| `val/working_arm_rad`, `val/idle_arm_rad` | `--eval-every` | the grasping arm's error, and the still arm's (it should stay near 0) |
| `val/joint_rad_left_grasps`, `val/joint_rad_right_grasps` | `--eval-every` | the error on each kind of episode |
| `val/working_arm_chunk` (Images) | `--eval-every` | one held-out chunk of the arm doing the grasp: prediction, recording, start |
| `args` (Text) | once | the run's settings |

`--no-tensorboard` keeps to `log.jsonl`.

## Inference

```bash
# How far the chunks land from what the arm did, on recorded episodes
# (a file, or a folder searched recursively):
python infer.py --checkpoint runs/va1/best.pt --episode /data/val --every 25 \
    --tensorboard runs/va1/tb_eval --out runs/va1/eval
```

prints, per episode, its working arm, the error 1, 10 and 50 steps ahead,
the working and the idle arm's error, and the time per chunk.
`--tensorboard` logs the same per episode (one run each, `all` for the mean)
at the checkpoint's training step, with a plot of the working arm's chunks
over the whole episode; `--out` keeps each episode's chunks as `.npz`. In code:

```python
from infer import VAInference
policy = VAInference("runs/va1/best.pt")        # dtype="fp16" before Ampere
chunk = policy.predict(rgb, depth_mm, qpos)
# rgb (2, 480, 640, 3) uint8, depth_mm (2, 480, 640) uint16, chest then front,
# qpos (16,) -> chunk (50, 16): absolute targets for t+1 .. t+50, radians,
# grippers in metres, in the dataset's joint order.
```

`--sample-steps` (DDIM, 10 by default) trades speed for smoothness.

## What it reads

| HDF5 key | Used as |
|---|---|
| `observations/images/{chest,front}_rgb` (T, 480, 640, 3) | Qwen3-VL, chest first |
| `observations/images/{chest,front}_depth` (T, 480, 640) mm | depth encoder: 0.2–1.2 m, 0 = not measured |
| `observations/qpos[t]` (T, 16) | state |
| `observations/qpos[t+1 : t+1+Ta]` | label |

The file's `action` is not used: the recorder fills it from `qpos[t]`, so it
holds where the arm already is. Each episode's last `Ta` steps repeat the
final pose and are left out of the loss.

The 16 values are `left_j1..j7, left_gripper, right_j1..j7, right_gripper`,
normalised by the mean/std of the training episodes (`stats.json`) and split
into 2 arms × 8.

## Settings worth knowing

- `--backbone`: any Qwen3-VL id or local folder. The expert gets half its
  layers as blocks (`--blocks` to choose), block *i* fed from layer 2*i*.
- `--chunk 50`: 1 s at 50 Hz.
- `--batch 8`: not measured on a GPU yet; lower it if it runs out of
  memory. Qwen runs in bf16 without gradients; only the expert trains.
- `--val-episodes 2`: the newest of each kind of episode is held out.
  Split by episode, never by frame.
- `--ema 0.9999`: the expert's weights are averaged as it trains; evaluation,
  `best.pt` and `infer.py` use the average. `0` turns it off.
- Training frames are colour-jittered and shifted up to `--shift-px 16`,
  depth with them; `--no-augment` trains on them as recorded. Evaluation
  grades the same held-out batches under the same noise every time.

## Departures from the paper design

- PRoPE needs each camera's intrinsics and extrinsics, which the episodes
  do not carry. Image tokens get a 2D sin-cos position and a camera
  embedding instead; action tokens a learned embedding of their step.
- Output is absolute joint targets, 8 per arm, not the end-effector pose:
  no camera extrinsics to put a pose in a camera frame. So `act_head` is
  768→8, there is no `abs_pos_enc`, and the APT checkpoint does not load.




  # 0. Lấy code (thay cho scp)
cd ~/hoanglmv
git clone https://github.com/DangTuyen0604/VA.git      # lần sau: cd VA && git pull
cd VA && mkdir -p runs

# 1. Kiểm tra môi trường: cần cuda True, transformers >= 4.57, GPU 1 trống
python -c "import torch, transformers; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| GPUs', torch.cuda.device_count(), '| transformers', transformers.__version__)"
nvidia-smi

# 2b. Danh sách tay của từng episode (mục 2)
python - <<'EOF' | tee runs/arms.txt
import glob, h5py
from data import working_arm
for p in sorted(glob.glob("/home/toann/hoanglmv/dataset/2026-10-13/*.hdf5")):
    arm, travel = working_arm(h5py.File(p, "r")["observations/qpos"][:])print(p.split("/")[-1], ["LEFT ", "RIGHT"][arm], f"left {travel[0]:.2f} rad, right {travel[1]:.2f} rad")
EOF

# 3. Chạy thử Qwen 25 bước (mục 1 và 3), nên chạy trong tmux
CUDA_VISIBLE_DEVICES=1 python -u train.py \
    --data /home/toann/hoanglmv/dataset/2026-10-13 \
    --out runs/check_qwen \
    --backbone Qwen/Qwen3-VL-2B-Instruct \
    --batch 32 --steps 25 --log-every 5 \
    --eval-every 20 --eval-batches 2 \
    --val-episodes 4 --workers 4 --device cuda \
    2>&1 | tee runs/check_qwen.out

# 4. Gửi mình
head -60 runs/check_qwen.out
cat runs/arms.txt


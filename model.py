"""The action expert (VA): a frozen Qwen3-VL sees the two RGB frames, a small
encoder the two depth frames, and a diffusion transformer predicts the noise
on the next Ta joint targets of both arms.

    rgb (2 cams) --Qwen3-VL, frozen--> hidden[0] ----------+
                                     hidden[l_i] -> highway i
    depth (2 cams) --conv 32/32--> tokens on the same 15x20 grid
    vl0 = proj(hidden[0]) + depth + xy + camera id   (2 x 300 tokens)

    blocks i = 1..n:  vision <- gate(proj_vl[i](hidden[l_i]), vision)
                      [vision | state 2 | action 2*Ta] -> attention, MLP,
                      both under AdaRMSNorm (FiLM) on the diffusion step
    head: RMSNorm -> Linear 768->8 on the action tokens -> eps (B, 2, Ta, 8)

Each Qwen3-VL image token covers 32x32 px (patch 16, merged 2x2), so a
640x480 frame gives 20x15 = 300 tokens, in raster order: token k is row
k // 20, column k % 20, the same cell as depth token k.

The paper design positions tokens with PRoPE, which needs each camera's
intrinsics and extrinsics. The episodes carry none, so the image tokens get
a 2D sin-cos position and a camera embedding instead, and the action tokens
a learned embedding of their step.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import ARM_DIM, CAMERAS, IMAGE_HW, NUM_ARMS

CELL = 32                                   # px per image token
GRID = (IMAGE_HW[0] // CELL, IMAGE_HW[1] // CELL)      # (15, 20)
TOKENS_PER_IMAGE = GRID[0] * GRID[1]                   # 300


def highway_layers(num_layers, count):
    """`count` of the backbone's layers 1..num_layers, evenly spread, the
    last always among them: 28 layers, 14 blocks -> 2, 4, ..., 28."""
    return [max(1, round(num_layers * (i + 1) / count)) for i in range(count)]


# ----------------------------------------------------------------- backbones

def download(name, every=5.0):
    """Fetch a hub model into the Hugging Face cache, a line every `every` s:
    GB so far, %, MB/s, time left. The hub's own bars redraw with \\r, so a
    pipe or a log file shows none of them; these lines survive both. A local
    folder, or a model already cached, returns at once."""
    import os
    import threading
    import time

    from huggingface_hub import HfApi, constants, snapshot_download
    from huggingface_hub.utils import disable_progress_bars, enable_progress_bars

    if os.path.isdir(name):
        return
    try:
        snapshot_download(name, local_files_only=True)
        return
    except Exception:
        pass
    try:
        info = HfApi().model_info(name, files_metadata=True)
        total = sum(f.size or 0 for f in info.siblings) or None
    except Exception:
        total = None                        # offline or gated: let the download say
    blobs = os.path.join(constants.HF_HUB_CACHE,
                         "models--" + name.replace("/", "--"), "blobs")

    def done():
        # Blocks on disk, not st_size: a file written out of order (Xet) is sparse.
        if not os.path.isdir(blobs):
            return 0
        sizes = (os.stat(e.path) for e in os.scandir(blobs) if e.is_file())
        return sum(min(s.st_size, s.st_blocks * 512) for s in sizes)

    def report(stop):
        last, last_t = done(), time.time()
        while not stop.wait(every):
            now, now_t = done(), time.time()
            rate = (now - last) / (now_t - last_t)
            last, last_t = now, now_t
            line = f"  {name}: {now / 1e9:.2f}"
            if total:
                line += f" / {total / 1e9:.2f} GB ({100 * now / total:.0f}%)"
            else:
                line += " GB"
            line += f", {rate / 1e6:.0f} MB/s"
            if total and rate > 0:
                left = max(0, total - now) / rate
                line += f", ~{left / 60:.0f} min left" if left >= 60 else f", ~{left:.0f} s left"
            print(line, flush=True)

    size = f"{total / 1e9:.2f} GB" if total else "size unknown"
    print(f"downloading {name} ({size}) into {constants.HF_HUB_CACHE}", flush=True)
    started = time.time()
    stop = threading.Event()
    thread = threading.Thread(target=report, args=(stop,), daemon=True)
    disable_progress_bars()
    thread.start()
    try:
        snapshot_download(name)
    finally:
        stop.set()
        thread.join()
        enable_progress_bars()
    print(f"downloaded {name} in {time.time() - started:.0f} s", flush=True)


class QwenInputs:
    """Qwen3-VL's processor, held at 640x480: run in the loader's workers.

    The chat holds the two images and nothing else, chest first. A frame is
    never resized: min and max pixels are both 640x480, which is a multiple
    of 32 each way, and every batch is checked to have come out 30x40 patches.
    """

    def __init__(self, name):
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(name)
        image = self.processor.image_processor
        pixels = IMAGE_HW[0] * IMAGE_HW[1]
        image.size = {"shortest_edge": pixels, "longest_edge": pixels}
        for attr in ("min_pixels", "max_pixels"):
            if hasattr(image, attr):
                setattr(image, attr, pixels)
        if image.patch_size * image.merge_size != CELL:
            raise ValueError(f"{name}: a token covers {image.patch_size} x "
                             f"{image.merge_size} px, not {CELL}")
        self.grid_thw = [1, IMAGE_HW[0] // image.patch_size,
                         IMAGE_HW[1] // image.patch_size]
        chat = [{"role": "user", "content": [{"type": "image"} for _ in CAMERAS]}]
        self.text = self.processor.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=False)

    def __call__(self, rgb):
        from PIL import Image

        images = [Image.fromarray(frame.numpy()) for sample in rgb for frame in sample]
        out = self.processor(text=[self.text] * len(rgb), images=images,
                             return_tensors="pt")
        if not (out["image_grid_thw"] == torch.tensor(self.grid_thw)).all():
            raise ValueError(f"the processor resized the frames: grid "
                             f"{out['image_grid_thw'][0].tolist()}, expected {self.grid_thw}")
        return {key: out[key] for key in
                ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}


class QwenBackbone(nn.Module):
    """Qwen3-VL without its LM head, frozen: the image tokens' hidden states
    at the input (vl0's source) and at the highway layers."""

    def __init__(self, name, num_highways, dtype=torch.bfloat16):
        super().__init__()
        from transformers import Qwen3VLForConditionalGeneration

        download(name)
        full =Qwen3VLForConditionalGeneration.from_pretrained(name, dtype=dtype)
        self.model = full.model
        self.model.requires_grad_(False)
        self.model.eval()
        text = full.config.text_config
        self.dim = text.hidden_size
        self.layers = highway_layers(text.num_hidden_layers, num_highways)
        self.image_token_id = full.config.image_token_id
        self.dtype = dtype

    def train(self, mode=True):
        return super().train(False)         # frozen: never dropout

    @torch.no_grad()
    def forward(self, batch):
        inputs = dict(batch["backbone"])
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        out = self.model(**inputs, output_hidden_states=True, use_cache=False)
        mask = inputs["input_ids"] == self.image_token_id
        count = len(CAMERAS) * TOKENS_PER_IMAGE
        if not (mask.sum(1) == count).all():
            raise ValueError(f"expected {count} image tokens a sample, got "
                             f"{mask.sum(1).tolist()}")

        def take(hidden):
            return hidden[mask].view(mask.shape[0], count, -1).float()

        return take(out.hidden_states[0]), [take(out.hidden_states[l]) for l in self.layers]


class DummyBackbone(nn.Module):
    """A frozen random stand-in with Qwen3-VL's token layout, for checking the
    data and the expert end to end on a CPU without downloading anything."""

    def __init__(self, num_highways, dim=64, num_layers=8):
        super().__init__()
        self.dim = dim
        self.layers = highway_layers(num_layers, num_highways)
        self.patch = nn.Conv2d(3, dim, CELL, CELL)
        self.mix = nn.ModuleList(nn.Linear(dim, dim) for _ in range(num_layers))
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, batch):
        rgb = batch["rgb"]                                  # (B, C, H, W, 3)
        b, c = rgb.shape[:2]
        x = rgb.flatten(0, 1).permute(0, 3, 1, 2).float() / 255.0
        x = self.patch(x).flatten(2).transpose(1, 2).reshape(b, c * TOKENS_PER_IMAGE, -1)
        hidden = [x]
        for layer in self.mix:
            hidden.append(torch.tanh(layer(hidden[-1])))
        return hidden[0], [hidden[l] for l in self.layers]


# -------------------------------------------------------------------- expert

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, affine=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x):
        # In fp32, back in the input's dtype: under bf16 autocast q and k
        # would otherwise come out fp32 beside a bf16 v, which attention refuses.
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            y = y * self.weight.float()
        return y.to(x.dtype)


def sincos_1d(positions, dim):
    half = dim // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    angle = positions.float()[:, None] * freq[None]
    return torch.cat([angle.sin(), angle.cos()], dim=1)


def sincos_2d(rows, cols, dim):
    """(rows*cols, dim): half the channels for the row, half for the column."""
    r, c = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
    return torch.cat([sincos_1d(r.flatten(), dim // 2),
                      sincos_1d(c.flatten(), dim // 2)], dim=1)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.q_norm = RMSNorm(dim // heads)
        self.k_norm = RMSNorm(dim // heads)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, d // self.heads).unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.out(x.transpose(1, 2).reshape(b, n, d))


class AttentionBlock(nn.Module):
    """Self-attention over the whole sequence, then an MLP, each under
    AdaRMSNorm: scale, shift and an output gate from the diffusion step,
    zero at start so every block begins as the identity."""

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim, affine=False)
        self.attn = Attention(dim, heads)
        self.norm2 = RMSNorm(dim, affine=False)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate="tanh"),
                                 nn.Linear(hidden, dim))
        self.film = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, cond):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.film(cond)[:, None].chunk(6, -1)
        x = x + gate1 * self.attn(self.norm1(x) * (1 + scale1) + shift1)
        return x + gate2 * self.mlp(self.norm2(x) * (1 + scale2) + shift2)


class GatedFusion(nn.Module):
    """vision <- vl*g + h*(1-g), g per token and channel from both."""

    def __init__(self, backbone_dim, dim):
        super().__init__()
        self.proj_vl = nn.Sequential(nn.LayerNorm(backbone_dim), nn.Linear(backbone_dim, dim))
        self.gate = nn.Linear(2 * dim, dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)      # g = 0.5 at start

    def forward(self, highway, h):
        vl = self.proj_vl(highway)
        g = torch.sigmoid(self.gate(torch.cat([vl, h], -1)))
        return vl * g + h * (1 - g)


def mlp(inp, dim):
    return nn.Sequential(nn.Linear(inp, dim), nn.SiLU(), nn.Linear(dim, dim))


class ActionExpert(nn.Module):
    def __init__(self, backbone_dim, num_blocks, chunk=50, dim=768, heads=12):
        super().__init__()
        self.chunk = chunk
        cams = len(CAMERAS)
        # Vision side.
        self.proj0 = nn.Sequential(nn.LayerNorm(backbone_dim), nn.Linear(backbone_dim, dim))
        self.depth = nn.Conv2d(2, dim, CELL, CELL)      # depth, valid -> one token a cell
        self.register_buffer("xy", sincos_2d(*GRID, dim).repeat(cams, 1), persistent=False)
        self.camera = nn.Embedding(cams, dim)
        self.register_buffer("camera_ids", torch.arange(cams).repeat_interleave(
            TOKENS_PER_IMAGE), persistent=False)
        self.fusion = nn.ModuleList(GatedFusion(backbone_dim, dim) for _ in range(num_blocks))
        # Action side.
        self.state_in = mlp(ARM_DIM, dim)
        self.action_in = mlp(ARM_DIM, dim)
        self.arm = nn.Embedding(NUM_ARMS, dim)
        self.step = nn.Parameter(sincos_1d(torch.arange(chunk), dim) * 0.02)
        self.kind = nn.Embedding(3, dim)                # vision, state, action
        # Diffusion step -> FiLM condition.
        self.time = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim), nn.SiLU())
        self.blocks = nn.ModuleList(AttentionBlock(dim, heads) for _ in range(num_blocks))
        self.head_norm = RMSNorm(dim)
        self.head = nn.Linear(dim, ARM_DIM)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def vision_tokens(self, vis0, depth):
        """vl0: image tokens + depth on the same cell + xy + camera id."""
        b, cams = depth.shape[:2]
        d = self.depth(depth.flatten(0, 1))             # (B*C, dim, 15, 20)
        d = d.flatten(2).transpose(1, 2).reshape(b, cams * TOKENS_PER_IMAGE, -1)
        return (self.proj0(vis0) + d + self.xy + self.camera(self.camera_ids)
                + self.kind.weight[0])

    def forward(self, vis0, highways, depth, state, noisy, k):
        """state (B, 2, 8), noisy (B, 2, Ta, 8), k (B,) -> eps (B, 2, Ta, 8)."""
        b, arms, ta, _ = noisy.shape
        vision = self.vision_tokens(vis0, depth)
        n_vision = vision.shape[1]
        arm = self.arm.weight                           # (2, dim)
        state_tok = self.state_in(state) + arm + self.kind.weight[1]
        action_tok = (self.action_in(noisy) + arm[:, None] + self.step[:ta]
                      + self.kind.weight[2]).flatten(1, 2)
        cond = self.time(sincos_1d(k, 256).to(vision.dtype))
        rest = torch.cat([state_tok, action_tok], 1)
        for fusion, block, highway in zip(self.fusion, self.blocks, highways):
            vision = fusion(highway, vision)
            x = block(torch.cat([vision, rest], 1), cond)
            vision, rest = x[:, :n_vision], x[:, n_vision:]
        actions = rest[:, arms:]                        # drop the state tokens
        return self.head(self.head_norm(actions)).view(b, arms, ta, ARM_DIM)


# ----------------------------------------------------------------- diffusion

class Diffusion:
    """DDPM training with a cosine schedule, DDIM sampling (deterministic)."""

    def __init__(self, steps=100):
        self.steps = steps
        s = 0.008
        t = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
        f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        betas = (1 - f[1:] / f[:-1]).clamp(max=0.999)
        self.alphas_cumprod = torch.cumprod(1 - betas, 0).float()

    def alpha(self, k, like):
        return self.alphas_cumprod.to(like.device)[k].view(-1, *[1] * (like.dim() - 1))

    def add_noise(self, x0, eps, k):
        a = self.alpha(k, x0)
        return a.sqrt() * x0 + (1 - a).sqrt() * eps

    def schedule(self, n):
        return torch.linspace(self.steps - 1, 0, n).round().long().tolist()


def to_arms(x):
    """(B, Ta, 16) -> (B, 2, Ta, 8); (B, 16) -> (B, 2, 8)."""
    if x.dim() == 2:
        return x.view(x.shape[0], NUM_ARMS, ARM_DIM)
    return x.view(x.shape[0], x.shape[1], NUM_ARMS, ARM_DIM).transpose(1, 2)


def from_arms(x):
    """(B, 2, Ta, 8) -> (B, Ta, 16)."""
    return x.transpose(1, 2).flatten(2)


class VAPolicy(nn.Module):
    def __init__(self, backbone, chunk=50, dim=768, heads=12, diffusion_steps=100):
        super().__init__()
        self.backbone = backbone
        self.expert = ActionExpert(backbone.dim, len(backbone.layers), chunk, dim, heads)
        self.diffusion = Diffusion(diffusion_steps)

    def loss(self, batch):
        vis0, highways = self.backbone(batch)
        x0 = to_arms(batch["actions"])
        eps = torch.randn_like(x0)
        k = torch.randint(0, self.diffusion.steps, (x0.shape[0],), device=x0.device)
        eps_hat = self.expert(vis0, highways, batch["depth"], to_arms(batch["state"]),
                              self.diffusion.add_noise(x0, eps, k), k)
        keep = (~batch["is_pad"])[:, None, :, None].float()     # (B, 1, Ta, 1)
        return ((eps_hat.float() - eps) ** 2 * keep).sum() / (keep.sum() * x0.shape[1]
                                                              * x0.shape[-1])

    @torch.no_grad()
    def sample(self, batch, steps=10, clip=6.0):
        """The next Ta steps, normalised: (B, Ta, 16)."""
        vis0, highways = self.backbone(batch)
        state = to_arms(batch["state"])
        b = state.shape[0]
        x = torch.randn(b, state.shape[1], self.expert.chunk, ARM_DIM, device=state.device)
        schedule = self.diffusion.schedule(steps)
        for i, k in enumerate(schedule):
            kk = torch.full((b,), k, device=x.device, dtype=torch.long)
            eps = self.expert(vis0, highways, batch["depth"], state, x, kk).float()
            a = self.diffusion.alpha(kk, x)
            x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-clip, clip)
            if i + 1 == len(schedule):
                x = x0
            else:
                a_next = self.diffusion.alpha(torch.full_like(kk, schedule[i + 1]), x)
                x = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps
        return from_arms(x)

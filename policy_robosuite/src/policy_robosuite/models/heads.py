"""Diffusion action head over the action chunk.

Adapted from diffusion_policy (Chi et al., 2023)
`diffusion_policy/model/diffusion/conditional_unet1d.py`: squaredcos beta
schedule, epsilon prediction, MSE loss, FiLM-style conditioning of the 1D
UNet blocks from the fused latent. Sampling is DDIM (eta=0, deterministic).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..config import ActionHeadConfig


# --------------------------------------------------------------------------- #
# 1D UNet backbone
# --------------------------------------------------------------------------- #

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        return torch.cat([(t[:, None] * emb[None]).sin(), (t[:, None] * emb[None]).cos()], dim=-1)


class ResidualBlock1D(nn.Module):
    """GN -> SiLU -> Conv1d -> FiLM(cond) -> GN -> SiLU -> Conv1d -> residual.

    Adapted from diffusion_policy's ConditionalResidualBlock1D. The FiLM
    heads are zero-initialized (scale applied as 1 + s) so the block starts
    as a plain residual block.
    """

    def __init__(self, in_dim: int, out_dim: int, cond_dim: int, kernel: int = 5):
        super().__init__()
        self.gn1 = nn.GroupNorm(8, in_dim)
        self.conv1 = nn.Conv1d(in_dim, out_dim, kernel, padding=kernel // 2)
        self.cond_mlp = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, out_dim * 2))
        nn.init.zeros_(self.cond_mlp[-1].weight)
        nn.init.zeros_(self.cond_mlp[-1].bias)
        self.gn2 = nn.GroupNorm(8, out_dim)
        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel, padding=kernel // 2)
        self.residual = nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T), cond: (B, cond_dim)
        scale, bias = self.cond_mlp(cond).chunk(2, dim=-1)  # (B, C) each
        h = self.conv1(self.act(self.gn1(x)))
        h = h * (1.0 + scale[:, :, None]) + bias[:, :, None]
        h = self.conv2(self.act(self.gn2(h)))
        return h + self.residual(x)


class DownBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, cond_dim: int, kernel: int = 5):
        super().__init__()
        self.res = ResidualBlock1D(in_dim, out_dim, cond_dim, kernel)
        self.down = nn.Conv1d(out_dim, out_dim, 3, stride=2, padding=1)

    def forward(self, x, cond):
        return self.down(self.res(x, cond))


class UpBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, cond_dim: int, kernel: int = 5):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_dim, in_dim, 4, stride=2, padding=1)
        self.res = ResidualBlock1D(in_dim * 2, out_dim, cond_dim, kernel)  # skip concat

    def forward(self, x, skip, cond):
        return self.res(torch.cat([self.up(x), skip], dim=1), cond)


class ConditionalUnet1D(nn.Module):
    """1D UNet over action chunks (B, T, D). Simplified dims vs
    diffusion_policy (256/512 instead of 256/512/1024)."""

    def __init__(self, input_dim: int, cond_dim: int, dims: tuple[int, int] = (256, 512), kernel: int = 5):
        super().__init__()
        d1, d2 = dims
        # FiLM blocks consume the concatenated [time | condition] feature (2*cond_dim)
        self.conv_in = nn.Conv1d(input_dim, d1, kernel, padding=kernel // 2)
        self.down1 = DownBlock(d1, d2, 2 * cond_dim, kernel)
        self.down2 = DownBlock(d2, d2, 2 * cond_dim, kernel)
        self.mid = ResidualBlock1D(d2, d2, 2 * cond_dim, kernel)
        self.up2 = UpBlock(d2, d1, 2 * cond_dim, kernel)   # skip from level 1
        self.up1 = UpBlock(d1, d1, 2 * cond_dim, kernel)   # skip from level 0
        self.conv_out = nn.Conv1d(d1, input_dim, 1)
        # global feature = time embedding + condition projection
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.cond_proj = nn.Linear(cond_dim, cond_dim)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, T, input_dim) noisy chunk.
            time_emb: (B, time_dim) sinusoidal embedding.
            cond: (B, cond_dim) fused latent (+proprio).
        Returns:
            (B, T, input_dim) prediction (noise or velocity).
        """
        x0 = self.conv_in(x.transpose(1, 2))  # (B, d1, T)      level 0
        global_feat = torch.cat([self.time_mlp(time_emb), self.cond_proj(cond)], dim=-1)
        s1 = self.down1(x0, global_feat)      # (B, d2, T/2)    level 1
        s2 = self.down2(s1, global_feat)      # (B, d2, T/4)    level 2
        m = self.mid(s2, global_feat)         # (B, d2, T/4)
        u1 = self.up2(m, s1, global_feat)     # (B, d1, T/2)
        u2 = self.up1(u1, x0, global_feat)    # (B, d1, T)
        return self.conv_out(u2).transpose(1, 2)


def _squaredcos_betas(num_steps: int, max_beta: float = 0.02) -> torch.Tensor:
    """diffusers DDPMScheduler `squaredcos_cap_v2` beta schedule (the one
    diffusion_policy uses via HuggingFace), with diffusers' beta_end=0.02 cap.

    The cap matters at the top of the schedule: raw squaredcos betas blow up
    near t=T (alpha_bar(T)=0), and without the 0.02 clip beta[T-1]=0.999,
    leaving sqrt(alpha_bar[T-1]) ~ 5e-4. DDIM then estimates
    x0_hat = (x - sqrt(1-alpha_bar)*eps) / sqrt(alpha_bar) with a ~2000x
    amplifier on the first step, so epsilon-level model error explodes into
    garbage chunks (measured |x0_hat - gt| ~ 40 on training windows while the
    training loss was ~0.007). With the 0.02 cap the first-step amplification
    is ~2.3x, the diffusion_policy regime.
    """
    def alpha_bar(t: float) -> float:
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    betas = []
    for i in range(num_steps):
        t1, t2 = i / num_steps, (i + 1) / num_steps
        betas.append(min(1.0 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Diffusion head
# --------------------------------------------------------------------------- #

class DiffusionActionHead(nn.Module):
    """DDPM (epsilon prediction, MSE) + deterministic DDIM sampling.

    Training/inference step counts follow diffusion_policy (100 train steps);
    we default to 10 DDIM steps at eval to keep rollout latency sane.
    """

    def __init__(self, cfg: ActionHeadConfig, cond_dim: int):
        super().__init__()
        self.unet = ConditionalUnet1D(cfg.action_dim, cond_dim)
        self.horizon = cfg.horizon
        self.action_dim = cfg.action_dim
        self.num_train_steps = cfg.num_train_steps
        self.inference_steps = cfg.num_inference_steps
        betas = _squaredcos_betas(cfg.num_train_steps)
        alphas_cumprod = (1.0 - betas).cumprod(dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def compute_loss(self, cond: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        B = actions.shape[0]
        t = torch.randint(0, self.num_train_steps, (B,), device=actions.device)
        noise = torch.randn_like(actions)
        ac = self.alphas_cumprod[t][:, None, None]
        x_t = ac.sqrt() * actions + (1.0 - ac).sqrt() * noise
        pred = self.unet(x_t, t, cond)  # the UNet embeds time internally
        return F.mse_loss(pred, noise)  # epsilon prediction, like diffusion_policy

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, num_steps: int | None = None) -> torch.Tensor:
        steps = num_steps or self.inference_steps
        device = cond.device
        B = cond.shape[0]
        x = torch.randn(B, self.horizon, self.action_dim, device=device)
        times = torch.linspace(self.num_train_steps - 1, 0, steps + 1, device=device).long()
        for i in range(steps):
            t_cur, t_prev = int(times[i]), int(times[i + 1])
            ac_cur, ac_prev = self.alphas_cumprod[t_cur], self.alphas_cumprod[t_prev]
            t_batch = torch.full((B,), t_cur, device=device, dtype=torch.long)
            eps = self.unet(x, t_batch, cond)
            x0 = (x - (1.0 - ac_cur).sqrt() * eps) / ac_cur.sqrt().clamp(min=1e-8)
            if t_prev == 0:
                x = ac_prev.sqrt() * x0
            else:  # DDIM, eta=0 (deterministic)
                x = ac_prev.sqrt() * x0 + (1.0 - ac_prev).sqrt() * eps
        return x

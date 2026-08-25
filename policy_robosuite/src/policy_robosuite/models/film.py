"""FiLM conditioning (Perez et al., 2018) for the vision encoder — plan hint 2.

The encoder takes plain RGB (3ch). All the per-pixel condition information —
the Plücker ray map (6ch) and the EE-trail map (1ch) — is encoded by a small
descriptor CNN into a condition vector, which (plu s the end-effector pose)
drives per-channel (gamma, beta) parameters applied to intermediate encoder
features. Because the descriptor is computed from geometry — not a learned
per-view lookup table — it transfers to the held-out front_camera at inference.
"""
from __future__ import annotations

import torch
from torch import nn


class ViewDescriptorNet(nn.Module):
    """Tiny conv net over a pixel-aligned condition map -> descriptor vector.

    Two uses (models/policy.py):
      - in_channels=6: the Plücker ray map alone -> *geometric* per-view
        signature for the per-view aux heads.
      - in_channels=7: [ray map | EE-trail] -> the spatial part of the
        encoder's FiLM condition.

    This is deliberately a function of the ray map rather than a learned
    view-id embedding: view-id tables cannot represent a novel camera.
    """

    def __init__(self, in_channels: int = 6, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, out_dim, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, cond_map: torch.Tensor) -> torch.Tensor:
        """Args: cond_map (B, in_channels, H, W). Returns: (B, out_dim)."""
        return self.net(cond_map)


class FiLMConditioner(nn.Module):
    """Maps a condition vector to (gamma, beta) for L FiLM layers.

    gamma is parametrized as (1 + scale) and both heads are zero-initialized,
    so at init the encoder behaves like an unconditional one.
    """

    def __init__(self, cond_dim: int, hidden_dim: int = 128, out_channels: tuple[int, ...] = (64, 128, 256, 512)):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.to_scale = nn.ModuleList([nn.Linear(hidden_dim, c) for c in out_channels])
        self.to_bias = nn.ModuleList([nn.Linear(hidden_dim, c) for c in out_channels])
        for lin in list(self.to_scale) + list(self.to_bias):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, cond: torch.Tensor):
        """Args: cond (B, cond_dim). Returns: list of (gamma, bias), each (B, C, 1, 1)."""
        h = self.mlp(cond)
        params = []
        for to_s, to_b in zip(self.to_scale, self.to_bias):
            gamma = 1.0 + to_s(h)[:, :, None, None]
            bias = to_b(h)[:, :, None, None]
            params.append((gamma, bias))
        return params

"""Fusion of per-view latents z_v into the global latent z_g (plan point 4).

Attention pooling with learned queries is permutation-invariant and robust to
(a) view order, (b) missing views (key padding mask + view dropout at train
time), and (c) a *single* view — case (c) is exactly the student/adapter path,
so the same module serves as the distillation target and the inference model.
"""
from __future__ import annotations

import torch
from torch import nn

from ..config import FusionConfig


class AttentionFusion(nn.Module):
    def __init__(self, latent_dim: int, cfg: FusionConfig):
        super().__init__()
        self.num_queries = cfg.num_queries
        self.queries = nn.Parameter(torch.randn(1, cfg.num_queries, latent_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            latent_dim, cfg.num_heads, dropout=cfg.dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        self.pool_norm = nn.LayerNorm(latent_dim)

    def forward(self, z_views: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Args:
            z_views: (B, V, C) per-view latents; V may be 1 (student path) or vary.
            key_padding_mask: (B, V) bool, True = dropped view.
        Returns:
            z_g: (B, C) global latent.
        """
        B = z_views.shape[0]
        q = self.queries.expand(B, -1, -1)
        attn_out, _ = self.cross_attn(q, z_views, z_views, key_padding_mask=key_padding_mask)
        q = self.norm(q + attn_out)
        q = q + self.mlp(q)
        return self.pool_norm(q.mean(dim=1))  # pool the learned queries

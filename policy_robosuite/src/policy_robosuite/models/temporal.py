"""Temporal aggregation: T_o per-frame tokens of one view -> per-view latent z_v.

This is the "temporal" half of the View-aware Temporal Encoder (plan point 3).
A learned time embedding marks each frame's position in the observation
window; a small transformer with a learned [CLS] token pools the sequence.
"""
from __future__ import annotations

import torch
from torch import nn

from ..config import TemporalConfig


class TemporalAggregator(nn.Module):
    def __init__(self, token_dim: int, cfg: TemporalConfig, max_len: int = 16):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.time_emb = nn.Embedding(max_len, token_dim)
        layer = nn.TransformerEncoderLayer(
            token_dim,
            cfg.num_heads,
            dim_feedforward=token_dim * 2,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, cfg.num_layers, enable_nested_tensor=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Args: tokens (B, T, C). Returns: z_v (B, C)."""
        B, T, _ = tokens.shape
        t = torch.arange(T, device=tokens.device)
        x = tokens + self.time_emb(t)[None]
        cls = self.cls.expand(B, -1, -1)
        out = self.transformer(torch.cat([cls, x], dim=1))
        return out[:, 0]

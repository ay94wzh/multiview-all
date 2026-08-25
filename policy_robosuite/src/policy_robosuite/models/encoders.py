"""Per-view image encoder (shared weights across all views).

Input per frame: RGB(3) only. The per-pixel camera geometry (Plücker ray map,
6ch) and the EE-trail (1ch) do NOT enter the input stack — they are encoded
into the FiLM condition vector instead (models/film.py, models/policy.py).

The encoder is a ResNet-18 with GroupNorm (diffusion_policy-style), trained
from scratch, with optional FiLM after each stage.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from .film import FiLMConditioner

# ResNet-18 stage output channels (layer1..layer4)
_RESNET18_STAGE_CHANNELS = (64, 128, 256, 512)


def _to_groupnorm(module: nn.Module) -> nn.Module:
    """Replace every BatchNorm2d with GroupNorm(32, C) (diffusion_policy-style)."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(32, child.num_features))
        else:
            _to_groupnorm(child)
    return module


class FiLMResNetEncoder(nn.Module):
    """ResNet-18 with a standard RGB stem (3 input channels) and optional FiLM
    after each stage.

    Args:
        embed_dim: output token dimension (feeds the temporal aggregator).
        cond_dim: FiLM condition dimension — descriptor of the
            [Plücker ray map | EE-trail] condition map plus EE pose.
        use_film: apply FiLM conditioning after each stage.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        cond_dim: int = 96,
        use_film: bool = True,
    ):
        super().__init__()
        self.use_film = use_film
        resnet = models.resnet18(weights=None)
        _to_groupnorm(resnet)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2 = resnet.layer1, resnet.layer2
        self.layer3, self.layer4 = resnet.layer3, resnet.layer4
        self.stages = (self.layer1, self.layer2, self.layer3, self.layer4)

        self.film = FiLMConditioner(cond_dim, out_channels=_RESNET18_STAGE_CHANNELS) if use_film else None
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(512, embed_dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None):
        """Args:
            x: (B, 3, H, W) RGB.
            cond: (B, cond_dim) FiLM context (ignored if use_film=False).
        Returns:
            feature_map (B, 512, H/32, W/32), token (B, embed_dim).
        """
        h = self.stem(x)
        film_params: list[tuple[torch.Tensor, torch.Tensor] | None] = (
            self.film(cond)
            if (self.use_film and self.film is not None and cond is not None)
            else [None] * 4
        )
        for i, stage in enumerate(self.stages):
            h = stage(h)
            if film_params[i] is not None:
                gamma, bias = film_params[i]
                h = gamma * h + bias
        return h, self.head(self.pool(h))

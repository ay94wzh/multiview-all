"""Loss weighting. The individual losses are computed in MultiViewPolicy.forward:

- action          main: base-frame action chunk from z_g
- aux_view        per-view camera-frame action chunks from z_v (plan point 3)
- distill_latent  student adapter reconstructs z_g from the single held-out view
- distill_action  optional: action-distribution matching for the student
"""
from __future__ import annotations

import torch

from ..config import ProjectConfig


def total_loss(losses: dict[str, torch.Tensor], cfg: ProjectConfig) -> torch.Tensor:
    d = cfg.distillation
    total = losses["action"]
    if "aux_view" in losses:
        total = total + cfg.loss.aux_view_weight * losses["aux_view"]
    if "distill_latent" in losses:
        total = total + d.latent_weight * losses["distill_latent"]
    if "distill_action" in losses:
        total = total + d.action_weight * losses["distill_action"]
    return total

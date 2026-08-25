"""Pure diffusion-policy baseline (arch="baseline").

The project's view-invariance machinery (FiLM conditioning on Plücker ray
maps + EE-trail, attention fusion over views, per-view aux heads, online
distillation, the temporal transformer) is the research contribution. This
module is the CONTROLLED BASELINE it must beat: a vanilla multi-camera
diffusion policy in the diffusion_policy (Chi et al.) recipe.

Each frame of each view goes through a SHARED ResNet-18 (GroupNorm, trained
from scratch, FiLM disabled) and the V*T_o frame tokens are concatenated into
ONE global conditioning vector (diffusion_policy's obs_as_global_cond style)
that conditions the SAME DiffusionActionHead / ConditionalUnet1D as the
multiview model. The action head is deliberately reused as-is — it is the
downstream network this baseline exists to validate.

Deliberately absent: FiLM, ray maps, EE-trails, temporal transformer,
attention fusion, per-view aux heads, distillation.
"""
from __future__ import annotations

import torch
from torch import nn

from ..config import ProjectConfig
from .act import ACTActionHead
from .encoders import FiLMResNetEncoder
from .heads import DiffusionActionHead

# Must match policy.py's PROPRIO_EMBED_DIM. Kept local on purpose: policy.py
# imports baseline.py (build_policy dispatch), so a shared constant would be
# a circular import.
PROPRIO_EMBED_DIM = 128


class BaselinePolicy(nn.Module):
    """Vanilla multi-camera diffusion policy: concat all frame tokens -> 1D-UNet.

    Consumes the same window semantics as MultiViewPolicy (window start s,
    chunk from t = s + T_o - 1) but only the batch keys frames, qpos,
    goal_pos, actions.
    The encoder attribute MUST be named ``encoder`` (the Trainer groups the
    backbone parameters via ``model.encoder.parameters()``) and the head
    ``global_head`` (eval/rollout scripts set
    ``model.global_head.inference_steps``).
    """

    def __init__(self, cfg: ProjectConfig):
        super().__init__()
        self.cfg = cfg
        # Plain ResNet18-GN; the cond_dim is unused with use_film=False.
        self.encoder = FiLMResNetEncoder(
            embed_dim=cfg.encoder.embed_dim, cond_dim=0, use_film=False
        )
        V = len(cfg.cameras.train_views)
        cond_dim = V * cfg.data.obs_horizon * cfg.encoder.embed_dim
        if cfg.action_head.use_proprio:
            cond_dim += PROPRIO_EMBED_DIM
        # goal_pos rides the proprio embedding: [qpos | goal_pos] -> 128.
        # goal_dim=0 (legacy checkpoints) builds the plain qpos-only embed.
        self.proprio_embed = nn.Linear(
            cfg.data.proprio_dim + cfg.data.goal_dim, PROPRIO_EMBED_DIM
        )
        if cfg.action_head.head_type == "diffusion":
            self.global_head = DiffusionActionHead(cfg.action_head, cond_dim)
        elif cfg.action_head.head_type == "act":
            # ACT head reconstructs its token sequence from the flat cond
            # vector: [V*T_o frame tokens | proprio embed] — see act.py.
            self.global_head = ACTActionHead(
                cfg.action_head, cond_dim,
                obs_token_dim=cfg.encoder.embed_dim,
                num_obs_tokens=V * cfg.data.obs_horizon,
                proprio_dim=PROPRIO_EMBED_DIM,
            )
        else:
            raise ValueError(
                f"unknown action_head.head_type {cfg.action_head.head_type!r} "
                "(expected 'diffusion' or 'act')"
            )

    def _cond(self, batch: dict) -> torch.Tensor:
        """(B, V, T_o, 3, H, W) frames + qpos (+ goal_pos) -> (B, cond_dim)."""
        frames = batch["frames"]
        B, V, T_o = frames.shape[:3]
        x = frames.reshape(B * V * T_o, *frames.shape[3:])
        _, tok = self.encoder(x, None)  # (B*V*T_o, embed_dim)
        cond = tok.reshape(B, -1)
        if self.cfg.action_head.use_proprio:
            proprio = batch["qpos"]
            if self.cfg.data.goal_dim > 0:
                proprio = torch.cat([proprio, batch["goal_pos"]], dim=-1)
            cond = torch.cat([cond, self.proprio_embed(proprio)], dim=-1)
        return cond

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        """The single training loss: epsilon-MSE on the base-frame chunk."""
        cond = self._cond(batch)
        return {"action": self.global_head.compute_loss(cond, batch["actions"])}

    @torch.no_grad()
    def _sample(self, batch: dict) -> torch.Tensor:
        cond = self._cond(batch)
        return self.global_head.sample(cond)

    @torch.no_grad()
    def sample_teacher(self, batch: dict) -> torch.Tensor:
        """Same as sample_student: the baseline has no teacher/student split.
        Both names exist so eval/rollout.py works unchanged."""
        return self._sample(batch)

    @torch.no_grad()
    def sample_student(self, batch: dict) -> torch.Tensor:
        return self._sample(batch)

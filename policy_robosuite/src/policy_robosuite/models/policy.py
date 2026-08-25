"""The multiview policy: teacher + per-view aux + student in one module.

Plan mapping:
  point 2 -- one shared FiLM encoder across views, conditioned per view
             (ray-map descriptor + EE context) -> view-aware features
  point 3 -- per-view latent z_v decoded by a *shared, view-conditioned* aux
             head into actions in that camera's frame (auxiliary constraint)
  point 4 -- {z_v} fused by attention pooling into the global latent z_g,
             decoded into actions in the robot base frame
  point 5 -- distillation: the student is the *same fusion* applied to the
             single held-out view, trained to reconstruct z_g (online, in
             the same training run)
  point 6 -- the objective: z_g (and the student path) must be view-invariant

IMPORTANT design decision (README, corrections #5): view-invariance is
enforced on the *fused* latent z_g, not on z_v. z_v should keep viewpoint
information by design — that is what lets the per-view aux heads work and what
makes the student adapter learnable.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import ProjectConfig
from .baseline import BaselinePolicy
from .encoders import FiLMResNetEncoder
from .film import ViewDescriptorNet
from .fusion import AttentionFusion
from .heads import DiffusionActionHead
from .temporal import TemporalAggregator

VIEW_DESC_DIM = 64    # ViewDescriptorNet output (geometric, for the aux heads)
FILM_DESC_DIM = 64    # ViewDescriptorNet output over [ray map | trail], for FiLM
EE_EMBED_DIM = 32     # EE pose embedding for FiLM
PROPRIO_EMBED_DIM = 128


class MultiViewPolicy(nn.Module):
    def __init__(self, cfg: ProjectConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.action_head.head_type == "act":
            raise ValueError(
                "action_head.head_type='act' is only supported with arch='baseline'; "
                "the multiview arch (teacher/aux/student) keeps the diffusion action head"
            )
        assert cfg.encoder.embed_dim == cfg.fusion.latent_dim, (
            "encoder.embed_dim must equal fusion.latent_dim "
            "(encoder tokens feed the fusion directly)"
        )
        cond_dim = FILM_DESC_DIM + EE_EMBED_DIM
        self.encoder = FiLMResNetEncoder(
            embed_dim=cfg.encoder.embed_dim, cond_dim=cond_dim, use_film=cfg.encoder.use_film
        )
        self.temporal = TemporalAggregator(cfg.encoder.embed_dim, cfg.temporal)
        self.fusion = AttentionFusion(cfg.fusion.latent_dim, cfg.fusion)

        # geometric view signature (ray map only) — conditions the aux heads
        self.view_desc = ViewDescriptorNet(in_channels=6, out_dim=VIEW_DESC_DIM)
        # spatial FiLM condition: [Plücker ray map | EE-trail] -> descriptor
        self.film_desc = ViewDescriptorNet(in_channels=7, out_dim=FILM_DESC_DIM)
        self.ee_embed = nn.Linear(7, EE_EMBED_DIM)  # xyz + wxyz

        # global head: base-frame actions from z_g (+ proprio)
        head_cond_dim = cfg.fusion.latent_dim + (PROPRIO_EMBED_DIM if cfg.action_head.use_proprio else 0)
        self.global_head = DiffusionActionHead(cfg.action_head, head_cond_dim)
        # per-view aux head: camera-frame actions from z_v (+ proprio + view desc).
        # Shared across views and conditioned on the *geometric* view descriptor,
        # so it also works for the held-out camera at inference.
        aux_cond_dim = cfg.fusion.latent_dim + VIEW_DESC_DIM + (
            PROPRIO_EMBED_DIM if cfg.action_head.use_proprio else 0
        )
        self.aux_head = DiffusionActionHead(cfg.action_head, aux_cond_dim)
        # goal_pos rides the proprio embedding: [qpos | goal_pos] -> 128.
        # goal_dim=0 (legacy checkpoints) builds the plain qpos-only embed.
        self.proprio_embed = nn.Linear(
            cfg.data.proprio_dim + cfg.data.goal_dim, PROPRIO_EMBED_DIM
        )

    # ------------------------------------------------------------------ #
    # encoding helpers
    # ------------------------------------------------------------------ #

    def _teacher_idxs(self, view_names) -> list[int]:
        """Indices of the teacher views in a batch: everything except the
        held-out student view (when distillation is enabled)."""
        idxs = list(range(len(view_names)))
        if self.cfg.distillation.enabled and self.cfg.distillation.student_view in view_names:
            idxs.remove(view_names.index(self.cfg.distillation.student_view))
        return idxs

    def _view_conds(self, batch: dict) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Per-view, per-frame conditioning for the encoder (FiLM) and the
        aux heads.

        The FiLM condition of frame t_i is the descriptor of the 7-channel
        [Plücker ray map | EE-trail(t_i)] condition map plus the EE pose at
        t_i — causal, so no future arm positions condition earlier frames.
        The aux heads get the geometric ray-map-only descriptor.

        Returns:
            film_conds: list over views of (B, T_o, FILM_DESC_DIM + EE_EMBED_DIM).
            view_descs: list over views of (B, VIEW_DESC_DIM).
        """
        frames = batch["frames"]
        B, V, T_o = frames.shape[:3]
        ray_maps = batch["ray_maps"].to(frames.device)   # (V, 6, H, W), static
        trails = batch["trails"].to(frames.device)       # (B, V, T_o, 1, H, W)
        ee_poses = batch["ee_pose"]                      # (B, T_o, 7)
        film_conds, view_descs = [], []
        for v in range(V):
            view_descs.append(self.view_desc(ray_maps[v][None]).expand(B, -1))
            cond_map = torch.cat(
                [ray_maps[v][None, None].expand(B, T_o, -1, -1, -1), trails[:, v]], dim=2
            )  # (B, T_o, 7, H, W) [plucker | trail(t_i)]
            desc = self.film_desc(cond_map.reshape(B * T_o, 7, *cond_map.shape[-2:]))
            desc = desc.reshape(B, T_o, -1)
            film_conds.append(torch.cat([desc, self.ee_embed(ee_poses)], dim=-1))
        return film_conds, view_descs

    def _encode_view(self, frames: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Encode one view's observation window into z_v.

        Args:
            frames: (B, T_o, 3, H, W) RGB frames.
            cond: (B, T_o, cond_dim) per-frame FiLM condition of this view
                (film_desc over [ray map | trail(t_i)] + EE pose(t_i)).
        Returns:
            z_v: (B, embed_dim).
        """
        B, T, C, H, W = frames.shape
        x = frames.reshape(B * T, C, H, W)
        cond_t = cond.reshape(B * T, -1)
        _, tok = self.encoder(x, cond_t)
        return self.temporal(tok.reshape(B, T, -1))

    def _embed_proprio(self, batch: dict) -> torch.Tensor:
        """[qpos | goal_pos] -> PROPRIO_EMBED_DIM (goal_pos optional via goal_dim)."""
        x = batch["qpos"]
        if self.cfg.data.goal_dim > 0:
            x = torch.cat([x, batch["goal_pos"]], dim=-1)
        return self.proprio_embed(x)

    def _proprio_cond(self, batch: dict, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([z, self._embed_proprio(batch)], dim=-1)

    # ------------------------------------------------------------------ #
    # training forward
    # ------------------------------------------------------------------ #

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        """Compute all training losses.

        Expected batch keys (see data/dataset.py):
            frames:      (B, V, T_o, 3, H, W) RGB
            ray_maps:    (V, 6, H, W) — shared across the batch (static cameras)
            trails:      (B, V, T_o, 1, H, W) — trail at each frame (FiLM cond)
            view_names:  list[str] of length V
            qpos:        (B, proprio_dim)
            goal_pos:    (B, goal_dim) — task goal position (conditioning,
                         embedded alongside qpos)
            ee_pose:     (B, T_o, 7) — pose at each frame (FiLM cond)
            actions:     (B, A_h, action_dim) base frame
            actions_cam: dict view -> (B, A_h, action_dim) camera frame
        Returns:
            dict of scalar losses: action, aux_view, distill_latent, ...
        """
        cfg = self.cfg
        frames = batch["frames"]
        B, V = frames.shape[:2]
        view_names = batch["view_names"]

        # 1. per-view encoding (shared encoder, per-view FiLM conditioning)
        film_conds, view_descs = self._view_conds(batch)
        z_vs = [self._encode_view(frames[:, v], film_conds[v]) for v in range(V)]
        z_all = torch.stack(z_vs, dim=1)  # (B, V, C)

        # 2. fusion -> z_g, teacher views only. The student view (held-out
        # front_camera) is encoded by the shared encoder but must NOT feed the
        # global latent — it enters only through the distillation loss below,
        # otherwise z_g would leak the novel viewpoint (README, corrections #5/6).
        teacher_idx = self._teacher_idxs(view_names)
        V_t = len(teacher_idx)
        drop_mask = None
        if cfg.cameras.view_dropout_prob > 0 and self.training:
            keep = torch.rand(B, V_t, device=frames.device) >= cfg.cameras.view_dropout_prob
            if not keep.any(dim=1).all():
                keep[:, 0] = True  # never drop every view
            drop_mask = ~keep
        z_g = self.fusion(z_all[:, teacher_idx], key_padding_mask=drop_mask)

        # 3. main loss: base-frame actions from z_g
        cond_g = self._proprio_cond(batch, z_g) if cfg.action_head.use_proprio else z_g
        losses = {"action": self.global_head.compute_loss(cond_g, batch["actions"])}

        # 4. aux loss: camera-frame actions from each teacher z_v (plan point 3)
        aux = 0.0
        for v in teacher_idx:
            cond_aux = torch.cat([z_vs[v], view_descs[v]], dim=-1)
            if cfg.action_head.use_proprio:
                cond_aux = torch.cat([cond_aux, self._embed_proprio(batch)], dim=-1)
            aux = aux + self.aux_head.compute_loss(cond_aux, batch["actions_cam"][view_names[v]])
        losses["aux_view"] = aux / V_t

        # 5. distillation: single held-out view -> z_g reconstruction (plan point 5)
        if cfg.distillation.enabled:
            assert cfg.distillation.student_view in view_names, (
                f"student_view {cfg.distillation.student_view!r} not in the batch views "
                f"{view_names}; make sure the dataset includes it"
            )
            idx = view_names.index(cfg.distillation.student_view)
            z_hat = self.fusion(z_all[:, idx : idx + 1])  # same fusion, V=1
            z_target = z_g.detach() if cfg.distillation.stop_teacher_grad else z_g
            losses["distill_latent"] = F.mse_loss(z_hat, z_target)
            if cfg.distillation.action_weight > 0.0:
                cond_s = self._proprio_cond(batch, z_hat) if cfg.action_head.use_proprio else z_hat
                losses["distill_action"] = self.global_head.compute_loss(cond_s, batch["actions"])
        return losses

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #

    def encode_global(self, batch: dict) -> torch.Tensor:
        """Teacher path: all train views -> z_g."""
        frames = batch["frames"]
        film_conds, _ = self._view_conds(batch)
        z_vs = [self._encode_view(frames[:, v], film_conds[v]) for v in range(frames.shape[1])]
        z_all = torch.stack(z_vs, dim=1)
        view_names = batch.get("view_names")
        if view_names is not None:  # drop the student view if it snuck into the batch
            z_all = z_all[:, self._teacher_idxs(view_names)]
        return self.fusion(z_all)

    @torch.no_grad()
    def sample_teacher(self, batch: dict) -> torch.Tensor:
        """Sample a base-frame action chunk from all train views."""
        z_g = self.encode_global(batch)
        cond = self._proprio_cond(batch, z_g) if self.cfg.action_head.use_proprio else z_g
        return self.global_head.sample(cond)

    @torch.no_grad()
    def sample_student(self, batch: dict) -> torch.Tensor:
        """Sample a base-frame action chunk from the single held-out view —
        the deployed inference path (plan point 5)."""
        frames = batch["frames"]
        film_conds, _ = self._view_conds(batch)
        idx = 0
        if "view_names" in batch and self.cfg.distillation.student_view in batch["view_names"]:
            idx = batch["view_names"].index(self.cfg.distillation.student_view)
        z_v = self._encode_view(frames[:, idx], film_conds[idx])
        z_g = self.fusion(z_v[:, None])
        cond = self._proprio_cond(batch, z_g) if self.cfg.action_head.use_proprio else z_g
        return self.global_head.sample(cond)


def build_policy(cfg: ProjectConfig) -> nn.Module:
    """Build the policy for the config's arch.

    "multiview": the view-invariant teacher + aux + student (this module).
    "baseline": the vanilla multi-camera diffusion policy (models/baseline.py).
    """
    if cfg.arch == "multiview":
        return MultiViewPolicy(cfg)
    if cfg.arch == "baseline":
        return BaselinePolicy(cfg)
    raise ValueError(f"unknown arch {cfg.arch!r} (expected 'multiview' or 'baseline')")

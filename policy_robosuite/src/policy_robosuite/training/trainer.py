"""Single-GPU training loop.

Deliberately simple (no accelerate/DDP) — single-GPU training is enough
here. Parameter groups follow act-plus-plus: a smaller lr for the encoder
backbone, the main lr for everything else.
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # tensorboard is optional
    SummaryWriter = None

try:
    import wandb
except ImportError:  # wandb is optional
    wandb = None

from ..config import ProjectConfig
from .losses import total_loss


class Trainer:
    def __init__(
        self,
        cfg: ProjectConfig,
        model: nn.Module,
        train_loader: DataLoader,
        device: str | None = None,
    ):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.device = torch.device(device or cfg.train.device)
        self.model.to(self.device)
        # flush_secs=10: scalars reach disk within 10 s of being written. The
        # default 120 s buffer is silently lost when the process is killed.
        self.writer = SummaryWriter("runs", flush_secs=10) if SummaryWriter is not None else None
        self.wandb = self._init_wandb()
        self.epoch = 0

        # Deterministic parameter-group ordering, in model.parameters() order.
        # A `set(model.encoder.parameters())` here would iterate in id()-hash
        # order — DIFFERENT in every process — so a resumed run's
        # optimizer.load_state_dict would map the saved exp_avg/step tensors
        # onto the wrong encoder parameters, silently corrupting training
        # (symptom: foreach-Adam shape-mismatch crashes in optimizer.step()).
        backbone_ids = {id(p) for p in model.encoder.parameters()}
        backbone = [p for p in model.parameters() if id(p) in backbone_ids]
        rest = [p for p in model.parameters() if id(p) not in backbone_ids]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": backbone, "lr": cfg.train.backbone_lr},
                {"params": rest, "lr": cfg.train.lr},
            ],
            weight_decay=1e-4,
        )

    def _init_wandb(self):
        """Start a wandb run; None if wandb is missing, disabled, or not logged in."""
        if wandb is None or not self.cfg.train.use_wandb:
            return None
        try:
            return wandb.init(
                project=self.cfg.train.wandb_project,
                name=self.cfg.train.wandb_run_name or None,
                config=self.cfg.to_dict(),
                # absolute path: wandb ignores relative `dir` values
                dir=str(Path(self.cfg.train.checkpoint_dir).resolve() / "wandb"),
            )
        except Exception as e:  # no api key (no-tty), no network, ...
            print(
                f"[train] wandb.init failed ({e}); training continues without wandb. "
                "Run `wandb login` once, or set WANDB_API_KEY, to record runs.",
                flush=True,
            )
            return None

    def _to_device(self, batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            if k == "actions_cam":
                out[k] = {name: t.to(self.device) for name, t in v.items()}
            elif k == "view_names":
                out[k] = v
            else:
                out[k] = v.to(self.device)
        return out

    def train_step(self, batch: dict) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        losses = self.model(self._to_device(batch))
        loss = total_loss(losses, self.cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
        self.optimizer.step()
        return {k: float(v.detach()) for k, v in losses.items()}

    def train(self):
        cfg = self.cfg
        step = 0
        for epoch in range(self.epoch, cfg.train.epochs):
            self.epoch = epoch
            for i, batch in enumerate(self.train_loader):
                if cfg.train.steps_per_epoch > 0 and i >= cfg.train.steps_per_epoch:
                    break
                metrics = self.train_step(batch)
                step += 1
                if step % cfg.train.log_interval == 0:
                    if self.writer is not None:
                        for k, v in metrics.items():
                            self.writer.add_scalar(f"train/{k}", v, step)
                        self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], step)
                    if self.wandb is not None:
                        self.wandb.log(
                            {f"train/{k}": v for k, v in metrics.items()}
                            | {"train/lr": self.optimizer.param_groups[0]["lr"]},
                            step=step,
                        )
                    # Console copy: the tensorboard/wandb buffers are lost if the
                    # process is killed; stdout in the shell/nohup log is not.
                    line = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                    print(f"[train] epoch {epoch} step {step}: {line}", flush=True)
            self.save("latest")
            print(
                f"[train] epoch {epoch} done -> saved {Path(cfg.train.checkpoint_dir) / 'latest.pt'}",
                flush=True,
            )
            if (epoch + 1) % cfg.train.eval_interval_epochs == 0:
                self.save(f"epoch-{epoch + 1}")
        if self.writer is not None:
            self.writer.close()
        if self.wandb is not None:
            self.wandb.finish()

    def save(self, tag: str):
        ckpt_dir = Path(self.cfg.train.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"{tag}.pt"
        # Atomic write: a kill mid-save must never leave a truncated .pt behind,
        # or the auto-resume in train.py would load a corrupted checkpoint.
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "model": self.model.state_dict(),
                "cfg": dataclasses.asdict(self.cfg),
                "epoch": self.epoch,
                "optimizer": self.optimizer.state_dict(),
            },
            tmp,
        )
        os.replace(tmp, path)

    def _check_resume_compat(self, path: str, ckpt: dict) -> None:
        """Refuse to resume a checkpoint trained under a different architecture.

        The ckpt carries the config it was trained with; any difference in a
        shape-affecting field (arch, goal_dim, proprio_dim, embed dims, ...)
        means load_state_dict would either crash on a size mismatch or, worse,
        silently misalign a field that only changes latent semantics. This
        gives the user an actionable message instead of a torch traceback
        (e.g. resuming a goal-blind ckpt under the new goal_dim=3 config).
        """
        if "cfg" not in ckpt:
            return
        ckpt_cfg = ckpt["cfg"]
        if isinstance(ckpt_cfg, dict) and "goal_dim" not in ckpt_cfg.get("data", {}):
            # Pre-goal checkpoint: it was trained goal-blind (Linear(9, 128)).
            ckpt_cfg = {**ckpt_cfg, "data": {**ckpt_cfg.get("data", {}), "goal_dim": 0}}
        if not isinstance(ckpt_cfg, ProjectConfig):
            ckpt_cfg = ProjectConfig.from_dict(ckpt_cfg)
        sig_a = self._shape_signature(ckpt_cfg)
        sig_b = self._shape_signature(self.cfg)
        diff = {k: (sig_a[k], sig_b[k]) for k in sig_a if sig_a[k] != sig_b[k]}
        if diff:
            detail = "; ".join(f"{k}: ckpt={a} vs config={b}" for k, (a, b) in diff.items())
            raise RuntimeError(
                f"cannot resume {path}: checkpoint was trained under a different "
                f"config ({detail}). This is not a resumable run — train fresh "
                "with a different train.checkpoint_dir (or move/delete the old "
                "checkpoint, or pass --no-resume)."
            )

    @staticmethod
    def _shape_signature(cfg: ProjectConfig) -> dict:
        """The config fields that change model/optimizer tensor shapes."""
        return {
            "arch": cfg.arch,
            "train_views": tuple(cfg.cameras.train_views),
            "image_size": tuple(cfg.cameras.image_size),
            "obs_horizon": cfg.data.obs_horizon,
            "proprio_dim": cfg.data.proprio_dim,
            "goal_dim": cfg.data.goal_dim,
            "embed_dim": cfg.encoder.embed_dim,
            "use_film": cfg.encoder.use_film,
            "num_queries": cfg.fusion.num_queries,
            "latent_dim": cfg.fusion.latent_dim,
            "temporal_layers": cfg.temporal.num_layers,
            "temporal_heads": cfg.temporal.num_heads,
            "action_dim": cfg.action_head.action_dim,
            "horizon": cfg.action_head.horizon,
            "num_train_steps": cfg.action_head.num_train_steps,
            "head_type": cfg.action_head.head_type,
            # ACT head shape-affecting fields (nhead/dropout don't change
            # tensor shapes, so they are deliberately not compared).
            "act_latent_dim": cfg.action_head.act.latent_dim,
            "act_hidden_dim": cfg.action_head.act.hidden_dim,
            "act_enc_layers": cfg.action_head.act.enc_layers,
            "act_dec_layers": cfg.action_head.act.dec_layers,
            "act_dim_feedforward": cfg.action_head.act.dim_feedforward,
        }

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._check_resume_compat(path, ckpt)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        # Fail loudly on optimizer-state/parameter shape mismatches (e.g. a
        # checkpoint saved by a process with a different parameter order).
        # Without this, foreach-Adam raises a confusing runtime error on the
        # first optimizer.step(), or worse, trains with silently corrupted
        # momentum buffers.
        # Momentum buffers (exp_avg, exp_avg_sq) must match their parameter's
        # shape; the scalar 'step' counter is expected to be shape (). A
        # mismatch in the momentum buffers means the checkpoint was saved by a
        # process with a different parameter order — resuming would corrupt
        # training.
        _MOMENTUM_KEYS = {"exp_avg", "exp_avg_sq"}
        for group_idx, group in enumerate(self.optimizer.param_groups):
            for param in group["params"]:
                state = self.optimizer.state[param]
                for k, v in state.items():
                    if k in _MOMENTUM_KEYS and isinstance(v, torch.Tensor) and v.shape != param.shape:
                        raise ValueError(
                            f"optimizer state '{k}' shape {tuple(v.shape)} does not match "
                            f"parameter shape {tuple(param.shape)} (group {group_idx}); "
                            "checkpoint optimizer state is incompatible — train with "
                            "--no-resume"
                        )
        self.epoch = ckpt["epoch"]

"""Typed configuration for the multiview project.

Plain dataclasses + YAML (argparse + pyyaml), with dotted `--overrides`.
"""
from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CameraConfig:
    train_views: list[str] = field(
        default_factory=lambda: ["top_camera", "left_camera", "right_camera", "bottom_camera"]
    )
    eval_view: str = "front_camera"        # held-out viewpoint used at inference
    image_size: tuple[int, int] = (160, 160)   # (H, W)
    view_dropout_prob: float = 0.2         # randomly drop views from the fusion at train time
    # radius (m) of the front-hemisphere sphere the cameras sit on. Robosuite
    # rig: 1.0 around the tabletop look target (0, 0, 0.8) — see
    # envs/robosuite_env.py; the ManiSkill rig used 0.6 around (0, 0, 0.2).
    camera_radius: float = 1.0


@dataclass
class TrajectoryConfig:
    # EE positions kept in the trail. 24 for robosuite: replayed robomimic
    # episodes run 59-127 steps, so the old 64 would reject every short Lift
    # episode (needs T >= window + obs_horizon + horizon).
    window_steps: int = 24
    recency_decay: float = 0.9             # per-step intensity decay of the trail
    line_width: int = 2                    # anti-aliased polyline width (px)


@dataclass
class DataConfig:
    task: str = "Lift"                     # robomimic tasks: Lift | Can | Square
    obs_horizon: int = 3                   # T_o stacked frames per view
    batch_size: int = 64
    dataset_dir: str = "demos"             # demos/<task>/episode_XXXXX.npz (replayed)
    proprio_dim: int = 7                   # robot0_joint_pos (7-dof Panda; ManiSkill was 9)
    goal_dim: int = 3                      # goal_pos conditioning vector; 0 = legacy goal-blind models
    seed: int = 0


@dataclass
class EncoderConfig:
    embed_dim: int = 256                   # must equal fusion.latent_dim
    use_film: bool = True                  # FiLM on/off — the key conditioning ablation


@dataclass
class TemporalConfig:
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.0


@dataclass
class FusionConfig:
    num_queries: int = 4                   # learned queries attending over views
    num_heads: int = 4
    latent_dim: int = 256                  # must equal encoder.embed_dim
    dropout: float = 0.0


@dataclass
class ActHeadConfig:
    """ACT (action chunking with transformers, CVAE) head hyperparameters —
    defaults follow the ACT reference (Zhao et al. / act-plus-plus)."""
    latent_dim: int = 32                   # CVAE latent z dim
    kl_weight: float = 10.0                # weight of KL(mu, logvar) in the loss
    hidden_dim: int = 256                  # d_model of the transformer layers
    enc_layers: int = 2                    # CVAE encoder layers (over [CLS, proprio, actions])
    dec_layers: int = 4                    # decoder layers (queries -> action chunk)
    nhead: int = 8
    dim_feedforward: int = 512
    dropout: float = 0.1


@dataclass
class ActionHeadConfig:
    action_dim: int = 7                    # OSC_POSE: 3 pos + 3 rot + 1 gripper
    horizon: int = 16                      # A_h predicted actions per chunk
    num_train_steps: int = 100             # DDPM diffusion steps (inert for head_type=act)
    num_inference_steps: int = 10          # DDIM sampling steps at eval (inert for head_type=act)
    use_proprio: bool = True
    head_type: str = "diffusion"           # "diffusion" (1D-UNet) | "act" (CVAE transformer)
    act: ActHeadConfig = field(default_factory=ActHeadConfig)


@dataclass
class DistillationConfig:
    enabled: bool = True                   # online distillation in the single training run
    student_view: str = "front_camera"
    latent_weight: float = 1.0
    action_weight: float = 0.0             # optional action-distribution matching
    stop_teacher_grad: bool = True         # detach z_g in the distillation loss


@dataclass
class LossConfig:
    aux_view_weight: float = 0.2           # per-view camera-frame action heads


@dataclass
class TrainConfig:
    lr: float = 1e-4
    backbone_lr: float = 1e-5              # encoder params (act-plus-plus convention)
    epochs: int = 500
    steps_per_epoch: int = 200
    grad_clip: float = 1.0
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 20
    eval_interval_epochs: int = 100        # how often to keep an epoch-N.pt snapshot
    device: str = "cuda"
    # wandb logging (optional; training continues without it if wandb is
    # missing or not logged in — see training/trainer.py)
    use_wandb: bool = False
    wandb_project: str = "multiview"
    wandb_run_name: str = ""               # empty = auto-generated name


@dataclass
class EvalConfig:
    checkpoint: str = "checkpoints/latest.pt"
    num_episodes: int = 50
    use_teacher: bool = False              # teacher (all train views) vs student (front only)
    render: bool = False
    seed: typing.Optional[int] = None      # override data.seed for the rollout episodes
    inference_steps: int = 0               # 0 = use the checkpoint's action_head value


@dataclass
class ProjectConfig:
    arch: str = "multiview"  # "multiview" (view-invariant policy) | "baseline" (pure diffusion policy)
    cameras: CameraConfig = field(default_factory=CameraConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    action_head: ActionHeadConfig = field(default_factory=ActionHeadConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: list[str] | None = None) -> "ProjectConfig":
        """Load config from YAML, then apply --overrides as dotted key=value pairs,
        e.g. `data.batch_size=32` or `encoder.use_film=false`."""
        raw = yaml.safe_load(Path(path).read_text()) or {}
        _validate_keys(cls, raw)
        cfg = _from_dict(cls, raw)
        for ov in overrides or []:
            key, _, value = ov.partition("=")
            if not key or not value:
                raise ValueError(f"Bad override {ov!r}; expected key=value")
            obj = _get_path(cfg, key)
            last = key.rsplit(".", 1)[-1]
            if not hasattr(obj, last):
                raise KeyError(
                    f"Unknown config key {key!r} in override {ov!r}; "
                    f"did you mean eval.num_episodes / eval.seed / eval.inference_steps?"
                )
            setattr(obj, last, _parse(value))
        return cfg

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectConfig":
        """Round-trip helper (e.g. reconstructing a config saved in a checkpoint)."""
        return _from_dict(cls, d)


def _validate_keys(cls, d: dict, prefix: str = ""):
    """Reject unknown yaml keys instead of silently ignoring them. A typo'd or
    unwrapped key (e.g. top-level `checkpoint:` instead of `eval:\n  checkpoint:`)
    would otherwise parse into dataclass defaults and silently evaluate the
    wrong checkpoint."""
    hints = typing.get_type_hints(cls)
    valid = {f.name for f in dataclasses.fields(cls)}
    for k, v in d.items():
        if k not in valid:
            raise ValueError(
                f"Unknown config key '{prefix}{k}' in yaml (eval keys must be "
                "nested under 'eval:', e.g. 'eval:\\n  checkpoint: ...')"
            )
        ftype = hints[k]
        if dataclasses.is_dataclass(ftype) and isinstance(v, dict):
            _validate_keys(ftype, v, prefix + k + ".")


def _from_dict(cls, d: dict):
    hints = typing.get_type_hints(cls)  # resolves string annotations (PEP 563)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        ftype = hints[f.name]
        if dataclasses.is_dataclass(ftype) and isinstance(v, dict):
            v = _from_dict(ftype, v)
        elif isinstance(v, list) and typing.get_origin(ftype) is tuple:
            v = tuple(v)
        kwargs[f.name] = v
    return cls(**kwargs)


def _get_path(cfg, key: str):
    obj = cfg
    for part in key.split(".")[:-1]:
        obj = getattr(obj, part)
    return obj


def _parse(value: str):
    """Best-effort typed parse for override values."""
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value

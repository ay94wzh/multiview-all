"""Episode dataset for the multiview policy.

Consumes the per-episode npz files written by scripts/collect_data.py:

    qpos:    (T, 7)   robot joint positions
    ee_pose: (T, 7)   xyz + wxyz quaternion (base frame)
    actions: (T, 7)   pd_ee_delta_pose actions (base frame)
    goal_pos: (3,)    task goal position (static within an episode)
    <view>/rgb: (T, H, W, 3) uint8     <view>/K, <view>/R, <view>/t: camera params
    success: bool

Per-episode preprocessing (done once at __init__, kept in RAM):
  - Plücker ray maps (6, H, W) per view, from the stored K/R/t. Static
    cameras, so they are computed from the first episode and shared.
  - EE-trail rendering (T, 1, H, W) per view (geometry/trajectory.py).

Window sampling follows the Diffusion Policy alignment: for a sampled window
start s and t = s + T_o - 1 (the last observation step),

    frames  = obs[s : s + T_o]         per view: RGB only (3 channels)
    trails  = trail[s : s + T_o]       per frame: trail[t_i] covers
                                       [t_i - window_steps + 1, t_i] (FiLM cond)
    ee_pose = pose[s : s + T_o]        per-frame EE pose (FiLM cond)
    actions = actions[t : t + A_h]     chunk the policy must reproduce
    qpos at step t.

The FiLM condition is per-frame and causal: the frame at step t_i is
conditioned by the trail ending at t_i and the EE pose at t_i — no future
arm positions leak into earlier frames of the window.

Batch views: train_views + the distillation student_view (held-out
front_camera), so the student path can be trained online. The policy's
teacher fusion excludes the student view (see models/policy.py). For
arch="baseline" (models/baseline.py) the view-invariance preprocessing is
skipped: no trails, no ray maps, no camera-frame actions, no student view —
items carry only frames / view_names / qpos / goal_pos / actions.

Memory note: episodes are fully loaded into RAM (~T x 160x160x3 x V bytes
each, a few MB) — fine on a workstation.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch

from ..config import ProjectConfig
from ..geometry.ray_maps import plucker_map, transform_delta_action
from ..geometry.trajectory import render_trails

_INPUT_CHANNELS = 3  # rgb only; plucker + trail are the FiLM condition


def _load_episode(path: Path, views: list[str], H: int, W: int, cfg: ProjectConfig) -> dict | None:
    """Load one episode npz and precompute its per-view trails.

    Returns None if the episode is shorter than one sample window
    (obs_horizon + action_head.horizon) — such episodes can never be sampled.
    """
    with np.load(path) as z:
        T = len(z["actions"])
        need = cfg.data.obs_horizon + cfg.action_head.horizon
        if T < need:
            warnings.warn(
                f"{path.name} has only {T} steps < obs_horizon+horizon={need}; skipping"
            )
            return None
        ep = {
            "qpos": z["qpos"].astype(np.float32),
            "ee_pose": z["ee_pose"].astype(np.float32),
            "actions": z["actions"].astype(np.float32),
            "rgb": {},
            "K": {},
            "R": {},
            "t": {},
            "trails": {},
        }
        if "goal_pos" not in z:
            raise RuntimeError(
                f"{path.name} lacks 'goal_pos'; re-collect with "
                "scripts/collect_data.py (the goal is static per episode but is "
                "not recoverable from an old npz)"
            )
        ep["goal_pos"] = z["goal_pos"].astype(np.float32)
        for v in views:
            ep["rgb"][v] = z[f"{v}/rgb"]
            K = z[f"{v}/K"].astype(np.float64)
            R = z[f"{v}/R"].astype(np.float64)
            t = z[f"{v}/t"].astype(np.float64)
            ep["K"][v], ep["R"][v], ep["t"][v] = K, R, t
            if cfg.arch != "baseline":
                # (T, 1, H, W) uint8 trail, one per step of the episode
                ep["trails"][v] = render_trails(
                    ep["ee_pose"][:, :3], K, R, t, H, W, cfg.trajectory
                )
    return ep


class MultiViewDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: ProjectConfig):
        self.cfg = cfg
        H, W = cfg.cameras.image_size

        task_dir = Path(cfg.data.dataset_dir) / cfg.data.task
        # Strict 5-digit glob: episode_00000.npz only — excludes tmp files and
        # double-suffix strays (see scripts/collect_data.py).
        episode_files = sorted(task_dir.glob("episode_[0-9][0-9][0-9][0-9][0-9].npz"))
        if not episode_files:
            raise FileNotFoundError(
                f"no episode_*.npz in {task_dir}; run scripts/collect_data.py first"
            )

        # Views in every batch: the teacher views plus (if distilling online)
        # the held-out student view. Order matters: policy.py indexes the
        # student view by name, teachers are the rest. The baseline (arch =
        # "baseline") has no student path, so it sees the train views only.
        self.views = list(cfg.cameras.train_views)
        if (
            cfg.distillation.enabled
            and cfg.arch != "baseline"
            and cfg.distillation.student_view not in self.views
        ):
            self.views.append(cfg.distillation.student_view)
        self.view_names = tuple(self.views)

        self.episodes = []
        for p in episode_files:
            ep = _load_episode(p, self.views, H, W, cfg)
            if ep is not None:
                self.episodes.append(ep)
        if not self.episodes:
            raise RuntimeError(f"no usable episodes in {task_dir}")

        # Static cameras: one ray map per view, shared across the dataset.
        # Skipped for the baseline (no FiLM conditioning to feed them to).
        if cfg.arch != "baseline":
            first = self.episodes[0]
            self.ray_maps = np.stack(
                [
                    np.transpose(
                        plucker_map(first["K"][v], first["R"][v], first["t"][v], H, W),
                        (2, 0, 1),
                    ).astype(np.float32)
                    for v in self.views
                ]
            )  # (V, 6, H, W)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict:
        """One sample (not batched): dict of numpy arrays; collate_fn stacks them."""
        cfg = self.cfg
        H, W = cfg.cameras.image_size
        T_o, A_h = cfg.data.obs_horizon, cfg.action_head.horizon
        ep = self.episodes[idx % len(self.episodes)]
        T = len(ep["actions"])

        # window start (obs). Valid range: obs [s, s+T_o) and actions
        # [s+T_o-1, s+T_o-1+A_h) must both fit in [0, T) -> s <= T - T_o - A_h + 1
        if np.random.rand() < 0.3:
            # Oversample the episode-start region, including PADDED starts
            # (s < 0: repeat frame/ee/trail index 0), which is exactly the
            # window the rollout feeds at its first replan. Without this the
            # model barely sees start windows and, being ambiguous with the
            # pre-grasp settle phase (static EE, empty trail), damps its
            # actions there (measured: constant ~0.05 |dpos| instead of the
            # demo's ~0.17 -> rollout falls behind and the grasp never lands).
            s = np.random.randint(-(T_o - 1), min(T_o, T - T_o - A_h + 2))
        else:
            s = np.random.randint(0, T - T_o - A_h + 2)
        t = s + T_o - 1  # last obs step = step whose action chunk we predict
        idx = np.clip(np.arange(s, s + T_o), 0, T - 1)  # pads s<0 with index 0

        # Baseline (arch="baseline"): the window semantics above are identical,
        # but no view-invariance preprocessing — RGB frames only.
        if cfg.arch == "baseline":
            frames = np.zeros((len(self.views), T_o, _INPUT_CHANNELS, H, W), dtype=np.float32)
            for vi, v in enumerate(self.views):
                rgb = ep["rgb"][v][idx] / 255.0  # (T_o, H, W, 3)
                frames[vi] = np.transpose(rgb, (0, 3, 1, 2)).astype(np.float32)
            return {
                "frames": frames,
                "view_names": self.view_names,
                "qpos": ep["qpos"][t],
                "goal_pos": ep["goal_pos"],
                "actions": ep["actions"][t : t + A_h],
            }

        frames = np.zeros((len(self.views), T_o, _INPUT_CHANNELS, H, W), dtype=np.float32)
        trails = np.zeros((len(self.views), T_o, 1, H, W), dtype=np.float32)
        actions_cam = {}
        for vi, v in enumerate(self.views):
            rgb = ep["rgb"][v][idx] / 255.0  # (T_o, H, W, 3)
            frames[vi] = np.transpose(rgb, (0, 3, 1, 2)).astype(np.float32)
            trails[vi] = ep["trails"][v][idx] / 255.0  # (T_o, 1, H, W)

            R = ep["R"][v]
            actions_cam[v] = transform_delta_action(ep["actions"][t : t + A_h], R)

        return {
            "frames": frames,
            "ray_maps": self.ray_maps,  # shared (V, 6, H, W), not batched
            "trails": trails,           # (V, T_o, 1, H, W) trail at each frame
            "view_names": self.view_names,
            "qpos": ep["qpos"][t],
            "goal_pos": ep["goal_pos"],
            "ee_pose": ep["ee_pose"][idx],  # (T_o, 7) per frame, padded like frames
            "actions": ep["actions"][t : t + A_h],
            "actions_cam": actions_cam,
        }


def collate_fn(items: list[dict]) -> dict:
    """Stack per-sample dicts into a training batch (all tensors).

    Key-driven: the baseline (arch="baseline") items carry only frames /
    view_names / qpos / goal_pos / actions, so those keys are simply absent
    from the batch; the multiview items add ray_maps / trails / ee_pose /
    actions_cam.
    """
    view_names = items[0]["view_names"]
    batch = {
        "frames": torch.from_numpy(np.stack([it["frames"] for it in items])),
        "view_names": view_names,
        "qpos": torch.from_numpy(np.stack([it["qpos"] for it in items])),
        "goal_pos": torch.from_numpy(np.stack([it["goal_pos"] for it in items])),
        "actions": torch.from_numpy(np.stack([it["actions"] for it in items])),
    }
    if "ray_maps" in items[0]:
        # ray maps are identical across samples (static cameras): keep them
        # unbatched, (V, 6, H, W) — policy.py indexes batch["ray_maps"][v].
        batch["ray_maps"] = torch.from_numpy(items[0]["ray_maps"])
    if "trails" in items[0]:
        batch["trails"] = torch.from_numpy(np.stack([it["trails"] for it in items]))
    if "ee_pose" in items[0]:
        batch["ee_pose"] = torch.from_numpy(np.stack([it["ee_pose"] for it in items]))
    if "actions_cam" in items[0]:
        batch["actions_cam"] = {
            v: torch.from_numpy(np.stack([it["actions_cam"][v] for it in items]))
            for v in view_names
        }
    return batch

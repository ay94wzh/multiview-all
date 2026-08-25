"""Evaluation rollouts.

ACT-style chunked execution: sample a chunk, execute it with temporal
ensembling (median vote over overlapping chunk predictions), re-plan when the
chunk is exhausted.

Ported from multiview's eval/rollout.py (ManiSkill) to robosuite 1.5.2:
  * env.reset() takes no seed -> np.random.seed(seed + ep) beforehand;
  * env.step() returns a 4-tuple and its done flag is horizon-only
    (ignore_done=True means it never fires) — success comes from
    env._check_success();
  * env.action_space does not exist -> env.action_dim;
  * goal_pos comes from get_goal_pos(env) at reset (same source the demo
    replay used to write the npz goal_pos);
  * robosuite has no human viewer for offscreen envs — the render flag is
    accepted for script parity and ignored.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch
from torch import nn

from ..config import ProjectConfig
from ..envs.robosuite_env import (
    get_camera_params,
    get_ee_pose,
    get_goal_pos,
    get_qpos,
    get_view_rgbs,
    make_env,
)
from ..geometry.ray_maps import plucker_map
from ..geometry.trajectory import TrajectoryRenderer

_INPUT_CHANNELS = 3


def make_policy_batch(
    cfg: ProjectConfig,
    obs_frames: dict[str, list[np.ndarray]],
    qpos: np.ndarray,
    views: list[str],
    ray_maps: dict[str, np.ndarray] | None = None,
    trails: dict[str, np.ndarray] | None = None,
    ee_pose: np.ndarray | None = None,
    goal_pos: np.ndarray | None = None,
) -> dict:
    """Assemble one policy input batch from stacked per-view frames.

    obs_frames[view] is a list of the last T_o RGB frames (H, W, 3) uint8,
    oldest first. trails[view] is the matching per-frame stack of rendered
    (T_o, H, W, 1) float trails (trail of frame k covers history up to that
    frame's own timestep), and ee_pose is (T_o, 7) aligned the same way.
    Trails, ray maps and poses all go into the batch as the per-frame FiLM
    condition (policy._view_conds), not as encoder input channels. For the
    baseline (arch="baseline") they are not passed: the batch carries
    frames / view_names / qpos / goal_pos only.
    """
    H, W = cfg.cameras.image_size
    T_o = cfg.data.obs_horizon
    frames = np.zeros((1, len(views), T_o, _INPUT_CHANNELS, H, W), dtype=np.float32)
    for vi, v in enumerate(views):
        rgb = np.stack(obs_frames[v]) / 255.0  # (T_o, H, W, 3)
        frames[0, vi] = np.transpose(rgb, (0, 3, 1, 2))
    batch = {
        "frames": torch.from_numpy(frames),
        "view_names": views,
        "qpos": torch.from_numpy(qpos[None].astype(np.float32)),
    }
    if cfg.data.goal_dim > 0:
        assert goal_pos is not None, "cfg.data.goal_dim > 0 requires goal_pos"
        batch["goal_pos"] = torch.from_numpy(goal_pos[None].astype(np.float32))
    if cfg.arch != "baseline":
        assert ray_maps is not None and trails is not None and ee_pose is not None
        batch_trails = np.zeros((1, len(views), T_o, 1, H, W), dtype=np.float32)
        for vi, v in enumerate(views):
            batch_trails[0, vi] = np.transpose(np.asarray(trails[v]), (0, 3, 1, 2))
        batch["ray_maps"] = torch.from_numpy(np.stack([ray_maps[v] for v in views]))
        batch["trails"] = torch.from_numpy(batch_trails)
        batch["ee_pose"] = torch.from_numpy(ee_pose[None].astype(np.float32))
    return batch


def evaluate(
    cfg: ProjectConfig,
    policy: nn.Module,
    num_episodes: int,
    use_teacher: bool = False,
    render: bool = False,
    device: str = "cuda",
) -> dict[str, float]:
    """Roll out the policy in robosuite and report success rate.

    Chunked execution: sample a chunk, execute obs_horizon actions with median
    temporal ensembling over overlapping chunk predictions, re-plan. Success
    per episode comes from env._check_success() at the end of the rollout.
    """
    cfg.eval.num_episodes = num_episodes
    # The baseline always observes the train views (it has no student path);
    # use_teacher only selects teacher-vs-student for arch="multiview".
    if cfg.arch == "baseline" or use_teacher:
        views = cfg.cameras.train_views
    else:
        views = [cfg.distillation.student_view]

    env = make_env(
        cfg.data.task,
        camera_names=tuple(views),
        image_size=cfg.cameras.image_size,
        camera_radius=cfg.cameras.camera_radius,
    )
    H, W = cfg.cameras.image_size
    if cfg.arch != "baseline":
        ray_maps = {v: np.transpose(plucker_map(*get_camera_params(env, v), H, W), (2, 0, 1)) for v in views}
        renderers = {v: TrajectoryRenderer(cfg.trajectory) for v in views}
    else:
        ray_maps, renderers = {}, {}

    policy = policy.to(device)
    policy.eval()
    successes = 0
    for ep in range(num_episodes):
        np.random.seed(cfg.data.seed + ep)  # robosuite has no env.seed()
        obs = env.reset()
        # The goal is static within an episode: fetch it once at reset. Same
        # source as the demo replay (get_goal_pos), so the training and eval
        # conditioning vectors are bit-identical per seed.
        goal_pos = get_goal_pos(env)
        first_rgbs = get_view_rgbs(obs, tuple(views))
        obs_frames = {v: [first_rgbs[v]] * cfg.data.obs_horizon for v in views}
        # EE-pose window aligned with obs_frames (padded with the first pose,
        # mirroring how obs_frames pads with the first frame). After this,
        # poses and trail points are appended once per ENV STEP, so at each
        # replan the window is [e_{t-2}, e_{t-1}, e_t] and the per-frame
        # trails cover the same dense per-step history as dataset.py windows.
        # Appending per *replan* (every obs_horizon steps) misaligns the FiLM
        # conditions with the frames — a train/eval mismatch that made every
        # rollout input out-of-distribution.
        first_ee = get_ee_pose(env)
        ee_history: deque = deque([first_ee] * cfg.data.obs_horizon, maxlen=cfg.data.obs_horizon)
        for r in renderers.values():
            r.reset()
        done, t = False, 0
        # Temporal ensembling (ACT/diffusion-policy style): chunk rows
        # predicted for the same absolute timestep by consecutive replans are
        # combined before execution. The MEDIAN (not the mean) rejects
        # wrong-mode samples outright instead of averaging them in — the model
        # is bimodal on some early windows (approach vs pre-grasp settle), and
        # mean-voting let one flipped sample dilute the executed motion.
        ensemble: dict[int, list[np.ndarray]] = {}
        while not done and t < 300:
            qpos = get_qpos(obs)
            if cfg.arch != "baseline":
                # Per-frame trails: frame k of the window (oldest first) gets
                # the trail truncated to its own timestep — no future EE
                # positions.
                trails = {
                    v: np.stack([
                        renderers[v].render_past(
                            *get_camera_params(env, v), H, W,
                            steps_ago=cfg.data.obs_horizon - 1 - k,
                        )
                        for k in range(cfg.data.obs_horizon)
                    ])
                    for v in views
                }  # (T_o, H, W, 1)
                batch = make_policy_batch(cfg, obs_frames, qpos, views,
                                          ray_maps, trails, np.stack(list(ee_history)),
                                          goal_pos)
            else:
                batch = make_policy_batch(cfg, obs_frames, qpos, views, goal_pos=goal_pos)
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            chunk = policy.sample_teacher(batch) if use_teacher else policy.sample_student(batch)
            chunk = chunk[0].cpu().numpy()  # (A_h, action_dim)
            # Window semantics (dataset.py): chunk row i is the action applied
            # from state t-1+i, i.e. row 0 PRODUCED the window's last frame.
            # At replan time the env is already at state t, so row 0 is the
            # action that was just applied and execution starts at row 1.
            # Voting rows at their absolute action index t-1+i lets consecutive
            # replans ensemble the same action (temporal ensembling).
            for i in range(chunk.shape[0]):
                ensemble.setdefault(t - 1 + i, []).append(chunk[i])
            # Execute obs_horizon actions per replan (execute horizon).
            for a in [np.median(ensemble.pop(t + i), axis=0) for i in range(cfg.data.obs_horizon)]:
                a = np.asarray(a, dtype=np.float32)
                # Snap gripper to the data's bimodal +-1 (robosuite: +1 =
                # close, -1 = open; the replay stored the demos' signs
                # verbatim, so the same rule holds as for ManiSkill).
                a[6] = 1.0 if a[6] >= 0.0 else -1.0
                action = np.zeros(env.action_dim)
                action[: len(a)] = a
                obs, _, _, _ = env.step(action)  # ignore_done=True -> no early stop
                # Update the EE/trail history per step so the next window's
                # FiLM conditions stay aligned with its frames (see above).
                ee_history.append(get_ee_pose(env))
                if renderers:  # empty for the baseline (no trail rendering)
                    for v in views:
                        renderers[v].add(ee_history[-1][:3])
                rgbs = get_view_rgbs(obs, tuple(views))
                for v in views:
                    obs_frames[v].pop(0)
                    obs_frames[v].append(rgbs[v])
                # robosuite offscreen envs have no human viewer; the render
                # flag is accepted for script parity and ignored.
                t += 1
                if t >= 300:
                    break
        successes += bool(env._check_success())
    env.close()
    return {"success_rate": successes / num_episodes, "episodes": num_episodes}

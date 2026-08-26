#!/usr/bin/env python
"""Record policy rollouts with frames + state, for failure diagnosis.

Mirrors the student rollout loop in src/policy_robosuite/eval/rollout.py
exactly (same policy-input construction, same temporal ensembling, same
execution cadence; for arch="baseline" it records the train-view rollout
instead) and additionally records, per env step:

  - every policy-input view RGB (JPEG-encoded, small on disk) — the 4 train
    views for the baseline, the 1 student view for the multiview arch
  - render_camera RGB (high-res, goal site visible — not a policy input)
  - EE pose, object pose, goal position, gripper qpos, is_grasped proxy
  - the executed (ensembled) action
  - every sampled chunk at replan time (to see what the head outputs)

Robosuite port notes (vs the multiview ManiSkill recorder):
  * reset has no seed -> np.random.seed(seed) beforehand;
  * step() returns a 4-tuple; done is horizon-only (ignore_done=True never
    fires) so the episode bound is the same t < 300 cap as eval/rollout.py;
  * success comes from env._check_success() at the end;
  * no env.action_space -> env.action_dim;
  * object pose per task from the obs keys (cube_pos / Can_pos /
    SquareNut_pos; obs quats are xyzw, stored as wxyz like ee_pose);
  * ManiSkill's obs["extra"]["is_grasped"] does not exist here — the proxy
    is: gripper fingers closed (|q0 - q1| < 8mm; open = +/-0.0208) AND the
    object within 6 cm of the EE site.

Output: <out>/episode_XXXXX.npz + summary.csv. No simulator viewer needed —
videos and plots are built afterwards by scripts/visualize_rollouts.py.

Usage:
    python scripts/rollout_record.py --checkpoint checkpoints/latest.pt \
        --num-episodes 10 --seed 0 --out runs/rollout_vis
"""
from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from _argpath import resolve  # scripts/ dir (sys.path[0]) — see _argpath.py
from policy_robosuite.config import ProjectConfig
from policy_robosuite.eval.rollout import make_policy_batch
from policy_robosuite.envs.robosuite_env import (
    get_camera_params,
    get_ee_pose,
    get_goal_pos,
    get_qpos,
    get_render_rgb,
    get_view_rgbs,
    make_env,
)
from policy_robosuite.geometry.ray_maps import plucker_map
from policy_robosuite.geometry.trajectory import TrajectoryRenderer
from policy_robosuite.models.policy import build_policy

# Grasp proxy thresholds (robosuite has no is_grasped obs key). The Panda
# gripper reads open = [+0.0208, -0.0208] -> |q0 - q1| = 0.0416, closed -> ~0.
_GRIPPER_CLOSED_MM = 0.008
_GRASP_REACH_M = 0.06


def _jpeg(rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def _object_pose(obs: dict) -> np.ndarray:
    """Object pose as (7,) xyz + wxyz — the task's key from the obs dict.

    Lift -> cube_pos/cube_quat, Can -> Can_pos/Can_quat,
    Square -> SquareNut_pos/SquareNut_quat (robosuite capitalizes the object
    names). Obs quats are xyzw; convert to wxyz like get_ee_pose.
    """
    from robosuite.utils.transform_utils import convert_quat

    for pos_key in ("cube_pos", "Can_pos", "SquareNut_pos"):
        if pos_key in obs:
            quat_key = pos_key.replace("_pos", "_quat")
            xyz = np.asarray(obs[pos_key], dtype=np.float32)
            quat = convert_quat(np.asarray(obs[quat_key]), to="wxyz")
            return np.concatenate([xyz, quat]).astype(np.float32)
    raise KeyError(f"no object pose in obs (keys: {sorted(obs)})")


def _is_grasped(obs: dict, ee_xyz: np.ndarray, obj_xyz: np.ndarray) -> float:
    """Grasp proxy: fingers closed AND object within reach of the EE site."""
    q = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
    closed = float(np.abs(q[0] - q[1])) < _GRIPPER_CLOSED_MM
    near = float(np.linalg.norm(obj_xyz - ee_xyz)) < _GRASP_REACH_M
    return float(closed and near)


def roll_out_episode(cfg: ProjectConfig, policy, env, ray_maps, renderers,
                     views: list[str], seed: int) -> dict:
    """One recorded student rollout — same loop as eval/rollout.py::evaluate."""
    T_o = cfg.data.obs_horizon
    H, W = cfg.cameras.image_size
    np.random.seed(seed)
    obs = env.reset()
    # The goal is static within an episode: fetch it once at reset (same
    # source as the demo replay's npz goal_pos, so conditioning matches).
    goal_pos = get_goal_pos(env)
    first_rgbs = get_view_rgbs(obs, tuple(views))
    obs_frames = {v: [first_rgbs[v]] * T_o for v in views}
    first_ee = get_ee_pose(env)
    ee_history: deque = deque([first_ee] * T_o, maxlen=T_o)
    for r in renderers.values():
        r.reset()

    # recordings: per-view RGB (JPEG bytes) + render + per-step state
    rec = {k: [] for k in ("ee_pose", "obj_pose", "is_grasped", "goal_pos",
                           "action", "gripper_qpos")}
    rec_rgb = {v: [] for v in views}
    rec["render_rgb"] = []
    obj0 = _object_pose(obs)
    for v in views:
        rec_rgb[v].append(_jpeg(first_rgbs[v]))
    rec["render_rgb"].append(_jpeg(get_render_rgb(env)))
    rec["ee_pose"].append(first_ee)
    rec["obj_pose"].append(obj0)
    rec["is_grasped"].append(_is_grasped(obs, first_ee[:3], obj0[:3]))
    rec["goal_pos"].append(goal_pos)
    rec["action"].append(np.zeros(cfg.action_head.action_dim, dtype=np.float32))
    rec["gripper_qpos"].append(np.asarray(obs["robot0_gripper_qpos"],
                                          dtype=np.float32))

    replan_actions, replan_at_step = [], []
    ensemble: dict[int, list[np.ndarray]] = {}
    t = 0
    while t < 300:
        qpos = get_qpos(obs)
        if cfg.arch != "baseline":
            trails = {
                v: np.stack([
                    renderers[v].render_past(
                        *get_camera_params(env, v), H, W,
                        steps_ago=T_o - 1 - k,
                    )
                    for k in range(T_o)
                ])
                for v in views
            }
            batch = make_policy_batch(cfg, obs_frames, qpos, views,
                                      ray_maps, trails, np.stack(list(ee_history)),
                                      goal_pos)
        else:
            batch = make_policy_batch(cfg, obs_frames, qpos, views, goal_pos=goal_pos)
        batch = {k: (v.to("cuda") if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        chunk = policy.sample_student(batch)[0].cpu().numpy()  # (A_h, 7)
        replan_actions.append(chunk)
        replan_at_step.append(t)
        for i in range(chunk.shape[0]):
            ensemble.setdefault(t - 1 + i, []).append(chunk[i])
        for i in range(T_o):
            a = np.asarray(np.median(ensemble.pop(t + i), axis=0), dtype=np.float32)
            a[6] = 1.0 if a[6] >= 0.0 else -1.0  # snap gripper to +/-1
            action = np.zeros(env.action_dim)
            action[: len(a)] = a
            obs, _, _, _ = env.step(action)
            ee_history.append(get_ee_pose(env))
            if renderers:  # empty for the baseline (no trail rendering)
                for v in views:
                    renderers[v].add(ee_history[-1][:3])
            rgbs = get_view_rgbs(obs, tuple(views))
            for v in views:
                obs_frames[v].pop(0)
                obs_frames[v].append(rgbs[v])
                rec_rgb[v].append(_jpeg(rgbs[v]))
            obj = _object_pose(obs)
            rec["render_rgb"].append(_jpeg(get_render_rgb(env)))
            rec["ee_pose"].append(ee_history[-1])
            rec["obj_pose"].append(obj)
            rec["is_grasped"].append(_is_grasped(obs, ee_history[-1][:3], obj[:3]))
            rec["goal_pos"].append(goal_pos)
            rec["action"].append(a)
            rec["gripper_qpos"].append(np.asarray(obs["robot0_gripper_qpos"],
                                                  dtype=np.float32))
            t += 1

    success = bool(env._check_success())
    return {
        "success": success,
        "seed": seed,
        "steps": t,
        "arch": cfg.arch,
        "views": views,
        **{f"{v}_rgb": np.asarray(rec_rgb[v], dtype=object) for v in views},
        "render_rgb": np.asarray(rec["render_rgb"], dtype=object),
        "ee_pose": np.stack(rec["ee_pose"]).astype(np.float32),
        "obj_pose": np.stack(rec["obj_pose"]).astype(np.float32),
        "is_grasped": np.asarray(rec["is_grasped"], dtype=np.float32),
        "goal_pos": np.stack(rec["goal_pos"]).astype(np.float32),
        "action": np.stack(rec["action"]).astype(np.float32),
        "gripper_qpos": np.stack(rec["gripper_qpos"]).astype(np.float32),
        "replan_actions": np.asarray(replan_actions, dtype=np.float32),
        "replan_at_step": np.asarray(replan_at_step, dtype=np.int64),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--train-config", type=str, default="configs/train_multiview.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0, help="first episode seed")
    parser.add_argument("--out", type=str, default="runs/rollout_vis")
    args = parser.parse_args()

    eval_cfg = ProjectConfig.from_yaml(resolve(args.config))
    train_cfg = ProjectConfig.from_yaml(resolve(args.train_config))
    cfg = train_cfg
    cfg.eval = eval_cfg.eval
    cfg.eval.checkpoint = str(resolve(args.checkpoint))

    ckpt = torch.load(cfg.eval.checkpoint, map_location="cpu", weights_only=False)
    if "cfg" in ckpt:
        ckpt_cfg = ckpt["cfg"]
        if isinstance(ckpt_cfg, dict) and "goal_dim" not in ckpt_cfg.get("data", {}):
            # Pre-goal checkpoint: keep it goal-blind (proprio embed shape).
            ckpt_cfg = {**ckpt_cfg, "data": {**ckpt_cfg.get("data", {}), "goal_dim": 0}}
        cfg = ProjectConfig.from_dict(ckpt_cfg)
        cfg.eval = eval_cfg.eval
        cfg.eval.checkpoint = args.checkpoint
    model = build_policy(cfg)
    model.load_state_dict(ckpt["model"])
    if cfg.eval.seed is not None:
        cfg.data.seed = cfg.eval.seed
    if cfg.eval.inference_steps > 0:
        model.global_head.inference_steps = cfg.eval.inference_steps
    model = model.to("cuda").eval()

    # The baseline has no student path: record its (train-view) rollouts.
    views = cfg.cameras.train_views if cfg.arch == "baseline" else [cfg.distillation.student_view]
    env = make_env(
        cfg.data.task,
        camera_names=tuple(views),
        image_size=cfg.cameras.image_size,
        camera_radius=cfg.cameras.camera_radius,
    )
    H, W = cfg.cameras.image_size
    if cfg.arch != "baseline":
        ray_maps = {v: np.transpose(plucker_map(*get_camera_params(env, v), H, W), (2, 0, 1))
                    for v in views}
        renderers = {v: TrajectoryRenderer(cfg.trajectory) for v in views}
    else:
        ray_maps, renderers = {}, {}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ep in range(args.num_episodes):
        seed = args.seed + ep
        rec = roll_out_episode(cfg, model, env, ray_maps, renderers, views, seed)
        np.savez_compressed(
            out_dir / f"episode_{ep:05d}.npz",
            success=np.asarray(rec["success"]),
            seed=np.asarray(seed),
            steps=np.asarray(rec["steps"]),
            arch=rec["arch"],
            views=rec["views"],
            **{f"{v}_rgb": rec[f"{v}_rgb"] for v in views},
            render_rgb=rec["render_rgb"],
            ee_pose=rec["ee_pose"], obj_pose=rec["obj_pose"],
            is_grasped=rec["is_grasped"], goal_pos=rec["goal_pos"],
            action=rec["action"], gripper_qpos=rec["gripper_qpos"],
            replan_actions=rec["replan_actions"],
            replan_at_step=rec["replan_at_step"],
        )
        obj = rec["obj_pose"][:, :3]
        goal = rec["goal_pos"][0]
        rows.append({
            "episode": ep,
            "seed": seed,
            "success": int(rec["success"]),
            "steps": rec["steps"],
            "final_dist_to_goal": float(np.linalg.norm(obj[-1] - goal)),
            "max_obj_disp": float(np.max(np.linalg.norm(obj - obj[0], axis=1))),
            "grasped_ever": int(rec["is_grasped"].max() > 0.5),
            "grasped_at_end": int(rec["is_grasped"][-1] > 0.5),
            "mean_ee_delta": float(np.mean(np.linalg.norm(rec["action"][1:, :3], axis=1))),
        })
        print(f"[{ep + 1}/{args.num_episodes}] seed {seed}: "
              f"success={rec['success']} steps={rec['steps']} "
              f"final_dist={rows[-1]['final_dist_to_goal']:.3f} "
              f"grasped_ever={rows[-1]['grasped_ever']}", flush=True)

    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    env.close()
    print(f"wrote {out_dir}/summary.csv", flush=True)


if __name__ == "__main__":
    main()

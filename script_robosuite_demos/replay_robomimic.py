#!/usr/bin/env python
"""Replay robomimic PH demonstrations through robosuite with the five-camera rig.

Robomimic v15 datasets (low_dim_v15.hdf5) record full MuJoCo states plus
actions; we reconstruct the source environment from the recorded `env_args`
(OSC_POSE world-frame delta controller, Panda), restore the initial state,
and re-execute the recorded actions. Cameras are injected into the recorded
per-episode `model_file` xml via a processor, so the sim matches the recorded
states exactly and our five cameras are always present.

Replay is *approximate*: low-level dynamics drift between the recorded and
the replayed trajectory, so some demos fail to re-satisfy the task success
criterion even though the original collection succeeded. Episodes are kept
only when `env._check_success()` holds at the final state.

Output schema is identical to multiview/scripts/collect_data.py (see
data/dataset.py): per-episode npz with qpos (T,7), ee_pose (T,7) xyz+wxyz,
actions (T,7), goal_pos (3,), success (bool), and per-view
<view>/rgb (T,H,W,3) uint8, <view>/K, <view>/R, <view>/t (CV convention,
X_cam = R @ X_world + t).

Usage:
    python replay_robomimic.py --task Lift   --source <hdf5> --out ../policy_robosuite/demos
    python replay_robomimic.py --task Can    --source <hdf5> --out ../policy_robosuite/demos
    python replay_robomimic.py --task Square --source <hdf5> --out ../policy_robosuite/demos
"""
from __future__ import annotations

import argparse
import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np
import robosuite as suite
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
)
from robosuite.utils.transform_utils import convert_quat, mat2quat

# Same five named cameras as multiview (VIEW_CAMERA_NAMES): top/left/right/
# bottom for training, front held out for inference.
VIEW_CAMERA_NAMES = (
    "top_camera",
    "left_camera",
    "right_camera",
    "bottom_camera",
    "front_camera",
)

# Nominal camera poses — the multiview rig. All cameras sit on the FRONT
# hemisphere (x > 0) of a sphere of radius `camera_radius` centered on the
# look target, which for robosuite is the tabletop workspace (the ManiSkill
# target was (0, 0, 0.2); robosuite's table surface is ~0.815 m).
LOOK_TARGET = [0.0, 0.0, 0.8]
CAMERA_RADIUS = 1.0  # tuned for fovy 60 so the scene fills the 160x160 frame
CAMERA_FOVY = 60.0
IMAGE_SIZE = (160, 160)

# (azimuth_deg, elevation_deg), identical to multiview's _CAMERA_SPHERICAL.
CAMERA_SPHERICAL = {
    "top_camera": (0.0, 45.0),
    "left_camera": (40.0, 20.0),
    "right_camera": (-40.0, 20.0),
    "bottom_camera": (0.0, 5.0),
    "front_camera": (0.0, 30.0),
}

# Shortest episode the dataset can use: trail window + obs horizon + action
# horizon (config.py trajectory.window_steps 24, data.obs_horizon 3,
# action_head.horizon 16). Same filter as multiview collect_data.py.
MIN_STEPS = 24 + 3 + 16

# A .tmp older than this is a leftover from a killed run (see atomic write).
STALE_TMP_SECONDS = 1800


def spherical_eye(name: str) -> np.ndarray:
    """Camera position: look target + radius * direction (multiview convention)."""
    az, el = np.deg2rad(CAMERA_SPHERICAL[name])
    off = CAMERA_RADIUS * np.array(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
    )
    return np.asarray(LOOK_TARGET) + off


def camera_elements() -> list[ET.Element]:
    """The five cameras as MuJoCo <camera> elements (pos + quat wxyz).

    Camera frame convention (same as CamPoseOpensource cam_utils.py): the
    columns of the world rotation are [right | up | -forward] — MuJoCo cameras
    look along -z. fovy must be explicit: MuJoCo defaults to 45 and the
    intrinsics below read sim.model.cam_fovy.

    Quat ordering gotcha: robosuite's mat2quat returns (x, y, z, w) while the
    MuJoCo XML quat attribute is (w, x, y, z); the reorder below is load-
    bearing (writing mat2quat's output verbatim points the cameras sideways).
    """
    elems = []
    for name in VIEW_CAMERA_NAMES:
        eye = spherical_eye(name)
        forward = np.asarray(LOOK_TARGET) - eye
        forward /= np.linalg.norm(forward)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)  # re-orthonormalize
        R_cam2world = np.stack([right, up, -forward], axis=1)  # columns
        quat = mat2quat(R_cam2world)  # robosuite returns (x, y, z, w)!
        quat = quat[[3, 0, 1, 2]]  # reorder to (w, x, y, z) for the MuJoCo XML quat
        elem = ET.Element("camera")
        elem.set("name", name)
        elem.set("mode", "fixed")
        elem.set("pos", " ".join(f"{v:.6f}" for v in eye))
        elem.set("quat", " ".join(f"{v:.6f}" for v in quat))
        elem.set("fovy", f"{CAMERA_FOVY:.3f}")
        elems.append(elem)
    return elems


def add_cameras_processor(xml: str) -> str:
    """XML processor: append the five cameras to the top-level worldbody.

    Registered via env.set_xml_processor; _initialize_sim runs it on every
    model string, including the per-episode model_file passed to
    reset_from_xml_string. The processor MUST return the xml (a documented
    robosuite quirk).
    """
    root = ET.fromstring(xml)
    wb = root.find("worldbody")
    assert wb is not None, "model xml has no top-level <worldbody>"
    # Drop any existing cameras with our names, then append ours.
    for elem in wb.findall("camera"):
        if elem.get("name") in VIEW_CAMERA_NAMES:
            wb.remove(elem)
    for elem in camera_elements():
        wb.append(elem)
    return ET.tostring(root, encoding="utf8").decode("utf8")


def _replay_env_class(env_cls):
    """Subclass of a robosuite env that injects the five cameras into every
    model it builds — the default model at construction AND the per-episode
    model_file at reset_from_xml_string.

    Why a subclass: robosuite validates camera observables (renders each one)
    during env __init__, against the *default* model — before any
    set_xml_processor could run — so our cameras must be in the default model
    too. edit_model_xml is the first element of the env's xml-processor chain
    and runs on every model string, so overriding it covers both paths.
    """
    class _ReplayEnv(env_cls):
        def edit_model_xml(self, xml_str: str) -> str:
            xml_str = super().edit_model_xml(xml_str)
            return add_cameras_processor(xml_str)

    _ReplayEnv.__name__ = f"Replay{env_cls.__name__}"
    return _ReplayEnv


def build_env(env_name: str, env_kwargs: dict) -> suite.Env:
    """Reconstruct the dataset's environment, with our camera obs enabled.

    env_kwargs come verbatim from the hdf5 env_args (controller_configs,
    control_freq, lite_physics, ignore_done, ...) — only the render/camera
    settings are overridden. The controller_configs is already in composite
    form (body_parts); legacy keys in it are ignored by 1.5.2 but coincide
    with the defaults, so replaying is exact.
    """
    from robosuite.environments import REGISTERED_ENVS

    env_kwargs = dict(env_kwargs)
    H, W = IMAGE_SIZE
    env_kwargs.update(
        {
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "use_camera_obs": True,
            "camera_names": list(VIEW_CAMERA_NAMES),
            "camera_heights": H,
            "camera_widths": W,
            "camera_depths": False,
        }
    )
    return _replay_env_class(REGISTERED_ENVS[env_name])(**env_kwargs)


def get_goal_pos(env) -> np.ndarray:
    """Task-specific goal position (3,) — static within an episode.

    Lift    -> current cube position (body frame of the cube)
    Can     -> center of the goal bin (static env param, bin2_pos)
    Square  -> midpoint of the two pegs (single square nut in single_object
               mode; the goal is *which* peg the nut goes on, so the midpoint
               is a static approximation of a two-valued goal — document it)
    """
    task = type(env).__name__
    from robosuite.environments.manipulation.lift import Lift
    from robosuite.environments.manipulation.pick_place import PickPlace
    from robosuite.environments.manipulation.nut_assembly import NutAssembly

    if isinstance(env, Lift):
        return np.asarray(env.sim.data.body_xpos[env.cube_body_id], dtype=np.float32)
    if isinstance(env, PickPlace):
        return np.asarray(env.bin2_pos, dtype=np.float32)
    if isinstance(env, NutAssembly):
        p1 = env.sim.data.body_xpos[env.peg1_body_id]
        p2 = env.sim.data.body_xpos[env.peg2_body_id]
        return np.asarray(0.5 * (p1 + p2), dtype=np.float32)
    raise NotImplementedError(f"goal_pos undefined for task {task!r}")


def camera_params(env) -> dict:
    """Per-camera (K, R, t) with X_cam = R @ X_world + t (CV convention).

    robosuite's get_camera_extrinsic_matrix returns the world->camera pose as
    make_pose(cam_xpos, cam_xmat) @ diag(1,-1,-1) — the camera-axis correction
    that makes the camera frame match OpenCV convention. From that 4x4 P we
    store R = P[:3,:3].T, t = -R @ cam_xpos; then the camera center
    o = -R.T @ t = cam_xpos and the center ray is the MuJoCo optical axis
    (verified numerically). K uses sim.model.cam_fovy (60 deg) — the same
    f the renderer uses, so K @ (R x + t) maps exactly into the rendered frame.
    """
    H, W = IMAGE_SIZE
    out = {}
    for name in VIEW_CAMERA_NAMES:
        K = get_camera_intrinsic_matrix(env.sim, name, H, W)
        P = get_camera_extrinsic_matrix(env.sim, name)  # 4x4 world->camera
        R = P[:3, :3].T
        t = -R @ env.sim.data.cam_xpos[env.sim.model.camera_name2id(name)]
        out[name] = (K.astype(np.float32), R.astype(np.float32), t.astype(np.float32))
    return out


def replay_demo(env, h5, demo_key: str) -> dict | None:
    """Replay one demo; return the npz payload, or None if it fails the gate."""
    demo = h5["data"][demo_key]
    actions = np.asarray(demo["actions"])
    states = np.asarray(demo["states"])
    model_file = demo.attrs["model_file"]

    # Rebuild the sim from the recorded model xml (state-size must match, else
    # the recorded states belong to a different model — refuse loudly).
    env.reset_from_xml_string(model_file)
    sim = env.sim
    # Robosuite state vectors are [time, qpos, qvel] (see binding_utils
    # MjSimState: from_flattened splits off the leading time scalar); the
    # robomimic-recorded states use the same convention.
    n_state = 1 + sim.model.nq + sim.model.nv + sim.model.na
    if states.shape[1] != n_state:
        raise RuntimeError(
            f"{demo_key}: recorded states have {states.shape[1]} floats but this "
            f"model has {n_state} (nq+nv+na); model_file mismatch?"
        )
    sim.set_state_from_flattened(states[0])
    sim.forward()

    rgbs = {v: [] for v in VIEW_CAMERA_NAMES}
    qpos, ee_poses, act = [], [], []
    goal_pos = get_goal_pos(env)

    def capture(obs):
        # Flip: robosuite's IMAGE_CONVENTION="opengl" leaves frames raw from
        # MuJoCo (row 0 = bottom); ManiSkill sensor frames are top-down, so
        # flipud before storing keeps the npz identical in orientation.
        for v in VIEW_CAMERA_NAMES:
            rgbs[v].append(np.flipud(obs[f"{v}_image"]))
        qpos.append(np.asarray(obs["robot0_joint_pos"], dtype=np.float32))
        ee_xyz = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
        ee_quat = convert_quat(np.asarray(obs["robot0_eef_quat_site"]), to="wxyz")
        ee_poses.append(np.concatenate([ee_xyz, ee_quat]).astype(np.float32))

    capture(env._get_observations())
    for a in actions:
        obs = env.step(a)[0]  # ignore_done=True -> never terminates early
        act.append(np.asarray(a, dtype=np.float32))
        capture(obs)

    success = bool(env._check_success())
    if not success:
        print(f"  {demo_key}: replay FAILED success check ({len(actions)} steps) — dropped")
        return None
    if len(actions) < MIN_STEPS:
        print(f"  {demo_key}: only {len(actions)} steps < {MIN_STEPS} — dropped")
        return None

    cam = camera_params(env)
    data = {
        # Schema (same as multiview collect_data.py): qpos (T, 7) — one row
        # per action, captured before it executes — while ee_pose/rgb keep
        # T+1 rows (one per frame, incl. the initial state).
        "qpos": np.stack(qpos[: len(actions)]).astype(np.float32),
        "ee_pose": np.stack(ee_poses).astype(np.float32),
        "actions": np.stack(act).astype(np.float32),
        "goal_pos": goal_pos,
        "success": np.asarray(True),
    }
    for v in VIEW_CAMERA_NAMES:
        data[f"{v}/rgb"] = np.stack(rgbs[v])
        K, R, t = cam[v]
        data[f"{v}/K"], data[f"{v}/R"], data[f"{v}/t"] = K, R, t
    return data


def replay_task(source: Path, out_dir: Path, num_episodes: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Resume-friendly: `num_episodes` is the total target; existing episodes
    # are kept and numbering continues after them.
    existing = sorted(p.stem for p in out_dir.glob("episode_[0-9][0-9][0-9][0-9][0-9].npz"))
    ep = int(existing[-1].split("_")[-1]) + 1 if existing else 0
    if existing:
        print(f"found {len(existing)} existing episodes; continuing at episode_{ep:05d}")
        num_episodes = max(num_episodes, ep)
    for stale in out_dir.glob("episode_*.npz.tmp"):
        age = time.time() - stale.stat().st_mtime
        if age > STALE_TMP_SECONDS:
            stale.unlink()
            print(f"removed stale temp file {stale}")
        else:
            print(f"warning: {stale} is fresh — another collector may be running; leaving it")

    with h5py.File(source, "r") as h5:
        env_args = json.loads(h5["data"].attrs["env_args"])
        env_name = env_args["env_name"]
        env_kwargs = env_args["env_kwargs"]
        demo_keys = sorted(k for k in h5["data"].keys() if k.startswith("demo"))
        print(f"[replay] task env {env_name} ({type_check(env_name)}), {len(demo_keys)} source demos", flush=True)
        env = build_env(env_name, env_kwargs)
        print(f"[replay] PID {os.getpid()}; writing to {out_dir}; target {num_episodes} episodes", flush=True)

        attempted = 0
        while ep < num_episodes:
            if attempted >= len(demo_keys) * 3:
                raise RuntimeError(
                    f"replay failed {attempted} attempts (source demos exhausted "
                    f"{len(demo_keys)}x3); {num_episodes - ep} episodes still missing"
                )
            demo_key = demo_keys[attempted % len(demo_keys)]
            attempted += 1
            try:
                data = replay_demo(env, h5, demo_key)
            except Exception as e:  # one bad demo must not kill the run
                print(f"  {demo_key}: replay crashed ({e!r}) — dropped")
                continue
            if data is None:
                continue
            path = out_dir / f"episode_{ep:05d}.npz"
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "wb") as f:
                np.savez_compressed(f, **data)
            os.replace(tmp, path)
            ep += 1
            print(f"saved {path} ({ep}/{num_episodes})", flush=True)
        env.close()
    print(f"[replay] done: {ep} episodes in {out_dir}")


def type_check(env_name: str) -> str:
    """Map the robomimic env_name to the robosuite task class name (cosmetic)."""
    return {"Lift": "Lift", "PickPlaceCan": "PickPlaceCan", "NutAssemblySquare": "NutAssemblySquare"}[env_name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, required=True, choices=["Lift", "Can", "Square"])
    parser.add_argument("--source", type=str, required=True, help="low_dim_v15.hdf5 path")
    parser.add_argument("--out", type=str, default="demos")
    parser.add_argument("--num-episodes", type=int, default=50)
    args = parser.parse_args()

    source = Path(args.source)
    out_dir = Path(args.out) / args.task
    replay_task(source, out_dir, args.num_episodes)


if __name__ == "__main__":
    main()

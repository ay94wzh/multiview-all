"""Robosuite 1.5.2 environment wrapper (robomimic tasks: Lift / Can / Square).

Same contract as multiview's maniskill_env.py: five named cameras
(top/left/right/bottom for training, front held out for inference) live in the
env with known intrinsics/extrinsics, and the helper functions mirror
get_camera_params / get_view_rgbs / get_ee_pose / get_qpos / get_goal_pos.

The camera rig is IDENTICAL to script_robosuite_demos/replay_robomimic.py:
all five cameras sit on the front hemisphere (x > 0) of a sphere of radius
1.0 m centered on the tabletop look target (0, 0, 0.8), fovy 60, 160x160.
Cameras are injected into every model the env builds (the default model at
construction AND per-episode model files at reset_from_xml_string) by
overriding edit_model_xml — the same subclass trick the replay uses, so the
same env class serves both replay and rollout.

Robosuite specifics (vs ManiSkill):
  * no env.seed() / no env.action_space: seed np.random before reset(),
    use env.action_dim.
  * step() returns a 4-tuple; done is horizon-only (no success signal) —
    success comes from env._check_success().
  * robosuite uses IMAGE_CONVENTION="opengl": raw renders are bottom-up, so
    frames are np.flipud'd here (the npz demos store the same orientation).
  * gripper sign: -1 = open, +1 = close (replay stores actions verbatim).
  * env.reset() returns a single obs dict (no (obs, info) tuple).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
)
from robosuite.utils.transform_utils import convert_quat

VIEW_CAMERA_NAMES = (
    "top_camera",
    "left_camera",
    "right_camera",
    "bottom_camera",
    "front_camera",
)

# Same rig as the demo replay (script_robosuite_demos/replay_robomimic.py).
LOOK_TARGET = [0.0, 0.0, 0.8]   # tabletop workspace (ManiSkill used (0,0,0.2))
CAMERA_RADIUS = 1.0
CAMERA_FOVY = 60.0
IMAGE_SIZE = (160, 160)

# (azimuth_deg, elevation_deg) — identical to multiview's _CAMERA_SPHERICAL.
_CAMERA_SPHERICAL = {
    "top_camera": (0.0, 45.0),
    "left_camera": (40.0, 20.0),
    "right_camera": (-40.0, 20.0),
    "bottom_camera": (0.0, 5.0),
    "front_camera": (0.0, 30.0),
}

# Task -> robosuite env name (the hdf5 env_args env_name).
TASK_TO_ENV_NAME = {
    "Lift": "Lift",
    "Can": "PickPlaceCan",
    "Square": "NutAssemblySquare",
}


def _spherical_eye(name: str) -> np.ndarray:
    """Camera position: look target + radius * direction (multiview convention)."""
    az, el = np.deg2rad(_CAMERA_SPHERICAL[name])
    off = CAMERA_RADIUS * np.array(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
    )
    return np.asarray(LOOK_TARGET) + off


def _camera_elements(camera_names: tuple[str, ...] = VIEW_CAMERA_NAMES) -> list[ET.Element]:
    """The cameras as MuJoCo <camera> elements (pos + quat wxyz).

    Frame columns [right | up | -forward]; MuJoCo cameras look along -z.
    Quat ordering gotcha: robosuite's mat2quat returns (x, y, z, w) while the
    MuJoCo XML quat attribute is (w, x, y, z) — the reorder below is
    load-bearing (verified empirically: without it the cameras point ~60-90 deg
    off the look target).
    """
    from robosuite.utils.transform_utils import mat2quat

    elems = []
    for name in camera_names:
        eye = _spherical_eye(name)
        forward = np.asarray(LOOK_TARGET) - eye
        forward /= np.linalg.norm(forward)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)  # re-orthonormalize
        R_cam2world = np.stack([right, up, -forward], axis=1)  # columns
        quat = mat2quat(R_cam2world)  # robosuite returns (x, y, z, w)!
        quat = quat[[3, 0, 1, 2]]  # reorder to (w, x, y, z) for the XML
        elem = ET.Element("camera")
        elem.set("name", name)
        elem.set("mode", "fixed")
        elem.set("pos", " ".join(f"{v:.6f}" for v in eye))
        elem.set("quat", " ".join(f"{v:.6f}" for v in quat))
        elem.set("fovy", f"{CAMERA_FOVY:.3f}")
        elems.append(elem)
    return elems


def _add_cameras_processor(xml: str) -> str:
    """XML processor: append the cameras to the top-level worldbody.

    Runs on every model string the env compiles (the processor must RETURN
    the xml — a documented robosuite quirk). Existing cameras with our names
    are removed first so repeated calls stay idempotent.
    """
    root = ET.fromstring(xml)
    wb = root.find("worldbody")
    assert wb is not None, "model xml has no top-level <worldbody>"
    for elem in wb.findall("camera"):
        if elem.get("name") in VIEW_CAMERA_NAMES:
            wb.remove(elem)
    for elem in _camera_elements():
        wb.append(elem)
    return ET.tostring(root, encoding="utf8").decode("utf8")


def _env_class_with_cameras(env_cls):
    """Subclass that injects the cameras into every model it builds.

    Robosuite validates camera observables (renders each one) during env
    __init__ against the *default* model — before any set_xml_processor could
    run — so the cameras must be in the default model too. edit_model_xml is
    the first element of the env's xml-processor chain and runs on every model
    string, so overriding it covers both the construction-time model and any
    per-episode model loaded via reset_from_xml_string.
    """
    class _MultiviewEnv(env_cls):
        def edit_model_xml(self, xml_str: str) -> str:
            xml_str = super().edit_model_xml(xml_str)
            return _add_cameras_processor(xml_str)

    _MultiviewEnv.__name__ = f"MultiView{env_cls.__name__}"
    return _MultiviewEnv


def _default_controller_configs() -> dict:
    """OSC_POSE world-frame delta controller — the robomimic dataset recipe.

    Verbatim copy of the v15 hdf5 `env_args.env_kwargs.controller_configs`
    (robomimic's 1.4-era dict: legacy keys `damping` / `control_delta` /
    `damping_limits` are accepted by robosuite 1.5.2 — proven by the demo
    replay, which passes this exact dict to robosuite.make). Replay and
    rollout must use the SAME controller or the demos' action semantics
    change. input_ref_frame='world' and the gripper sub-config are the
    critical bits — the BASIC composite defaults are 'base' and a plain
    position gripper.
    """
    return {
        "type": "BASIC",
        "body_parts": {
            "right": {
                "type": "OSC_POSE",
                "input_max": 1,
                "input_min": -1,
                "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                "kp": 150,
                "damping": 1,
                "impedance_mode": "fixed",
                "kp_limits": [0, 300],
                "damping_limits": [0, 10],
                "position_limits": None,
                "orientation_limits": None,
                "uncouple_pos_ori": True,
                "control_delta": True,
                "interpolation": None,
                "ramp_ratio": 0.2,
                "input_ref_frame": "world",
                "gripper": {"type": "GRIP"},
            }
        },
    }


def make_env(
    task: str,
    camera_names: tuple[str, ...] = VIEW_CAMERA_NAMES,
    image_size: tuple[int, int] = IMAGE_SIZE,
    camera_radius: float = CAMERA_RADIUS,
    seed: int | None = None,
    **env_kwargs,
):
    """Build the task env with our cameras, mirroring the dataset env recipe.

    Additional env_kwargs pass through to the env constructor (robosuite.make
    style). Camera obs are forced on at the given resolution; the controller,
    control_freq (20), lite_physics and reward settings match the robomimic
    hdf5 env_args so rollouts act in the same space the demos were recorded.
    """
    import robosuite as suite
    from robosuite.environments import REGISTERED_ENVS

    H, W = image_size
    kwargs = dict(env_kwargs)
    kwargs.setdefault("robots", ["Panda"])
    kwargs.setdefault("controller_configs", _default_controller_configs())
    kwargs.setdefault("control_freq", 20)
    kwargs.setdefault("lite_physics", False)
    kwargs.setdefault("ignore_done", True)
    kwargs.setdefault("reward_shaping", False)
    kwargs.setdefault("use_object_obs", True)
    kwargs.setdefault("has_renderer", False)
    kwargs.setdefault("has_offscreen_renderer", True)
    kwargs.setdefault("use_camera_obs", True)
    kwargs.setdefault("camera_names", list(camera_names))
    kwargs.setdefault("camera_heights", H)
    kwargs.setdefault("camera_widths", W)
    kwargs.setdefault("camera_depths", False)
    if seed is not None:
        np.random.seed(seed)  # robosuite has no env.seed(); seed numpy instead
    return _env_class_with_cameras(REGISTERED_ENVS[TASK_TO_ENV_NAME[task]])(**kwargs)


def get_camera_params(env, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Camera K, R, t with X_cam = R @ X_world + t (CV convention).

    Same extraction as the demo replay: R/t from the sim's cam_xpos/cam_xmat
    (world->camera pose with the OpenCV axis correction), K from cam_fovy.
    """
    H, W = IMAGE_SIZE
    K = get_camera_intrinsic_matrix(env.sim, name, H, W)
    P = get_camera_extrinsic_matrix(env.sim, name)  # 4x4 world->camera
    R = P[:3, :3].T
    t = -R @ env.sim.data.cam_xpos[env.sim.model.camera_name2id(name)]
    return K.astype(np.float32), R.astype(np.float32), t.astype(np.float32)


def get_view_rgbs(obs: dict, camera_names: tuple[str, ...] = VIEW_CAMERA_NAMES) -> dict[str, np.ndarray]:
    """Per-camera RGB frames (H, W, 3) uint8 from a robosuite obs dict.

    Robosuite's IMAGE_CONVENTION="opengl" leaves renders bottom-up; frames are
    flipped here to the top-down orientation the demos store.
    """
    return {name: np.flipud(obs[f"{name}_image"]) for name in camera_names}


def get_ee_pose(env) -> np.ndarray:
    """End-effector pose as (7,) xyz + wxyz quaternion (robosuite obs is xyzw)."""
    obs = env._get_observations()
    xyz = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    quat = convert_quat(np.asarray(obs["robot0_eef_quat_site"]), to="wxyz")
    return np.concatenate([xyz, quat]).astype(np.float32)


def get_qpos(obs: dict) -> np.ndarray:
    """Robot joint positions (7,) from a robosuite obs dict."""
    return np.asarray(obs["robot0_joint_pos"], dtype=np.float32)


def get_goal_pos(env) -> np.ndarray:
    """Task-specific goal position (3,) — static within an episode.

    Lift    -> current cube position (body frame of the cube);
    Can     -> center of the goal bin (static env param, bin2_pos);
    Square  -> midpoint of the two pegs (the goal is *which* peg the nut goes
               on, so the midpoint is a static approximation — see README).
    """
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
    raise NotImplementedError(f"goal_pos undefined for task {type(env).__name__!r}")

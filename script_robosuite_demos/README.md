# script_robosuite_demos — replay robomimic demos with the multiview camera rig

The `multiview` policy uses five named cameras (`top_camera`, `left_camera`,
`right_camera`, `bottom_camera`, `front_camera`). The official robomimic PH
datasets were recorded without them, so demos are replayed through robosuite
with the cameras injected, and re-saved in the same per-episode npz schema as
`multiview/scripts/collect_data.py` — the dataset code in `policy_robosuite`
consumes them directly.

## Usage

```bash
conda activate multiview-robosuite   # or: conda run -n multiview-robosuite python ...

# Lift / Can / Square — 50 demos each, written to demos/<task>/episode_XXXXX.npz
python replay_robomimic.py --task Lift   --source /path/to/robomimic/datasets/lift/ph/low_dim_v15.hdf5   --out demos
python replay_robomimic.py --task Can    --source /path/to/robomimic/datasets/can/ph/low_dim_v15.hdf5    --out demos
python replay_robomimic.py --task Square --source /path/to/robomimic/datasets/square/ph/low_dim_v15.hdf5 --out demos

# visual check: one mp4 montage per episode (2x3 grid of the five cameras)
python play_dataset.py --task Lift
```

Flags: `--num-episodes` (default 50; runs are resume-friendly — existing
episodes are kept and numbering continues), `--source` (the robomimic
`low_dim_v15.hdf5`), `--out` (parent dir; task subdir is appended).

## How the replay works

1. `env_args` (recorded in the hdf5 `data` attrs) reconstruct the dataset
   environment: `robosuite.make(env_name, **env_kwargs)` with the original
   composite OSC_POSE controller (`control_delta: True`, world frame), Panda,
   `control_freq: 20`, `lite_physics: False`. Only the render settings are
   overridden (`has_offscreen_renderer: True`, `use_camera_obs: True`, the
   five cameras at 160x160).
2. The per-episode `model_file` xml (also recorded in the hdf5) is loaded via
   `reset_from_xml_string` so the sim matches the recorded states exactly.
   robosuite's own `edit_model_xml` processor rewrites the recorded absolute
   asset paths to the local install.
3. The initial sim state is restored with `set_state_from_flattened(states[0])`
   and the recorded actions are re-executed with `env.step`.
4. Frames, qpos, ee pose, actions and camera intrinsics/extrinsics are captured
   at every step; K/R/t come from the live sim (`cam_xpos`/`cam_xmat`/`cam_fovy`).
5. Episodes are kept only if `env._check_success()` holds at the final state —
   low-level dynamics drift between collection and replay, so some source
   demos fail the gate and are dropped. Episodes shorter than the dataset's
   minimum window (`window_steps 24 + obs_horizon 3 + horizon 16 = 43`) are
   also dropped.

## Camera rig

Identical to `multiview`'s `_CAMERA_SPHERICAL`: all five cameras sit on the
front hemisphere (x > 0) of a sphere of radius 1.0 m centered on the look
target (0, 0, 0.8) — the robosuite tabletop workspace (ManiSkill used
(0, 0, 0.2); the robosuite table surface is ~0.815 m).

| camera        | azimuth | elevation |
|---------------|---------|-----------|
| top_camera    |   0°    |    45°    |
| left_camera   | +40°    |    20°    |
| right_camera  |  -40°   |    20°    |
| bottom_camera |   0°    |     5°    |
| front_camera  |   0°    |    30°    |

Cameras are injected as fixed `<camera>` elements (pos + quat **wxyz**, frame
columns `[right | up | -forward]`) by overriding the env's `edit_model_xml`
processor. Note the quat reorder: robosuite's `mat2quat` returns (x,y,z,w)
while the MuJoCo XML `quat` attribute is (w,x,y,z) — the replay reorders
before writing, or the cameras point sideways (verified empirically). The
processor, so they are present both at env construction (robosuite validates
camera observables by rendering them during `__init__`) and in every
per-episode model.

## Conventions

- **Frames are stored top-down**: robosuite's `IMAGE_CONVENTION="opengl"`
  leaves MuJoCo renders unflipped (row 0 = bottom); the replay applies
  `np.flipud` before storing, matching ManiSkill sensor frames.
- **Gripper sign**: robosuite uses −1 = open, +1 = close (opposite of the
  ManiSkill collector). Replayed actions are stored verbatim, so the npz
  dataset is internally consistent — never mix these demos with ManiSkill
  demos.
- **Camera params**: `X_cam = R @ X_world + t` (CV convention, same as
  `plucker_map`); `K = [[f, 0, W/2], [0, f, H/2], [0, 0, 1]]` with
  `f = 0.5 * H / tan(fovy/2)`, fovy 60°.
- **goal_pos (3,) per task** (static within an episode):
  - Lift → current cube position (`body_xpos[cube_body_id]`);
  - Can → center of the goal bin (`env.bin2_pos`);
  - Square → midpoint of the two pegs (the task's goal is *which* peg the nut
    goes on, so the midpoint is a static two-valued-goal approximation — note
    this when interpreting Square results).

## npz schema (per episode)

```
qpos       (T, 7)      joint positions
ee_pose    (T+1, 7)    xyz + wxyz quaternion (one row per frame, incl. init)
actions    (T, 7)      recorded OSC_POSE delta actions
goal_pos   (3,)        task goal
success    bool
<view>/rgb (T+1, H, W, 3) uint8   (one frame more than actions, as usual)
<view>/K, <view>/R, <view>/t      camera params (float32)
```

T+1 frames vs T actions is the standard diffusion-policy alignment (the
dataset indexes frames/actions independently and pads safely).

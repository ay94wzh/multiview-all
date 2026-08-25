"""Plücker ray maps: the geometric camera conditioning of the encoder.

Math (world <-> camera). Extrinsics are given as [R | t] such that

    X_cam = R @ X_world + t

Then:
    camera center in world:    o   = -R.T @ t
    pixel ray direction (cam): d_c = normalize(K^{-1} @ [u, v, 1])
    ray direction in world:    d   = R.T @ d_c
    Plücker coordinates:       (o x d, d)   -> (H, W, 6)

For *static* cameras the map is constant per camera: computed once, cached.
At inference the same function computes the map of the held-out front_camera,
which is what forces the encoder to read geometry instead of memorizing views.

Naming note: "Plücker" honors Julius Plücker — the ü matters in papers,
not in code.
"""
from __future__ import annotations

import numpy as np


def pixel_directions_cam(K: np.ndarray, H: int, W: int) -> np.ndarray:
    """Normalized per-pixel ray directions in the camera frame.

    Args:
        K: (3, 3) intrinsics.
        H, W: image height/width.

    Returns:
        (H, W, 3) float32, unit-norm directions.
    """
    us, vs = np.meshgrid(np.arange(W), np.arange(H))  # (H, W) each
    uv1 = np.stack([us, vs, np.ones_like(us)], axis=-1).astype(np.float64)  # (H, W, 3)
    Kinv = np.linalg.inv(K)
    dirs = uv1 @ Kinv.T  # (H, W, 3), unnormalized
    return (dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)).astype(np.float32)


def plucker_map(K: np.ndarray, R: np.ndarray, t: np.ndarray, H: int, W: int) -> np.ndarray:
    """Per-pixel Plücker ray coordinates (o x d, d) in the world frame.

    Args:
        K: (3, 3) intrinsics.
        R, t: (3, 3), (3,) extrinsics with X_cam = R @ X_world + t.
        H, W: image height/width.

    Returns:
        (H, W, 6) float32; channels [3:6] are unit-norm directions.
    """
    d_c = pixel_directions_cam(K, H, W)  # (H, W, 3)
    # d_w = R.T @ d_c  <=>  d_c @ R  (row-vector convention of the last axis)
    d = d_c @ R.astype(np.float64)
    o = -(R.T @ t).reshape(1, 1, 3)
    m = np.cross(np.broadcast_to(o, d.shape), d)  # moment o x d
    return np.concatenate([m, d], axis=-1).astype(np.float32)


def transform_delta_action(delta: np.ndarray, R_w2c: np.ndarray) -> np.ndarray:
    """Transform a delta-EE action [dp | domega | ...] from base to camera frame.

    dp_cam = R @ dp, domega_cam = R @ domega. The axis-angle delta rotates as a
    vector because R (omega^) R^T = (R omega)^. Any extra channels (e.g. the
    gripper dimension of pd_ee_delta_pose) are scalar and pass through unchanged.

    NOTE: this applies to *delta* actions only. For absolute 3D poses the
    rotation part must be conjugated (R @ q @ R^T) — see README, corrections #4.
    """
    delta = np.asarray(delta, dtype=np.float32)
    R = np.asarray(R_w2c, dtype=np.float32)
    out = delta.copy()
    out[..., :3] = delta[..., :3] @ R.T  # R @ dp == dp @ R.T
    out[..., 3:6] = delta[..., 3:6] @ R.T
    return out


def transform_delta_action_to_base(delta_cam: np.ndarray, R_w2c: np.ndarray) -> np.ndarray:
    """Inverse of :func:`transform_delta_action` (camera frame -> base frame)."""
    return transform_delta_action(delta_cam, np.asarray(R_w2c, dtype=np.float32).T)

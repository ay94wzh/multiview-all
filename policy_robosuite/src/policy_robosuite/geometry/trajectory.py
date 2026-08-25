"""Per-view rendering of the recent end-effector trajectory as a grayscale
trail image — one extra encoder input channel.

Plan point 1: "The trajectory of end effector in a certain time window is
expressed as grayscale image from each view point." Because the trail is
projected through each camera's known intrinsics/extrinsics, it is
view-dependent by construction and carries both motion and viewpoint signal.

Design choices (see README, corrections #3):
- recency-faded segments: intensity = recency_decay ** age
- anti-aliased polylines (cv2 LINE_AA) instead of single-pixel lines
- bright dot at the current EE position

Note: rendering every frame in the dataset sampler is too slow; datasets
precompute trails per episode once (see data/dataset.py).
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from ..config import TrajectoryConfig


def _project(P: np.ndarray, p_world: np.ndarray):
    """Project a 3D world point to (u, v) image coords; None if behind camera."""
    x = P @ np.append(p_world, 1.0)
    if x[2] <= 1e-6:
        return None
    return (float(x[0] / x[2]), float(x[1] / x[2]))


def render_trail(
    positions_world: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    H: int,
    W: int,
    cfg: TrajectoryConfig,
) -> np.ndarray:
    """Render the trail of `positions_world` (T, 3) through one camera.

    Args:
        positions_world: (T, 3) EE positions, oldest first.
        K, R, t: camera intrinsics/extrinsics (X_cam = R @ X_world + t).
        H, W: output image size.
        cfg: trail styling.

    Returns:
        (H, W, 1) float32 in [0, 1].
    """
    img = np.zeros((H, W), dtype=np.float32)
    if len(positions_world) < 2:
        return img[..., None]
    P = K @ np.concatenate([R, np.asarray(t).reshape(3, 1)], axis=1)
    n = len(positions_world)
    for i in range(n - 1):
        a = _project(P, positions_world[i])
        b = _project(P, positions_world[i + 1])
        if a is None or b is None:
            continue
        age = n - 1 - i  # 0 = newest segment
        intensity = float(cfg.recency_decay ** age)
        cv2.line(
            img,
            (int(round(a[0])), int(round(a[1]))),
            (int(round(b[0])), int(round(b[1]))),
            color=intensity,
            thickness=cfg.line_width,
            lineType=cv2.LINE_AA,
        )
    cur = _project(P, positions_world[-1])
    if cur is not None:
        cv2.circle(
            img,
            (int(round(cur[0])), int(round(cur[1]))),
            radius=cfg.line_width + 1,
            color=1.0,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    return img[..., None]


def render_trails(
    positions_world: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    H: int,
    W: int,
    cfg: TrajectoryConfig,
) -> np.ndarray:
    """Trail image for every timestep of an episode.

    Args:
        positions_world: (T, 3) EE positions of the whole episode.

    Returns:
        (T, 1, H, W) uint8 trails; trail[t] covers
        positions[max(0, t - window_steps + 1) .. t].
    """
    T = len(positions_world)
    out = np.zeros((T, 1, H, W), dtype=np.uint8)
    window: deque = deque(maxlen=cfg.window_steps)
    for i in range(T):  # note: loop var must not shadow the translation `t`
        window.append(positions_world[i])
        trail = render_trail(np.asarray(window), K, R, t, H, W, cfg)  # (H, W, 1)
        out[i] = np.transpose(trail * 255.0, (2, 0, 1)).astype(np.uint8)
    return out


class TrajectoryRenderer:
    """Online renderer for rollout/eval: keeps the last `window_steps` EE
    positions in a deque and renders on demand (used by eval/rollout.py)."""

    def __init__(self, cfg: TrajectoryConfig):
        self.cfg = cfg
        self._points: deque = deque(maxlen=cfg.window_steps)

    def reset(self) -> None:
        self._points.clear()

    def add(self, ee_pos_world: np.ndarray) -> None:
        self._points.append(np.asarray(ee_pos_world, dtype=np.float64))

    def render(self, K: np.ndarray, R: np.ndarray, t: np.ndarray, H: int, W: int) -> np.ndarray:
        if not self._points:
            return np.zeros((H, W, 1), dtype=np.float32)
        return render_trail(np.asarray(self._points), K, R, t, H, W, self.cfg)

    def render_past(
        self, K: np.ndarray, R: np.ndarray, t: np.ndarray, H: int, W: int, steps_ago: int = 0
    ) -> np.ndarray:
        """Trail as of `steps_ago` steps back (0 = up to the current EE).

        Used for per-frame FiLM conditioning: the frame at window position k
        gets the trail covering history up to that frame's own timestep.
        Returns zeros if `steps_ago` exceeds the stored history.
        """
        pts = list(self._points)
        if steps_ago >= len(pts):
            return np.zeros((H, W, 1), dtype=np.float32)
        return render_trail(np.asarray(pts[: len(pts) - steps_ago]), K, R, t, H, W, self.cfg)

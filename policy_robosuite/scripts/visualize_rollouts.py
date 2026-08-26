#!/usr/bin/env python
"""Turn recorded rollouts (scripts/rollout_record.py) into diagnosis assets.

Reads <in_dir>/episode_*.npz (JPEG-encoded policy-input views + high-res
render frames + per-step state) and writes into the same dir:

  episode_XXXXX.mp4    side-by-side policy-views | render video of the
                       rollout (left: the policy's actual input views — 2x2
                       grid of the 4 train views for the baseline, single
                       student view for the multiview arch; right: high-res
                       render_camera showing the goal site)
  montage_grid.png     at-a-glance grid: one row per episode, 4 keyframes
  trajectories.png     top-down object path per episode (small multiples)
  diagnostics.png      2x2 panel of aggregate failure diagnostics

Robosuite port of multiview's visualize_rollouts.py: per-view frame keys
(<view>_rgb) instead of a single front_camera, obj_pose instead of cube_pose,
a data-driven plot limit (the ManiSkill 0.35 m fixed limit does not fit the
robosuite tabletop layouts), and no top_rgb fallback (robosuite recordings
always carry render_rgb).

No simulator needed. Usage:
    python scripts/visualize_rollouts.py --in runs/rollout_vis
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # headless: save to png, never open a window
import matplotlib.pyplot as plt
import numpy as np

from policy_robosuite.viz.video import write_video

# Reference palette (validated default, light surface) — see dataviz palette.md.
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_BLUE = "#2a78d6"          # categorical slot 1 / sequential base
_ORANGE = "#eb6834"        # categorical slot 2
_AQUA = "#1baf7a"          # categorical slot 3
_GOOD = "#0ca30c"          # status: success
_BAD = "#d03b3b"           # status: failure
_BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6",
              "#1c5cab", "#104281"]  # sequential blue 100..650
_STEP_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("step", _BLUE_RAMP)

_SERIES = [_BLUE, _ORANGE, _AQUA]  # first 3 categorical slots (all-pairs validated)


def _decode(buf) -> np.ndarray:
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(bytes(buf), np.uint8),
                                     cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _short(name: str) -> str:
    return name.removesuffix("_camera")


def load_episode(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        if "render_rgb" not in z:
            raise SystemExit(f"{path}: no render_rgb (robosuite recordings always have it; "
                             f"is this a multiview/ManiSkill recording?)")
        views = [str(v) for v in z["views"]] if "views" in z else ["front_camera"]
        rec = {
            "success": bool(z["success"]),
            "steps": int(z["steps"]),
            "views": views,
            "frames": {v: [_decode(b) for b in z[f"{v}_rgb"]] for v in views},
            "render": [_decode(b) for b in z["render_rgb"]],
            "arch": str(z["arch"]) if "arch" in z else "multiview",
            "obj": z["obj_pose"][:, :3],
            "ee": z["ee_pose"][:, :3],
            "goal": z["goal_pos"][0],
            "action": z["action"],
            "is_grasped": z["is_grasped"],
        }
    return rec


def _policy_panel(rec: dict, t: int, Hr: int, Wr: int) -> np.ndarray:
    """Left panel: the policy's input views, scaled to the render size.

    One view -> direct upscale (multiview arch: the student view); four views
    -> 2x2 grid of upscaled tiles (baseline: the train views). Both panels
    align in time and the input views' low resolution stays visibly honest.
    """
    views = rec["views"]
    if len(views) == 1:
        frame = cv2.resize(rec["frames"][views[0]][t], (Wr, Hr),
                           interpolation=cv2.INTER_LINEAR)
        cv2.putText(frame, f"{views[0]} (policy input)", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return frame
    cols = int(np.ceil(np.sqrt(len(views))))
    rows = int(np.ceil(len(views) / cols))
    tile_w, tile_h = Wr // cols, Hr // rows
    panel = np.full((Hr, Wr, 3), 24, dtype=np.uint8)
    for i, v in enumerate(views):
        r, c = i // cols, i % cols
        tile = cv2.resize(rec["frames"][v][t], (tile_w, tile_h),
                          interpolation=cv2.INTER_LINEAR)
        cv2.putText(tile, _short(v), (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
        panel[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = tile
    return panel


def render_episode_video(rec: dict, ep_id: str, out_path: Path, fps: int = 15) -> None:
    """Side-by-side policy-views | render montage with step counter + success flag."""
    T = len(rec["render"])
    Hr, Wr = rec["render"][0].shape[:2]
    pad = np.full((Hr, 8, 3), 40, dtype=np.uint8)

    def frames():
        for t in range(T):
            render = rec["render"][t].copy()
            cv2.putText(render, "render_camera (goal visible)", (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            row = np.concatenate([_policy_panel(rec, t, Hr, Wr), pad, render], axis=1)
            frame = np.full((Hr + 22, 2 * Wr + 8, 3), 24, dtype=np.uint8)
            frame[:Hr] = row
            label = f"{ep_id}  step {t}/{T - 1}"
            if t == T - 1:
                label += "   SUCCESS" if rec["success"] else "   FAIL"
            cv2.putText(frame, label, (6, Hr + 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            yield frame

    write_video(frames(), out_path, fps, (Hr + 22, 2 * Wr + 8))


def montage_grid(recs: list[dict], out_path: Path) -> None:
    """One row per episode, keyframes at 0 / 33% / 66% / last of the first view."""
    view = recs[0]["views"][0]
    H, W = recs[0]["frames"][view][0].shape[:2]
    n = len(recs)
    cols = ["start", "33%", "66%", "end"]
    tile_w, tile_h = W + 6, H + 6
    canvas = np.full((n * tile_h + 30, 4 * tile_w + 90, 3), 255, dtype=np.uint8)
    for i, name in enumerate(cols):
        cv2.putText(canvas, name, (90 + i * tile_w + 6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _ink_bgr(_INK_2), 1, cv2.LINE_AA)
    for r, rec in enumerate(recs):
        frames = rec["frames"][view]
        T = len(frames)
        idxs = [0, max(T - 1, 0) * 1 // 3, max(T - 1, 0) * 2 // 3, T - 1]
        y0 = 30 + r * tile_h
        cv2.putText(canvas, f"ep {r:02d}", (6, y0 + H // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, _ink_bgr(_INK_2), 1, cv2.LINE_AA)
        for c, t in enumerate(idxs):
            x0 = 90 + c * tile_w
            canvas[y0 : y0 + H, x0 : x0 + W] = frames[t]
        status = "ok" if rec["success"] else "x"
        color = _GOOD if rec["success"] else _BAD
        cv2.putText(canvas, status, (90 + 4 * tile_w + 4, y0 + H // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _ink_bgr(color), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"wrote {out_path}", flush=True)


def _ink_bgr(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (4, 2, 0))


def _style_ax(ax, xlabel: str, ylabel: str, xlim=None, ylim=None) -> None:
    ax.set_facecolor(_SURFACE)
    ax.set_xlabel(xlabel, color=_INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=_INK_2, fontsize=9)
    ax.tick_params(colors=_MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color("#c3c2b7")
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True, color=_GRID, lw=0.8)


def trajectories(recs: list[dict], out_path: Path) -> None:
    """Top-down object path per episode, colored by step; goal = orange star.

    Plot limit is data-driven: the ManiSkill fixed 0.35 m around the goal does
    not fit the robosuite tabletop layouts (Lift ~0.2 m around the goal, Can's
    bins ~0.5 m apart), so the limit follows the recorded extents.
    """
    cols = 4
    rows = (len(recs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.6 * rows))
    fig.patch.set_facecolor(_SURFACE)
    all_pts = np.concatenate([np.vstack([r["obj"], r["goal"][None]]) for r in recs])
    extent = max(float(np.ptp(all_pts[:, 0])), float(np.ptp(all_pts[:, 1])))
    lim = max(0.25, 0.5 * extent + 0.05)
    for i, (rec, ax) in enumerate(zip(recs, np.ravel(axes))):
        obj = rec["obj"]
        goal = rec["goal"]
        T = len(obj)
        pts = ax.scatter(obj[:, 0], obj[:, 1], c=np.arange(T), cmap=_STEP_CMAP,
                         s=6, linewidths=0, zorder=3)
        ax.plot(obj[:, 0], obj[:, 1], color=_BLUE, lw=1.0, alpha=0.45, zorder=2)
        ax.scatter(obj[0, 0], obj[0, 1], marker="o", s=28, facecolor="none",
                   edgecolor=_INK_2, lw=1.2, zorder=4)
        ax.scatter(goal[0], goal[1], marker="*", s=130, color=_ORANGE, zorder=5)
        ax.set_xlim(goal[0] - lim, goal[0] + lim)
        ax.set_ylim(goal[1] - lim, goal[1] + lim)
        ax.set_aspect("equal")
        _style_ax(ax, "", "")
        ax.set_xticks([])
        ax.set_yticks([])
        status = "success" if rec["success"] else "fail"
        color = _GOOD if rec["success"] else _BAD
        ax.set_title(f"ep {i:02d} · {status} · {T - 1} steps", color=color, fontsize=9)
    for ax in np.ravel(axes)[len(recs):]:
        ax.axis("off")
    fig.suptitle("object path (top-down) — dot = start, star = goal",
                 color=_INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def _final_dist(rec: dict) -> float:
    return float(np.linalg.norm(rec["obj"][-1] - rec["goal"]))


def diagnostics(recs: list[dict], out_path: Path) -> None:
    """2x2 panel: final dist bars, steps bars, d_goal(t) for 3 eps, gripper(t)."""
    final = np.array([_final_dist(r) for r in recs])
    steps = np.array([r["steps"] for r in recs])
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.patch.set_facecolor(_SURFACE)

    # 1. final object->goal distance per episode (success bar in green)
    ax = axes[0][0]
    colors = [_GOOD if r["success"] else _BLUE for r in recs]
    ax.bar(np.arange(len(recs)), final, color=colors, width=0.7, zorder=3)
    ax.axhline(0.025, color=_INK_2, lw=1, ls="--")
    ax.text(len(recs) - 1, 0.025, "  goal radius 2.5 cm", color=_INK_2,
            fontsize=8, va="bottom")
    _style_ax(ax, "episode", "final object→goal distance (m)")
    ax.set_xticks(np.arange(len(recs)), [str(i) for i in range(len(recs))])
    ax.set_title("did the object reach the goal?", color=_INK, fontsize=10)

    # 2. steps per episode
    ax = axes[0][1]
    ax.bar(np.arange(len(recs)), steps, color=_BLUE, width=0.7, zorder=3)
    _style_ax(ax, "episode", "steps taken (max 300)")
    ax.set_xticks(np.arange(len(recs)), [str(i) for i in range(len(recs))])
    ax.set_title("how long did the rollout last?", color=_INK, fontsize=10)

    # 3. d_goal(t) — the success episode plus two representative failures
    ax = axes[1][0]
    succ = [i for i, r in enumerate(recs) if r["success"]]
    pick = (succ + [i for i in range(len(recs)) if i not in succ])[:3]
    for k, i in enumerate(pick):
        r = recs[i]
        d = np.linalg.norm(r["obj"] - r["goal"][None], axis=1)
        ax.plot(np.arange(len(d)), d, color=_SERIES[k], lw=2,
                label=f"ep {i}" + (" (success)" if r["success"] else ""))
    ax.axhline(0.025, color=_INK_2, lw=1, ls="--")
    _style_ax(ax, "step", "object→goal distance (m)")
    ax.set_title("distance to goal over the episode", color=_INK, fontsize=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=_INK)

    # 4. gripper action for the same three episodes
    ax = axes[1][1]
    for k, i in enumerate(pick):
        r = recs[i]
        ax.plot(np.arange(len(r["action"])), r["action"][:, 6],
                color=_SERIES[k], lw=1.4, label=f"ep {i}")
        grasp = np.where(r["is_grasped"] > 0.5)[0]
        if len(grasp):
            ax.scatter(grasp, np.full_like(grasp, 1.15), marker="^", s=24,
                       color=_SERIES[k])
    _style_ax(ax, "step", "gripper action (+1 open / -1 close)", ylim=(-1.3, 1.35))
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set_title("gripper command — did it ever grasp? (▲ = grasped)",
                 color=_INK, fontsize=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=_INK)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_dir", type=str, default="runs/rollout_vis")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    files = sorted(in_dir.glob("episode_[0-9][0-9][0-9][0-9][0-9].npz"))
    if not files:
        raise SystemExit(f"no episode_*.npz in {in_dir}; run scripts/rollout_record.py first")

    recs = [load_episode(f) for f in files]
    for f, rec in zip(files, recs):
        render_episode_video(rec, f.stem, f.with_suffix(".mp4"))
        print(f"wrote {f.with_suffix('.mp4')}", flush=True)
    montage_grid(recs, in_dir / "montage_grid.png")
    trajectories(recs, in_dir / "trajectories.png")
    diagnostics(recs, in_dir / "diagnostics.png")


if __name__ == "__main__":
    main()

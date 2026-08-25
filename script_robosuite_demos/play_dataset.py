#!/usr/bin/env python
"""Render montage videos of replayed robosuite episodes for visual checks.

Reads the per-episode npz files written by replay_robomimic.py and writes one
mp4 per episode under demos/<task>/videos/: a 2x3 grid of the five camera
views with view labels and a step counter. No simulator needed — the videos
come straight from the stored RGB frames.

Usage:
    python play_dataset.py --task Lift
    python play_dataset.py --task Can --max 5
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

VIEW_CAMERA_NAMES = (
    "top_camera",
    "left_camera",
    "right_camera",
    "bottom_camera",
    "front_camera",
)

# 2x3 grid: row 1 = top / left / right, row 2 = bottom / front / (blank)
_GRID = (2, 3)


def write_video(frames, out_path: Path, fps: int, size: tuple[int, int]) -> None:
    """Write an iterator of uint8 RGB frames to an H.264 mp4 (ffmpeg, or OpenCV fallback)."""
    H, W = size
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-pix_fmt", "yuv420p", str(out_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for frame in frames:
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {out_path}")
        return
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {out_path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def make_montage(rgb: dict[str, np.ndarray], t: int, ep_id: str, H: int, W: int) -> np.ndarray:
    """One 2x3 grid frame from the five per-view images at step t."""
    rows = []
    for r in range(_GRID[0]):
        cols = []
        for c in range(_GRID[1]):
            cell = np.full((H, W, 3), 32, dtype=np.uint8)  # dark blank cell
            idx = r * _GRID[1] + c
            name = VIEW_CAMERA_NAMES[idx] if idx < len(VIEW_CAMERA_NAMES) else ""
            if name in rgb:
                cell = rgb[name][t]
                cv2.putText(
                    cell, name, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA
                )
            cols.append(cell)
        rows.append(np.concatenate(cols, axis=1))
    frame = np.concatenate(rows, axis=0)
    cv2.putText(
        frame, f"{ep_id}  step {t}", (6, frame.shape[0] - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return frame


def render_episode_video(path: Path, out_path: Path, H: int, W: int, fps: int) -> None:
    with np.load(path) as z:
        T = len(z["actions"])
        rgb = {v: z[f"{v}/rgb"] for v in VIEW_CAMERA_NAMES}
    write_video(
        (make_montage(rgb, t, path.stem, H, W) for t in range(T)),
        out_path, fps, (H * _GRID[0], W * _GRID[1]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["Lift", "Can", "Square"])
    parser.add_argument("--episodes", type=str, default="",
                        help="comma-separated episode numbers, e.g. 0,5,12 (default: all)")
    parser.add_argument("--max", type=int, default=0, help="render at most the first N episodes")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--out", type=str, default="demos")
    parser.add_argument("--first-frame", type=str, default="",
                        help="optional: dump the first montage frame of the first video to this png")
    args = parser.parse_args()

    task_dir = Path(args.out) / args.task
    files = sorted(task_dir.glob("episode_[0-9][0-9][0-9][0-9][0-9].npz"))
    if not files:
        raise SystemExit(f"no episode_*.npz in {task_dir}; run replay_robomimic.py first")
    if args.episodes:
        wanted = {f"episode_{int(n):05d}" for n in args.episodes.split(",")}
        files = [f for f in files if f.stem in wanted]
    if args.max:
        files = files[: args.max]

    with np.load(files[0]) as z:
        H, W = z[f"{VIEW_CAMERA_NAMES[0]}/rgb"].shape[1:3]
    out_dir = task_dir / "videos"
    for i, f in enumerate(files):
        out_path = out_dir / f"{f.stem}.mp4"
        render_episode_video(f, out_path, H, W, args.fps)
        print(f"[{i + 1}/{len(files)}] wrote {out_path}", flush=True)
        if args.first_frame and i == 0:
            with np.load(f) as z:
                rgb = {v: z[f"{v}/rgb"] for v in VIEW_CAMERA_NAMES}
            first = make_montage(rgb, 0, f.stem, H, W)
            cv2.imwrite(args.first_frame, cv2.cvtColor(first, cv2.COLOR_RGB2BGR))
            print(f"wrote first frame to {args.first_frame}", flush=True)


if __name__ == "__main__":
    main()

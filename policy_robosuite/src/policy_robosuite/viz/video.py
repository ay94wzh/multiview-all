"""Small video-writing helper shared by the scripts that render mp4s.

OpenCV's `mp4v` (MPEG-4 Part 2) videos are not playable in VSCode's media
viewer or in browsers, so all videos go out as H.264: raw RGB frames are
piped to ffmpeg/libx264. If ffmpeg is not on PATH we fall back to OpenCV
mp4v (still viewable with VLC etc.).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def write_video(frames, out_path: Path, fps: int, size: tuple[int, int]) -> None:
    """Write an iterator/list of uint8 RGB frames (H, W, 3) to an H.264 mp4.

    Args:
        frames: iterable of (H, W, 3) uint8 RGB arrays, or a (T, H, W, 3) array.
        out_path: destination .mp4 path.
        fps: playback frame rate.
        size: (H, W) of one frame.
    """
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
    # no ffmpeg: OpenCV mp4v fallback
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {out_path}")
    try:
        for frame in frames:
            # frames are RGB; VideoWriter expects BGR
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

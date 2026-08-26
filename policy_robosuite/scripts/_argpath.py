"""Resolve relative path args so scripts work from any CWD.

Scripts default to CWD-relative paths (configs/eval.yaml, checkpoints/...),
which is the documented convention of running them from policy_robosuite/.
When a script is invoked from elsewhere (e.g. the repo root via
`python policy_robosuite/scripts/eval.py`), a relative path that does NOT
exist under the CWD is looked up relative to the repo's policy_robosuite/
dir (this file's parent) and then the repo root — so that invocation finds
configs/checkpoints too.

Paths the script CREATES (--out dirs, logs) are never redirected: call
resolve() only for paths that must already exist.
"""
from __future__ import annotations

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
_POLICY_DIR = SCRIPT_DIR.parent    # the repo's policy_robosuite/ dir
_REPO_ROOT = SCRIPT_DIR.parents[2]  # the MULTIVIEW repo root


def resolve(path: str | Path) -> Path:
    """CWD-relative with repo-tree fallback; absolute paths pass through.

    Returns the CWD-relative path unchanged when nothing exists anywhere —
    so callers keep a clear FileNotFoundError for genuinely missing files
    instead of an out-of-nowhere path.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    cwd_path = Path.cwd() / p
    if cwd_path.exists():
        return cwd_path
    for base in (_POLICY_DIR, _REPO_ROOT):
        cand = base / p
        if cand.exists():
            return cand
    return cwd_path

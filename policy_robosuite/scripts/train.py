#!/usr/bin/env python
"""Train the policy_robosuite policy.

Usage:
    python scripts/train.py --config configs/train_multiview.yaml \
        [--overrides key=value ...]
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

# Reduce allocator fragmentation (gradient-checkpoint recompute and mixed-size tensors).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from torch.utils.data import DataLoader, RandomSampler

from _argpath import resolve  # scripts/ dir (sys.path[0]) — see _argpath.py
from policy_robosuite.config import ProjectConfig
from policy_robosuite.data.dataset import MultiViewDataset, collate_fn
from policy_robosuite.models.policy import build_policy
from policy_robosuite.training.trainer import Trainer


def seed_worker(worker_id: int):
    """Per-worker, per-epoch RNG seed.

    Without this, every fork'd worker inherits the parent's global `random`
    and numpy RNG state and draws the *same* window-start sequence (and
    repeats it every epoch) — the dataset samples would be correlated across
    workers. The dataset's episode/window choice uses np.random, hence both.
    """
    import numpy as np
    import torch

    worker_info = torch.utils.data.get_worker_info()
    random.seed(worker_info.seed)
    # numpy's legacy RandomState only accepts seeds up to 2**32 - 1, while
    # torch hands workers 64-bit seeds; fold into range (decorrelation is all
    # we need here, not cryptographic strength).
    np.random.seed(worker_info.seed % (2**32 - 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_multiview.yaml")
    parser.add_argument("--overrides", nargs="*", default=None)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="start from scratch instead of resuming checkpoints/latest.pt",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="checkpoint to resume from (default: <checkpoint_dir>/latest.pt)",
    )
    args = parser.parse_args()

    cfg = ProjectConfig.from_yaml(resolve(args.config), args.overrides)
    dataset = MultiViewDataset(cfg)
    # Sample windows WITH REPLACEMENT so each epoch contains exactly
    # steps_per_epoch batches, regardless of the dataset size. Without this,
    # a 100-episode dataset at batch 16 yields only 6 batches/epoch (the
    # loader exhausts after one pass and steps_per_epoch silently caps at 6):
    # 500 "epochs" = 3000 updates, far too few to fit the demos (measured:
    # train loss plateaus at ~0.007 and rollouts diverge at episode starts).
    sampler = RandomSampler(
        dataset,
        replacement=True,
        num_samples=cfg.train.steps_per_epoch * cfg.data.batch_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        sampler=sampler,
        num_workers=4,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
    )
    model = build_policy(cfg)
    trainer = Trainer(cfg, model, loader)
    ckpt_path = (resolve(args.resume_from) if args.resume_from
                 else resolve(Path(cfg.train.checkpoint_dir) / "latest.pt"))
    if not args.no_resume and ckpt_path.exists():
        trainer.load(ckpt_path)
        print(f"[train] resumed from {ckpt_path} (epoch {trainer.epoch})", flush=True)
    trainer.train()


if __name__ == "__main__":
    main()

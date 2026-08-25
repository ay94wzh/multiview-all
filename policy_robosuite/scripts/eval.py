#!/usr/bin/env python
"""Evaluate a trained checkpoint.

Usage:
    python scripts/eval.py --config configs/eval.yaml
      [--train-config configs/train_multiview.yaml]   # for model/dataset settings
"""
from __future__ import annotations

import argparse
import dataclasses

import torch

from policy_robosuite.config import ProjectConfig
from policy_robosuite.eval.rollout import evaluate
from policy_robosuite.models.policy import build_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--train-config", type=str, default="configs/train_multiview.yaml")
    parser.add_argument("--overrides", nargs="*", default=None)
    # Convenience flags for the knobs you tweak most often at eval time.
    # They win over both the yaml and --overrides.
    parser.add_argument("--num-episodes", type=int, default=None,
                        help="override eval.num_episodes (default: yaml value)")
    parser.add_argument("--seed", type=int, default=None,
                        help="override eval.seed, the episode seed base (default: yaml value)")
    parser.add_argument("--inference-steps", type=int, default=None,
                        help="override eval.inference_steps (default: yaml value)")
    args = parser.parse_args()

    eval_cfg = ProjectConfig.from_yaml(args.config, args.overrides)
    if args.num_episodes is not None:
        eval_cfg.eval.num_episodes = args.num_episodes
    if args.seed is not None:
        eval_cfg.eval.seed = args.seed
    if args.inference_steps is not None:
        eval_cfg.eval.inference_steps = args.inference_steps
    train_cfg = ProjectConfig.from_yaml(args.train_config)
    cfg = train_cfg
    cfg.eval = eval_cfg.eval

    ckpt = torch.load(cfg.eval.checkpoint, map_location="cpu", weights_only=False)
    if "cfg" in ckpt:  # a checkpoint also carries the config it was trained with
        ckpt_cfg = ckpt["cfg"]
        if isinstance(ckpt_cfg, dict) and "goal_dim" not in ckpt_cfg.get("data", {}):
            # Pre-goal checkpoint (trained goal-blind): keep it goal-blind —
            # its proprio embed is Linear(9, 128), which won't load under the
            # new default goal_dim=3.
            ckpt_cfg = {**ckpt_cfg, "data": {**ckpt_cfg.get("data", {}), "goal_dim": 0}}
        cfg = ProjectConfig.from_dict(ckpt_cfg)
        cfg.eval = eval_cfg.eval
    model = build_policy(cfg)
    model.load_state_dict(ckpt["model"])
    if cfg.eval.seed is not None:
        cfg.data.seed = cfg.eval.seed
    if cfg.eval.inference_steps > 0:
        model.global_head.inference_steps = cfg.eval.inference_steps

    metrics = evaluate(
        cfg,
        model,
        num_episodes=cfg.eval.num_episodes,
        use_teacher=cfg.eval.use_teacher,
        render=cfg.eval.render,
    )
    print(metrics)


if __name__ == "__main__":
    main()

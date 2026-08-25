"""Dataset + policy-forward contract tests on synthetic episodes (no simulator).

These guard the npz schema written by scripts/collect_data.py and the batch
keys consumed by models/policy.py + training/trainer.py. Images are tiny so
the whole model runs on CPU in seconds.
"""
import numpy as np
import pytest
import torch

from policy_robosuite.config import ProjectConfig
from policy_robosuite.data.dataset import MultiViewDataset, collate_fn

VIEWS = ("top_camera", "left_camera", "right_camera", "bottom_camera", "front_camera")


def make_episode(path, T=40, H=16, W=16, seed=0):
    rng = np.random.default_rng(seed)
    K = np.array([[30.0, 0.0, W / 2], [0.0, 30.0, H / 2], [0.0, 0.0, 1.0]])
    data = {
        "qpos": rng.standard_normal((T, 7)).astype(np.float32),
        "ee_pose": rng.standard_normal((T, 7)).astype(np.float32),
        "actions": rng.standard_normal((T, 7)).astype(np.float32),
        "goal_pos": rng.standard_normal(3).astype(np.float32),
        "success": np.asarray(True),
    }
    for v in VIEWS:
        data[f"{v}/rgb"] = rng.integers(0, 255, (T, H, W, 3), dtype=np.uint8)
        data[f"{v}/K"] = K
        data[f"{v}/R"] = np.eye(3)
        data[f"{v}/t"] = np.zeros(3)
    # NOTE: the path must end in ".npz" here — np.savez_compressed appends the
    # suffix to any other name (the bug fixed in collect_data.py).
    np.savez_compressed(path, **data)


@pytest.fixture()
def small_cfg(tmp_path):
    (tmp_path / "Lift").mkdir()
    make_episode(tmp_path / "Lift" / "episode_00000.npz", seed=0)
    make_episode(tmp_path / "Lift" / "episode_00001.npz", seed=1)
    make_episode(tmp_path / "Lift" / "episode_00002.npz", T=10, seed=2)  # too short
    cfg = ProjectConfig()
    cfg.cameras.image_size = (16, 16)
    cfg.trajectory.window_steps = 8
    cfg.data.dataset_dir = str(tmp_path)
    cfg.data.batch_size = 2
    return cfg


def test_dataset_shapes_and_batch(small_cfg):
    ds = MultiViewDataset(small_cfg)
    assert len(ds) == 2  # the T=10 episode is skipped
    assert ds.views[-1] == "front_camera"  # student view appended for distillation

    item = ds[0]
    H = W = 16
    assert item["frames"].shape == (5, 3, 3, H, W)  # V, T_o, rgb, H, W
    assert item["ray_maps"].shape == (5, 6, H, W)
    assert item["trails"].shape == (5, 3, 1, H, W)  # V, T_o, trail, H, W (per-frame FiLM cond)
    assert item["actions"].shape == (16, 7)  # A_h, action_dim
    assert item["qpos"].shape == (7,) and item["ee_pose"].shape == (3, 7)  # T_o, 7
    assert item["goal_pos"].shape == (3,)

    batch = collate_fn([ds[0], ds[1]])
    assert batch["frames"].shape == (2, 5, 3, 3, H, W)
    assert batch["ray_maps"].shape == (5, 6, H, W)  # unbatched, shared
    assert batch["trails"].shape == (2, 5, 3, 1, H, W)
    assert batch["ee_pose"].shape == (2, 3, 7)
    assert batch["actions"].shape == (2, 16, 7)
    assert batch["goal_pos"].shape == (2, 3)
    assert set(batch["actions_cam"]) == set(VIEWS)
    assert batch["view_names"] == tuple(ds.views)


def test_camera_frame_actions_are_rotated(small_cfg):
    ds = MultiViewDataset(small_cfg)
    item = ds[0]
    from policy_robosuite.geometry.ray_maps import transform_delta_action_to_base

    # identity cameras in the synthetic episodes: camera-frame == base-frame
    for v in VIEWS:
        np.testing.assert_allclose(
            transform_delta_action_to_base(item["actions_cam"][v], np.eye(3)),
            item["actions"],
            atol=1e-6,
        )


def test_policy_forward_consumes_batch(small_cfg):
    from policy_robosuite.models.policy import build_policy
    from policy_robosuite.training.losses import total_loss

    ds = MultiViewDataset(small_cfg)
    batch = collate_fn([ds[0], ds[1]])
    policy = build_policy(small_cfg)
    losses = policy(batch)
    for key in ("action", "aux_view", "distill_latent"):
        assert key in losses, f"missing loss {key}"
    assert losses["action"].shape == ()
    total = total_loss(losses, small_cfg)
    total.backward()
    assert all(p.grad is not None for p in policy.global_head.parameters())
    # the student view must NOT be able to influence z_g (teacher-only fusion)
    assert losses["distill_latent"].requires_grad


# --------------------------------------------------------------------------- #
# arch="baseline": the pure diffusion policy (models/baseline.py)
# --------------------------------------------------------------------------- #

def test_old_checkpoint_cfg_without_arch_defaults_to_multiview():
    # checkpoints saved before the arch field existed load as multiview
    assert ProjectConfig.from_dict({}).arch == "multiview"
    assert ProjectConfig.from_dict({"data": {"batch_size": 32}}).arch == "multiview"


def test_baseline_dataset_shapes_and_batch(small_cfg):
    from policy_robosuite.data.dataset import MultiViewDataset, collate_fn

    small_cfg.arch = "baseline"
    ds = MultiViewDataset(small_cfg)
    assert len(ds) == 2  # the T=10 episode is still skipped
    # train views only: no student front_camera, no ray maps / trails
    assert list(ds.views) == list(small_cfg.cameras.train_views)

    item = ds[0]
    assert set(item) == {"frames", "view_names", "qpos", "goal_pos", "actions"}
    H = W = 16
    assert item["frames"].shape == (4, 3, 3, H, W)  # V=4, T_o, rgb, H, W
    assert item["actions"].shape == (16, 7)
    assert item["qpos"].shape == (7,)
    assert item["goal_pos"].shape == (3,)

    batch = collate_fn([ds[0], ds[1]])
    assert set(batch) == {"frames", "view_names", "qpos", "goal_pos", "actions"}
    assert batch["frames"].shape == (2, 4, 3, 3, H, W)
    assert batch["actions"].shape == (2, 16, 7)
    assert batch["goal_pos"].shape == (2, 3)
    assert batch["view_names"] == tuple(ds.views)


def test_build_policy_dispatch(small_cfg):
    from policy_robosuite.models.baseline import BaselinePolicy
    from policy_robosuite.models.policy import MultiViewPolicy, build_policy

    assert isinstance(build_policy(small_cfg), MultiViewPolicy)
    small_cfg.arch = "baseline"
    assert isinstance(build_policy(small_cfg), BaselinePolicy)
    small_cfg.arch = "bogus"
    with pytest.raises(ValueError):
        build_policy(small_cfg)


def test_baseline_policy_forward_consumes_batch(small_cfg):
    from policy_robosuite.data.dataset import MultiViewDataset, collate_fn
    from policy_robosuite.models.policy import build_policy
    from policy_robosuite.training.losses import total_loss

    small_cfg.arch = "baseline"
    ds = MultiViewDataset(small_cfg)
    batch = collate_fn([ds[0], ds[1]])
    policy = build_policy(small_cfg)

    losses = policy(batch)
    assert set(losses) == {"action"}  # no aux_view / distill_latent
    assert losses["action"].shape == ()
    total = total_loss(losses, small_cfg)
    total.backward()
    # the diffusion head (under test) AND the shared encoder must learn
    assert all(p.grad is not None for p in policy.global_head.parameters())
    assert all(p.grad is not None for p in policy.encoder.parameters())

    # sampling: no teacher/student split in the baseline — same condition and
    # same noise must give the same chunk (they share one sampling path)
    student = policy.sample_student(batch)
    assert student.shape == (2, 16, 7)
    torch.manual_seed(0)
    teacher = policy.sample_teacher(batch)
    torch.manual_seed(0)
    student = policy.sample_student(batch)
    assert torch.equal(student, teacher)


def _unzero_film_heads(policy):
    """The FiLM cond MLPs are zero-initialized by design (heads.py), so an
    untrained UNet is condition-blind: scale=1, bias=0 regardless of cond.
    Give them nonzero weights so the sampled chunk actually depends on the
    conditioning vector (what training would do)."""
    with torch.no_grad():
        for m in policy.global_head.unet.modules():
            if hasattr(m, "cond_mlp"):
                m.cond_mlp[-1].weight.normal_()
                m.cond_mlp[-1].bias.normal_()


def test_goal_conditioning_changes_the_sampled_action(small_cfg):
    """The whole point: with goal_dim=3 the goal must reach the action — same
    noise, different goal_pos, different sampled chunk."""
    from policy_robosuite.data.dataset import MultiViewDataset, collate_fn
    from policy_robosuite.models.policy import build_policy

    small_cfg.arch = "baseline"
    ds = MultiViewDataset(small_cfg)
    batch = collate_fn([ds[0], ds[1]])
    policy = build_policy(small_cfg)
    assert policy.proprio_embed.in_features == small_cfg.data.proprio_dim + small_cfg.data.goal_dim
    _unzero_film_heads(policy)

    torch.manual_seed(0)
    a_goal = policy.sample_student(batch)
    batch["goal_pos"][:, 0] += 0.5  # nudge the goal away
    torch.manual_seed(0)
    b_goal = policy.sample_student(batch)
    assert not torch.equal(a_goal, b_goal)


def test_goal_dim_zero_builds_legacy_goal_blind_policy(small_cfg):
    """goal_dim=0 = the pre-goal architecture: qpos-only proprio embed; the
    policy must ignore goal_pos entirely (same noise, same action regardless
    of the goal) while still conditioning on qpos — the old checkpoint path."""
    from policy_robosuite.data.dataset import MultiViewDataset, collate_fn
    from policy_robosuite.models.policy import build_policy

    small_cfg.arch = "baseline"
    small_cfg.data.goal_dim = 0
    ds = MultiViewDataset(small_cfg)
    batch = collate_fn([ds[0], ds[1]])
    assert "goal_pos" in batch  # dataset schema always carries it
    policy = build_policy(small_cfg)
    assert policy.proprio_embed.in_features == small_cfg.data.proprio_dim
    losses = policy(batch)  # goal_dim=0 -> never reads batch["goal_pos"]
    assert set(losses) == {"action"}
    assert policy.sample_student(batch).shape == (2, 16, 7)
    _unzero_film_heads(policy)

    # same-noise sampling: goal must NOT matter, qpos must
    torch.manual_seed(0)
    a = policy.sample_student(batch)
    batch["goal_pos"][:, 0] += 0.5
    torch.manual_seed(0)
    b = policy.sample_student(batch)
    assert torch.equal(a, b)  # goal ignored
    batch["qpos"] = batch["qpos"] + 0.1
    torch.manual_seed(0)
    c = policy.sample_student(batch)
    assert not torch.equal(b, c)  # qpos still conditions


# --------------------------------------------------------------------------- #
# arch="baseline" + action_head.head_type="act": the ACT (CVAE) baseline
# --------------------------------------------------------------------------- #

def test_act_config_defaults_and_round_trip():
    # old checkpoints have no head_type: they must default to diffusion
    assert ProjectConfig.from_dict({"action_head": {"horizon": 16}}).action_head.head_type == "diffusion"
    cfg = ProjectConfig()
    cfg.action_head.head_type = "act"
    cfg.action_head.act.kl_weight = 5.0
    cfg2 = ProjectConfig.from_dict(cfg.to_dict())  # checkpoint round-trip
    assert cfg2.action_head.head_type == "act"
    assert cfg2.action_head.act.kl_weight == 5.0
    assert cfg2.action_head.act.dec_layers == 4


def test_build_policy_rejects_multiview_act(small_cfg):
    from policy_robosuite.models.policy import build_policy

    small_cfg.action_head.head_type = "act"  # arch still "multiview"
    with pytest.raises(ValueError):
        build_policy(small_cfg)


def test_act_baseline_policy_forward_and_sample(small_cfg):
    from policy_robosuite.data.dataset import MultiViewDataset, collate_fn
    from policy_robosuite.models.act import ACTActionHead
    from policy_robosuite.models.policy import build_policy
    from policy_robosuite.training.losses import total_loss

    small_cfg.arch = "baseline"
    small_cfg.action_head.head_type = "act"
    ds = MultiViewDataset(small_cfg)
    batch = collate_fn([ds[0], ds[1]])
    policy = build_policy(small_cfg)
    assert isinstance(policy.global_head, ACTActionHead)

    losses = policy(batch)
    assert set(losses) == {"action"}  # no aux_view / distill_latent
    assert losses["action"].shape == ()
    total = total_loss(losses, small_cfg)
    total.backward()
    # the ACT head (under test) AND the shared encoder must learn
    assert all(p.grad is not None for p in policy.global_head.parameters())
    assert all(p.grad is not None for p in policy.encoder.parameters())

    # no teacher/student split in the baseline — same condition and (none of)
    # the noise must give the same chunk (they share one sampling path)
    student = policy.sample_student(batch)
    assert student.shape == (2, 16, 7)
    torch.manual_seed(0)
    teacher = policy.sample_teacher(batch)
    torch.manual_seed(0)
    student = policy.sample_student(batch)
    assert torch.equal(student, teacher)


def test_act_head_contract():
    # direct head on random cond (no encoder): surface + split-cond logic
    from policy_robosuite.models.act import ACTActionHead

    cfg = ProjectConfig()
    cfg.action_head.head_type = "act"
    head = ACTActionHead(cfg.action_head, cond_dim=12 * 256 + 128,
                         obs_token_dim=256, num_obs_tokens=12, proprio_dim=128)
    head.eval()  # dropout off: sample has no RNG -> deterministic
    cond = torch.randn(2, 3200)
    actions = torch.randn(2, 16, 7)
    loss = head.compute_loss(cond, actions)
    assert loss.shape == ()
    loss.backward()
    assert all(p.grad is not None for p in head.parameters())
    out = head.sample(cond)
    assert out.shape == (2, 16, 7)
    assert head.sample(cond, num_steps=50).shape == (2, 16, 7)  # num_steps ignored
    assert torch.equal(head.sample(cond), head.sample(cond))    # deterministic


def test_act_baseline_yaml_parses():
    from pathlib import Path

    cfg = ProjectConfig.from_yaml(
        Path(__file__).resolve().parents[1] / "configs" / "train_act_baseline.yaml"
    )
    assert cfg.action_head.head_type == "act"
    assert cfg.data.task == "Lift"
    assert cfg.action_head.act.kl_weight == 10.0
    assert cfg.action_head.horizon == 30


def test_eval_yamls_apply_checkpoint_and_inference_steps():
    """Regression: eval keys were written at the yaml top level, which
    _from_dict silently ignored -> every eval.py run loaded the default
    checkpoint (checkpoints/latest.pt) and evaluated the wrong model."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    # Baseline/ACT eval runs point eval.checkpoint at their own dir (the
    # checkpoint_dir of the matching train config); only eval.yaml is shipped.
    expected = {
        "eval.yaml": "checkpoints/latest.pt",
    }
    for name, ckpt in expected.items():
        cfg = ProjectConfig.from_yaml(root / "configs" / name)
        assert cfg.eval.checkpoint == ckpt, name
        assert cfg.eval.inference_steps == 50, name
        assert cfg.eval.num_episodes == 50, name


def test_from_yaml_rejects_unwrapped_eval_key(tmp_path):
    """Top-level `checkpoint:` must raise, not silently parse into defaults."""
    f = tmp_path / "bad.yaml"
    f.write_text("checkpoint: checkpoints_act_baseline_push/latest.pt\nnum_episodes: 50\n")
    with pytest.raises(ValueError, match="Unknown config key"):
        ProjectConfig.from_yaml(f)

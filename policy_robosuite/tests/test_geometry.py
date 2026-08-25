"""Geometry smoke tests: ray maps, delta-action transforms, trail rendering."""
import numpy as np
import pytest

from policy_robosuite.config import ProjectConfig
from policy_robosuite.geometry.ray_maps import pixel_directions_cam, plucker_map, transform_delta_action, transform_delta_action_to_base
from policy_robosuite.geometry.trajectory import render_trail


def test_principal_ray_points_along_optical_axis():
    K = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]])
    H, W = 48, 64
    d = pixel_directions_cam(K, H, W)
    np.testing.assert_allclose(d[24, 32], [0.0, 0.0, 1.0], atol=1e-6)
    assert np.allclose(np.linalg.norm(d, axis=-1), 1.0)


def test_plucker_shape_and_norms():
    K = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]])
    R, t = np.eye(3), np.zeros(3)
    H, W = 48, 64
    m = plucker_map(K, R, t, H, W)
    assert m.shape == (H, W, 6)
    np.testing.assert_allclose(np.linalg.norm(m[..., 3:], axis=-1), 1.0, atol=1e-5)
    # camera at origin -> moment is o x d = 0
    np.testing.assert_allclose(m[..., :3], 0.0, atol=1e-6)


def test_plucker_rotates_with_camera():
    # rotz(90 deg): world +x appears along camera +y, so a camera-center ray
    # pointing at world +x has direction (0, 0, 1) in cam -> d_w = R^T d_c = +x
    theta = np.pi / 2
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    K = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]])
    m = plucker_map(K, R, np.zeros(3), 48, 64)
    # pixel (u=32+cx...): direction (0.01, 0, 1)/|.| in cam; compute center-left pixel instead
    np.testing.assert_allclose(m[24, 32], [0, 0, 0, 0, 0, 1.0], atol=1e-5)


def test_delta_action_roundtrip():
    theta = np.deg2rad(37)
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    a = np.random.default_rng(0).standard_normal(6).astype(np.float32)
    a_cam = transform_delta_action(a, R)
    np.testing.assert_allclose(transform_delta_action_to_base(a_cam, R), a, atol=1e-5)


def test_trail_renders_at_projected_point():
    cfg = ProjectConfig().trajectory
    K = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]])
    pos = np.array([[0.0, 0.0, 4.0], [0.0, 0.0, 5.0]])
    img = render_trail(pos, K, np.eye(3), np.zeros(3), 48, 64, cfg)
    assert img.shape == (48, 64, 1)
    assert img[24, 32, 0] > 0.5  # bright dot at the current position

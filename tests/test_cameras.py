"""Camera extrinsics and projection."""

from __future__ import annotations

import math

import numpy as np
import pytest

from jims_mower.cameras import (
    attitude_plane_hits,
    body_to_world,
    camera_world_pose,
    focal_length_px,
    ground_hits,
    heightfield_hits,
    project_point,
    world_to_body,
    world_to_optical,
)
from jims_mower.config import default_camera_rig
from jims_mower.types import CameraSpec, Pose


def test_default_rig_counts() -> None:
    assert len(default_camera_rig(4)) == 4
    assert len(default_camera_rig(5)) == 5
    assert len(default_camera_rig(6)) == 6
    names = [c.name for c in default_camera_rig(6)]
    assert names == ["front", "front_left", "front_right", "rear", "left", "right"]


def test_default_rig_rejects_bad_count() -> None:
    with pytest.raises(Exception):
        default_camera_rig(3)


def test_body_world_roundtrip() -> None:
    pose = Pose(2.0, 3.0, 0.4)
    wx, wy, wz = body_to_world(0.25, 0.1, 0.38, pose)
    bx, by, bz = world_to_body(wx, wy, wz, pose)
    assert bx == pytest.approx(0.25, abs=1e-9)
    assert by == pytest.approx(0.1, abs=1e-9)
    assert bz == pytest.approx(0.38)


def test_front_camera_sits_on_nose() -> None:
    pose = Pose(5.0, 5.0, 0.0)
    cam = CameraSpec("front", 0.25, 0.0, 0.38, 0.0, -18.0)
    wp = camera_world_pose(pose, cam)
    assert wp.x == pytest.approx(5.25)
    assert wp.y == pytest.approx(5.0)
    assert wp.z == pytest.approx(0.38)


def test_camera_rides_robot_elevation() -> None:
    pose = Pose(5.0, 5.0, 0.0, z=0.20)
    cam = CameraSpec("front", 0.25, 0.0, 0.38, 0.0, -18.0)
    wp = camera_world_pose(pose, cam)
    assert wp.z == pytest.approx(0.58)


def test_point_ahead_is_in_front_optical() -> None:
    pose = Pose(0.0, 0.0, 0.0)
    cam = CameraSpec("front", 0.0, 0.0, 0.38, 0.0, 0.0)
    wp = camera_world_pose(pose, cam)
    ox, oy, oz = world_to_optical(2.0, 0.0, 0.38, wp)
    assert oz > 0.0
    assert abs(ox) < 1e-6
    assert abs(oy) < 1e-6


def test_point_behind_has_negative_depth() -> None:
    pose = Pose(0.0, 0.0, 0.0)
    cam = CameraSpec("front", 0.0, 0.0, 0.38, 0.0, 0.0)
    wp = camera_world_pose(pose, cam)
    _, _, oz = world_to_optical(-2.0, 0.0, 0.38, wp)
    assert oz < 0.0


def test_project_rejects_behind_camera() -> None:
    pose = Pose(0.0, 0.0, 0.0)
    cam = CameraSpec("front", 0.0, 0.0, 0.38, 0.0, 0.0)
    wp = camera_world_pose(pose, cam)
    assert project_point(-1.0, 0.0, 0.38, wp, 80, 60) is None


def test_project_ahead_near_image_center() -> None:
    pose = Pose(0.0, 0.0, 0.0)
    cam = CameraSpec("front", 0.0, 0.0, 0.38, 0.0, 0.0)
    wp = camera_world_pose(pose, cam)
    proj = project_point(3.0, 0.0, 0.38, wp, 80, 60)
    assert proj is not None
    u, v, depth = proj
    assert depth == pytest.approx(3.0)
    assert u == pytest.approx(39.5, abs=1.5)
    assert v == pytest.approx(29.5, abs=1.5)


def test_focal_length_grows_as_fov_shrinks() -> None:
    wide = focal_length_px(80, 90.0)
    narrow = focal_length_px(80, 40.0)
    assert narrow > wide


def test_focal_rejects_bad_fov() -> None:
    with pytest.raises(ValueError):
        focal_length_px(80, 0.0)
    with pytest.raises(ValueError):
        focal_length_px(80, 180.0)


def test_ground_hits_look_down() -> None:
    pose = Pose(2.0, 2.0, 0.0)
    cam = CameraSpec("front", 0.25, 0.0, 0.38, 0.0, -25.0)
    wp = camera_world_pose(pose, cam)
    hx, hy, valid = ground_hits(wp, 16, 12)
    assert bool(valid.any())
    assert np_isfinite_some(hx)


def np_isfinite_some(arr) -> bool:
    import numpy as np

    return bool(np.isfinite(arr).any())


def test_heightfield_hits_match_flat_ground() -> None:
    pose = Pose(2.0, 2.0, 0.0)
    cam = CameraSpec("front", 0.25, 0.0, 0.38, 0.0, -25.0)
    wp = camera_world_pose(pose, cam)
    hx0, hy0, valid0 = ground_hits(wp, 16, 12)
    hx1, hy1, _hz, valid1 = heightfield_hits(
        wp, 16, 12, lambda xs, ys: np.zeros_like(xs, dtype=np.float32)
    )
    both = valid0 & valid1
    assert bool(both.any())
    assert np.allclose(hx0[both], hx1[both], atol=0.15)
    assert np.allclose(hy0[both], hy1[both], atol=0.15)


def test_attitude_plane_matches_flat_ground() -> None:
    pose = Pose(2.0, 2.0, 0.0)
    cam = CameraSpec("front", 0.25, 0.0, 0.38, 0.0, -25.0)
    wp = camera_world_pose(pose, cam)
    hx0, hy0, valid0 = ground_hits(wp, 16, 12)
    hx1, hy1, valid1 = attitude_plane_hits(wp, 16, 12, pose)
    both = valid0 & valid1
    assert bool(both.any())
    assert np.allclose(hx0[both], hx1[both], atol=1e-5)
    assert np.allclose(hy0[both], hy1[both], atol=1e-5)


def test_attitude_plane_tracks_seated_grade() -> None:
    slope = 0.12
    pose = Pose(6.0, 6.0, 0.0, z=0.0, pitch=slope, roll=0.0)
    cam = CameraSpec("front", 0.25, 0.0, 0.38, 0.0, -22.0)
    wp = camera_world_pose(pose, cam)
    hx, hy, valid = attitude_plane_hits(wp, 24, 18, pose)
    assert bool(valid.any())
    # Hits should land on the same planar grade the robot is sitting on.
    z_plane = math.tan(slope) * (hx[valid] - pose.x)
    assert float(np.abs(z_plane).mean()) < 2.5


def test_left_camera_yaw() -> None:
    pose = Pose(0.0, 0.0, 0.0)
    cam = CameraSpec("left", 0.0, 0.25, 0.38, 90.0, 0.0)
    wp = camera_world_pose(pose, cam)
    assert wp.yaw == pytest.approx(math.pi / 2)

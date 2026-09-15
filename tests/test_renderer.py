"""Synthetic multi-camera renderer."""

from __future__ import annotations

import numpy as np
import pytest

from jims_mower.constants import TERRAIN_DRAIN
from jims_mower.maps import GrassCoverageMap
from jims_mower.renderer import render_camera, render_scalar_map, render_topdown
from jims_mower.terrain import HeightField
from jims_mower.types import CameraSpec, Obstacle, Pose


def _coverage() -> GrassCoverageMap:
    return GrassCoverageMap(8.0, 8.0, 0.2)


def test_camera_frame_shape_and_dtype() -> None:
    img = render_camera(
        Pose(4.0, 4.0, 0.0),
        CameraSpec("front", 0.25, 0.0, 0.38, 0.0, -18.0),
        _coverage(),
        [],
        32,
        24,
        (8.0, 8.0),
    )
    assert img.shape == (24, 32, 3)
    assert img.dtype == np.uint8


def test_person_ahead_tints_front_view() -> None:
    coverage = _coverage()
    pose = Pose(2.0, 4.0, 0.0)
    person = Obstacle("person", 3.5, 4.0, 0.25, z=0.9)
    cam = CameraSpec("front", 0.25, 0.0, 0.38, 0.0, -10.0)
    empty = render_camera(pose, cam, coverage, [], 40, 30, (8.0, 8.0))
    with_person = render_camera(pose, cam, coverage, [person], 40, 30, (8.0, 8.0))
    assert np.abs(with_person.astype(int) - empty.astype(int)).sum() > 0


def test_rear_camera_does_not_see_object_in_front() -> None:
    coverage = _coverage()
    pose = Pose(2.0, 4.0, 0.0)
    person = Obstacle("person", 4.0, 4.0, 0.25, z=0.9)
    rear = CameraSpec("rear", -0.25, 0.0, 0.38, 180.0, -8.0)
    img = render_camera(pose, rear, coverage, [person], 40, 30, (8.0, 8.0))
    # Red-ish person pixels should be rare in the rear view.
    red = (img[:, :, 0] > 180) & (img[:, :, 1] < 120)
    assert int(red.sum()) < 8


def test_cut_grass_changes_ground_color() -> None:
    coverage = _coverage()
    pose = Pose(4.0, 4.0, 0.0)
    cam = CameraSpec("front", 0.25, 0.0, 0.38, 0.0, -28.0)
    before = render_camera(pose, cam, coverage, [], 32, 24, (8.0, 8.0))
    coverage.mark_circle(4.4, 4.0, 1.2)
    after = render_camera(pose, cam, coverage, [], 32, 24, (8.0, 8.0))
    assert not np.array_equal(before, after)


def test_topdown_contains_robot_and_obstacle() -> None:
    coverage = _coverage()
    img = render_topdown(
        Pose(4.0, 4.0, 0.0),
        coverage,
        [Obstacle("tree", 6.0, 6.0, 0.35)],
        trimmer_xy=(4.32, 4.0),
        trimmer_on=True,
        image_size=80,
    )
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.dtype == np.uint8


def test_different_cameras_differ() -> None:
    coverage = _coverage()
    pose = Pose(4.0, 4.0, 0.0)
    obst = [Obstacle("toy", 5.0, 4.3, 0.1, z=0.08)]
    front = render_camera(
        pose,
        CameraSpec("front", 0.25, 0.0, 0.38, 0.0, -12.0),
        coverage,
        obst,
        32,
        24,
        (8.0, 8.0),
    )
    left = render_camera(
        pose,
        CameraSpec("left", 0.0, 0.25, 0.38, 90.0, -12.0),
        coverage,
        obst,
        32,
        24,
        (8.0, 8.0),
    )
    assert not np.array_equal(front, left)


def test_drain_changes_camera_pixels() -> None:
    coverage = _coverage()
    pose = Pose(2.0, 4.0, 0.0)
    cam = CameraSpec("front", 0.25, 0.0, 0.38, 0.0, -22.0)
    flat = render_camera(pose, cam, coverage, [], 40, 30, (8.0, 8.0), terrain=None)
    hf = HeightField.empty(8.0, 8.0, 0.2)
    # Ditch across the view 1.2 m ahead.
    for row in range(hf.rows):
        for col in range(hf.cols):
            x = (col + 0.5) * 0.2
            y = (row + 0.5) * 0.2
            if 3.0 <= x <= 3.6 and 3.2 <= y <= 4.8:
                hf.elevation[row, col] = -0.18
                hf.labels[row, col] = TERRAIN_DRAIN
    hf.recompute_slope()
    with_drain = render_camera(pose, cam, coverage, [], 40, 30, (8.0, 8.0), terrain=hf)
    assert not np.array_equal(flat, with_drain)
    # Drain pixels are darker / browner than uncut grass.
    brown = (
        (with_drain[:, :, 0] < 100)
        & (with_drain[:, :, 1] < 90)
        & (with_drain[:, :, 2] < 80)
    )
    assert int(brown.sum()) > 5


def test_topdown_shows_drain() -> None:
    coverage = _coverage()
    hf = HeightField.empty(8.0, 8.0, 0.2)
    hf.labels[10:14, 8:20] = TERRAIN_DRAIN
    hf.elevation[10:14, 8:20] = -0.16
    img = render_topdown(
        Pose(4.0, 4.0, 0.0),
        coverage,
        [],
        image_size=80,
        terrain=hf,
    )
    flat = render_topdown(Pose(4.0, 4.0, 0.0), coverage, [], image_size=80)
    assert not np.array_equal(img, flat)


def test_scalar_map_shapes() -> None:
    grid = np.linspace(-0.2, 0.4, 20 * 20, dtype=np.float32).reshape(20, 20)
    img = render_scalar_map(grid, cmap="elev", image_size=40)
    assert img.shape[2] == 3
    assert img.dtype == np.uint8
    haz = render_scalar_map(np.zeros((10, 10), dtype=np.float32), cmap="hazard", image_size=20)
    assert haz.shape[2] == 3

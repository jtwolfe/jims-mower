"""RGB / ToF terrain cues: classification and back-projection, not mAP."""

from __future__ import annotations

import numpy as np
import pytest

from jims_mower.constants import (
    BANK_RGB,
    DRAIN_EDGE_RGB,
    DRAIN_RGB,
    HAZARD_DRAIN,
    HAZARD_DRAIN_EDGE,
    HAZARD_STEEP,
    TERRAIN_DRAIN,
    TERRAIN_DRAIN_EDGE,
    UNCUT_GRASS_RGB,
)
from jims_mower.maps import GrassCoverageMap
from jims_mower.perception import (
    HeuristicTerrainObserver,
    OracleTerrainObserver,
    classify_terrain_rgb,
    drain_pixel_fraction,
)
from jims_mower.perception.cv_terrain import project_labels_to_maps, stamp_tof_corners
from jims_mower.perception.fuse import gate_isolated_lips
from jims_mower.renderer import render_camera
from jims_mower.terrain import HeightField
from jims_mower.types import CameraSpec, PerceptionContext, Pose


def test_classify_palette_swatches() -> None:
    img = np.zeros((12, 12, 3), dtype=np.uint8)
    img[:4, :] = DRAIN_RGB
    img[4:8, :] = DRAIN_EDGE_RGB
    img[8:, :] = UNCUT_GRASS_RGB
    labels = classify_terrain_rgb(img)
    assert int((labels[:4] == HAZARD_DRAIN).mean()) > 0.8
    assert int((labels[4:8] >= HAZARD_DRAIN_EDGE).mean()) > 0.8
    assert int((labels[8:] == 0).mean()) > 0.8


def test_classify_bank_olive_not_uncut() -> None:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[:] = BANK_RGB
    labels = classify_terrain_rgb(img)
    assert int((labels == HAZARD_STEEP).sum()) > 20
    grass = np.zeros((8, 8, 3), dtype=np.uint8)
    grass[:] = UNCUT_GRASS_RGB
    assert int((classify_terrain_rgb(grass) == HAZARD_STEEP).sum()) == 0


def test_classify_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        classify_terrain_rgb(np.zeros((8, 8), dtype=np.uint8))


def test_drain_view_has_brown_fraction() -> None:
    coverage = GrassCoverageMap(8.0, 8.0, 0.2)
    pose = Pose(2.0, 4.0, 0.0)
    cam = CameraSpec("front", 0.25, 0.0, 0.38, 0.0, -22.0)
    hf = HeightField.empty(8.0, 8.0, 0.2)
    for row in range(hf.rows):
        for col in range(hf.cols):
            x = (col + 0.5) * 0.2
            y = (row + 0.5) * 0.2
            if 3.0 <= x <= 3.6 and 3.2 <= y <= 4.8:
                hf.elevation[row, col] = -0.18
                hf.labels[row, col] = TERRAIN_DRAIN
            elif 2.8 <= x <= 3.8 and 3.1 <= y <= 4.9 and hf.labels[row, col] == 0:
                hf.labels[row, col] = TERRAIN_DRAIN_EDGE
    hf.recompute_slope()
    img = render_camera(pose, cam, coverage, [], 40, 30, (8.0, 8.0), terrain=hf)
    frac = drain_pixel_fraction(img)
    assert frac > 0.02
    labels = classify_terrain_rgb(img)
    assert int((labels >= HAZARD_DRAIN_EDGE).sum()) > 5


def test_backproject_stamps_cells_ahead() -> None:
    img = np.zeros((24, 32, 3), dtype=np.uint8)
    img[16:, :] = DRAIN_RGB
    labels = classify_terrain_rgb(img)
    pose = Pose(2.0, 4.0, 0.0)
    cam = CameraSpec("front", 0.25, 0.0, 0.38, 0.0, -22.0)
    elev = np.zeros((40, 40), dtype=np.float32)
    slope = np.zeros((40, 40), dtype=np.float32)
    hazard = np.zeros((40, 40), dtype=np.float32)
    hits = project_labels_to_maps(
        img,
        labels,
        cam,
        pose,
        elevation=elev,
        slope=slope,
        hazard=hazard,
        resolution_m=0.2,
        world_size=(8.0, 8.0),
    )
    assert hits > 0
    assert int((hazard >= HAZARD_DRAIN_EDGE).sum()) > 0
    # Hits should sit in front of the robot, not behind.
    rows, cols = np.where(hazard >= HAZARD_DRAIN_EDGE)
    xs = (cols + 0.5) * 0.2
    assert float(xs.mean()) > pose.x


def test_tof_stamps_dropped_wheel() -> None:
    pose = Pose(2.0, 2.0, 0.0)
    hazard = np.zeros((40, 40), dtype=np.float32)
    elev = np.zeros_like(hazard)
    slope = np.zeros_like(hazard)
    tof = np.array([0.06, 0.28, 0.06, 0.06], dtype=np.float32)
    n = stamp_tof_corners(
        tof,
        pose,
        hazard=hazard,
        elevation=elev,
        slope=slope,
        resolution_m=0.1,
        length_m=0.5,
        track_m=0.4,
    )
    assert n == 1
    assert float(hazard.max()) == float(HAZARD_DRAIN)


def test_heuristic_ignores_god_view_terrain() -> None:
    pose = Pose(2.0, 4.0, 0.0)
    cam = CameraSpec("front", 0.25, 0.0, 0.38, 0.0, -22.0)
    fake = HeightField.empty(8.0, 8.0, 0.2)
    fake.elevation[:, :] = 3.0
    ctx = PerceptionContext(
        pose,
        [cam],
        [],
        (32, 24),
        map_shape=(40, 40),
        resolution_m=0.2,
        world_size=(8.0, 8.0),
        terrain=fake,
    )
    images = {"front": np.zeros((24, 32, 3), dtype=np.uint8)}
    images["front"][:] = UNCUT_GRASS_RGB
    imu = np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0], dtype=np.float32)
    gps = np.array([2.0, 4.0, 0.0, 1.0], dtype=np.float32)
    est = HeuristicTerrainObserver().estimate(images, imu, gps, ctx)
    assert est.source == "heuristic"
    assert float(est.elevation.max()) < 1.0
    oracle = OracleTerrainObserver().estimate(images, imu, gps, ctx)
    assert float(oracle.elevation.max()) == pytest.approx(3.0)


def test_gate_isolated_lips_keeps_channel_neighbours() -> None:
    haz = np.zeros((8, 8), dtype=np.float32)
    haz[3:5, 3:5] = HAZARD_DRAIN
    haz[2, 3] = HAZARD_DRAIN_EDGE
    haz[0, 0] = HAZARD_DRAIN_EDGE  # isolated dirt
    gated = gate_isolated_lips(haz, dilate_cells=2)
    assert float(gated[2, 3]) == float(HAZARD_DRAIN_EDGE)
    assert float(gated[0, 0]) == 0.0


def test_heuristic_reset_clears_maps() -> None:
    obs = HeuristicTerrainObserver()
    pose = Pose(2.0, 4.0, 0.0)
    cam = CameraSpec("front", 0.25, 0.0, 0.38, 0.0, -22.0)
    ctx = PerceptionContext(
        pose,
        [cam],
        [],
        (16, 12),
        map_shape=(20, 20),
        resolution_m=0.4,
        world_size=(8.0, 8.0),
    )
    img = np.zeros((12, 16, 3), dtype=np.uint8)
    img[:] = DRAIN_RGB
    imu = np.zeros(6, dtype=np.float32)
    imu[2] = 9.81
    gps = np.zeros(4, dtype=np.float32)
    first = obs.estimate({"front": img}, imu, gps, ctx)
    assert int((first.hazard >= HAZARD_DRAIN_EDGE).sum()) > 0
    obs.reset()
    blank = np.zeros((12, 16, 3), dtype=np.uint8)
    blank[:] = UNCUT_GRASS_RGB
    second = obs.estimate({"front": blank}, imu, gps, ctx)
    assert int((second.hazard >= HAZARD_DRAIN_EDGE).sum()) == 0

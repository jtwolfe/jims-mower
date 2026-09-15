"""Multi-camera BEV fuse merges ground-plane stamps with confidence."""

from __future__ import annotations

import numpy as np

from jims_mower.constants import HAZARD_DRAIN, HAZARD_DRAIN_EDGE
from jims_mower.perception.fuse import BevStamp, fuse_stamps, gate_isolated_lips
from jims_mower.perception import HeuristicTerrainObserver
from jims_mower.types import CameraSpec, PerceptionContext, Pose


def test_fuse_higher_confidence_wins() -> None:
    weak = BevStamp(
        rows=np.array([5], dtype=np.int32),
        cols=np.array([5], dtype=np.int32),
        labels=np.array([HAZARD_DRAIN_EDGE], dtype=np.float32),
        confidence=np.array([0.2], dtype=np.float32),
        radii=np.array([0], dtype=np.int32),
        camera="left",
    )
    strong = BevStamp(
        rows=np.array([5], dtype=np.int32),
        cols=np.array([5], dtype=np.int32),
        labels=np.array([HAZARD_DRAIN], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        radii=np.array([0], dtype=np.int32),
        camera="front",
    )
    hazard, conf = fuse_stamps([weak, strong], (12, 12))
    assert float(hazard[5, 5]) == float(HAZARD_DRAIN)
    assert float(conf[5, 5]) == np.float32(0.9)


def test_fuse_two_cameras_cover_more_than_one() -> None:
    a = BevStamp(
        rows=np.array([2], dtype=np.int32),
        cols=np.array([2], dtype=np.int32),
        labels=np.array([HAZARD_DRAIN], dtype=np.float32),
        confidence=np.array([0.8], dtype=np.float32),
        radii=np.array([0], dtype=np.int32),
        camera="front",
    )
    b = BevStamp(
        rows=np.array([8], dtype=np.int32),
        cols=np.array([8], dtype=np.int32),
        labels=np.array([HAZARD_DRAIN_EDGE], dtype=np.float32),
        confidence=np.array([0.7], dtype=np.float32),
        radii=np.array([0], dtype=np.int32),
        camera="rear",
    )
    one, _ = fuse_stamps([a], (12, 12))
    both, _ = fuse_stamps([a, b], (12, 12))
    assert int((one > 0).sum()) == 1
    assert int((both > 0).sum()) == 2


def test_gate_keeps_ditch_stripe_drops_speck() -> None:
    hazard = np.zeros((20, 20), dtype=np.float32)
    hazard[4:16, 8] = HAZARD_DRAIN_EDGE
    hazard[4:16, 9] = HAZARD_DRAIN
    hazard[2, 15] = HAZARD_DRAIN_EDGE
    gated = gate_isolated_lips(hazard)
    assert int((gated[4:16, 8:10] >= HAZARD_DRAIN_EDGE).sum()) >= 20
    assert float(gated[2, 15]) == 0.0


def test_heuristic_fuse_path_stamps_drain_swatch() -> None:
    pose = Pose(2.0, 4.0, 0.0)
    cam = CameraSpec("front", 0.25, 0.0, 0.38, 0.0, -22.0)
    ctx = PerceptionContext(
        pose,
        [cam],
        [],
        (32, 24),
        map_shape=(40, 40),
        resolution_m=0.2,
        world_size=(8.0, 8.0),
    )
    img = np.zeros((24, 32, 3), dtype=np.uint8)
    img[16:, :] = (58, 42, 28)  # DRAIN_RGB
    imu = np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0], dtype=np.float32)
    gps = np.array([2.0, 4.0, 0.0, 1.0], dtype=np.float32)
    est = HeuristicTerrainObserver().estimate({"front": img}, imu, gps, ctx)
    assert est.source == "heuristic"
    assert int((est.hazard >= HAZARD_DRAIN_EDGE).sum()) > 0
    assert est.confidence is not None
    assert float(est.confidence.max()) > 0.0

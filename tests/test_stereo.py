"""Near-field metric stereo + frozen elev. Not COLMAP, not mAP / FPS."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jims_mower.config import default_camera_rig, load_config
from jims_mower.mission_flow import MissionPolicy
from jims_mower.perception.stereo import (
    STEREO_BASELINE_MAX_M,
    STEREO_BASELINE_MIN_M,
    STEREO_NOTE,
    depth_resolution_m,
    disparity_px,
    find_stereo_pair,
    range_from_disparity,
    rasterize_points,
    synthetic_stereo_points,
)
from jims_mower.planning.observed import ObservedMap
from jims_mower.types import CameraSpec, Pose

_STEREO_YAML = Path("configs/orin/extrinsics_stereo.yaml")


def _true_pair() -> list[CameraSpec]:
    return [
        CameraSpec("stereo_left", 0.24, 0.04, 0.38, 0.0, -22.0),
        CameraSpec("stereo_right", 0.24, -0.04, 0.38, 0.0, -22.0),
    ]


def test_disparity_roundtrip() -> None:
    d = disparity_px(range_m=2.0, baseline_m=0.08, focal_px=40.0)
    z = range_from_disparity(disparity_px=d, baseline_m=0.08, focal_px=40.0)
    assert z == pytest.approx(2.0)


def test_depth_resolution_worsens_with_range() -> None:
    near = depth_resolution_m(range_m=1.0, baseline_m=0.08, focal_px=50.0)
    far = depth_resolution_m(range_m=4.0, baseline_m=0.08, focal_px=50.0)
    assert far / near == pytest.approx(16.0)
    assert far > near


def test_default_gym_lookaround_is_not_a_pair() -> None:
    assert find_stereo_pair(default_camera_rig(6)) is None
    wide = [
        CameraSpec("front_left", 0.20, 0.20, 0.38, 0.0, -22.0),
        CameraSpec("front_right", 0.20, -0.20, 0.38, 0.0, -22.0),
    ]
    assert find_stereo_pair(wide) is None


def test_named_pair_in_six_to_twelve_cm_band() -> None:
    pair = find_stereo_pair(_true_pair())
    assert pair is not None
    assert pair.left.name == "stereo_left"
    assert pair.right.name == "stereo_right"
    assert STEREO_BASELINE_MIN_M <= pair.baseline_m <= STEREO_BASELINE_MAX_M
    assert pair.baseline_m == pytest.approx(0.08, abs=1e-6)
    info = pair.as_info()
    assert info["not_colmap"] is True
    assert info["fps_claim"] is None
    assert info["map_claim"] is None
    assert "COLMAP" not in STEREO_NOTE or "not COLMAP" in STEREO_NOTE or "not" in STEREO_NOTE


def test_yaw_mismatch_rejected() -> None:
    cams = [
        CameraSpec("stereo_left", 0.24, 0.04, 0.38, 8.0, -22.0),
        CameraSpec("stereo_right", 0.24, -0.04, 0.38, 0.0, -22.0),
    ]
    assert find_stereo_pair(cams) is None


def test_baseline_outside_band_rejected() -> None:
    tight = [
        CameraSpec("stereo_left", 0.24, 0.02, 0.38, 0.0, -22.0),
        CameraSpec("stereo_right", 0.24, -0.02, 0.38, 0.0, -22.0),
    ]
    wide = [
        CameraSpec("stereo_left", 0.24, 0.08, 0.38, 0.0, -22.0),
        CameraSpec("stereo_right", 0.24, -0.08, 0.38, 0.0, -22.0),
    ]
    assert find_stereo_pair(tight) is None
    assert find_stereo_pair(wide) is None


def test_stereo_yaml_loads_a_true_pair() -> None:
    cfg = load_config(_STEREO_YAML)
    pair = find_stereo_pair(cfg.resolved_cameras())
    assert pair is not None
    assert pair.left.name == "stereo_left"
    names = {c.name for c in cfg.resolved_cameras()}
    assert {"rear", "left", "right"}.issubset(names)


def test_synthetic_reconstruct_on_plane() -> None:
    pair = find_stereo_pair(_true_pair())
    assert pair is not None
    pose = Pose(4.0, 4.0, 0.0, z=0.10)
    xs, ys, zs = synthetic_stereo_points(
        pose,
        pair,
        width=40,
        height=30,
        height_at=lambda _x, _y: 0.10,
        pixel_stride=2,
    )
    assert xs.size > 8
    assert float(np.mean(zs)) == pytest.approx(0.10, abs=0.05)
    elev, hits = rasterize_points(
        xs,
        ys,
        zs,
        shape=(32, 32),
        resolution_m=0.25,
        width_m=8.0,
        height_m=8.0,
    )
    assert int(hits.sum()) > 0
    assert float(elev[hits].mean()) == pytest.approx(0.10, abs=0.05)


def test_stamp_metric_elevation_skips_locked_cells() -> None:
    omap = ObservedMap.empty(6.0, 6.0, 0.25)
    omap.stamp_disk(3.0, 3.0, 1.5)
    first = np.full(omap.elevation.shape, 0.20, dtype=np.float32)
    hits = np.asarray(omap.observed, dtype=bool)
    n = omap.stamp_metric_elevation(first, hits, respect_lock=True)
    assert n > 0
    frozen = omap.elevation.copy()
    omap.lock_observed()
    flop = np.full(omap.elevation.shape, 1.50, dtype=np.float32)
    n2 = omap.stamp_metric_elevation(flop, hits, respect_lock=True)
    assert n2 == 0
    locked = np.asarray(omap.locked, dtype=bool)
    np.testing.assert_allclose(omap.elevation[locked], frozen[locked], atol=1e-5)
    omap.ingest_observer({"elevation": flop}, only_observed=True)
    np.testing.assert_allclose(omap.elevation[locked], frozen[locked], atol=1e-5)


def test_mission_stamp_writes_stereo_then_freeze_holds() -> None:
    cfg = load_config(_STEREO_YAML)
    cfg.world.width_m = 8.0
    cfg.world.height_m = 8.0
    cfg.world.resolution_m = 0.25
    policy = MissionPolicy(cfg)
    policy.observed = ObservedMap.empty(8.0, 8.0, 0.25)
    elev = np.zeros(policy.observed.elevation.shape, dtype=np.float32)
    for row in range(elev.shape[0]):
        elev[row, :] = 0.04 * row
    policy.observed.stamp_disk(4.0, 4.0, 2.4)
    pose = Pose(4.0, 4.0, 0.0, z=float(elev[16, 16]))
    info: dict = {}
    policy._stamp({"elevation": elev, "cameras": {}}, info, pose, explored=False)
    assert info.get("stereo")
    assert info["stereo"]["not_colmap"] is True
    assert info["stereo"]["fps_claim"] is None
    assert info["stereo"]["map_claim"] is None
    assert int(info["stereo"]["n_points"]) > 0
    frozen = policy.observed.elevation.copy()
    policy.observed.lock_observed()
    flop = elev + 0.80
    info2: dict = {}
    policy._stamp({"elevation": flop, "cameras": {}}, info2, pose, explored=False)
    locked = np.asarray(policy.observed.locked, dtype=bool)
    np.testing.assert_allclose(policy.observed.elevation[locked], frozen[locked], atol=1e-5)


def test_default_gym_mission_does_not_claim_stereo() -> None:
    from jims_mower.config import EnvConfig

    cfg = EnvConfig()
    cfg.world.width_m = 6.0
    cfg.world.height_m = 6.0
    cfg.world.resolution_m = 0.25
    policy = MissionPolicy(cfg)
    policy.observed = ObservedMap.empty(6.0, 6.0, 0.25)
    policy.observed.stamp_disk(3.0, 3.0, 1.2)
    info: dict = {}
    elev = np.zeros(policy.observed.elevation.shape, dtype=np.float32)
    policy._stamp({"elevation": elev}, info, Pose(3.0, 3.0, 0.0), explored=False)
    assert "stereo" not in info

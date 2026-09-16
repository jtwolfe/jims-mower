"""Gym MAP-2 elev fuse: stereo + ToF + IMU, lock, MAP-2b prior slot."""

from __future__ import annotations

import numpy as np
import pytest

from jims_mower.perception.calibration import LipFixture, build_lip_elevation, expected_lip_cells
from jims_mower.perception.elev_fuse import (
    NullMonoDepthPrior,
    apply_mono_prior,
    fuse_elev_stereo_tof_imu,
)
from jims_mower.perception.stereo import (
    find_stereo_pair,
    height_sampler_from_raster,
    rasterize_points,
    synthetic_stereo_points,
)
from jims_mower.planning.observed import ObservedMap
from jims_mower.types import CameraSpec, Pose

# Gym kerb/lip step vs authored tape. Not a field matcher residual.
KERB_STEP_TOL_M = 0.05


def _pair() -> list[CameraSpec]:
    return [
        CameraSpec("stereo_left", 0.24, 0.04, 0.38, 0.0, -22.0),
        CameraSpec("stereo_right", 0.24, -0.04, 0.38, 0.0, -22.0),
    ]


class _FlopPrior:
    def sample(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.full(shape, 1.80, dtype=np.float32),
            np.ones(shape, dtype=bool),
        )


def test_kerb_step_vs_tape_within_tolerance() -> None:
    pair = find_stereo_pair(_pair())
    assert pair is not None
    fix = LipFixture(tape_m=2.0, lip_height_m=0.12, resolution_m=0.25, world_m=8.0)
    elev = build_lip_elevation(fix)
    xs, ys, zs = synthetic_stereo_points(
        fix.robot,
        pair,
        width=64,
        height=48,
        height_at=height_sampler_from_raster(elev, resolution_m=fix.resolution_m),
        pixel_stride=2,
    )
    raster, hits = rasterize_points(
        xs,
        ys,
        zs,
        shape=elev.shape,
        resolution_m=fix.resolution_m,
        width_m=fix.world_m,
        height_m=fix.world_m,
    )
    imu = np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0], dtype=np.float32)
    tof = np.array([0.06, 0.06, 0.06, 0.06], dtype=np.float32)
    result = fuse_elev_stereo_tof_imu(
        shape=elev.shape,
        resolution_m=fix.resolution_m,
        world_size=(fix.world_m, fix.world_m),
        pose=fix.robot,
        stereo_elev=raster,
        stereo_hits=hits,
        tof=tof,
        imu=imu,
        elevation=np.zeros_like(elev),
    )
    assert result.not_matcher is True
    assert result.fps_claim is None
    assert result.map_claim is None
    assert result.n_stereo > 0
    cells = expected_lip_cells(fix)
    assert cells
    heights = [float(result.elevation[r, c]) for r, c in cells if result.stereo_hits[r, c]]
    assert heights, "gym stereo should hit the lip cells"
    mean_z = float(np.mean(heights))
    expect = float(fix.robot.z + fix.lip_height_m)
    assert abs(mean_z - expect) <= KERB_STEP_TOL_M


def test_locked_cells_do_not_flop_after_fuse() -> None:
    omap = ObservedMap.empty(6.0, 6.0, 0.25)
    omap.stamp_disk(3.0, 3.0, 1.8)
    pose = Pose(3.0, 3.0, 0.0, z=0.10)
    first = np.full(omap.elevation.shape, 0.12, dtype=np.float32)
    hits = np.asarray(omap.observed, dtype=bool)
    omap.stamp_metric_elevation(first, hits, respect_lock=True)
    frozen = omap.elevation.copy()
    omap.lock_observed()
    flop = np.full(omap.elevation.shape, 1.40, dtype=np.float32)
    result = fuse_elev_stereo_tof_imu(
        shape=omap.elevation.shape,
        resolution_m=omap.resolution_m,
        world_size=(omap.width_m, omap.height_m),
        pose=pose,
        stereo_elev=flop,
        stereo_hits=hits,
        tof=np.array([0.04, 0.04, 0.04, 0.04], dtype=np.float32),
        imu=np.array([2.0, 0.0, 9.0, 0.0, 0.0, 0.0], dtype=np.float32),
        elevation=omap.elevation,
        locked=omap.locked,
        elevation_set=omap.elevation_set,
        mono_prior=_FlopPrior(),
    )
    omap.fuse_metric(result, respect_lock=True)
    locked = np.asarray(omap.locked, dtype=bool)
    np.testing.assert_allclose(omap.elevation[locked], frozen[locked], atol=1e-5)
    omap.ingest_observer({"elevation": flop}, only_observed=True, pose=pose)
    np.testing.assert_allclose(omap.elevation[locked], frozen[locked], atol=1e-5)


def test_mono_prior_cannot_override_valid_stereo() -> None:
    shape = (12, 12)
    stereo = np.full(shape, 0.22, dtype=np.float32)
    hits = np.zeros(shape, dtype=bool)
    hits[4:8, 4:8] = True
    prior = np.full(shape, 1.75, dtype=np.float32)
    written = hits.copy()
    out = apply_mono_prior(stereo, written, prior, np.ones(shape, dtype=bool))
    np.testing.assert_allclose(out[hits], 0.22, atol=1e-6)
    assert float(out[~hits].max()) == pytest.approx(1.75)

    result = fuse_elev_stereo_tof_imu(
        shape=shape,
        resolution_m=0.25,
        world_size=(3.0, 3.0),
        pose=Pose(1.5, 1.5, 0.0, z=0.10),
        stereo_elev=stereo,
        stereo_hits=hits,
        elevation=np.zeros(shape, dtype=np.float32),
        mono_prior=_FlopPrior(),
    )
    np.testing.assert_allclose(result.elevation[hits], 0.22, atol=1e-5)
    assert result.as_info()["not_matcher"] is True


def test_null_mono_prior_writes_nothing() -> None:
    prior = NullMonoDepthPrior()
    elev, valid = prior.sample((4, 4))
    assert elev.shape == (4, 4)
    assert not np.any(valid)

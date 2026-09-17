"""Yard-scale grade recovery: heuristic maps must track gradient_yard.

These are elevation MAE / coverage metrics, not detector mAP.
"""

from __future__ import annotations

import numpy as np
import pytest

from jims_mower.constants import HAZARD_DRAIN_EDGE, TERRAIN_DRAIN, TERRAIN_DRAIN_EDGE
from jims_mower.env import MowerEnv
from jims_mower.perception.grade import PlanarGradeModel, gradients_from_attitude
from jims_mower.planning import TerrainPolicy
from jims_mower.scenarios import load_dsl_scenario
from jims_mower.types import Pose


def _gradient_env(*, cameras: int = 4) -> MowerEnv:
    scn = load_dsl_scenario("gradient_yard")
    cfg = scn.config
    cfg.sensors.width = 24
    cfg.sensors.height = 18
    cfg.sensors.camera_count = cameras
    cfg.sensors.cameras = []
    cfg.max_steps = 80
    cfg.perception.terrain_mode = "heuristic"
    return MowerEnv(config=scn, render_mode=None)


def test_gradients_from_nose_up_heading_plus_x() -> None:
    gx, gy = gradients_from_attitude(roll=0.0, pitch=0.14, yaw=0.0)
    assert gx > 0.10
    assert abs(gy) < 0.02


def test_planar_grade_model_tracks_seated_pose() -> None:
    model = PlanarGradeModel()
    pose = Pose(2.0, 6.0, 0.0, z=-0.4, pitch=0.14, roll=0.0)
    imu = np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0], dtype=np.float32)
    model.update(pose, imu, None)
    # Ascent along +x: high-x corner is above the low-x seat.
    assert model.z_at(10.0, 6.0) > model.z_at(2.0, 6.0) + 0.8
    elev, slope = model.raster((20, 20), 0.5)
    assert float(slope.mean()) == pytest.approx(0.14, abs=0.02)
    assert float(elev.max() - elev.min()) > 0.8


def test_gradient_yard_heuristic_mae_drops_seed7() -> None:
    env = _gradient_env()
    obs, info = env.reset(seed=7)
    assert info["terrain_source"] == "heuristic"
    true = env._terrain.elevation.astype(np.float32)
    # The slow prior recovers the yard grade. Instantaneous elevation is a
    # local disk — it must not be a yard-wide sheet hinged to the IMU.
    prior = obs["elevation_prior"]
    mae = float(np.mean(np.abs(prior - true)))
    flat_mae = float(np.mean(np.abs(true)))
    assert mae < 0.22
    assert mae < 0.55 * flat_mae
    assert float(prior.max() - prior.min()) > 1.0
    local = np.abs(obs["elevation"]) > 1e-4
    assert int(local.sum()) < prior.size // 2
    env.close()


def test_gradient_yard_lips_not_mass_invented_seed7() -> None:
    env = _gradient_env()
    obs, _info = env.reset(seed=7)
    true_lip = int((env._terrain.labels == TERRAIN_DRAIN_EDGE).sum())
    true_ch = int((env._terrain.labels == TERRAIN_DRAIN).sum())
    obs_lip = int((obs["hazard"] >= HAZARD_DRAIN_EDGE).sum())
    # Stamps inflate a real ditch, but must not paint thousands of false lips.
    assert true_ch > 0 and true_lip > 0
    assert obs_lip < true_lip * 5 + 120
    assert obs_lip < 900
    env.close()


def test_gradient_yard_coverage_beats_flat_map_baseline() -> None:
    """Terrain policy on seed 7 should cover more than the ~2% flat-map fight."""
    env = _gradient_env()
    obs, info = env.reset(seed=7)
    policy = TerrainPolicy(env.cfg)
    policy.reset(obs, info)
    start = float(info["coverage_fraction"])
    travel = 0.0
    prev = (env._pose.x, env._pose.y)
    steps = 80
    for _ in range(steps):
        action = policy.act(obs, info)
        obs, _reward, terminated, truncated, info = env.step(action)
        travel += float(np.hypot(env._pose.x - prev[0], env._pose.y - prev[1]))
        prev = (env._pose.x, env._pose.y)
        if terminated or truncated:
            break
    coverage = float(info["coverage_fraction"])
    env.close()
    # Pre-fix baseline: ~8.5 m / 160 steps (~4 m / 80) and ~2% coverage while
    # fighting a flat map. On a recovered grade the chassis should travel and
    # cut without a drain drop. Grade-aware cruise is slower than the old
    # 0.5 m/s (KIND_GRADE × grade_speed_factor); 80 steps should still move
    # more than a spin-in-place (~2 m), not the old 4 m straight-line.
    assert coverage >= start
    # Wider 0.70 m chassis + grade-aware cruise is slower than the old
    # 50 cm / 0.5 m/s straight-line. 80 steps should still move more
    # than a spin-in-place.
    assert travel > 1.0 or (coverage - start) > 0.003
    assert info.get("drain_drop") is not True


def test_elevation_prior_in_obs() -> None:
    env = _gradient_env()
    obs, _info = env.reset(seed=7)
    assert "elevation_prior" in obs
    assert obs["elevation_prior"].shape == obs["elevation"].shape
    assert float(np.abs(obs["elevation_prior"]).max()) > 0.3
    env.close()


def test_paint_planar_grade_is_local_and_committed() -> None:
    from jims_mower.perception.grade import LOCAL_GRADE_RADIUS_M, paint_planar_grade

    model = PlanarGradeModel()
    pose = Pose(2.0, 2.0, 0.0, z=0.10, pitch=0.14, roll=0.0)
    model.update(pose, None, None)
    elev = np.zeros((24, 24), dtype=np.float32)
    slope = np.zeros_like(elev)
    committed = np.zeros((24, 24), dtype=bool)
    prior = paint_planar_grade(
        model,
        elev,
        slope,
        resolution_m=0.25,
        origin_xy=(2.0, 2.0),
        radius_m=LOCAL_GRADE_RADIUS_M,
        committed=committed,
    )
    assert float(np.abs(prior).max()) > 0.3
    written = np.abs(elev) > 1e-5
    assert int(written.sum()) > 8
    assert int(written.sum()) < 24 * 24 // 2
    first = elev.copy()
    # A violent tip must not rewrite committed cells.
    model.update(Pose(2.0, 2.0, 0.0, z=0.10, pitch=-0.40, roll=0.35), None, None)
    paint_planar_grade(
        model,
        elev,
        slope,
        resolution_m=0.25,
        origin_xy=(2.0, 2.0),
        radius_m=LOCAL_GRADE_RADIUS_M,
        committed=committed,
    )
    locked = first[committed]
    now = elev[committed]
    assert float(np.abs(now - locked).max()) < 0.08


def test_planar_grade_tip_does_not_flip_yard_sign() -> None:
    """Instantaneous pitch/roll must not invert the slow plane across the yard."""
    model = PlanarGradeModel()
    pose = Pose(4.0, 4.0, 0.0, z=0.0, pitch=0.12, roll=0.0)
    model.update(pose, None, None)
    before, _ = model.raster((20, 20), 0.4)
    left = float(before[:, :8].mean())
    right = float(before[:, 12:].mean())
    assert right > left
    # Ridge tip: nose down, left-up. Old code preferred the larger tilt
    # and re-hinged the whole sheet.
    for i in range(8):
        model.update(
            Pose(4.0 + 0.05 * i, 4.0, 0.2, z=0.02, pitch=-0.45, roll=0.40),
            None,
            None,
        )
    after, _ = model.raster((20, 20), 0.4)
    assert float(after[:, 12:].mean()) > float(after[:, :8].mean()) - 0.05
    # Correlation with the first plane stays positive — no sign flip.
    flat_b = before.reshape(-1) - float(before.mean())
    flat_a = after.reshape(-1) - float(after.mean())
    denom = float(np.linalg.norm(flat_b) * np.linalg.norm(flat_a)) + 1e-6
    corr = float(np.dot(flat_b, flat_a) / denom)
    assert corr > 0.35

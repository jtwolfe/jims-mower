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
    est = obs["elevation"]
    mae = float(np.mean(np.abs(est - true)))
    flat_mae = float(np.mean(np.abs(true)))
    # Flattened maps sat near 0.43 m MAE. The plane must recover the grade.
    assert mae < 0.22
    assert mae < 0.55 * flat_mae
    assert float(est.max() - est.min()) > 1.0
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
    steps = 40
    for _ in range(steps):
        action = policy.act(obs, info)
        obs, _reward, terminated, truncated, info = env.step(action)
        travel += float(np.hypot(env._pose.x - prev[0], env._pose.y - prev[1]))
        prev = (env._pose.x, env._pose.y)
        if terminated or truncated:
            break
    coverage = float(info["coverage_fraction"])
    env.close()
    # Pre-fix baseline: ~8.5 m / 160 steps and ~2% coverage. 40 steps on a
    # recovered grade should move and cut more than a blocked flat map.
    assert coverage >= start
    assert travel > 6.0 or (coverage - start) > 0.015
    assert info.get("drain_drop") is not True


def test_elevation_prior_in_obs() -> None:
    env = _gradient_env()
    obs, _info = env.reset(seed=7)
    assert "elevation_prior" in obs
    assert obs["elevation_prior"].shape == obs["elevation"].shape
    assert float(np.abs(obs["elevation_prior"]).max()) > 0.3
    env.close()

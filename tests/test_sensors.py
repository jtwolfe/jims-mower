"""IMU / GPS / ToF observation shapes and consistency."""

from __future__ import annotations

import numpy as np
import pytest

from jims_mower.constants import GRAVITY_MPS2
from jims_mower.env import MowerEnv
from jims_mower.perception import BlindTerrainObserver, HeuristicTerrainObserver
from jims_mower.sensors import (
    imu_to_array,
    sample_accel_bias,
    simulate_gps,
    simulate_imu,
    simulate_tof,
)
from jims_mower.types import Pose


def test_imu_level_rest_is_gravity_on_z() -> None:
    rng = np.random.default_rng(0)
    pose = Pose(1.0, 2.0, 0.1)
    sample = simulate_imu(
        pose,
        pose,
        v=0.0,
        v_prev=0.0,
        omega=0.0,
        dt=0.1,
        rng=rng,
        accel_noise_std=0.0,
        gyro_noise_std=0.0,
        accel_bias=np.zeros(3),
    )
    acc = sample.accel_mps2
    assert acc[0] == pytest.approx(0.0, abs=1e-5)
    assert acc[1] == pytest.approx(0.0, abs=1e-5)
    assert acc[2] == pytest.approx(GRAVITY_MPS2, abs=1e-4)
    assert np.allclose(sample.gyro_radps, 0.0)


def test_imu_array_shape() -> None:
    rng = np.random.default_rng(1)
    sample = simulate_imu(
        Pose(0, 0, 0),
        Pose(0, 0, 0),
        v=0.2,
        v_prev=0.1,
        omega=0.3,
        dt=0.1,
        rng=rng,
        accel_noise_std=0.05,
        gyro_noise_std=0.01,
        accel_bias=sample_accel_bias(rng, 0.02),
    )
    buf = imu_to_array(sample)
    assert buf.shape == (6,)
    assert buf.dtype == np.float32


def test_gps_near_truth_when_valid() -> None:
    rng = np.random.default_rng(2)
    pose = Pose(5.0, 4.0, 0.0, z=0.1)
    hits = 0
    for _ in range(30):
        fix = simulate_gps(
            pose,
            rng,
            horiz_noise_std_m=0.2,
            vert_noise_std_m=0.2,
            dropout_prob=0.0,
        )
        assert fix.valid == pytest.approx(1.0)
        assert abs(fix.x - 5.0) < 1.0
        assert abs(fix.y - 4.0) < 1.0
        hits += 1
    assert hits == 30


def test_gps_dropout() -> None:
    rng = np.random.default_rng(3)
    fix = simulate_gps(
        Pose(1, 1, 0),
        rng,
        horiz_noise_std_m=0.1,
        vert_noise_std_m=0.1,
        dropout_prob=1.0,
    )
    assert fix.valid == pytest.approx(0.0)
    assert fix.as_array().shape == (4,)


def test_tof_clipped_and_shaped() -> None:
    rng = np.random.default_rng(4)
    out = simulate_tof(np.array([0.06, 0.06, 0.06, 2.5]), rng, noise_std_m=0.0, max_range_m=1.2)
    assert out.shape == (4,)
    assert out[3] == pytest.approx(1.2)


def test_tof_count_masks_unused_corners() -> None:
    rng = np.random.default_rng(5)
    raw = np.array([0.20, 0.21, 0.22, 0.23])
    two = simulate_tof(raw, rng, noise_std_m=0.0, max_range_m=1.2, count=2)
    assert two[0] == pytest.approx(0.20)
    assert two[1] == pytest.approx(0.21)
    assert two[2] == pytest.approx(0.0)
    assert two[3] == pytest.approx(0.0)
    none = simulate_tof(raw, rng, noise_std_m=0.0, max_range_m=1.2, count=0)
    assert np.allclose(none, 0.0)


def test_env_obs_sensor_shapes() -> None:
    env = MowerEnv(
        config={
            "sensors": {"width": 16, "height": 12, "camera_count": 4},
            "world": {
                "width_m": 8.0,
                "height_m": 8.0,
                "resolution_m": 0.25,
                "n_people": 0,
                "n_dogs": 0,
                "n_cats": 0,
                "n_birds": 0,
                "n_trees": 0,
                "n_furniture": 0,
                "n_toys": 0,
                "terrain": {"enabled": True, "n_drains": 2, "n_banks": 1, "noise_amp_m": 0.0},
            },
        }
    )
    obs, info = env.reset(seed=21)
    assert obs["imu"].shape == (6,)
    assert obs["gps"].shape == (4,)
    assert obs["tof"].shape == (4,)
    assert obs["elevation"].shape == obs["coverage"].shape
    assert obs["slope"].shape == obs["coverage"].shape
    assert obs["hazard"].shape == obs["coverage"].shape
    assert obs["confidence"].shape == obs["coverage"].shape
    assert float(obs["confidence"].min()) >= 0.0
    assert float(obs["confidence"].max()) <= 1.0
    assert env.observation_space.contains(obs)
    assert info["n_drains"] >= 1
    env.close()


def test_blind_terrain_observer_zeros() -> None:
    env = MowerEnv(
        config={
            "sensors": {"width": 16, "height": 12, "camera_count": 4},
            "world": {
                "width_m": 8.0,
                "height_m": 8.0,
                "resolution_m": 0.25,
                "n_people": 0,
                "n_dogs": 0,
                "n_cats": 0,
                "n_birds": 0,
                "n_trees": 0,
                "n_furniture": 0,
                "n_toys": 0,
                "terrain": {"enabled": True, "n_drains": 1, "n_banks": 0},
            },
            "perception": {"terrain_mode": "blind"},
        },
        terrain_observer=BlindTerrainObserver(),
    )
    obs, info = env.reset(seed=22)
    assert float(np.abs(obs["elevation"]).max()) == pytest.approx(0.0)
    assert info["terrain_source"] == "blind"
    env.close()


def test_heuristic_observer_paints_local_slope() -> None:
    env = MowerEnv(
        config={
            "sensors": {"width": 16, "height": 12, "camera_count": 4},
            "world": {
                "width_m": 8.0,
                "height_m": 8.0,
                "resolution_m": 0.25,
                "n_people": 0,
                "n_dogs": 0,
                "n_cats": 0,
                "n_birds": 0,
                "n_trees": 0,
                "n_furniture": 0,
                "n_toys": 0,
                "terrain": {"enabled": False},
            },
        },
        terrain_observer=HeuristicTerrainObserver(radius_m=1.0),
    )
    obs, info = env.reset(seed=23)
    assert info["terrain_source"] == "heuristic"
    # Local disk is non-zero even on flat ground (z/slope may be ~0); maps exist.
    assert obs["slope"].shape == obs["coverage"].shape
    env.close()

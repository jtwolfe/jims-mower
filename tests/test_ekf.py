"""EKF observability sanity (not a paper) + terrain-policy wiring."""

from __future__ import annotations

import math

import numpy as np
import pytest

from jims_mower.config import EnvConfig
from jims_mower.constants import GRAVITY_MPS2
from jims_mower.planning.controller import TerrainPolicy
from jims_mower.planning.fusion import (
    ComplementaryPoseFilter,
    EkfNoise,
    EkfPoseFilter,
    make_pose_filter,
)


def _level_imu(yaw_rate: float = 0.0) -> np.ndarray:
    return np.array([0.0, 0.0, GRAVITY_MPS2, 0.0, 0.0, yaw_rate], dtype=np.float32)


def test_dead_reckon_matches_commanded_odom() -> None:
    ekf = EkfPoseFilter(EkfNoise(q_xy=0.0, q_yaw=0.0, q_odom=0.0, q_tilt=0.0, q_z=0.0))
    ekf.reset(0.0, 0.0, 0.0)
    gps = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    pose = ekf.update(gps, _level_imu(), 0.2, commanded_v=1.0)
    pose = ekf.update(gps, _level_imu(), 0.2, commanded_v=1.0)
    assert pose.x == pytest.approx(0.4, abs=1e-5)
    assert pose.y == pytest.approx(0.0, abs=1e-5)


def test_xy_covariance_grows_without_gps() -> None:
    ekf = EkfPoseFilter(EkfNoise(q_xy=0.25, q_odom=0.20))
    ekf.reset(0.0, 0.0, 0.0)
    # Start certain so process noise (no GPS) is the observability story.
    ekf._P[0, 0] = 0.04
    ekf._P[1, 1] = 0.04
    p0 = float(ekf.covariance()[0, 0])
    gps = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(40):
        ekf.update(gps, _level_imu(), 0.1, commanded_v=0.8)
    assert float(ekf.covariance()[0, 0]) > p0 * 2.0


def test_gps_keeps_xy_tighter_than_odom_only() -> None:
    blind = EkfPoseFilter()
    blind.reset(0.0, 0.0, 0.0)
    fused = EkfPoseFilter()
    fused.reset(0.0, 0.0, 0.0)
    drop = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    imu = _level_imu()
    x = 0.0
    for _ in range(40):
        x += 0.05
        blind.update(drop, imu, 0.1, commanded_v=0.5)
        fused.update(np.array([x, 0.0, 0.0, 1.0], dtype=np.float32), imu, 0.1, commanded_v=0.5)
    assert float(fused.covariance()[0, 0]) < float(blind.covariance()[0, 0])
    assert float(fused.covariance()[0, 0]) < 2.5


def test_yaw_more_observable_when_moving_under_gps() -> None:
    sit = EkfPoseFilter()
    sit.reset(0.0, 0.0, 0.0)
    move = EkfPoseFilter()
    move.reset(0.0, 0.0, 0.0)
    imu = _level_imu()
    x = 0.0
    for _ in range(50):
        sit.update(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), imu, 0.1, commanded_v=0.0)
        x += 0.10
        move.update(np.array([x, 0.0, 0.0, 1.0], dtype=np.float32), imu, 0.1, commanded_v=1.0)
    assert float(move.covariance()[3, 3]) < float(sit.covariance()[3, 3])


def test_accel_tilt_shrinks_pitch_variance() -> None:
    ekf = EkfPoseFilter(EkfNoise(r_tilt=0.05))
    ekf.reset(0.0, 0.0, 0.0, pitch=0.0)
    p0 = float(ekf.covariance()[4, 4])
    pitch = 0.20
    imu = np.array(
        [-GRAVITY_MPS2 * math.sin(pitch), 0.0, GRAVITY_MPS2 * math.cos(pitch), 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    gps = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(20):
        ekf.update(gps, imu, 0.1)
    assert float(ekf.covariance()[4, 4]) < p0
    assert ekf.pose().pitch == pytest.approx(pitch, abs=0.08)


def test_optional_gps_z_is_observable_only_when_enabled() -> None:
    with_z = EkfPoseFilter(EkfNoise(use_gps_z=True, r_gps_z=0.35))
    no_z = EkfPoseFilter(EkfNoise(use_gps_z=False))
    with_z.reset(0.0, 0.0, 0.0, z=0.0)
    no_z.reset(0.0, 0.0, 0.0, z=0.0)
    imu = _level_imu()
    gps = np.array([0.0, 0.0, 0.55, 1.0], dtype=np.float32)
    for _ in range(25):
        with_z.update(gps, imu, 0.1)
        no_z.update(gps, imu, 0.1)
    assert float(with_z.covariance()[2, 2]) < float(no_z.covariance()[2, 2])
    assert with_z.pose().z == pytest.approx(0.55, abs=0.2)


def test_gps_gate_rejects_jumps() -> None:
    ekf = EkfPoseFilter(EkfNoise(gps_gate_m=1.0, r_gps_xy=0.2))
    ekf.reset(0.0, 0.0, 0.0)
    imu = _level_imu()
    ekf.update(np.array([20.0, 0.0, 0.0, 1.0], dtype=np.float32), imu, 0.1)
    assert ekf.pose().x == pytest.approx(0.0, abs=1e-6)


def test_rejects_negative_noise() -> None:
    with pytest.raises(ValueError):
        EkfNoise(q_xy=-0.1)


def test_terrain_policy_uses_ekf_by_default() -> None:
    policy = TerrainPolicy(EnvConfig())
    assert isinstance(policy.fusion, EkfPoseFilter)


def test_make_pose_filter_complementary_stub() -> None:
    cfg = EnvConfig()
    cfg.planner.pose_filter = "complementary"
    filt = make_pose_filter(cfg)
    assert isinstance(filt, ComplementaryPoseFilter)

"""Complementary GPS + IMU pose stub."""

from __future__ import annotations

import math

import numpy as np
import pytest

from jims_mower.constants import GRAVITY_MPS2
from jims_mower.planning.fusion import ComplementaryPoseFilter, attitude_from_accel


def test_level_accel_is_flat() -> None:
    roll, pitch = attitude_from_accel(np.array([0.0, 0.0, GRAVITY_MPS2], dtype=np.float32))
    assert roll == pytest.approx(0.0, abs=1e-5)
    assert pitch == pytest.approx(0.0, abs=1e-5)


def test_pitch_from_forward_accel() -> None:
    # Nose-up: gravity has a −x component in the body frame.
    pitch_true = 0.30
    imu = np.array(
        [-GRAVITY_MPS2 * math.sin(pitch_true), 0.0, GRAVITY_MPS2 * math.cos(pitch_true)],
        dtype=np.float32,
    )
    _roll, pitch = attitude_from_accel(imu)
    assert pitch == pytest.approx(pitch_true, abs=0.02)


def test_gps_valid_nudges_xy() -> None:
    filt = ComplementaryPoseFilter(gps_blend=0.5, accel_blend=1.0)
    filt.reset(0.0, 0.0, 0.0)
    imu = np.array([0.0, 0.0, GRAVITY_MPS2, 0.0, 0.0, 0.0], dtype=np.float32)
    gps = np.array([4.0, 2.0, 0.1, 1.0], dtype=np.float32)
    pose = filt.update(gps, imu, 0.1)
    assert pose.x == pytest.approx(2.0, abs=1e-5)
    assert pose.y == pytest.approx(1.0, abs=1e-5)


def test_gps_dropout_keeps_xy() -> None:
    filt = ComplementaryPoseFilter(gps_blend=0.5, accel_blend=1.0)
    filt.reset(1.5, 2.5, 0.0)
    imu = np.array([0.0, 0.0, GRAVITY_MPS2, 0.0, 0.0, 0.0], dtype=np.float32)
    gps = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    pose = filt.update(gps, imu, 0.1)
    assert pose.x == pytest.approx(1.5)
    assert pose.y == pytest.approx(2.5)


def test_dead_reckon_forward() -> None:
    filt = ComplementaryPoseFilter(gps_blend=0.0, accel_blend=1.0)
    filt.reset(0.0, 0.0, 0.0)
    imu = np.array([0.0, 0.0, GRAVITY_MPS2, 0.0, 0.0, 0.0], dtype=np.float32)
    gps = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    pose = filt.update(gps, imu, 0.2, commanded_v=1.0, commanded_omega=0.0)
    assert pose.x == pytest.approx(0.2, abs=1e-5)
    assert pose.y == pytest.approx(0.0, abs=1e-5)


def test_rejects_bad_blend() -> None:
    with pytest.raises(ValueError):
        ComplementaryPoseFilter(gps_blend=1.5)

"""Simulated IMU, GNSS, and downward ToF — noisy but physically consistent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from jims_mower.cameras import _body_to_world_matrix
from jims_mower.constants import GRAVITY_MPS2
from jims_mower.kinematics import wheel_positions
from jims_mower.types import Pose

# ICD ToF order: FL, FR, RL, RR. Used by the gym board-under-wheel fixture.
TOF_CORNER_INDEX = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}


@dataclass(frozen=True)
class IMUSample:
    accel_mps2: np.ndarray  # body frame, specific force (level rest ≈ [0,0,g])
    gyro_radps: np.ndarray  # body frame, rad/s
    bias_mps2: np.ndarray


@dataclass(frozen=True)
class GPSSample:
    x: float
    y: float
    z: float
    valid: float  # 1.0 fix, 0.0 dropout

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.valid], dtype=np.float32)


def _euler_rates_to_body(
    roll: float, pitch: float, roll_rate: float, pitch_rate: float, yaw_rate: float
) -> np.ndarray:
    """Map ZYX Euler rates to body (p, q, r). Small-angle fallback is fine on grass."""
    sr, cr = np.sin(roll), np.cos(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    p = roll_rate - yaw_rate * sp
    q = pitch_rate * cr + yaw_rate * cp * sr
    r = -pitch_rate * sr + yaw_rate * cp * cr
    return np.array([p, q, r], dtype=np.float64)


def simulate_imu(
    pose: Pose,
    prev: Pose,
    *,
    v: float,
    v_prev: float,
    omega: float,
    dt: float,
    rng: np.random.Generator,
    accel_noise_std: float,
    gyro_noise_std: float,
    accel_bias: np.ndarray,
    enabled: bool = True,
) -> IMUSample:
    """Accel / gyro consistent with attitude and planar motion + white noise."""
    if not enabled:
        return IMUSample(
            accel_mps2=np.zeros(3, dtype=np.float32),
            gyro_radps=np.zeros(3, dtype=np.float32),
            bias_mps2=np.zeros(3, dtype=np.float32),
        )
    dt = max(float(dt), 1e-6)
    r = _body_to_world_matrix(pose.theta, pose.pitch, pose.roll)
    a_world = np.array(
        [
            (v * np.cos(pose.theta) - v_prev * np.cos(prev.theta)) / dt,
            (v * np.sin(pose.theta) - v_prev * np.sin(prev.theta)) / dt,
            0.0,
        ],
        dtype=np.float64,
    )
    g_world = np.array([0.0, 0.0, -GRAVITY_MPS2], dtype=np.float64)
    accel_true = r.T @ (a_world - g_world)
    roll_rate = (pose.roll - prev.roll) / dt
    pitch_rate = (pose.pitch - prev.pitch) / dt
    gyro_true = _euler_rates_to_body(pose.roll, pose.pitch, roll_rate, pitch_rate, omega)
    accel = accel_true + accel_bias + rng.normal(0.0, accel_noise_std, size=3)
    gyro = gyro_true + rng.normal(0.0, gyro_noise_std, size=3)
    return IMUSample(
        accel_mps2=accel.astype(np.float32),
        gyro_radps=gyro.astype(np.float32),
        bias_mps2=np.asarray(accel_bias, dtype=np.float32),
    )


def simulate_gps(
    pose: Pose,
    rng: np.random.Generator,
    *,
    horiz_noise_std_m: float,
    vert_noise_std_m: float,
    dropout_prob: float,
    include_altitude: bool = True,
    enabled: bool = True,
) -> GPSSample:
    if not enabled:
        return GPSSample(0.0, 0.0, 0.0, 0.0)
    if float(rng.random()) < dropout_prob:
        return GPSSample(0.0, 0.0, 0.0, 0.0)
    nx, ny = rng.normal(0.0, horiz_noise_std_m, size=2)
    nz = float(rng.normal(0.0, vert_noise_std_m)) if include_altitude else 0.0
    return GPSSample(
        float(pose.x + nx),
        float(pose.y + ny),
        float(pose.z + nz) if include_altitude else 0.0,
        1.0,
    )


def simulate_tof(
    clearances_m: np.ndarray,
    rng: np.random.Generator,
    *,
    noise_std_m: float,
    max_range_m: float,
    enabled: bool = True,
    count: int = 4,
) -> np.ndarray:
    """Downward ranges at the four wheel corners (FL, FR, RL, RR).

    ``count`` is 0 (all zero), 2 (front pair only), or 4 (full). Unused
    corners stay 0 so ``stamp_tof_corners`` skips them.
    """
    n = 4
    n_active = int(count)
    if not enabled or n_active <= 0:
        return np.zeros(n, dtype=np.float32)
    noisy = np.asarray(clearances_m, dtype=np.float64).reshape(-1)[:n]
    if noisy.size < n:
        noisy = np.pad(noisy, (0, n - noisy.size))
    noisy = noisy + rng.normal(0.0, noise_std_m, size=n)
    out = np.clip(noisy, 0.0, max_range_m).astype(np.float32)
    if n_active < n:
        out[n_active:] = 0.0
    return out


def imu_to_array(sample: IMUSample) -> np.ndarray:
    return np.concatenate([sample.accel_mps2, sample.gyro_radps]).astype(np.float32)


def slide_board_under_wheel(
    height_field: Any,
    pose: Pose,
    corner: str,
    *,
    length_m: float,
    track_m: float,
    thickness_m: float = 0.04,
    radius_m: float = 0.10,
) -> tuple[float, float]:
    """Raise a small disk under one wheel (gym fixture, not a real ToF board).

    Returns the world XY of that wheel. Next :func:`simulate_tof` on
    :func:`wheel_clearances` should show a shorter range at that corner
    if the chassis is not re-seated first.
    """
    key = str(corner).upper()
    if key not in TOF_CORNER_INDEX:
        raise ValueError(f"ToF corner must be one of {tuple(TOF_CORNER_INDEX)}; got {corner!r}")
    wheels = wheel_positions(pose, length_m, track_m)
    wx, wy = wheels[TOF_CORNER_INDEX[key]]
    res = float(height_field.resolution_m)
    rad = max(float(radius_m), res * 1.5)
    thick = float(thickness_m)
    rows, cols = height_field.rows, height_field.cols
    yy = (np.arange(rows) + 0.5) * res
    xx = (np.arange(cols) + 0.5) * res
    grid_x, grid_y = np.meshgrid(xx, yy)
    disk = (grid_x - wx) ** 2 + (grid_y - wy) ** 2 <= rad * rad
    height_field.elevation = np.where(
        disk, height_field.elevation + np.float32(thick), height_field.elevation
    ).astype(np.float32)
    if hasattr(height_field, "recompute_slope"):
        height_field.recompute_slope()
    return float(wx), float(wy)


def sample_accel_bias(rng: np.random.Generator, std: float) -> np.ndarray:
    if std <= 0:
        return np.zeros(3, dtype=np.float64)
    return rng.normal(0.0, std, size=3)

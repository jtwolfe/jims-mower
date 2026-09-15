"""Simulated IMU, GNSS, and downward ToF — noisy but physically consistent."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from jims_mower.cameras import _body_to_world_matrix
from jims_mower.constants import GRAVITY_MPS2
from jims_mower.types import Pose


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


def sample_accel_bias(rng: np.random.Generator, std: float) -> np.ndarray:
    if std <= 0:
        return np.zeros(3, dtype=np.float64)
    return rng.normal(0.0, std, size=3)

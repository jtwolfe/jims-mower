"""GPS + IMU complementary-filter pose stub (not a full EKF)."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from jims_mower.constants import GRAVITY_MPS2
from jims_mower.kinematics import wrap_angle
from jims_mower.types import Pose


def attitude_from_accel(imu: np.ndarray) -> tuple[float, float]:
    """Roll / pitch from specific force, assuming near-static (no VIO)."""
    if imu is None or np.asarray(imu).size < 3:
        return 0.0, 0.0
    ax, ay, az = float(imu[0]), float(imu[1]), float(imu[2])
    roll = math.atan2(ay, az) if (ay * ay + az * az) > 1e-6 else 0.0
    pitch = math.atan2(-ax, math.hypot(ay, az))
    return roll, pitch


class ComplementaryPoseFilter:
    """Loosely-coupled GPS XY + IMU attitude. Working onboard stub.

    Position: integrate commanded (or last) forward speed, then nudge toward
    a valid GNSS fix. Attitude: gyro-integrate, then blend accel tilt.
    Replace with your EKF / complementary filter on the Orin Nano.
    """

    def __init__(
        self,
        *,
        gps_blend: float = 0.08,
        accel_blend: float = 0.10,
    ) -> None:
        if not 0.0 <= gps_blend <= 1.0 or not 0.0 <= accel_blend <= 1.0:
            raise ValueError("blend weights must be in [0, 1]")
        self.gps_blend = float(gps_blend)
        self.accel_blend = float(accel_blend)
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.theta = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self._seeded = False

    def reset(
        self,
        x: float = 0.0,
        y: float = 0.0,
        theta: float = 0.0,
        z: float = 0.0,
        pitch: float = 0.0,
        roll: float = 0.0,
    ) -> Pose:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.theta = wrap_angle(theta)
        self.pitch = float(pitch)
        self.roll = float(roll)
        self._seeded = True
        return self.pose()

    def pose(self) -> Pose:
        return Pose(self.x, self.y, self.theta, self.z, self.pitch, self.roll)

    def update(
        self,
        gps: np.ndarray,
        imu: np.ndarray,
        dt: float,
        *,
        commanded_v: float = 0.0,
        commanded_omega: float = 0.0,
        seed_xy: Optional[tuple[float, float]] = None,
    ) -> Pose:
        dt = max(float(dt), 1e-6)
        imu = np.asarray(imu, dtype=np.float32).reshape(-1)
        gps = np.asarray(gps, dtype=np.float32).reshape(-1)

        if not self._seeded:
            if gps.size >= 4 and float(gps[3]) > 0.5:
                self.reset(float(gps[0]), float(gps[1]), 0.0, float(gps[2]) if gps.size >= 3 else 0.0)
            elif seed_xy is not None:
                self.reset(seed_xy[0], seed_xy[1])
            else:
                self._seeded = True

        gyro_yaw = float(imu[5]) if imu.size >= 6 else 0.0
        gyro_roll = float(imu[3]) if imu.size >= 6 else 0.0
        gyro_pitch = float(imu[4]) if imu.size >= 6 else 0.0
        # Commanded yaw is the motion we just asked for; gyro is a noisy check.
        yaw_rate = float(commanded_omega)
        if abs(gyro_yaw) > 1e-6:
            yaw_rate = 0.75 * yaw_rate + 0.25 * gyro_yaw
        self.theta = wrap_angle(self.theta + yaw_rate * dt)

        spec = float(np.linalg.norm(imu[:3])) if imu.size >= 3 else GRAVITY_MPS2
        self.roll = self.roll + gyro_roll * dt
        self.pitch = self.pitch + gyro_pitch * dt
        # Accel tilt is only trustworthy when specific force ≈ g (not a launch).
        if abs(spec - GRAVITY_MPS2) < 0.75:
            roll_a, pitch_a = attitude_from_accel(imu)
            a = self.accel_blend
            self.roll = (1.0 - a) * self.roll + a * roll_a
            self.pitch = (1.0 - a) * self.pitch + a * pitch_a

        self.x += float(commanded_v) * math.cos(self.theta) * dt
        self.y += float(commanded_v) * math.sin(self.theta) * dt

        if gps.size >= 4 and float(gps[3]) > 0.5:
            innov = math.hypot(float(gps[0]) - self.x, float(gps[1]) - self.y)
            # Gate large GNSS jumps (yard-scale noise / multipath). Keep odom.
            if innov < 5.0:
                b = self.gps_blend
                self.x = (1.0 - b) * self.x + b * float(gps[0])
                self.y = (1.0 - b) * self.y + b * float(gps[1])
            if gps.size >= 3:
                self.z = (1.0 - self.gps_blend) * self.z + self.gps_blend * float(gps[2])
        return self.pose()

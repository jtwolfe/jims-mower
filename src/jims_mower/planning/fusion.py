"""GPS + IMU + wheel-odometry pose filters (EKF plus complementary stub)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Protocol, Union

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


# State: [x, y, z, yaw, pitch, roll]. Wheel odom + gyro are the process
# inputs; GPS XY (+ optional Z) and accel tilt are the measurements.
_STATE_N = 6
_IDX_X, _IDX_Y, _IDX_Z, _IDX_YAW, _IDX_PITCH, _IDX_ROLL = range(_STATE_N)


@dataclass
class EkfNoise:
    """Process / measurement stds. Units: metres, radians, per-sqrt(s) for Q."""

    q_xy: float = 0.04
    q_z: float = 0.08
    q_yaw: float = 0.03
    q_tilt: float = 0.04
    q_odom: float = 0.06
    r_gps_xy: float = 1.2
    r_gps_z: float = 2.0
    r_tilt: float = 0.10
    use_gps_z: bool = True
    gps_gate_m: float = 5.0
    gyro_yaw_mix: float = 0.30

    def __post_init__(self) -> None:
        for name in (
            "q_xy",
            "q_z",
            "q_yaw",
            "q_tilt",
            "q_odom",
            "r_gps_xy",
            "r_gps_z",
            "r_tilt",
            "gps_gate_m",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"EkfNoise.{name} must be >= 0")
        if not 0.0 <= float(self.gyro_yaw_mix) <= 1.0:
            raise ValueError("EkfNoise.gyro_yaw_mix must be in [0, 1]")


class PoseFilter(Protocol):
    def reset(
        self,
        x: float = 0.0,
        y: float = 0.0,
        theta: float = 0.0,
        z: float = 0.0,
        pitch: float = 0.0,
        roll: float = 0.0,
    ) -> Pose: ...

    def pose(self) -> Pose: ...

    def update(
        self,
        gps: np.ndarray,
        imu: np.ndarray,
        dt: float,
        *,
        commanded_v: float = 0.0,
        commanded_omega: float = 0.0,
        seed_xy: Optional[tuple[float, float]] = None,
    ) -> Pose: ...


class EkfPoseFilter:
    """Loosely-coupled EKF: wheel odom + gyro predict; GPS / accel-tilt update.

    Not a paper filter. Yard-scale, numpy-only, configurable noise. Absolute
    XY is GPS-observable; yaw becomes observable once the robot moves under
    GPS; pitch/roll are observable from specific force when |a| ≈ g.
    """

    def __init__(self, noise: Optional[EkfNoise] = None) -> None:
        self.noise = noise or EkfNoise()
        self._x = np.zeros(_STATE_N, dtype=np.float64)
        self._P = np.eye(_STATE_N, dtype=np.float64)
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
        self._x = np.array(
            [float(x), float(y), float(z), wrap_angle(theta), float(pitch), float(roll)],
            dtype=np.float64,
        )
        self._P = np.diag([2.25, 2.25, 4.0, 0.40, 0.08, 0.08]).astype(np.float64)
        self._seeded = True
        return self.pose()

    def pose(self) -> Pose:
        return Pose(
            float(self._x[_IDX_X]),
            float(self._x[_IDX_Y]),
            wrap_angle(float(self._x[_IDX_YAW])),
            float(self._x[_IDX_Z]),
            float(self._x[_IDX_PITCH]),
            float(self._x[_IDX_ROLL]),
        )

    def covariance(self) -> np.ndarray:
        return self._P.copy()

    def variance(self) -> np.ndarray:
        return np.diag(self._P).astype(np.float64)

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
        imu = np.asarray(imu, dtype=np.float64).reshape(-1)
        gps = np.asarray(gps, dtype=np.float64).reshape(-1)

        if not self._seeded:
            if gps.size >= 4 and float(gps[3]) > 0.5:
                self.reset(
                    float(gps[0]),
                    float(gps[1]),
                    0.0,
                    float(gps[2]) if gps.size >= 3 else 0.0,
                )
            elif seed_xy is not None:
                self.reset(seed_xy[0], seed_xy[1])
            else:
                self._seeded = True

        gyro_roll = float(imu[3]) if imu.size >= 6 else 0.0
        gyro_pitch = float(imu[4]) if imu.size >= 6 else 0.0
        gyro_yaw = float(imu[5]) if imu.size >= 6 else 0.0
        mix = self.noise.gyro_yaw_mix
        omega = (1.0 - mix) * float(commanded_omega) + mix * gyro_yaw
        self._predict(float(commanded_v), omega, gyro_roll, gyro_pitch, dt)
        self._update_gps(gps)
        self._update_tilt(imu)
        self._x[_IDX_YAW] = wrap_angle(float(self._x[_IDX_YAW]))
        return self.pose()

    def _predict(
        self,
        v: float,
        omega: float,
        gyro_roll: float,
        gyro_pitch: float,
        dt: float,
    ) -> None:
        yaw = float(self._x[_IDX_YAW])
        c, s = math.cos(yaw), math.sin(yaw)
        self._x[_IDX_X] += v * c * dt
        self._x[_IDX_Y] += v * s * dt
        self._x[_IDX_YAW] = wrap_angle(yaw + omega * dt)
        self._x[_IDX_PITCH] += gyro_pitch * dt
        self._x[_IDX_ROLL] += gyro_roll * dt

        F = np.eye(_STATE_N, dtype=np.float64)
        F[_IDX_X, _IDX_YAW] = -v * s * dt
        F[_IDX_Y, _IDX_YAW] = v * c * dt

        n = self.noise
        slip = n.q_odom * (abs(v) + 0.25 * abs(omega))
        q = np.array(
            [
                (n.q_xy + slip) ** 2 * dt,
                (n.q_xy + slip) ** 2 * dt,
                n.q_z**2 * dt,
                (n.q_yaw + n.q_odom * abs(omega)) ** 2 * dt,
                n.q_tilt**2 * dt,
                n.q_tilt**2 * dt,
            ],
            dtype=np.float64,
        )
        self._P = F @ self._P @ F.T + np.diag(q)
        self._P = 0.5 * (self._P + self._P.T)

    def _update_gps(self, gps: np.ndarray) -> None:
        if gps.size < 4 or float(gps[3]) <= 0.5:
            return
        mx, my = float(gps[0]), float(gps[1])
        innov_xy = math.hypot(mx - float(self._x[_IDX_X]), my - float(self._x[_IDX_Y]))
        if innov_xy > self.noise.gps_gate_m:
            return
        use_z = bool(self.noise.use_gps_z) and gps.size >= 3
        if use_z:
            z = np.array([mx, my, float(gps[2])], dtype=np.float64)
            H = np.zeros((3, _STATE_N), dtype=np.float64)
            H[0, _IDX_X] = 1.0
            H[1, _IDX_Y] = 1.0
            H[2, _IDX_Z] = 1.0
            R = np.diag(
                [
                    self.noise.r_gps_xy**2,
                    self.noise.r_gps_xy**2,
                    self.noise.r_gps_z**2,
                ]
            )
        else:
            z = np.array([mx, my], dtype=np.float64)
            H = np.zeros((2, _STATE_N), dtype=np.float64)
            H[0, _IDX_X] = 1.0
            H[1, _IDX_Y] = 1.0
            R = np.diag([self.noise.r_gps_xy**2, self.noise.r_gps_xy**2])
        _joseph_update(self._x, self._P, z, H, R, wrap_idx=())

    def _update_tilt(self, imu: np.ndarray) -> None:
        if imu.size < 3:
            return
        spec = float(np.linalg.norm(imu[:3]))
        if abs(spec - GRAVITY_MPS2) >= 0.75:
            return
        roll_m, pitch_m = attitude_from_accel(imu)
        z = np.array([pitch_m, roll_m], dtype=np.float64)
        H = np.zeros((2, _STATE_N), dtype=np.float64)
        H[0, _IDX_PITCH] = 1.0
        H[1, _IDX_ROLL] = 1.0
        r = self.noise.r_tilt**2
        R = np.diag([r, r])
        _joseph_update(self._x, self._P, z, H, R, wrap_idx=(0, 1))


def _joseph_update(
    x: np.ndarray,
    P: np.ndarray,
    z: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    *,
    wrap_idx: tuple[int, ...],
) -> None:
    """In-place linear Kalman update (Joseph form). ``wrap_idx`` are z rows."""
    innov = z - H @ x
    for i in wrap_idx:
        innov[i] = wrap_angle(float(innov[i]))
    S = H @ P @ H.T + R
    try:
        K = P @ H.T @ np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return
    x += K @ innov
    i_kh = np.eye(x.size, dtype=np.float64) - K @ H
    P[:, :] = i_kh @ P @ i_kh.T + K @ R @ K.T
    P[:, :] = 0.5 * (P + P.T)


def ekf_noise_from_planner(planner: object) -> EkfNoise:
    """Build ``EkfNoise`` from a ``PlannerConfig`` (or any object with ``ekf``)."""
    ekf = getattr(planner, "ekf", None)
    if ekf is None:
        return EkfNoise()
    return EkfNoise(
        q_xy=float(ekf.q_xy),
        q_z=float(ekf.q_z),
        q_yaw=float(ekf.q_yaw),
        q_tilt=float(ekf.q_tilt),
        q_odom=float(ekf.q_odom),
        r_gps_xy=float(ekf.r_gps_xy),
        r_gps_z=float(ekf.r_gps_z),
        r_tilt=float(ekf.r_tilt),
        use_gps_z=bool(ekf.use_gps_z),
        gps_gate_m=float(ekf.gps_gate_m),
        gyro_yaw_mix=float(ekf.gyro_yaw_mix),
    )


def make_pose_filter(cfg: object) -> Union[EkfPoseFilter, ComplementaryPoseFilter]:
    """Terrain-policy factory. Default is the EKF; complementary stays as a stub."""
    planner = getattr(cfg, "planner", cfg)
    kind = str(getattr(planner, "pose_filter", "ekf") or "ekf").strip().lower()
    if kind in {"complementary", "comp", "stub"}:
        return ComplementaryPoseFilter(
            gps_blend=float(getattr(planner, "gps_blend", 0.08)),
            accel_blend=float(getattr(planner, "accel_blend", 0.10)),
        )
    return EkfPoseFilter(ekf_noise_from_planner(planner))

"""Yard-scale planar grade from IMU pitch/roll + pose (no god-view DEM).

The heuristic observer used to stamp only a local disk of ``pose.z``, so a
property-scale slope flattened to ~0 in the maps the planner sees. This
module keeps a rolling tangent-plane fit so elevation / slope rasters track
the grade the IMU is sitting on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from jims_mower.types import Pose


def gradients_from_attitude(roll: float, pitch: float, yaw: float) -> tuple[float, float]:
    """World ``dz/dx``, ``dz/dy`` from body pitch (nose-up) and roll (left-up)."""
    tan_p = math.tan(float(pitch))
    tan_r = math.tan(float(roll))
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    # Body-forward (cos, sin); body-left (-sin, cos).
    gx = tan_p * c + tan_r * (-s)
    gy = tan_p * s + tan_r * c
    return float(gx), float(gy)


def attitude_from_accel(imu: np.ndarray) -> tuple[float, float]:
    """Roll / pitch from specific force, assuming near-static (no VIO)."""
    from jims_mower.constants import GRAVITY_MPS2

    arr = np.asarray(imu, dtype=np.float32).reshape(-1)
    ax, ay, az = (float(arr[0]), float(arr[1]), float(arr[2])) if arr.size >= 3 else (0.0, 0.0, GRAVITY_MPS2)
    roll = math.atan2(ay, az) if (ay * ay + az * az) > 1e-6 else 0.0
    pitch = math.atan2(-ax, math.hypot(ay, az))
    return roll, pitch


@dataclass
class PlanarGradeModel:
    """Exponential-moving planar fit ``z = z0 + gx*(x-x0) + gy*(y-y0)``."""

    alpha: float = 0.28
    gx: float = 0.0
    gy: float = 0.0
    x0: float = 0.0
    y0: float = 0.0
    z0: float = 0.0
    n: int = 0
    _samples: list[tuple[float, float, float]] = field(default_factory=list)
    _max_samples: int = 24

    def reset(self) -> None:
        self.gx = 0.0
        self.gy = 0.0
        self.x0 = 0.0
        self.y0 = 0.0
        self.z0 = 0.0
        self.n = 0
        self._samples.clear()

    def update(
        self,
        pose: Pose,
        imu: Optional[np.ndarray] = None,
        gps: Optional[np.ndarray] = None,
    ) -> None:
        roll, pitch = float(pose.roll), float(pose.pitch)
        if imu is not None:
            imu_roll, imu_pitch = attitude_from_accel(imu)
            # Prefer the larger of pose vs IMU so a seated grade is not lost
            # when one source is briefly near zero.
            if abs(imu_roll) >= abs(roll):
                roll = imu_roll
            if abs(imu_pitch) >= abs(pitch):
                pitch = imu_pitch
        cx, cy, cz = float(pose.x), float(pose.y), float(pose.z)
        if gps is not None:
            g = np.asarray(gps, dtype=np.float32).reshape(-1)
            if g.size >= 4 and float(g[3]) > 0.5:
                cx, cy = float(g[0]), float(g[1])
                if g.size >= 3 and abs(float(g[2])) > 1e-4:
                    cz = 0.65 * cz + 0.35 * float(g[2])
        gx, gy = gradients_from_attitude(roll, pitch, pose.theta)
        if self.n == 0:
            self.gx, self.gy = gx, gy
            self.x0, self.y0, self.z0 = cx, cy, cz
        else:
            a = float(np.clip(self.alpha, 0.05, 0.8))
            self.gx = (1.0 - a) * self.gx + a * gx
            self.gy = (1.0 - a) * self.gy + a * gy
            pred = self.z_at(cx, cy)
            self.z0 += a * (cz - pred)
        self.n += 1
        self._samples.append((cx, cy, cz))
        if len(self._samples) > self._max_samples:
            self._samples = self._samples[-self._max_samples :]
        self._maybe_refit()

    def _maybe_refit(self) -> None:
        """Least-squares plane when enough distinct XY samples exist."""
        if len(self._samples) < 6:
            return
        pts = np.asarray(self._samples, dtype=np.float64)
        span = float(np.hypot(pts[:, 0].ptp(), pts[:, 1].ptp()))
        if span < 0.8:
            return
        a = np.column_stack((pts[:, 0], pts[:, 1], np.ones(len(pts))))
        try:
            coef, *_ = np.linalg.lstsq(a, pts[:, 2], rcond=None)
        except np.linalg.LinAlgError:
            return
        gx, gy, c = (float(coef[0]), float(coef[1]), float(coef[2]))
        if not (math.isfinite(gx) and math.isfinite(gy) and math.isfinite(c)):
            return
        mix = 0.35
        self.gx = (1.0 - mix) * self.gx + mix * gx
        self.gy = (1.0 - mix) * self.gy + mix * gy
        new_z0 = c + self.gx * self.x0 + self.gy * self.y0
        self.z0 = (1.0 - mix) * self.z0 + mix * new_z0

    def z_at(self, x: float, y: float) -> float:
        return float(self.z0 + self.gx * (x - self.x0) + self.gy * (y - self.y0))

    def slope_rad(self) -> float:
        return float(math.atan(math.hypot(self.gx, self.gy)))

    def raster(
        self,
        shape: tuple[int, int],
        resolution_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (elevation, slope) rasters for the current plane."""
        rows, cols = int(shape[0]), int(shape[1])
        res = max(float(resolution_m), 1e-6)
        yy = (np.arange(rows, dtype=np.float32) + 0.5) * res
        xx = (np.arange(cols, dtype=np.float32) + 0.5) * res
        grid_x, grid_y = np.meshgrid(xx, yy)
        elev = (
            np.float32(self.z0)
            + np.float32(self.gx) * (grid_x - np.float32(self.x0))
            + np.float32(self.gy) * (grid_y - np.float32(self.y0))
        ).astype(np.float32)
        slope = np.full((rows, cols), np.float32(self.slope_rad()), dtype=np.float32)
        return elev, slope


def paint_planar_grade(
    model: PlanarGradeModel,
    elevation: np.ndarray,
    slope: np.ndarray,
    *,
    resolution_m: float,
    confidence: Optional[np.ndarray] = None,
    conf_value: float = 0.40,
) -> np.ndarray:
    """Overwrite elevation / floor slope with the yard-scale plane. Returns prior."""
    prior, plane_slope = model.raster(elevation.shape, resolution_m)
    elevation[:, :] = prior
    slope[:, :] = np.maximum(slope, plane_slope)
    if confidence is not None and confidence.shape == elevation.shape:
        confidence[:, :] = np.maximum(confidence, np.float32(conf_value))
    return prior

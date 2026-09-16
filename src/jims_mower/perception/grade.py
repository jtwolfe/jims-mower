"""Local grade from IMU pitch/roll + pose (no god-view DEM).

A real mower does not rebuild a global tilted plane of the whole yard
from instantaneous chassis tip.

* **IMU tilt (pitch/roll)** is the local surface normal under the chassis.
  Useful for tip risk and as a *slow* prior for nearby cells.
* **Wheel / pose ``z``** is the local height sample.
* Yard-scale ``gx, gy`` come from a slow EMA plus a short XY+z fit — not
  a license to re-orient every mapped cell every frame.

``elevation_prior`` is still the current slow plane raster (costmap
floor). ``paint_planar_grade`` writes height only in a neighborhood
around the robot, and never clobbers cells that already have a sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from jims_mower.types import Pose

# Neighborhood that may take a local z + slow-grade sample. Camera
# footprints are larger; those cells become *seen* without inheriting
# the current chassis hinge.
LOCAL_GRADE_RADIUS_M = 2.5
# Instantaneous attitude must not yank the yard-scale plane.
ATTITUDE_ALPHA = 0.06
# Least-squares mix when the robot has actually travelled.
REFIT_MIX = 0.12
# First-stamp lock; neighborhood revisits may creep this much per step.
REVISIT_ALPHA = 0.05


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


def local_disk_mask(
    shape: tuple[int, int],
    origin_xy: tuple[float, float],
    radius_m: float,
    resolution_m: float,
) -> np.ndarray:
    """True on cells whose centres sit inside ``radius_m`` of ``origin_xy``."""
    rows, cols = int(shape[0]), int(shape[1])
    res = max(float(resolution_m), 1e-6)
    yy = (np.arange(rows, dtype=np.float32) + 0.5) * res
    xx = (np.arange(cols, dtype=np.float32) + 0.5) * res
    grid_x, grid_y = np.meshgrid(xx, yy)
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    r = max(float(radius_m), res * 0.5)
    return (grid_x - ox) ** 2 + (grid_y - oy) ** 2 <= r * r


@dataclass
class PlanarGradeModel:
    """Slow planar fit ``z = z0 + gx*(x-x0) + gy*(y-y0)``.

    ``gx, gy`` are a *slow* yard prior (EMA + XY+z). ``local_gx, local_gy``
    are the instantaneous chassis normal for tip / nearby cells only.
    """

    alpha: float = ATTITUDE_ALPHA
    gx: float = 0.0
    gy: float = 0.0
    local_gx: float = 0.0
    local_gy: float = 0.0
    x0: float = 0.0
    y0: float = 0.0
    z0: float = 0.0
    n: int = 0
    _samples: list[tuple[float, float, float]] = field(default_factory=list)
    _max_samples: int = 24

    def reset(self) -> None:
        self.gx = 0.0
        self.gy = 0.0
        self.local_gx = 0.0
        self.local_gy = 0.0
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
            # Blend — do not take the larger of pose vs IMU. A ridge tip
            # used to yank the whole plane because the extreme won.
            roll = 0.5 * roll + 0.5 * imu_roll
            pitch = 0.5 * pitch + 0.5 * imu_pitch
        cx, cy, cz = float(pose.x), float(pose.y), float(pose.z)
        if gps is not None:
            g = np.asarray(gps, dtype=np.float32).reshape(-1)
            if g.size >= 4 and float(g[3]) > 0.5:
                cx, cy = float(g[0]), float(g[1])
                if g.size >= 3 and abs(float(g[2])) > 1e-4:
                    cz = 0.65 * cz + 0.35 * float(g[2])
        inst_gx, inst_gy = gradients_from_attitude(roll, pitch, pose.theta)
        self.local_gx, self.local_gy = inst_gx, inst_gy
        if self.n == 0:
            self.gx, self.gy = inst_gx, inst_gy
            self.x0, self.y0, self.z0 = cx, cy, cz
        else:
            a = float(np.clip(self.alpha, 0.02, 0.25))
            self.gx = (1.0 - a) * self.gx + a * inst_gx
            self.gy = (1.0 - a) * self.gy + a * inst_gy
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
        span = float(np.hypot(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))
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
        mix = REFIT_MIX
        self.gx = (1.0 - mix) * self.gx + mix * gx
        self.gy = (1.0 - mix) * self.gy + mix * gy
        new_z0 = c + self.gx * self.x0 + self.gy * self.y0
        self.z0 = (1.0 - mix) * self.z0 + mix * new_z0

    def z_at(self, x: float, y: float) -> float:
        return float(self.z0 + self.gx * (x - self.x0) + self.gy * (y - self.y0))

    def slope_rad(self) -> float:
        return float(math.atan(math.hypot(self.gx, self.gy)))

    def local_slope_rad(self) -> float:
        return float(math.atan(math.hypot(self.local_gx, self.local_gy)))

    def raster(
        self,
        shape: tuple[int, int],
        resolution_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (elevation, slope) rasters for the current *slow* plane.

        This is a prior, not a license to stamp the owner mesh. Instantaneous
        chassis tip lives in ``local_gx, local_gy``.
        """
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
    origin_xy: Optional[tuple[float, float]] = None,
    radius_m: float = LOCAL_GRADE_RADIUS_M,
    committed: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Write the slow plane into a *local* disk. Returns the full prior.

    Far cells are left alone. Already-committed cells are not replaced
    by the current chassis attitude — they may take a tiny revisit mix
    so ToF / wheel-z can creep, not leap.
    """
    prior, plane_slope = model.raster(elevation.shape, resolution_m)
    if origin_xy is None:
        return prior
    local = local_disk_mask(elevation.shape, origin_xy, radius_m, resolution_m)
    if committed is not None and committed.shape == local.shape:
        locked = np.asarray(committed, dtype=bool)
        fresh = local & ~locked
        revisit = local & locked
    else:
        fresh = local
        revisit = np.zeros_like(local, dtype=bool)
    if np.any(fresh):
        elevation[fresh] = prior[fresh]
        if committed is not None and committed.shape == local.shape:
            committed[fresh] = True
    if np.any(revisit):
        a = np.float32(REVISIT_ALPHA)
        elevation[revisit] = (1.0 - a) * elevation[revisit] + a * prior[revisit]
    if np.any(local):
        slope[local] = np.maximum(slope[local], plane_slope[local])
        # Instantaneous tip: floor slope under the chassis, do not rewrite z.
        tip = np.float32(model.local_slope_rad())
        slope[local] = np.maximum(slope[local], tip)
        if confidence is not None and confidence.shape == elevation.shape:
            confidence[local] = np.maximum(confidence[local], np.float32(conf_value))
    return prior

"""Pluggable terrain observers. Oracle for training; stubs for real fusion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from jims_mower.constants import GRAVITY_MPS2, HAZARD_DRAIN, HAZARD_STEEP
from jims_mower.terrain import HeightField
from jims_mower.types import PerceptionContext, Pose


@dataclass
class TerrainEstimate:
    """Per-cell maps the policy sees. Same shape as coverage / occupancy."""

    elevation: np.ndarray
    slope: np.ndarray
    hazard: np.ndarray
    source: str


@runtime_checkable
class TerrainObserver(Protocol):
    """Elevation / slope / drain-hazard maps. Swap for real CV + IMU/GNSS fusion."""

    def estimate(
        self,
        images: dict[str, np.ndarray],
        imu: np.ndarray,
        gps: np.ndarray,
        context: PerceptionContext,
    ) -> TerrainEstimate:
        ...


def _empty_maps(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros(shape, dtype=np.float32),
        np.zeros(shape, dtype=np.float32),
        np.zeros(shape, dtype=np.float32),
    )


class BlindTerrainObserver:
    """Working stub: no elevation, slope, or drain knowledge."""

    def estimate(
        self,
        images: dict[str, np.ndarray],
        imu: np.ndarray,
        gps: np.ndarray,
        context: PerceptionContext,
    ) -> TerrainEstimate:
        del images, imu, gps
        elev, slope, hazard = _empty_maps(context.map_shape)
        return TerrainEstimate(elev, slope, hazard, source="blind")


class OracleTerrainObserver:
    """God-view copy of the height field. For training loops, like MockDetector."""

    def estimate(
        self,
        images: dict[str, np.ndarray],
        imu: np.ndarray,
        gps: np.ndarray,
        context: PerceptionContext,
    ) -> TerrainEstimate:
        del images, imu, gps
        terrain = context.terrain
        if not isinstance(terrain, HeightField):
            elev, slope, hazard = _empty_maps(context.map_shape)
            return TerrainEstimate(elev, slope, hazard, source="oracle")
        if terrain.slope is None:
            terrain.recompute_slope()
        assert terrain.slope is not None
        return TerrainEstimate(
            elevation=terrain.elevation.astype(np.float32).copy(),
            slope=terrain.slope.astype(np.float32).copy(),
            hazard=terrain.hazard_map(steep_slope_rad=context.steep_slope_rad),
            source="oracle",
        )


def _attitude_from_accel(imu: np.ndarray) -> tuple[float, float]:
    """Roll / pitch from specific force, assuming near-static (no VIO)."""
    ax, ay, az = (float(imu[0]), float(imu[1]), float(imu[2])) if imu.size >= 3 else (0.0, 0.0, GRAVITY_MPS2)
    roll = math.atan2(ay, az) if (ay * ay + az * az) > 1e-6 else 0.0
    pitch = math.atan2(-ax, math.hypot(ay, az))
    return roll, pitch


class HeuristicTerrainObserver:
    """Cheap IMU + local blob stub. Not SLAM / VIO — replace on the Orin."""

    def __init__(self, radius_m: float = 1.2, steep_rad: float = 0.30) -> None:
        self.radius_m = radius_m
        self.steep_rad = steep_rad

    def estimate(
        self,
        images: dict[str, np.ndarray],
        imu: np.ndarray,
        gps: np.ndarray,
        context: PerceptionContext,
    ) -> TerrainEstimate:
        elev, slope, hazard = _empty_maps(context.map_shape)
        pose: Pose = context.pose
        roll, pitch = _attitude_from_accel(np.asarray(imu, dtype=np.float32))
        slope_est = math.hypot(roll, pitch)
        rows, cols = context.map_shape
        res = context.resolution_m
        # Local disk under the robot; GPS XY is used when the fix is valid.
        cx, cy = pose.x, pose.y
        if gps.size >= 4 and float(gps[3]) > 0.5:
            cx, cy = float(gps[0]), float(gps[1])
        r = max(self.radius_m, res)
        c0 = int((cx - r) / res)
        c1 = int((cx + r) / res)
        r0 = int((cy - r) / res)
        r1 = int((cy + r) / res)
        r2 = r * r
        for row in range(max(0, r0), min(rows, r1 + 1)):
            wy = (row + 0.5) * res
            for col in range(max(0, c0), min(cols, c1 + 1)):
                wx = (col + 0.5) * res
                if (wx - cx) ** 2 + (wy - cy) ** 2 > r2:
                    continue
                slope[row, col] = slope_est
                elev[row, col] = pose.z
                if slope_est >= self.steep_rad:
                    hazard[row, col] = HAZARD_STEEP
        # Brown pixels in the cameras are a drain-lip hint, not a map.
        if any(_looks_like_drain(frame) for frame in images.values()):
            cell = (
                int(np.clip(cy / res, 0, rows - 1)),
                int(np.clip(cx / res, 0, cols - 1)),
            )
            hazard[cell] = max(float(hazard[cell]), float(HAZARD_DRAIN))
        return TerrainEstimate(elev, slope, hazard, source="heuristic")


def _looks_like_drain(image: np.ndarray) -> bool:
    if image.ndim != 3:
        return False
    r = image[:, :, 0].astype(np.int16)
    g = image[:, :, 1].astype(np.int16)
    b = image[:, :, 2].astype(np.int16)
    brown = (r > 40) & (r < 110) & (g < r) & (b < g) & ((g + b) < 140)
    return bool(brown.mean() > 0.04)


def terrain_observer_from_mode(mode: str) -> TerrainObserver:
    key = (mode or "oracle").strip().lower()
    if key == "blind":
        return BlindTerrainObserver()
    if key == "heuristic":
        return HeuristicTerrainObserver()
    return OracleTerrainObserver()

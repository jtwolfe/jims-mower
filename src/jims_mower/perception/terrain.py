"""Pluggable terrain observers. Oracle for training; CV heuristic / blind stubs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from jims_mower.constants import GRAVITY_MPS2, HAZARD_STEEP
from jims_mower.perception.cv_terrain import (
    classify_terrain_rgb,
    project_labels_to_maps,
    stamp_tof_corners,
)
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

    def reset(self) -> None:
        return None

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

    def reset(self) -> None:
        return None

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
    """RGB + ToF drain/bank heuristic. Not a learned net — replace on the Orin.

    Colour cues match the gym renderer's ditch/bank palette. Pixels are
    back-projected onto a flat-yard plane (the onboard approximation; this
    class ignores ``context.terrain``). Downward ToF stamps a local wheel
    drop. IMU paints a slope disk under the chassis. Maps persist for the
    episode so the planner can replan as new lips enter the cameras.
    """

    def __init__(
        self,
        radius_m: float = 1.2,
        steep_rad: float = 0.30,
        max_range_m: float = 9.0,
    ) -> None:
        self.radius_m = radius_m
        self.steep_rad = steep_rad
        self.max_range_m = max_range_m
        self._elevation: Optional[np.ndarray] = None
        self._slope: Optional[np.ndarray] = None
        self._hazard: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._elevation = None
        self._slope = None
        self._hazard = None

    def _ensure_maps(self, shape: tuple[int, int]) -> None:
        if (
            self._elevation is None
            or self._elevation.shape != shape
            or self._slope is None
            or self._hazard is None
        ):
            self._elevation, self._slope, self._hazard = _empty_maps(shape)

    def estimate(
        self,
        images: dict[str, np.ndarray],
        imu: np.ndarray,
        gps: np.ndarray,
        context: PerceptionContext,
    ) -> TerrainEstimate:
        # Intentionally unused: god-view height field is training-only.
        _ = context.terrain
        self._ensure_maps(context.map_shape)
        assert self._elevation is not None and self._slope is not None and self._hazard is not None
        pose: Pose = context.pose
        cams = {c.name: c for c in context.cameras}
        for name, frame in images.items():
            cam = cams.get(name)
            if cam is None or frame.ndim != 3:
                continue
            labels = classify_terrain_rgb(frame)
            project_labels_to_maps(
                frame,
                labels,
                cam,
                pose,
                elevation=self._elevation,
                slope=self._slope,
                hazard=self._hazard,
                resolution_m=context.resolution_m,
                world_size=context.world_size,
                ground_z=0.0,
                max_range_m=self.max_range_m,
                steep_rad=max(self.steep_rad, context.steep_slope_rad),
            )
        tof = _tof_from_context(context)
        if tof is not None:
            stamp_tof_corners(
                tof,
                pose,
                hazard=self._hazard,
                elevation=self._elevation,
                slope=self._slope,
                resolution_m=context.resolution_m,
                length_m=context.length_m,
                track_m=context.track_m,
                hover_m=context.chassis_hover_m,
                steep_rad=max(self.steep_rad, context.steep_slope_rad),
            )
        self._paint_local_imu(imu, gps, context)
        return TerrainEstimate(
            self._elevation.copy(),
            self._slope.copy(),
            self._hazard.copy(),
            source="heuristic",
        )

    def _paint_local_imu(
        self,
        imu: np.ndarray,
        gps: np.ndarray,
        context: PerceptionContext,
    ) -> None:
        assert self._elevation is not None and self._slope is not None and self._hazard is not None
        pose = context.pose
        roll, pitch = _attitude_from_accel(np.asarray(imu, dtype=np.float32))
        slope_est = math.hypot(roll, pitch)
        rows, cols = context.map_shape
        res = context.resolution_m
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
                self._slope[row, col] = max(float(self._slope[row, col]), slope_est)
                if abs(float(self._elevation[row, col])) < 1e-6:
                    self._elevation[row, col] = pose.z
                if slope_est >= self.steep_rad:
                    self._hazard[row, col] = max(float(self._hazard[row, col]), float(HAZARD_STEEP))


def _tof_from_context(context: PerceptionContext) -> Optional[np.ndarray]:
    raw = context.tof
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    if arr.size < 4:
        return None
    return arr


def terrain_observer_from_mode(mode: str) -> TerrainObserver:
    key = (mode or "heuristic").strip().lower()
    if key == "blind":
        return BlindTerrainObserver()
    if key == "oracle":
        return OracleTerrainObserver()
    return HeuristicTerrainObserver()

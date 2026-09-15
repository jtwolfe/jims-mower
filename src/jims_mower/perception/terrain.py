"""Pluggable terrain observers. Oracle for training; CV / learned / blind stubs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Union, runtime_checkable

import numpy as np

from jims_mower.constants import GRAVITY_MPS2, HAZARD_DRAIN, HAZARD_DRAIN_EDGE, HAZARD_STEEP
from jims_mower.perception.cv_terrain import (
    classify_structure_rgb,
    classify_terrain_rgb,
    stamp_structure_labels,
    stamp_tof_corners,
)
from jims_mower.perception.fuse import fuse_camera_labels, gate_isolated_lips, paint_geometry_from_hazard
from jims_mower.perception.grade import PlanarGradeModel, paint_planar_grade
from jims_mower.perception.learn import TerrainMLP, classify_image, load_weights
from jims_mower.perception.temporal import HazardHysteresis
from jims_mower.terrain import HeightField
from jims_mower.types import PerceptionContext, Pose


@dataclass
class TerrainEstimate:
    """Per-cell maps the policy sees. Same shape as coverage / occupancy."""

    elevation: np.ndarray
    slope: np.ndarray
    hazard: np.ndarray
    source: str
    confidence: Optional[np.ndarray] = None
    elevation_prior: Optional[np.ndarray] = None
    structure: Optional[np.ndarray] = None


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
        conf = np.zeros(context.map_shape, dtype=np.float32)
        return TerrainEstimate(elev, slope, hazard, source="blind", confidence=conf)


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
            return TerrainEstimate(
                elev,
                slope,
                hazard,
                source="oracle",
                confidence=np.ones(context.map_shape, dtype=np.float32),
            )
        if terrain.slope is None:
            terrain.recompute_slope()
        assert terrain.slope is not None
        hazard = terrain.hazard_map(steep_slope_rad=context.steep_slope_rad)
        return TerrainEstimate(
            elevation=terrain.elevation.astype(np.float32).copy(),
            slope=terrain.slope.astype(np.float32).copy(),
            hazard=hazard,
            source="oracle",
            confidence=np.ones_like(hazard, dtype=np.float32),
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
    back-projected onto the seated tangent plane (the onboard approximation;
    this class ignores ``context.terrain``). IMU pitch/roll + pose recover a
    yard-scale planar grade so elevation/slope maps are not flattened.
    Downward ToF stamps a local wheel drop. Isolated brown lips (dirt /
    shade on a smooth grade) are gated unless they sit next to a channel.
    Maps persist for the episode so the planner can replan as new lips enter
    the cameras.
    """

    def __init__(
        self,
        radius_m: float = 1.2,
        steep_rad: float = 0.30,
        max_range_m: float = 9.0,
        temporal: bool = False,
    ) -> None:
        self.radius_m = radius_m
        self.steep_rad = steep_rad
        self.max_range_m = max_range_m
        self.temporal = bool(temporal)
        self._filter = HazardHysteresis() if self.temporal else None
        self._elevation: Optional[np.ndarray] = None
        self._slope: Optional[np.ndarray] = None
        self._hazard: Optional[np.ndarray] = None
        self._confidence: Optional[np.ndarray] = None
        self._prior: Optional[np.ndarray] = None
        self._structure: Optional[np.ndarray] = None
        self._grade = PlanarGradeModel()

    def reset(self) -> None:
        self._elevation = None
        self._slope = None
        self._hazard = None
        self._confidence = None
        self._prior = None
        self._structure = None
        self._grade.reset()
        if self._filter is not None:
            self._filter.reset()

    def _ensure_maps(self, shape: tuple[int, int]) -> None:
        if (
            self._elevation is None
            or self._elevation.shape != shape
            or self._slope is None
            or self._hazard is None
            or self._confidence is None
        ):
            self._elevation, self._slope, self._hazard = _empty_maps(shape)
            # Unobserved cells stay cheap-but-uncertain (not a hard block).
            self._confidence = np.full(shape, 0.08, dtype=np.float32)
            self._structure = np.zeros(shape, dtype=np.uint8)

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
        assert (
            self._elevation is not None
            and self._slope is not None
            and self._hazard is not None
            and self._confidence is not None
        )
        pose: Pose = context.pose
        self._grade.update(pose, imu, gps)
        self._prior = paint_planar_grade(
            self._grade,
            self._elevation,
            self._slope,
            resolution_m=context.resolution_m,
            confidence=self._confidence,
        )
        prev_hazard = self._hazard.copy()
        fused, conf = fuse_camera_labels(
            images,
            lambda frame, _cam: classify_terrain_rgb(frame),
            context.cameras,
            pose,
            resolution_m=context.resolution_m,
            world_size=context.world_size,
            map_shape=context.map_shape,
            max_range_m=self.max_range_m,
        )
        if self._filter is not None:
            self._hazard = self._filter.update(fused, conf)
            self._confidence = conf
        else:
            self._hazard = np.maximum(self._hazard, fused)
            self._confidence = (
                conf if self._confidence is None else np.maximum(self._confidence, conf)
            )
        self._hazard = gate_isolated_lips(self._hazard)
        paint_geometry_from_hazard(
            self._hazard,
            elevation=self._elevation,
            slope=self._slope,
            pose_z=pose.z,
            steep_rad=max(self.steep_rad, context.steep_slope_rad),
        )
        self._grow_drain_gaps(prev_hazard)
        self._raise_confidence(self._hazard > 0.0, 0.72)
        tof = _tof_from_context(context)
        if tof is not None:
            before = self._hazard.copy()
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
            self._raise_confidence(self._hazard > before, 0.68)
        if self._structure is None or self._structure.shape != context.map_shape:
            self._structure = np.zeros(context.map_shape, dtype=np.uint8)
        cams = {c.name: c for c in context.cameras}
        for name, frame in images.items():
            cam = cams.get(name)
            if cam is None or frame.ndim != 3:
                continue
            stamp_structure_labels(
                frame,
                classify_structure_rgb(frame),
                cam,
                pose,
                structure=self._structure,
                resolution_m=context.resolution_m,
                world_size=context.world_size,
                max_range_m=self.max_range_m,
            )
        self._paint_local_imu(imu, gps, context)
        return TerrainEstimate(
            self._elevation.copy(),
            self._slope.copy(),
            self._hazard.copy(),
            source="heuristic",
            confidence=self._confidence.copy(),
            elevation_prior=None if self._prior is None else self._prior.copy(),
            structure=self._structure.copy(),
        )

    def _raise_confidence(self, mask: np.ndarray, value: float) -> None:
        assert self._confidence is not None
        hit = np.asarray(mask, dtype=bool)
        if hit.shape != self._confidence.shape or not np.any(hit):
            return
        self._confidence[hit] = np.maximum(self._confidence[hit], np.float32(value))

    def _grow_drain_gaps(self, prev_hazard: np.ndarray) -> None:
        """One-cell grow on *new* lip/channel stamps so a broken stripe blocks."""
        assert self._hazard is not None
        channel = self._hazard >= HAZARD_DRAIN
        fresh = (self._hazard >= HAZARD_DRAIN_EDGE) & (prev_hazard < HAZARD_DRAIN_EDGE)
        # Only grow lips that already touch a channel — do not invent a ditch.
        if np.any(channel):
            near = channel.copy()
            near[1:, :] |= channel[:-1, :]
            near[:-1, :] |= channel[1:, :]
            near[:, 1:] |= channel[:, :-1]
            near[:, :-1] |= channel[:, 1:]
            fresh = fresh & near
        if not np.any(fresh):
            return
        grown = fresh.copy()
        grown[1:, :] |= fresh[:-1, :]
        grown[:-1, :] |= fresh[1:, :]
        grown[:, 1:] |= fresh[:, :-1]
        grown[:, :-1] |= fresh[:, 1:]
        promote = grown & (self._hazard < HAZARD_DRAIN_EDGE)
        self._hazard[promote] = HAZARD_DRAIN_EDGE
        self._raise_confidence(promote, 0.55)

    def _paint_local_imu(
        self,
        imu: np.ndarray,
        gps: np.ndarray,
        context: PerceptionContext,
    ) -> None:
        assert (
            self._elevation is not None
            and self._slope is not None
            and self._hazard is not None
            and self._confidence is not None
        )
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
                if slope_est >= self.steep_rad:
                    self._hazard[row, col] = max(float(self._hazard[row, col]), float(HAZARD_STEEP))
                self._confidence[row, col] = max(float(self._confidence[row, col]), 0.42)


class LearnedTerrainObserver:
    """Load exporter-trained colour+position weights. Same contract as Blind.

    Ignores ``context.terrain``. Multi-camera BEV fuse + hysteresis. Not a
    production segmentation net — sim stub, no claimed accuracy.
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        *,
        radius_m: float = 1.2,
        steep_rad: float = 0.30,
        max_range_m: float = 9.0,
        temporal: bool = True,
        stride: int = 1,
    ) -> None:
        self.weights_path = Path(weights_path) if weights_path else None
        self.radius_m = radius_m
        self.steep_rad = steep_rad
        self.max_range_m = max_range_m
        self.stride = max(1, int(stride))
        self._model: Optional[TerrainMLP] = None
        if self.weights_path is not None:
            self._model = load_weights(self.weights_path)
        self._filter = HazardHysteresis() if temporal else None
        self._elevation: Optional[np.ndarray] = None
        self._slope: Optional[np.ndarray] = None
        self._hazard: Optional[np.ndarray] = None
        self._confidence: Optional[np.ndarray] = None
        self._prior: Optional[np.ndarray] = None
        self._grade = PlanarGradeModel()

    def reset(self) -> None:
        self._elevation = None
        self._slope = None
        self._hazard = None
        self._confidence = None
        self._prior = None
        self._grade.reset()
        if self._filter is not None:
            self._filter.reset()

    def _ensure_maps(self, shape: tuple[int, int]) -> None:
        if (
            self._elevation is None
            or self._elevation.shape != shape
            or self._slope is None
            or self._hazard is None
        ):
            self._elevation, self._slope, self._hazard = _empty_maps(shape)
            self._confidence = np.full(shape, 0.08, dtype=np.float32)

    def estimate(
        self,
        images: dict[str, np.ndarray],
        imu: np.ndarray,
        gps: np.ndarray,
        context: PerceptionContext,
    ) -> TerrainEstimate:
        _ = context.terrain
        self._ensure_maps(context.map_shape)
        assert (
            self._elevation is not None
            and self._slope is not None
            and self._hazard is not None
            and self._confidence is not None
        )
        pose: Pose = context.pose
        self._grade.update(pose, imu, gps)
        self._prior = paint_planar_grade(
            self._grade,
            self._elevation,
            self._slope,
            resolution_m=context.resolution_m,
            confidence=self._confidence,
        )
        prev_hazard = self._hazard.copy()

        def _label(frame: np.ndarray, cam) -> tuple[np.ndarray, np.ndarray]:
            if self._model is None:
                h, w = frame.shape[:2]
                return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.float32)
            return classify_image(
                frame,
                self._model,
                cam=cam,
                pose=pose,
                world_size=context.world_size,
                stride=self.stride,
            )

        fused, conf = fuse_camera_labels(
            images,
            _label,
            context.cameras,
            pose,
            resolution_m=context.resolution_m,
            world_size=context.world_size,
            map_shape=context.map_shape,
            max_range_m=self.max_range_m,
        )
        if self._filter is not None:
            self._hazard = self._filter.update(fused, conf)
        else:
            self._hazard = np.maximum(self._hazard, fused)
        self._confidence = conf
        self._hazard = gate_isolated_lips(self._hazard)
        paint_geometry_from_hazard(
            self._hazard,
            elevation=self._elevation,
            slope=self._slope,
            pose_z=pose.z,
            steep_rad=max(self.steep_rad, context.steep_slope_rad),
        )
        self._grow_drain_gaps(prev_hazard)
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
            source="learned",
            confidence=self._confidence.copy(),
            elevation_prior=None if self._prior is None else self._prior.copy(),
        )

    def _grow_drain_gaps(self, prev_hazard: np.ndarray) -> None:
        assert self._hazard is not None
        channel = self._hazard >= HAZARD_DRAIN
        fresh = (self._hazard >= HAZARD_DRAIN_EDGE) & (prev_hazard < HAZARD_DRAIN_EDGE)
        if np.any(channel):
            near = channel.copy()
            near[1:, :] |= channel[:-1, :]
            near[:-1, :] |= channel[1:, :]
            near[:, 1:] |= channel[:, :-1]
            near[:, :-1] |= channel[:, 1:]
            fresh = fresh & near
        if not np.any(fresh):
            return
        grown = fresh.copy()
        grown[1:, :] |= fresh[:-1, :]
        grown[:-1, :] |= fresh[1:, :]
        grown[:, 1:] |= fresh[:, :-1]
        grown[:, :-1] |= fresh[:, 1:]
        promote = grown & (self._hazard < HAZARD_DRAIN_EDGE)
        self._hazard[promote] = HAZARD_DRAIN_EDGE

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


TERRAIN_MODES = frozenset({"oracle", "blind", "heuristic", "learned"})


def default_weights_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "terrain_mlp.npz"


def terrain_observer_from_mode(
    mode: str,
    *,
    weights_path: Optional[Union[str, Path]] = None,
    temporal: Optional[bool] = None,
) -> TerrainObserver:
    key = (mode or "heuristic").strip().lower()
    if key == "blind":
        return BlindTerrainObserver()
    if key == "oracle":
        return OracleTerrainObserver()
    if key == "learned":
        path = Path(weights_path) if weights_path else default_weights_path()
        if not path.is_file():
            path = None
        use_temporal = True if temporal is None else bool(temporal)
        return LearnedTerrainObserver(path, temporal=use_temporal)
    return HeuristicTerrainObserver(temporal=bool(temporal) if temporal else False)

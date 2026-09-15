"""First-class path / building / bunker / garden / pond layers (not a geofence).

Scenario YAML authors paved ribbons as polylines and buildings / beds as
polygons. The rasters feed coverage (no-mow) and the costmap (block or
heavy cost). This is not a claimed detector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from jims_mower.constants import (
    STRUCTURE_BUILDING,
    STRUCTURE_BUNKER,
    STRUCTURE_GARDEN,
    STRUCTURE_GREEN,
    STRUCTURE_NONE,
    STRUCTURE_PATH,
    STRUCTURE_POND,
    TERRAIN_BUILDING,
    TERRAIN_BUNKER,
    TERRAIN_DRAIN,
    TERRAIN_GARDEN,
    TERRAIN_GREEN,
    TERRAIN_PATH,
    TERRAIN_POND,
)
from jims_mower.safety import point_in_polygon


@dataclass(frozen=True)
class PathFeature:
    """Paved / cart-path ribbon. Vertices are world XY metres."""

    vertices: tuple[tuple[float, float], ...]
    width_m: float = 1.2
    kind: str = "path_paved"

    def as_segments(self) -> list[tuple[float, float, float, float]]:
        pts = list(self.vertices)
        return [(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]) for i in range(len(pts) - 1)]


@dataclass(frozen=True)
class PolygonFeature:
    """Closed human structure: building, garden bed, or green."""

    vertices: tuple[tuple[float, float], ...]
    kind: str = "building"
    height_m: float = 0.0


@dataclass(frozen=True)
class BunkerFeature:
    """Sand-bowl bunker. Carved into the height field."""

    x: float
    y: float
    radius_m: float
    depth_m: float = 0.22
    kind: str = "bunker"


@dataclass
class StructureLayer:
    """uint8 class raster aligned with the grass / hazard grids."""

    grid: np.ndarray
    width_m: float
    height_m: float
    resolution_m: float

    @property
    def rows(self) -> int:
        return int(self.grid.shape[0])

    @property
    def cols(self) -> int:
        return int(self.grid.shape[1])

    def no_mow_mask(self) -> np.ndarray:
        return self.grid > STRUCTURE_NONE

    def blocked_mask(self) -> np.ndarray:
        return np.isin(
            self.grid,
            (
                STRUCTURE_BUILDING,
                STRUCTURE_BUNKER,
                STRUCTURE_GARDEN,
                STRUCTURE_GREEN,
                STRUCTURE_POND,
            ),
        )

    def path_mask(self) -> np.ndarray:
        return self.grid == STRUCTURE_PATH


def empty_structure_layer(rows: int, cols: int, width_m: float, height_m: float, resolution_m: float) -> StructureLayer:
    return StructureLayer(
        grid=np.zeros((int(rows), int(cols)), dtype=np.uint8),
        width_m=float(width_m),
        height_m=float(height_m),
        resolution_m=float(resolution_m),
    )


def rasterize_polygon(
    shape: tuple[int, int],
    resolution_m: float,
    vertices: Iterable[tuple[float, float]],
) -> np.ndarray:
    poly = [(float(x), float(y)) for x, y in vertices]
    rows, cols = shape
    mask = np.zeros((rows, cols), dtype=bool)
    if len(poly) < 3:
        return mask
    res = max(float(resolution_m), 1e-6)
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    c0 = max(0, int(math.floor(min(xs) / res) - 1))
    c1 = min(cols, int(math.ceil(max(xs) / res) + 1))
    r0 = max(0, int(math.floor(min(ys) / res) - 1))
    r1 = min(rows, int(math.ceil(max(ys) / res) + 1))
    for row in range(r0, r1):
        y = (row + 0.5) * res
        for col in range(c0, c1):
            x = (col + 0.5) * res
            if point_in_polygon(x, y, poly):
                mask[row, col] = True
    return mask


def rasterize_polyline(
    shape: tuple[int, int],
    resolution_m: float,
    vertices: Iterable[tuple[float, float]],
    width_m: float,
) -> np.ndarray:
    pts = [(float(x), float(y)) for x, y in vertices]
    rows, cols = shape
    mask = np.zeros((rows, cols), dtype=bool)
    if len(pts) < 2 or width_m <= 0:
        return mask
    res = max(float(resolution_m), 1e-6)
    half = 0.5 * float(width_m)
    yy = (np.arange(rows) + 0.5) * res
    xx = (np.arange(cols) + 0.5) * res
    grid_x, grid_y = np.meshgrid(xx, yy)
    for x0, y0, x1, y1 in zip(
        [p[0] for p in pts[:-1]],
        [p[1] for p in pts[:-1]],
        [p[0] for p in pts[1:]],
        [p[1] for p in pts[1:]],
    ):
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-9:
            mask |= (grid_x - x0) ** 2 + (grid_y - y0) ** 2 <= half * half
            continue
        ux, uy = dx / length, dy / length
        s = (grid_x - x0) * ux + (grid_y - y0) * uy
        d = (grid_x - x0) * (-uy) + (grid_y - y0) * ux
        along = (s >= -half) & (s <= length + half)
        mask |= along & (np.abs(d) <= half)
    return mask


def rasterize_disk(
    shape: tuple[int, int],
    resolution_m: float,
    x: float,
    y: float,
    radius_m: float,
) -> np.ndarray:
    rows, cols = shape
    res = max(float(resolution_m), 1e-6)
    yy = (np.arange(rows) + 0.5) * res
    xx = (np.arange(cols) + 0.5) * res
    grid_x, grid_y = np.meshgrid(xx, yy)
    return (grid_x - float(x)) ** 2 + (grid_y - float(y)) ** 2 <= float(radius_m) ** 2


STRUCTURE_FROM_TERRAIN = {
    TERRAIN_PATH: STRUCTURE_PATH,
    TERRAIN_BUILDING: STRUCTURE_BUILDING,
    TERRAIN_BUNKER: STRUCTURE_BUNKER,
    TERRAIN_GARDEN: STRUCTURE_GARDEN,
    TERRAIN_GREEN: STRUCTURE_GREEN,
    TERRAIN_POND: STRUCTURE_POND,
}


def layer_from_terrain_labels(
    labels: np.ndarray,
    width_m: float,
    height_m: float,
    resolution_m: float,
) -> StructureLayer:
    grid = np.zeros(labels.shape, dtype=np.uint8)
    for terr, struct in STRUCTURE_FROM_TERRAIN.items():
        grid[labels == terr] = struct
    return StructureLayer(
        grid=grid,
        width_m=float(width_m),
        height_m=float(height_m),
        resolution_m=float(resolution_m),
    )


def apply_paths(labels: np.ndarray, elevation: np.ndarray, resolution_m: float, paths: Iterable[PathFeature]) -> None:
    for path in paths:
        mask = rasterize_polyline(labels.shape, resolution_m, path.vertices, path.width_m)
        writable = mask & (labels != TERRAIN_DRAIN)
        labels[writable] = TERRAIN_PATH
        # Flatten the ribbon slightly so it reads as a hard surface.
        if np.any(writable):
            elevation[writable] = elevation[writable] * np.float32(0.15) + np.mean(elevation[writable]) * np.float32(
                0.85
            )


def apply_polygons(
    labels: np.ndarray,
    elevation: np.ndarray,
    resolution_m: float,
    polys: Iterable[PolygonFeature],
) -> None:
    kind_to_label = {
        "building": TERRAIN_BUILDING,
        "garden_bed": TERRAIN_GARDEN,
        "garden": TERRAIN_GARDEN,
        "green": TERRAIN_GREEN,
    }
    for poly in polys:
        mask = rasterize_polygon(labels.shape, resolution_m, poly.vertices)
        writable = mask & (labels != TERRAIN_DRAIN)
        label = kind_to_label.get(poly.kind, TERRAIN_BUILDING)
        labels[writable] = label
        if label == TERRAIN_BUILDING and np.any(writable):
            elevation[writable] += np.float32(max(0.08, poly.height_m) if poly.height_m else 0.12)
        if label == TERRAIN_GREEN and np.any(writable):
            elevation[writable] += np.float32(0.02)


def apply_bunkers(
    labels: np.ndarray,
    elevation: np.ndarray,
    resolution_m: float,
    bunkers: Iterable[BunkerFeature],
) -> None:
    for bunker in bunkers:
        mask = rasterize_disk(labels.shape, resolution_m, bunker.x, bunker.y, bunker.radius_m)
        writable = mask & (labels != TERRAIN_DRAIN)
        if not np.any(writable):
            continue
        yy = (np.arange(labels.shape[0]) + 0.5) * resolution_m
        xx = (np.arange(labels.shape[1]) + 0.5) * resolution_m
        grid_x, grid_y = np.meshgrid(xx, yy)
        dist = np.hypot(grid_x - bunker.x, grid_y - bunker.y)
        profile = 0.5 * (1.0 + np.cos(np.pi * np.clip(dist / max(bunker.radius_m, 1e-6), 0.0, 1.0)))
        elevation[writable] -= np.float32(bunker.depth_m) * profile[writable].astype(np.float32)
        labels[writable] = TERRAIN_BUNKER


def apply_ponds(
    labels: np.ndarray,
    elevation: np.ndarray,
    resolution_m: float,
    ponds: Iterable[PolygonFeature],
) -> None:
    """Stamp a water keep-out bowl. Not a hydro / water-level simulator."""
    for pond in ponds:
        mask = rasterize_polygon(labels.shape, resolution_m, pond.vertices)
        writable = mask & (labels != TERRAIN_DRAIN)
        if not np.any(writable):
            continue
        depth_m = float(pond.height_m) if pond.height_m > 0.0 else 0.28
        elevation[writable] -= np.float32(depth_m)
        labels[writable] = TERRAIN_POND


def no_mow_from_labels(labels: np.ndarray) -> np.ndarray:
    """True where coverage must ignore the cell (drain + human structures)."""
    return np.isin(
        labels,
        (
            TERRAIN_DRAIN,
            TERRAIN_PATH,
            TERRAIN_BUILDING,
            TERRAIN_BUNKER,
            TERRAIN_GARDEN,
            TERRAIN_GREEN,
            TERRAIN_POND,
        ),
    )

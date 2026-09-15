"""Grass-coverage and occupancy grids."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np

from jims_mower.types import Detection, Obstacle


class GrassCoverageMap:
    """Binary cut/uncut grass raster plus a non-grass mask (trees, etc.)."""

    def __init__(self, width_m: float, height_m: float, resolution_m: float) -> None:
        if resolution_m <= 0:
            raise ValueError("resolution_m must be positive")
        self.width_m = float(width_m)
        self.height_m = float(height_m)
        self.resolution_m = float(resolution_m)
        self.cols = max(1, int(round(width_m / resolution_m)))
        self.rows = max(1, int(round(height_m / resolution_m)))
        self.cut = np.zeros((self.rows, self.cols), dtype=bool)
        self.grass = np.ones((self.rows, self.cols), dtype=bool)

    def reset(self) -> None:
        self.cut.fill(False)
        self.grass.fill(True)

    def world_to_cell(self, x: float, y: float) -> Optional[tuple[int, int]]:
        if x < 0.0 or y < 0.0 or x >= self.width_m or y >= self.height_m:
            return None
        col = int(x / self.resolution_m)
        row = int(y / self.resolution_m)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return row, col
        return None

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        return (
            (col + 0.5) * self.resolution_m,
            (row + 0.5) * self.resolution_m,
        )

    def _disk_indices(self, x: float, y: float, radius: float) -> tuple[np.ndarray, np.ndarray]:
        r = max(radius, self.resolution_m * 0.5)
        c0 = int((x - r) / self.resolution_m)
        c1 = int((x + r) / self.resolution_m)
        r0 = int((y - r) / self.resolution_m)
        r1 = int((y + r) / self.resolution_m)
        rows: list[int] = []
        cols: list[int] = []
        r2 = r * r
        for row in range(max(0, r0), min(self.rows, r1 + 1)):
            cy = (row + 0.5) * self.resolution_m
            for col in range(max(0, c0), min(self.cols, c1 + 1)):
                cx = (col + 0.5) * self.resolution_m
                if (cx - x) ** 2 + (cy - y) ** 2 <= r2:
                    rows.append(row)
                    cols.append(col)
        if not rows:
            return np.array([], dtype=int), np.array([], dtype=int)
        return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)

    def exclude_circle(self, x: float, y: float, radius: float) -> None:
        rows, cols = self._disk_indices(x, y, radius)
        if rows.size:
            self.grass[rows, cols] = False
            self.cut[rows, cols] = False

    def mark_circle(self, x: float, y: float, radius: float) -> int:
        """Cut grass under a disk. Returns the number of newly cut cells."""
        rows, cols = self._disk_indices(x, y, radius)
        if rows.size == 0:
            return 0
        newly = self.grass[rows, cols] & ~self.cut[rows, cols]
        self.cut[rows, cols] = self.cut[rows, cols] | self.grass[rows, cols]
        return int(newly.sum())

    def grass_cell_count(self) -> int:
        return int(self.grass.sum())

    def cut_cell_count(self) -> int:
        return int((self.cut & self.grass).sum())

    def coverage_fraction(self) -> float:
        total = self.grass_cell_count()
        if total == 0:
            return 1.0
        return self.cut_cell_count() / total

    def as_float(self) -> np.ndarray:
        """1 = cut grass, 0 = uncut grass, -1 = non-grass."""
        out = np.zeros((self.rows, self.cols), dtype=np.float32)
        out[~self.grass] = -1.0
        out[self.grass & self.cut] = 1.0
        return out

    def exclude_mask(self, mask: np.ndarray) -> None:
        """Mark cells False in ``mask`` as non-grass (drain channels, etc.)."""
        if mask.shape != self.grass.shape:
            raise ValueError("exclude mask shape must match the grass grid")
        drop = ~np.asarray(mask, dtype=bool)
        self.grass[drop] = False
        self.cut[drop] = False

    def sample_world(self, x: float, y: float) -> float:
        """Return 1 (cut), 0 (uncut), or -1 (non-grass / OOB) at a world point."""
        cell = self.world_to_cell(x, y)
        if cell is None:
            return -1.0
        row, col = cell
        if not self.grass[row, col]:
            return -1.0
        return 1.0 if self.cut[row, col] else 0.0

    def regenerate(self, rng: np.random.Generator, frac: float) -> int:
        """Turn a fraction of cut grass back to uncut. Returns cells grown."""
        if frac <= 0.0:
            return 0
        cut_idx = np.argwhere(self.cut & self.grass)
        if cut_idx.size == 0:
            return 0
        n = int(np.ceil(float(len(cut_idx)) * float(frac)))
        n = min(n, len(cut_idx))
        pick = rng.choice(len(cut_idx), size=n, replace=False)
        grown = cut_idx[pick]
        self.cut[grown[:, 0], grown[:, 1]] = False
        return int(n)

    def save_state(self, path: Union[str, Path]) -> Path:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dest,
            cut=self.cut.astype(np.bool_),
            grass=self.grass.astype(np.bool_),
            width_m=np.float32(self.width_m),
            height_m=np.float32(self.height_m),
            resolution_m=np.float32(self.resolution_m),
        )
        return dest

    def load_state(self, path: Union[str, Path]) -> None:
        data = np.load(Path(path))
        cut = np.asarray(data["cut"], dtype=bool)
        grass = np.asarray(data["grass"], dtype=bool)
        if cut.shape != self.cut.shape or grass.shape != self.grass.shape:
            raise ValueError(
                f"grass state shape {cut.shape} does not match map {self.cut.shape}"
            )
        self.cut = cut
        self.grass = grass


class OccupancyMap:
    """0 = free, 1 = occupied. Filled from detections or known footprints."""

    def __init__(self, rows: int, cols: int) -> None:
        self.grid = np.zeros((rows, cols), dtype=np.float32)

    def clear(self) -> None:
        self.grid.fill(0.0)

    def paint_circle(
        self,
        x: float,
        y: float,
        radius: float,
        resolution_m: float,
        value: float = 1.0,
    ) -> None:
        rows, cols = self.grid.shape
        r = max(radius, resolution_m * 0.5)
        c0 = int((x - r) / resolution_m)
        c1 = int((x + r) / resolution_m)
        r0 = int((y - r) / resolution_m)
        r1 = int((y + r) / resolution_m)
        r2 = r * r
        for row in range(max(0, r0), min(rows, r1 + 1)):
            cy = (row + 0.5) * resolution_m
            for col in range(max(0, c0), min(cols, c1 + 1)):
                cx = (col + 0.5) * resolution_m
                if (cx - x) ** 2 + (cy - y) ** 2 <= r2:
                    self.grid[row, col] = max(self.grid[row, col], value)


def occupancy_from_detections(
    shape: tuple[int, int],
    detections: Iterable[Detection],
    resolution_m: float,
    inflate_m: float = 0.15,
    radii: Optional[dict[str, float]] = None,
) -> np.ndarray:
    """CV hook: rasterize detections that carry a world XY."""
    from jims_mower.constants import DEFAULT_RADII

    radii = radii or DEFAULT_RADII
    occ = OccupancyMap(*shape)
    for det in detections:
        if det.world_xy is None:
            continue
        radius = radii.get(det.label, 0.2) + inflate_m
        occ.paint_circle(det.world_xy[0], det.world_xy[1], radius, resolution_m)
    return occ.grid


def occupancy_from_obstacles(
    shape: tuple[int, int],
    obstacles: Iterable[Obstacle],
    resolution_m: float,
    inflate_m: float = 0.0,
) -> np.ndarray:
    occ = OccupancyMap(*shape)
    for obst in obstacles:
        occ.paint_circle(obst.x, obst.y, obst.radius + inflate_m, resolution_m)
    return occ.grid

"""Yard height field: steep banks and small earth drains / swales."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from jims_mower.constants import (
    HAZARD_DRAIN,
    HAZARD_DRAIN_EDGE,
    HAZARD_NONE,
    HAZARD_STEEP,
    TERRAIN_BANK,
    TERRAIN_DRAIN,
    TERRAIN_DRAIN_EDGE,
    TERRAIN_FLAT,
)


@dataclass(frozen=True)
class DrainFeature:
    """Shallow open drain / swale / drainage ditch a wheel must not drop into."""

    x0: float
    y0: float
    x1: float
    y1: float
    width_m: float
    depth_m: float
    side_slope: float
    kind: str = "drain"

    @property
    def length_m(self) -> float:
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)

    @property
    def heading(self) -> float:
        return math.atan2(self.y1 - self.y0, self.x1 - self.x0)


@dataclass(frozen=True)
class BankFeature:
    """Raised berm / steep bank."""

    x0: float
    y0: float
    x1: float
    y1: float
    width_m: float
    height_m: float
    kind: str = "bank"

    @property
    def length_m(self) -> float:
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)


@dataclass
class HeightField:
    """Elevation, slope, and terrain-label rasters aligned with the grass map."""

    width_m: float
    height_m: float
    resolution_m: float
    elevation: np.ndarray
    labels: np.ndarray
    drains: list[DrainFeature] = field(default_factory=list)
    banks: list[BankFeature] = field(default_factory=list)
    slope: Optional[np.ndarray] = None
    dzdx: Optional[np.ndarray] = None
    dzdy: Optional[np.ndarray] = None

    @property
    def rows(self) -> int:
        return int(self.elevation.shape[0])

    @property
    def cols(self) -> int:
        return int(self.elevation.shape[1])

    @classmethod
    def empty(cls, width_m: float, height_m: float, resolution_m: float) -> "HeightField":
        if resolution_m <= 0:
            raise ValueError("resolution_m must be positive")
        cols = max(1, int(round(width_m / resolution_m)))
        rows = max(1, int(round(height_m / resolution_m)))
        hf = cls(
            width_m=float(width_m),
            height_m=float(height_m),
            resolution_m=float(resolution_m),
            elevation=np.zeros((rows, cols), dtype=np.float32),
            labels=np.zeros((rows, cols), dtype=np.uint8),
        )
        hf.recompute_slope()
        return hf

    @classmethod
    def from_function(
        cls,
        width_m: float,
        height_m: float,
        resolution_m: float,
        fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        label: int = TERRAIN_FLAT,
    ) -> "HeightField":
        hf = cls.empty(width_m, height_m, resolution_m)
        yy = (np.arange(hf.rows) + 0.5) * resolution_m
        xx = (np.arange(hf.cols) + 0.5) * resolution_m
        grid_x, grid_y = np.meshgrid(xx, yy)
        hf.elevation = np.asarray(fn(grid_x, grid_y), dtype=np.float32)
        hf.labels.fill(label)
        hf.recompute_slope()
        return hf

    def recompute_slope(self) -> None:
        dzdy, dzdx = np.gradient(self.elevation, self.resolution_m)
        self.dzdx = dzdx.astype(np.float32)
        self.dzdy = dzdy.astype(np.float32)
        self.slope = np.arctan(np.hypot(dzdx, dzdy)).astype(np.float32)

    def world_to_cell(self, x: float, y: float) -> Optional[tuple[int, int]]:
        if x < 0.0 or y < 0.0 or x >= self.width_m or y >= self.height_m:
            return None
        col = int(x / self.resolution_m)
        row = int(y / self.resolution_m)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return row, col
        return None

    def sample(self, x: float, y: float) -> float:
        return float(self.sample_many(np.array([x]), np.array([y]))[0])

    def sample_label(self, x: float, y: float) -> int:
        cell = self.world_to_cell(x, y)
        if cell is None:
            return TERRAIN_FLAT
        return int(self.labels[cell])

    def sample_slope(self, x: float, y: float) -> float:
        if self.slope is None:
            self.recompute_slope()
        assert self.slope is not None
        cell = self.world_to_cell(x, y)
        if cell is None:
            return 0.0
        return float(self.slope[cell])

    def sample_many(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Bilinear elevation at arrays of world XY. Outside the yard is 0."""
        res = self.resolution_m
        col_f = np.asarray(xs, dtype=np.float32) / res - 0.5
        row_f = np.asarray(ys, dtype=np.float32) / res - 0.5
        c0 = np.floor(col_f).astype(int)
        r0 = np.floor(row_f).astype(int)
        c1 = c0 + 1
        r1 = r0 + 1
        tx = col_f - c0
        ty = row_f - r0
        z = np.zeros(col_f.shape, dtype=np.float32)
        inside = (xs >= 0.0) & (ys >= 0.0) & (xs < self.width_m) & (ys < self.height_m)
        if not np.any(inside):
            return z
        rows, cols = self.rows, self.cols

        def at(rr: np.ndarray, cc: np.ndarray) -> np.ndarray:
            rr_c = np.clip(rr, 0, rows - 1)
            cc_c = np.clip(cc, 0, cols - 1)
            return self.elevation[rr_c, cc_c]

        z00 = at(r0, c0)
        z10 = at(r0, c1)
        z01 = at(r1, c0)
        z11 = at(r1, c1)
        z = (1.0 - ty) * ((1.0 - tx) * z00 + tx * z10) + ty * ((1.0 - tx) * z01 + tx * z11)
        return np.where(inside, z.astype(np.float32), np.float32(0.0))

    def normal_at(self, x: float, y: float) -> tuple[float, float, float]:
        if self.dzdx is None or self.dzdy is None:
            self.recompute_slope()
        assert self.dzdx is not None and self.dzdy is not None
        cell = self.world_to_cell(x, y)
        if cell is None:
            return (0.0, 0.0, 1.0)
        gx = float(self.dzdx[cell])
        gy = float(self.dzdy[cell])
        n = np.array([-gx, -gy, 1.0], dtype=np.float64)
        n /= max(np.linalg.norm(n), 1e-9)
        return float(n[0]), float(n[1]), float(n[2])

    def normals_many(self, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.dzdx is None or self.dzdy is None:
            self.recompute_slope()
        assert self.dzdx is not None and self.dzdy is not None
        res = self.resolution_m
        cols_i = np.clip((np.asarray(xs) / res).astype(int), 0, self.cols - 1)
        rows_i = np.clip((np.asarray(ys) / res).astype(int), 0, self.rows - 1)
        gx = self.dzdx[rows_i, cols_i]
        gy = self.dzdy[rows_i, cols_i]
        nx, ny, nz = -gx, -gy, np.ones_like(gx)
        norm = np.sqrt(nx * nx + ny * ny + nz * nz)
        norm = np.maximum(norm, 1e-9)
        return nx / norm, ny / norm, nz / norm

    def hazard_map(self, steep_slope_rad: float) -> np.ndarray:
        """0 free, 1 steep bank, 2 drain lip, 3 drain channel."""
        if self.slope is None:
            self.recompute_slope()
        assert self.slope is not None
        out = np.zeros(self.elevation.shape, dtype=np.float32)
        out[self.slope >= steep_slope_rad] = HAZARD_STEEP
        out[self.labels == TERRAIN_DRAIN_EDGE] = HAZARD_DRAIN_EDGE
        out[self.labels == TERRAIN_DRAIN] = HAZARD_DRAIN
        return out

    def feature_keepouts(self, extra_radius_m: float = 0.25) -> list[tuple[float, float, float]]:
        """Circles along drain/bank centerlines so spawners stay off the features."""
        keepout: list[tuple[float, float, float]] = []
        for feat in (*self.drains, *self.banks):
            length = feat.length_m
            if length < 1e-6:
                continue
            radius = 0.5 * feat.width_m + extra_radius_m
            steps = max(2, int(math.ceil(length / max(radius, 0.3))))
            for i in range(steps + 1):
                t = i / steps
                x = feat.x0 + t * (feat.x1 - feat.x0)
                y = feat.y0 + t * (feat.y1 - feat.y0)
                keepout.append((x, y, radius))
        return keepout


def _segment_coords(
    xs: np.ndarray,
    ys: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Along-track ``s`` in metres and perpendicular distance ``d`` to a segment."""
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return np.zeros_like(xs), np.hypot(xs - x0, ys - y0)
    ux, uy = dx / length, dy / length
    s = (xs - x0) * ux + (ys - y0) * uy
    d = (xs - x0) * (-uy) + (ys - y0) * ux
    return s, d


def _carve_drain(hf: HeightField, drain: DrainFeature) -> None:
    yy = (np.arange(hf.rows) + 0.5) * hf.resolution_m
    xx = (np.arange(hf.cols) + 0.5) * hf.resolution_m
    grid_x, grid_y = np.meshgrid(xx, yy)
    s, d = _segment_coords(grid_x, grid_y, drain.x0, drain.y0, drain.x1, drain.y1)
    half_top = 0.5 * drain.width_m
    side_run = drain.depth_m / max(drain.side_slope, 0.15)
    half_bottom = max(hf.resolution_m * 0.25, half_top - side_run)
    along = (s >= -half_top) & (s <= drain.length_m + half_top)
    ad = np.abs(d)
    channel = along & (ad <= half_bottom)
    edge = along & (ad > half_bottom) & (ad <= half_top)
    hf.elevation[channel] -= np.float32(drain.depth_m)
    if np.any(edge):
        t = (ad[edge] - half_bottom) / max(half_top - half_bottom, 1e-6)
        hf.elevation[edge] -= np.float32(drain.depth_m) * (1.0 - t).astype(np.float32)
    # Rounded ends so the ditch does not have a vertical wall at the termini.
    ends = ((s < 0.0) & (s >= -half_top) | (s > drain.length_m) & (s <= drain.length_m + half_top)) & (
        ad <= half_top
    )
    if np.any(ends):
        # Soften labels only; elevation already ramped by along-track half_top.
        pass
    hf.labels[edge] = np.maximum(hf.labels[edge], TERRAIN_DRAIN_EDGE)
    hf.labels[channel] = TERRAIN_DRAIN


def _carve_bank(hf: HeightField, bank: BankFeature) -> None:
    yy = (np.arange(hf.rows) + 0.5) * hf.resolution_m
    xx = (np.arange(hf.cols) + 0.5) * hf.resolution_m
    grid_x, grid_y = np.meshgrid(xx, yy)
    s, d = _segment_coords(grid_x, grid_y, bank.x0, bank.y0, bank.x1, bank.y1)
    half_w = 0.5 * bank.width_m
    along = (s >= 0.0) & (s <= bank.length_m)
    ad = np.abs(d)
    mask = along & (ad <= half_w)
    if not np.any(mask):
        return
    # Cosine berm: peak on the centerline, zero at the sides.
    profile = 0.5 * (1.0 + np.cos(np.pi * ad / max(half_w, 1e-6)))
    bump = (bank.height_m * profile).astype(np.float32)
    # Do not fill in a drain with a bank.
    writable = mask & (hf.labels != TERRAIN_DRAIN) & (hf.labels != TERRAIN_DRAIN_EDGE)
    hf.elevation[writable] += bump[writable]
    bank_only = writable & (hf.labels == TERRAIN_FLAT)
    hf.labels[bank_only] = TERRAIN_BANK


def _smooth_noise(
    rng: np.random.Generator, rows: int, cols: int, amp: float
) -> np.ndarray:
    if amp <= 0:
        return np.zeros((rows, cols), dtype=np.float32)
    noise = rng.normal(0.0, amp, size=(rows, cols)).astype(np.float32)
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)
    kernel /= kernel.sum()
    padded = np.pad(noise, 1, mode="edge")
    out = np.zeros_like(noise)
    for i in range(3):
        for j in range(3):
            out += kernel[i, j] * padded[i : i + rows, j : j + cols]
    return out


def _place_segment(
    rng: np.random.Generator,
    width_m: float,
    height_m: float,
    length_m: float,
    keepout: list[tuple[float, float, float]],
    margin: float,
    tries: int = 60,
) -> Optional[tuple[float, float, float, float]]:
    for _ in range(tries):
        heading = float(rng.uniform(-math.pi, math.pi))
        x0 = float(rng.uniform(margin, width_m - margin))
        y0 = float(rng.uniform(margin, height_m - margin))
        x1 = x0 + length_m * math.cos(heading)
        y1 = y0 + length_m * math.sin(heading)
        if not (margin <= x1 <= width_m - margin and margin <= y1 <= height_m - margin):
            continue
        blocked = False
        steps = 6
        for i in range(steps + 1):
            t = i / steps
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            for kx, ky, kr in keepout:
                if math.hypot(x - kx, y - ky) < kr + margin * 0.25:
                    blocked = True
                    break
            if blocked:
                break
        if not blocked:
            return x0, y0, x1, y1
    return None


def generate_terrain(
    rng: np.random.Generator,
    width_m: float,
    height_m: float,
    resolution_m: float,
    *,
    n_drains: int = 2,
    n_banks: int = 2,
    drain_width_m: float = 0.40,
    drain_depth_m: float = 0.16,
    drain_length_m: float = 4.0,
    drain_side_slope: float = 1.5,
    bank_height_m: float = 0.40,
    bank_width_m: float = 1.8,
    bank_length_m: float = 3.5,
    max_slope_rad: float = 0.45,
    noise_amp_m: float = 0.015,
    keepout: Optional[list[tuple[float, float, float]]] = None,
    enabled: bool = True,
) -> HeightField:
    """Procedural yard elevation. Disabled → a flat field (still labeled)."""
    hf = HeightField.empty(width_m, height_m, resolution_m)
    if not enabled:
        return hf
    keep = list(keepout or [])
    # Cap berm height so generated bank faces stay under max_slope_rad.
    max_bank_h = math.tan(max(max_slope_rad, 1e-3)) * (0.5 * bank_width_m)
    bank_h = min(bank_height_m, max_bank_h)

    for _ in range(max(0, int(n_drains))):
        seg = _place_segment(
            rng,
            width_m,
            height_m,
            drain_length_m,
            keep,
            margin=max(0.8, drain_width_m),
        )
        if seg is None:
            continue
        drain = DrainFeature(
            *seg,
            width_m=drain_width_m,
            depth_m=drain_depth_m,
            side_slope=drain_side_slope,
        )
        _carve_drain(hf, drain)
        hf.drains.append(drain)
        keep.extend(hf.feature_keepouts()[-8:])

    for _ in range(max(0, int(n_banks))):
        seg = _place_segment(
            rng,
            width_m,
            height_m,
            bank_length_m,
            keep,
            margin=max(0.8, 0.5 * bank_width_m),
        )
        if seg is None:
            continue
        bank = BankFeature(*seg, width_m=bank_width_m, height_m=bank_h)
        _carve_bank(hf, bank)
        hf.banks.append(bank)

    if noise_amp_m > 0:
        noise = _smooth_noise(rng, hf.rows, hf.cols, noise_amp_m)
        # Keep drain channels as designed — only rumble the grass/banks.
        grass = (hf.labels == TERRAIN_FLAT) | (hf.labels == TERRAIN_BANK)
        hf.elevation[grass] += noise[grass]

    hf.recompute_slope()
    return hf

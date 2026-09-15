"""Yard height field: property-scale grade, steep banks, and drains / swales."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

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
    TERRAIN_PUDDLE,
)
from jims_mower.structures import (
    BunkerFeature,
    PathFeature,
    PolygonFeature,
    apply_bunkers,
    apply_paths,
    apply_polygons,
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


@dataclass(frozen=True)
class PuddleFeature:
    """Shallow rain puddle — temporary wet hazard, not a drain channel."""

    x: float
    y: float
    radius_m: float
    depth_m: float


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
    puddles: list[PuddleFeature] = field(default_factory=list)
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
        # Puddles are caution (steep slot) — not a terminating channel.
        out[self.labels == TERRAIN_PUDDLE] = HAZARD_STEEP
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
        for puddle in self.puddles:
            keepout.append((puddle.x, puddle.y, puddle.radius_m + extra_radius_m))
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


def _carve_puddle(hf: HeightField, puddle: PuddleFeature) -> None:
    yy = (np.arange(hf.rows) + 0.5) * hf.resolution_m
    xx = (np.arange(hf.cols) + 0.5) * hf.resolution_m
    grid_x, grid_y = np.meshgrid(xx, yy)
    dist = np.hypot(grid_x - puddle.x, grid_y - puddle.y)
    mask = dist <= puddle.radius_m
    if not np.any(mask):
        return
    # Cosine bowl so the lip is walkable; skip existing channels.
    profile = 0.5 * (1.0 + np.cos(np.pi * dist / max(puddle.radius_m, 1e-6)))
    writable = mask & (hf.labels != TERRAIN_DRAIN) & (hf.labels != TERRAIN_DRAIN_EDGE)
    hf.elevation[writable] -= np.float32(puddle.depth_m) * profile[writable].astype(np.float32)
    puddle_only = writable & (hf.labels == TERRAIN_FLAT)
    hf.labels[puddle_only] = TERRAIN_PUDDLE


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


def _apply_base_gradient(
    hf: HeightField,
    *,
    slope_rad: float,
    yaw_rad: float,
    undulation_m: float,
) -> None:
    """Tilt the whole yard, then add optional gentle undulation.

    The planar grade is centered on the yard so spawn-keepout at mid-field
    stays near grade. Drains and banks are carved *after* this so they sit
    on the tilted surface rather than on a flat plane.
    """
    if slope_rad <= 0.0 and undulation_m <= 0.0:
        return
    yy = (np.arange(hf.rows) + 0.5) * hf.resolution_m
    xx = (np.arange(hf.cols) + 0.5) * hf.resolution_m
    grid_x, grid_y = np.meshgrid(xx, yy)
    if slope_rad > 0.0:
        tan_s = math.tan(float(slope_rad))
        cx = 0.5 * hf.width_m
        cy = 0.5 * hf.height_m
        along = (grid_x - cx) * math.cos(yaw_rad) + (grid_y - cy) * math.sin(yaw_rad)
        hf.elevation += np.float32(tan_s) * along.astype(np.float32)
    if undulation_m > 0.0:
        nx = (grid_x / max(hf.width_m, 1e-6)) - 0.5
        ny = (grid_y / max(hf.height_m, 1e-6)) - 0.5
        roll = np.sin(2.0 * math.pi * nx) * np.cos(2.0 * math.pi * ny)
        dish = nx * nx + ny * ny
        hf.elevation += np.float32(undulation_m) * (0.75 * roll + 0.25 * dish).astype(
            np.float32
        )


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
    layout: str = "random",
    n_puddles: int = 0,
    puddle_radius_m: float = 0.45,
    puddle_depth_m: float = 0.04,
    explicit_drains: Optional[list[DrainFeature]] = None,
    explicit_banks: Optional[list[BankFeature]] = None,
    gradient_slope_rad: float = 0.0,
    gradient_yaw_rad: float = 0.0,
    gradient_undulation_m: float = 0.0,
    multi_scale_amp_m: float = 0.0,
    swale_amp_m: float = 0.0,
    dem_path: Optional[str] = None,
    paths: Optional[list[PathFeature]] = None,
    buildings: Optional[list[PolygonFeature]] = None,
    bunkers: Optional[list[BunkerFeature]] = None,
    garden_beds: Optional[list[PolygonFeature]] = None,
) -> HeightField:
    """Procedural yard elevation. Disabled → a flat field (still labeled).

    When enabled, a yard-scale planar grade (plus optional undulation) is
    applied first. ``explicit_drains`` / ``explicit_banks`` (scenario DSL)
    are carved on top of that base. Layout features and remaining random
    counts apply after that.
    """
    hf = HeightField.empty(width_m, height_m, resolution_m)
    if not enabled and not explicit_drains and not explicit_banks:
        return hf
    if enabled:
        _apply_base_gradient(
            hf,
            slope_rad=float(gradient_slope_rad),
            yaw_rad=float(gradient_yaw_rad),
            undulation_m=float(gradient_undulation_m),
        )
        _apply_multi_scale(
            hf,
            rng,
            amp_m=float(multi_scale_amp_m),
        )
        if dem_path:
            apply_dem_npy(hf, dem_path)
    keep = list(keepout or [])
    # Cap berm height so generated bank faces stay under max_slope_rad.
    max_bank_h = math.tan(max(max_slope_rad, 1e-3)) * (0.5 * bank_width_m)
    bank_h = min(bank_height_m, max_bank_h)

    for drain in explicit_drains or []:
        _carve_drain(hf, drain)
        hf.drains.append(drain)
        keep.extend(hf.feature_keepouts()[-8:])

    for bank in explicit_banks or []:
        capped = BankFeature(
            bank.x0,
            bank.y0,
            bank.x1,
            bank.y1,
            bank.width_m,
            min(bank.height_m, max_bank_h),
            kind=bank.kind,
        )
        _carve_bank(hf, capped)
        hf.banks.append(capped)

    if not enabled:
        hf.recompute_slope()
        return hf

    _apply_layout_features(
        hf,
        rng,
        layout=layout,
        width_m=width_m,
        height_m=height_m,
        drain_width_m=drain_width_m,
        drain_depth_m=drain_depth_m,
        drain_length_m=drain_length_m,
        drain_side_slope=drain_side_slope,
        bank_width_m=bank_width_m,
        bank_h=bank_h,
        bank_length_m=bank_length_m,
        keep=keep,
    )

    remaining_drains = max(0, int(n_drains) - len(hf.drains))
    remaining_banks = max(0, int(n_banks) - len(hf.banks))
    if layout in {"terrace", "kerb_gutter", "swale", "golf_rough", "golf_fairway"}:
        remaining_drains = 0
        remaining_banks = 0

    for _ in range(remaining_drains):
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

    for _ in range(remaining_banks):
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

    _place_puddles(
        hf,
        rng,
        n_puddles=n_puddles,
        radius_m=puddle_radius_m,
        depth_m=puddle_depth_m,
        keep=keep,
        width_m=width_m,
        height_m=height_m,
    )

    if noise_amp_m > 0:
        noise = _smooth_noise(rng, hf.rows, hf.cols, noise_amp_m)
        # Keep drain channels as designed — only rumble the grass/banks.
        grass = (hf.labels == TERRAIN_FLAT) | (hf.labels == TERRAIN_BANK)
        hf.elevation[grass] += noise[grass]

    _apply_authored_structures(
        hf,
        paths=paths or [],
        buildings=buildings or [],
        bunkers=bunkers or [],
        garden_beds=garden_beds or [],
        swale_amp_m=float(swale_amp_m) if enabled else 0.0,
        rng=rng,
        layout=layout,
    )

    hf.recompute_slope()
    return hf


def _apply_layout_features(
    hf: HeightField,
    rng: np.random.Generator,
    *,
    layout: str,
    width_m: float,
    height_m: float,
    drain_width_m: float,
    drain_depth_m: float,
    drain_length_m: float,
    drain_side_slope: float,
    bank_width_m: float,
    bank_h: float,
    bank_length_m: float,
    keep: list[tuple[float, float, float]],
) -> None:
    """Structured drains/banks for terrace, kerb+gutter, and swale yards."""
    margin = 0.7
    if layout == "terrace":
        n_walls = 3
        usable = max(1.0, height_m - 2.0 * margin)
        for i in range(n_walls):
            y = margin + (i + 1) * usable / (n_walls + 1)
            x0, x1 = margin + 0.4, width_m - margin - 0.4
            bank = BankFeature(
                x0,
                y,
                x1,
                y,
                width_m=min(bank_width_m, 1.4),
                height_m=bank_h * (0.65 + 0.18 * i),
                kind="retaining_wall",
            )
            _carve_bank(hf, bank)
            hf.banks.append(bank)
        keep.extend(hf.feature_keepouts())
        return
    if layout == "kerb_gutter":
        y = min(0.85, 0.12 * height_m)
        drain = DrainFeature(
            margin,
            y,
            width_m - margin,
            y,
            width_m=max(drain_width_m, 0.35),
            depth_m=drain_depth_m,
            side_slope=drain_side_slope,
            kind="gutter",
        )
        _carve_drain(hf, drain)
        hf.drains.append(drain)
        kerb = BankFeature(
            margin,
            y + 0.55,
            width_m - margin,
            y + 0.55,
            width_m=min(0.70, bank_width_m),
            height_m=min(0.18, bank_h),
            kind="kerb",
        )
        _carve_bank(hf, kerb)
        hf.banks.append(kerb)
        keep.extend(hf.feature_keepouts())
        return
    if layout in {"golf_rough", "golf_fairway"}:
        # Multi-scale rumble + optional swale already applied; no extra walls.
        _ = (rng, drain_length_m, bank_length_m)
        return
    if layout == "swale":
        x0, x1 = margin + 0.3, width_m - margin - 0.3
        y = 0.5 * height_m
        drain = DrainFeature(
            x0,
            y,
            x1,
            y,
            width_m=max(drain_width_m, 0.70),
            depth_m=max(drain_depth_m, 0.14),
            side_slope=max(drain_side_slope, 1.1),
            kind="swale",
        )
        _carve_drain(hf, drain)
        hf.drains.append(drain)
        keep.extend(hf.feature_keepouts())
        return
    _ = (rng, drain_length_m, bank_length_m)


def _place_puddles(
    hf: HeightField,
    rng: np.random.Generator,
    *,
    n_puddles: int,
    radius_m: float,
    depth_m: float,
    keep: list[tuple[float, float, float]],
    width_m: float,
    height_m: float,
) -> None:
    margin = radius_m + 0.6
    for _ in range(max(0, int(n_puddles))):
        placed = False
        for _try in range(50):
            x = float(rng.uniform(margin, width_m - margin))
            y = float(rng.uniform(margin, height_m - margin))
            blocked = False
            for kx, ky, kr in keep:
                if math.hypot(x - kx, y - ky) < kr + radius_m * 0.4:
                    blocked = True
                    break
            if blocked:
                continue
            puddle = PuddleFeature(x, y, radius_m, depth_m)
            _carve_puddle(hf, puddle)
            hf.puddles.append(puddle)
            keep.append((x, y, radius_m + 0.2))
            placed = True
            break
        if not placed:
            break


def _apply_multi_scale(
    hf: HeightField,
    rng: np.random.Generator,
    *,
    amp_m: float,
) -> None:
    """Two-octave sine + smoothed noise so golf yards undulate, not just tilt."""
    if amp_m <= 0.0:
        return
    yy = (np.arange(hf.rows) + 0.5) * hf.resolution_m
    xx = (np.arange(hf.cols) + 0.5) * hf.resolution_m
    grid_x, grid_y = np.meshgrid(xx, yy)
    nx = grid_x / max(hf.width_m, 1e-6)
    ny = grid_y / max(hf.height_m, 1e-6)
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    coarse = np.sin(2.0 * math.pi * nx + phase) * np.cos(1.6 * math.pi * ny)
    fine = np.sin(4.4 * math.pi * nx + 0.7) * np.cos(3.2 * math.pi * ny + 1.1)
    rumble = _smooth_noise(rng, hf.rows, hf.cols, 0.35 * amp_m)
    hf.elevation += np.float32(amp_m) * (0.70 * coarse + 0.30 * fine).astype(np.float32)
    hf.elevation += rumble


def apply_dem_npy(hf: HeightField, path: Union[str, Path]) -> None:
    """Add a vendored height patch. Resamples to the yard grid. No network."""
    dest = Path(path)
    if not dest.is_file():
        raise FileNotFoundError(f"DEM fixture not found: {dest}")
    patch = np.asarray(np.load(dest), dtype=np.float32)
    if patch.ndim != 2:
        raise ValueError("DEM .npy must be a 2-D height raster")
    resampled = _resample_height(patch, hf.rows, hf.cols)
    # Center so a raw SRTM crop does not lift the whole yard off spawn.
    resampled = resampled - float(np.mean(resampled))
    hf.elevation += resampled


def _resample_height(src: np.ndarray, rows: int, cols: int) -> np.ndarray:
    if src.shape == (rows, cols):
        return src.astype(np.float32, copy=True)
    yy = np.linspace(0.0, src.shape[0] - 1.0, rows)
    xx = np.linspace(0.0, src.shape[1] - 1.0, cols)
    grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")
    r0 = np.floor(grid_y).astype(int)
    c0 = np.floor(grid_x).astype(int)
    r1 = np.clip(r0 + 1, 0, src.shape[0] - 1)
    c1 = np.clip(c0 + 1, 0, src.shape[1] - 1)
    r0 = np.clip(r0, 0, src.shape[0] - 1)
    c0 = np.clip(c0, 0, src.shape[1] - 1)
    ty = grid_y - r0
    tx = grid_x - c0
    z00 = src[r0, c0]
    z10 = src[r0, c1]
    z01 = src[r1, c0]
    z11 = src[r1, c1]
    z = (1.0 - ty) * ((1.0 - tx) * z00 + tx * z10) + ty * ((1.0 - tx) * z01 + tx * z11)
    return z.astype(np.float32)


def _apply_authored_structures(
    hf: HeightField,
    *,
    paths: list[PathFeature],
    buildings: list[PolygonFeature],
    bunkers: list[BunkerFeature],
    garden_beds: list[PolygonFeature],
    swale_amp_m: float,
    rng: np.random.Generator,
    layout: str,
) -> None:
    if swale_amp_m > 0.0 and layout in {"golf_rough", "golf_fairway", "random", "swale"}:
        _carve_gentle_swales(hf, rng, amp_m=swale_amp_m, count=2 if layout == "golf_rough" else 1)
    apply_paths(hf.labels, hf.elevation, hf.resolution_m, paths)
    apply_polygons(hf.labels, hf.elevation, hf.resolution_m, buildings)
    apply_polygons(hf.labels, hf.elevation, hf.resolution_m, garden_beds)
    apply_bunkers(hf.labels, hf.elevation, hf.resolution_m, bunkers)


def _carve_gentle_swales(
    hf: HeightField,
    rng: np.random.Generator,
    *,
    amp_m: float,
    count: int,
) -> None:
    """Long, shallow bowls — not drain channels, just uneven ground."""
    if amp_m <= 0.0 or count <= 0:
        return
    yy = (np.arange(hf.rows) + 0.5) * hf.resolution_m
    xx = (np.arange(hf.cols) + 0.5) * hf.resolution_m
    grid_x, grid_y = np.meshgrid(xx, yy)
    for _ in range(int(count)):
        cx = float(rng.uniform(0.25 * hf.width_m, 0.75 * hf.width_m))
        cy = float(rng.uniform(0.25 * hf.height_m, 0.75 * hf.height_m))
        rx = float(rng.uniform(1.8, 3.4))
        ry = float(rng.uniform(1.1, 2.2))
        yaw = float(rng.uniform(-0.6, 0.6))
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = grid_x - cx, grid_y - cy
        lx = dx * c + dy * s
        ly = -dx * s + dy * c
        bowl = np.exp(-0.5 * ((lx / rx) ** 2 + (ly / ry) ** 2))
        writable = hf.labels == TERRAIN_FLAT
        hf.elevation[writable] -= np.float32(amp_m) * bowl[writable].astype(np.float32)

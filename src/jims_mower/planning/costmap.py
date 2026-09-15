"""Costmap from slope + hazard (+ occupancy). Channels forbidden; lips reroute."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from jims_mower.constants import (
    BUNKER_COST,
    HAZARD_DRAIN,
    HAZARD_DRAIN_EDGE,
    HAZARD_STEEP,
    PATH_COST,
    STRUCTURE_BUILDING,
    STRUCTURE_BUNKER,
    STRUCTURE_GARDEN,
    STRUCTURE_GREEN,
    STRUCTURE_PATH,
    STRUCTURE_POND,
)
from jims_mower.geofence import GeofenceSpec, rasterize_geofence

FREE_COST = 1.0
STEEP_COST = 5.0
BLOCKED_COST = math.inf


@dataclass
class Costmap:
    """Scalar traversal cost aligned with the grass / hazard rasters."""

    cost: np.ndarray
    blocked: np.ndarray
    width_m: float
    height_m: float
    resolution_m: float
    confidence: Optional[np.ndarray] = None

    @property
    def rows(self) -> int:
        return int(self.cost.shape[0])

    @property
    def cols(self) -> int:
        return int(self.cost.shape[1])

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

    def is_blocked_world(self, x: float, y: float) -> bool:
        cell = self.world_to_cell(x, y)
        if cell is None:
            return True
        return bool(self.blocked[cell])

    def nearest_free(self, x: float, y: float) -> Optional[tuple[int, int]]:
        """Closest unblocked cell to a world point (spiral, then brute)."""
        start = self.world_to_cell(x, y)
        if start is not None and not self.blocked[start]:
            return start
        if not np.any(~self.blocked):
            return None
        rows, cols = self.rows, self.cols
        if start is None:
            sr = int(np.clip(y / self.resolution_m, 0, rows - 1))
            sc = int(np.clip(x / self.resolution_m, 0, cols - 1))
        else:
            sr, sc = start
        max_r = max(rows, cols)
        for rad in range(1, max_r + 1):
            r0, r1 = max(0, sr - rad), min(rows, sr + rad + 1)
            c0, c1 = max(0, sc - rad), min(cols, sc + rad + 1)
            window = ~self.blocked[r0:r1, c0:c1]
            if not np.any(window):
                continue
            ys, xs = np.where(window)
            best_i = int(np.argmin((ys + r0 - sr) ** 2 + (xs + c0 - sc) ** 2))
            return int(ys[best_i] + r0), int(xs[best_i] + c0)
        return None


def _disk_dilate(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    """Boolean dilation by a disk of ``radius_cells`` (no scipy)."""
    if radius_cells <= 0:
        return mask.astype(bool, copy=True)
    out = mask.astype(bool, copy=True)
    rows, cols = np.where(mask)
    r2 = radius_cells * radius_cells
    h, w = mask.shape
    for i, j in zip(rows.tolist(), cols.tolist()):
        r0, r1 = max(0, i - radius_cells), min(h, i + radius_cells + 1)
        c0, c1 = max(0, j - radius_cells), min(w, j + radius_cells + 1)
        for rr in range(r0, r1):
            di = rr - i
            for cc in range(c0, c1):
                dj = cc - j
                if di * di + dj * dj <= r2:
                    out[rr, cc] = True
    return out


def build_costmap(
    hazard: np.ndarray,
    slope: np.ndarray,
    *,
    resolution_m: float,
    width_m: float,
    height_m: float,
    max_climb_slope_rad: float,
    drain_clearance_m: float = 0.40,
    occupancy: Optional[np.ndarray] = None,
    occupancy_inflate_m: float = 0.0,
    margin_m: float = 0.28,
    extra_blocked: Optional[np.ndarray] = None,
    confidence: Optional[np.ndarray] = None,
    uncertainty_inflate: float = 0.0,
    uncertain_hazard_boost: float = 0.0,
    uncertain_confidence_floor: float = 0.25,
    geofence: Optional[GeofenceSpec] = None,
    geofence_inflate_m: float = 0.30,
    wet: bool = False,
    wet_slope_extra: float = 3.0,
    structure: Optional[np.ndarray] = None,
    path_cost: float = PATH_COST,
    bunker_cost: float = BUNKER_COST,
    elevation: Optional[np.ndarray] = None,
    elevation_prior: Optional[np.ndarray] = None,
    prior_blend: float = 0.0,
) -> Costmap:
    """Build a traversal costmap.

    Hazard labels (observation contract):
      0 free — cost 1
      1 steep — slow corridor if ``slope < max_climb_slope_rad``, else blocked
      2 drain lip — reroute (blocked, plus clearance inflation)
      3 drain channel — forbidden (blocked, plus clearance inflation)

    Occupancy > 0.5 is blocked (trees / detections). A yard-edge margin keeps
    the body inside ``in_yard``. Cells outside a keep-in polygon (or inside a
    keep-out) are blocked after ``geofence_inflate_m``.

    When ``wet`` is set (scenario ``weather.wet``), steep corridors get
    ``wet_slope_extra`` added so A* prefers a drier path — still traversable
    below ``max_climb_slope_rad``.

    ``confidence`` (0–1, same shape) inflates finite costs when low:
    ``cost *= 1 + uncertainty_inflate * (1 - conf)``, plus
    ``uncertain_hazard_boost`` on steep/lip/channel hints. Uncertain free
    cells stay traversable so a blind map does not lock the planner.

    ``structure`` (uint8 path/building/bunker/garden/green/pond) blocks
    buildings / beds / greens / ponds, heavy-costs paved ribbons and bunkers.
    ``elevation_prior`` is a low-frequency grade; when ``prior_blend > 0``
    the climb check uses ``max(observer_slope, blend * prior_slope)`` so a
    flattened CV map does not fight physics.
    """
    hazard = np.asarray(hazard, dtype=np.float32)
    slope = np.asarray(slope, dtype=np.float32)
    slope = _blend_slope_with_prior(
        slope,
        elevation=elevation,
        elevation_prior=elevation_prior,
        prior_blend=prior_blend,
        resolution_m=resolution_m,
    )
    if hazard.shape != slope.shape:
        raise ValueError("hazard and slope must have the same shape")
    rows, cols = hazard.shape
    cost = np.full((rows, cols), FREE_COST, dtype=np.float32)
    blocked = np.zeros((rows, cols), dtype=bool)

    too_steep = slope >= float(max_climb_slope_rad)
    steep_corridor = (hazard >= HAZARD_STEEP) & (hazard < HAZARD_DRAIN_EDGE) & ~too_steep
    cost[steep_corridor] = STEEP_COST
    if wet and wet_slope_extra > 0.0:
        cost[steep_corridor] = cost[steep_corridor] + float(wet_slope_extra)
    # Unlabeled but still over the climb cap (heuristic maps, etc.).
    cost[too_steep] = BLOCKED_COST
    blocked[too_steep] = True

    channel = hazard >= HAZARD_DRAIN
    lip = (hazard >= HAZARD_DRAIN_EDGE) & (hazard < HAZARD_DRAIN)
    drain = channel | lip
    clear_cells = int(math.ceil(max(0.0, float(drain_clearance_m)) / max(resolution_m, 1e-6)))
    drain_keepout = _disk_dilate(drain, clear_cells)
    blocked[drain_keepout] = True
    cost[drain_keepout] = BLOCKED_COST

    if occupancy is not None:
        occ = np.asarray(occupancy, dtype=np.float32)
        if occ.shape != hazard.shape:
            raise ValueError("occupancy shape must match hazard")
        occ_mask = occ > 0.5
        occ_r = int(math.ceil(max(0.0, float(occupancy_inflate_m)) / max(resolution_m, 1e-6)))
        occ_keep = _disk_dilate(occ_mask, occ_r)
        blocked[occ_keep] = True
        cost[occ_keep] = BLOCKED_COST

    if extra_blocked is not None:
        extra = np.asarray(extra_blocked, dtype=bool)
        if extra.shape != hazard.shape:
            raise ValueError("extra_blocked shape must match hazard")
        blocked[extra] = True
        cost[extra] = BLOCKED_COST

    if structure is not None:
        struct = np.asarray(structure)
        if struct.shape != hazard.shape:
            raise ValueError("structure shape must match hazard")
        path = struct == STRUCTURE_PATH
        bunker = struct == STRUCTURE_BUNKER
        hard = np.isin(
            struct,
            (STRUCTURE_BUILDING, STRUCTURE_GARDEN, STRUCTURE_GREEN, STRUCTURE_POND),
        )
        finite = np.isfinite(cost) & ~blocked
        cost[finite & path] = np.maximum(cost[finite & path], float(path_cost))
        cost[finite & bunker] = np.maximum(cost[finite & bunker], float(bunker_cost))
        blocked[hard] = True
        cost[hard] = BLOCKED_COST
        # Bunkers are no-mow bowls — block rather than hope A* skirts them.
        blocked[bunker] = True
        cost[bunker] = BLOCKED_COST

    if geofence is not None and geofence.has_polygons():
        geo = rasterize_geofence(hazard.shape, resolution_m=resolution_m, spec=geofence)
        geo_r = int(math.ceil(max(0.0, float(geofence_inflate_m)) / max(resolution_m, 1e-6)))
        geo_keep = _disk_dilate(geo, geo_r)
        blocked[geo_keep] = True
        cost[geo_keep] = BLOCKED_COST

    margin_cells = int(math.floor(max(0.0, float(margin_m)) / max(resolution_m, 1e-6)))
    if margin_cells > 0:
        blocked[:margin_cells, :] = True
        blocked[-margin_cells:, :] = True
        blocked[:, :margin_cells] = True
        blocked[:, -margin_cells:] = True
        cost[blocked] = BLOCKED_COST

    if confidence is None:
        conf = np.ones((rows, cols), dtype=np.float32)
    else:
        conf = np.clip(np.asarray(confidence, dtype=np.float32), 0.0, 1.0)
        if conf.shape != hazard.shape:
            raise ValueError("confidence shape must match hazard")

    # Uncertain cells cost more so A* prefers well-observed free grass.
    inflate = max(0.0, float(uncertainty_inflate))
    boost = max(0.0, float(uncertain_hazard_boost))
    if inflate > 0.0 or boost > 0.0:
        unc = 1.0 - conf
        finite = np.isfinite(cost) & ~blocked
        if inflate > 0.0:
            cost[finite] = cost[finite] * (1.0 + inflate * unc[finite])
        if boost > 0.0:
            hinted = finite & (hazard >= HAZARD_STEEP)
            cost[hinted] = cost[hinted] + boost * unc[hinted]
            # Very low confidence on a drain-like hint: extra slow corridor,
            # still not a hard block (blind maps must stay traversable).
            floor = float(np.clip(uncertain_confidence_floor, 0.0, 1.0))
            shaky = hinted & (conf < floor)
            cost[shaky] = cost[shaky] + boost * (floor - conf[shaky])

    return Costmap(
        cost=cost,
        blocked=blocked,
        width_m=float(width_m),
        height_m=float(height_m),
        resolution_m=float(resolution_m),
        confidence=conf,
    )


def _blend_slope_with_prior(
    slope: np.ndarray,
    *,
    elevation: Optional[np.ndarray],
    elevation_prior: Optional[np.ndarray],
    prior_blend: float,
    resolution_m: float,
) -> np.ndarray:
    """Floor observer slope with a low-frequency prior so grades stay visible."""
    blend = float(np.clip(prior_blend, 0.0, 1.0))
    if blend <= 0.0 or elevation_prior is None:
        return slope
    prior = np.asarray(elevation_prior, dtype=np.float32)
    if prior.shape != slope.shape:
        return slope
    src = prior
    if elevation is not None:
        elev = np.asarray(elevation, dtype=np.float32)
        if elev.shape == slope.shape:
            src = (1.0 - blend) * elev + blend * prior
    res = max(float(resolution_m), 1e-6)
    dzdy, dzdx = np.gradient(src, res)
    prior_slope = np.arctan(np.hypot(dzdx, dzdy)).astype(np.float32)
    return np.maximum(slope, blend * prior_slope)

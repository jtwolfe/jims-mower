"""GPS polygon keep-in / keep-out helpers (numpy rasters, Orin-class size)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from jims_mower.types import Pose
from jims_mower.world import point_to_segment_distance


def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray-cast test. Vertices are (x, y) in metres. Degenerate poly → False."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


@dataclass(frozen=True)
class GeofenceSpec:
    """Keep-in is the allowed work area. Keep-out polygons are holes / no-go."""

    keep_in: list[tuple[float, float]] = field(default_factory=list)
    keep_out: list[list[tuple[float, float]]] = field(default_factory=list)
    inflate_m: float = 0.30

    def has_polygons(self) -> bool:
        return len(self.keep_in) >= 3 or any(len(p) >= 3 for p in self.keep_out)

    def as_info(self) -> dict:
        return {
            "keep_in": [list(p) for p in self.keep_in],
            "keep_out": [[list(p) for p in poly] for poly in self.keep_out],
            "inflate_m": self.inflate_m,
        }


def polygon_edge_distance(x: float, y: float, polygon: list[tuple[float, float]]) -> float:
    """Minimum distance from a point to any edge of a closed polygon."""
    n = len(polygon)
    if n < 2:
        return math.inf
    best = math.inf
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        best = min(best, point_to_segment_distance(x, y, x0, y0, x1, y1))
    return float(best)


def inside_keep_in(x: float, y: float, keep_in: list[tuple[float, float]]) -> bool:
    if len(keep_in) < 3:
        return True
    return point_in_polygon(x, y, keep_in)


def inside_any_keepout(x: float, y: float, keep_out: Iterable[list[tuple[float, float]]]) -> bool:
    for poly in keep_out:
        if len(poly) >= 3 and point_in_polygon(x, y, poly):
            return True
    return False


def allowed_xy(
    x: float,
    y: float,
    spec: Optional[GeofenceSpec],
) -> bool:
    if spec is None or not spec.has_polygons():
        return True
    if not inside_keep_in(x, y, spec.keep_in):
        return False
    if inside_any_keepout(x, y, spec.keep_out):
        return False
    return True


def geofence_clearance(
    x: float,
    y: float,
    spec: Optional[GeofenceSpec],
) -> float:
    """Signed-ish clearance: distance to the nearest fence. Inf if unconstrained."""
    if spec is None or not spec.has_polygons():
        return math.inf
    best = math.inf
    if len(spec.keep_in) >= 3:
        best = min(best, polygon_edge_distance(x, y, spec.keep_in))
    for poly in spec.keep_out:
        if len(poly) >= 3:
            best = min(best, polygon_edge_distance(x, y, poly))
    return float(best)


def geofence_advice(
    pose: Pose,
    spec: Optional[GeofenceSpec],
    *,
    slow_m: float = 0.80,
    stop_m: float = 0.28,
    look_ahead_m: float = 0.55,
) -> str:
    """Pre-touch slow / stop before the body leaves keep-in or enters keep-out.

    Physics still terminates on a true geofence violation (``in_yard``).
    """
    if spec is None or not spec.has_polygons():
        return "ok"
    if not allowed_xy(pose.x, pose.y, spec):
        return "stop"
    ahead = (
        pose.x + look_ahead_m * math.cos(pose.theta),
        pose.y + look_ahead_m * math.sin(pose.theta),
    )
    if not allowed_xy(ahead[0], ahead[1], spec):
        return "stop" if geofence_clearance(pose.x, pose.y, spec) <= stop_m else "slow"
    clearance = geofence_clearance(pose.x, pose.y, spec)
    if clearance <= stop_m:
        return "stop"
    if clearance <= slow_m:
        return "slow"
    return "ok"


def rasterize_geofence(
    shape: tuple[int, int],
    *,
    resolution_m: float,
    spec: Optional[GeofenceSpec],
    inflate_m: Optional[float] = None,
) -> np.ndarray:
    """Boolean mask of cells the planner must treat as blocked (outside keep-in)."""
    rows, cols = shape
    blocked = np.zeros((rows, cols), dtype=bool)
    if spec is None or not spec.has_polygons():
        return blocked
    res = max(float(resolution_m), 1e-6)
    for row in range(rows):
        y = (row + 0.5) * res
        for col in range(cols):
            x = (col + 0.5) * res
            if not allowed_xy(x, y, spec):
                blocked[row, col] = True
    return blocked

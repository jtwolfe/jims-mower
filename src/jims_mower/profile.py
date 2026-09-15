"""YardProfile: taught geofence + home pose + mesh ref (WAVE UX-A)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from jims_mower.constants import YARD_PROFILE_SCHEMA
from jims_mower.geofence import GeofenceSpec
from jims_mower.types import Pose


class ProfileError(ValueError):
    """Invalid YardProfile JSON."""


def _xy(item: Any, *, field: str) -> tuple[float, float]:
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return float(item[0]), float(item[1])
    if isinstance(item, dict) and "x" in item and "y" in item:
        return float(item["x"]), float(item["y"])
    raise ProfileError(f"{field} entries must be [x, y] or {{x, y}}")


def _polygon(raw: Any, *, field: str) -> list[tuple[float, float]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ProfileError(f"{field} must be a list of vertices")
    return [_xy(p, field=field) for p in raw]


def _point_line_distance(
    px: float,
    py: float,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> float:
    dx = x1 - x0
    dy = y1 - y0
    length2 = dx * dx + dy * dy
    if length2 < 1e-12:
        return math.hypot(px - x0, py - y0)
    t = ((px - x0) * dx + (py - y0) * dy) / length2
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))


def ramer_douglas_peucker(
    points: list[tuple[float, float]],
    epsilon: float,
) -> list[tuple[float, float]]:
    """Polyline simplification. ``epsilon`` is metres."""
    pts = list(points)
    if len(pts) < 3:
        return pts
    start, end = pts[0], pts[-1]
    best_i = -1
    best_d = -1.0
    for i in range(1, len(pts) - 1):
        d = _point_line_distance(pts[i][0], pts[i][1], start[0], start[1], end[0], end[1])
        if d > best_d:
            best_d = d
            best_i = i
    if best_d > epsilon and best_i > 0:
        left = ramer_douglas_peucker(pts[: best_i + 1], epsilon)
        right = ramer_douglas_peucker(pts[best_i:], epsilon)
        return left[:-1] + right
    return [start, end]


def _dedupe_trail(
    trail: Iterable[tuple[float, float]],
    min_step_m: float = 0.05,
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for raw in trail:
        x, y = float(raw[0]), float(raw[1])
        if out and math.hypot(x - out[-1][0], y - out[-1][1]) < min_step_m:
            continue
        out.append((x, y))
    return out


def trail_to_polygon(
    trail: Iterable[tuple[float, float]],
    *,
    epsilon_m: float = 0.35,
    min_vertices: int = 3,
    fallback: Optional[list[tuple[float, float]]] = None,
) -> list[tuple[float, float]]:
    """Smooth a pose trail into an open keep-in ring.

    Closes the loop before RDP when the last point is near the first, then
    drops the repeated closer so ``keep_in`` stays an open ring.
    """
    pts = _dedupe_trail(trail)
    if len(pts) >= 3:
        closed = list(pts)
        if math.hypot(closed[0][0] - closed[-1][0], closed[0][1] - closed[-1][1]) > epsilon_m:
            closed.append(closed[0])
        simple = ramer_douglas_peucker(closed, epsilon_m)
        if len(simple) >= 2 and math.hypot(
            simple[0][0] - simple[-1][0], simple[0][1] - simple[-1][1]
        ) <= max(epsilon_m, 1e-6):
            simple = simple[:-1]
        if len(simple) >= min_vertices:
            return [(float(x), float(y)) for x, y in simple]
        if len(pts) >= min_vertices:
            return [(float(x), float(y)) for x, y in pts]
    if fallback is not None and len(fallback) >= min_vertices:
        return [(float(x), float(y)) for x, y in fallback]
    return []


def perimeter_waypoints(
    width_m: float,
    height_m: float,
    *,
    margin_m: float = 0.80,
    keep_in: Optional[list[tuple[float, float]]] = None,
) -> list[tuple[float, float]]:
    """Closed perimeter the teach policy follows (last = first)."""
    if keep_in is not None and len(keep_in) >= 3:
        ring = [(float(x), float(y)) for x, y in keep_in]
        if math.hypot(ring[0][0] - ring[-1][0], ring[0][1] - ring[-1][1]) > 1e-6:
            ring.append(ring[0])
        return ring
    m = max(0.15, float(margin_m))
    w = max(float(width_m) - m, m + 0.2)
    h = max(float(height_m) - m, m + 0.2)
    return [(m, m), (w, m), (w, h), (m, h), (m, m)]


def circle_polygon(
    x: float,
    y: float,
    radius_m: float,
    *,
    n: int = 8,
) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    r = max(float(radius_m), 0.05)
    for i in range(max(3, int(n))):
        ang = 2.0 * math.pi * i / n
        pts.append((float(x + r * math.cos(ang)), float(y + r * math.sin(ang))))
    return pts


@dataclass
class YardProfile:
    """Taught yard: keep-in / keep-out, home pose, optional mesh path."""

    schema: str = YARD_PROFILE_SCHEMA
    name: str = "taught"
    width_m: float = 12.0
    height_m: float = 12.0
    resolution_m: float = 0.20
    keep_in: list[tuple[float, float]] = field(default_factory=list)
    keep_out: list[list[tuple[float, float]]] = field(default_factory=list)
    home: dict[str, float] = field(default_factory=lambda: {"x": 1.0, "y": 1.0, "theta": 0.0})
    mesh: str = "yard.glb"
    trail: list[tuple[float, float]] = field(default_factory=list)
    inflate_m: float = 0.30
    not_a_benchmark: bool = True

    def geofence_spec(self) -> GeofenceSpec:
        return GeofenceSpec(
            keep_in=list(self.keep_in),
            keep_out=[list(p) for p in self.keep_out],
            inflate_m=float(self.inflate_m),
        )

    def home_pose(self) -> Pose:
        h = self.home or {}
        return Pose(
            float(h.get("x", 1.0)),
            float(h.get("y", 1.0)),
            float(h.get("theta", 0.0)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema or YARD_PROFILE_SCHEMA,
            "name": self.name,
            "width_m": float(self.width_m),
            "height_m": float(self.height_m),
            "resolution_m": float(self.resolution_m),
            "keep_in": [list(p) for p in self.keep_in],
            "keep_out": [[list(p) for p in poly] for poly in self.keep_out],
            "home": {
                "x": float(self.home.get("x", 1.0)),
                "y": float(self.home.get("y", 1.0)),
                "theta": float(self.home.get("theta", 0.0)),
            },
            "mesh": self.mesh,
            "trail": [list(p) for p in self.trail],
            "inflate_m": float(self.inflate_m),
            "not_a_benchmark": True,
        }


def is_yard_profile(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    schema = str(data.get("schema") or "").strip()
    if schema == YARD_PROFILE_SCHEMA:
        return True
    return bool(data.get("keep_in") and data.get("home") is not None)


def parse_yard_profile(data: dict[str, Any]) -> YardProfile:
    if not isinstance(data, dict):
        raise ProfileError("YardProfile must be a mapping")
    schema = str(data.get("schema") or "").strip()
    if schema and schema != YARD_PROFILE_SCHEMA:
        raise ProfileError(f"unsupported yard schema {schema!r}; expected {YARD_PROFILE_SCHEMA}")
    keep_in = _polygon(data.get("keep_in") or data.get("geofence"), field="keep_in")
    raw_out = data.get("keep_out") or data.get("keepout") or []
    keep_out: list[list[tuple[float, float]]] = []
    if raw_out:
        if not isinstance(raw_out, list):
            raise ProfileError("keep_out must be a list of polygons")
        if raw_out and isinstance(raw_out[0], (list, tuple)) and len(raw_out[0]) == 2:
            keep_out.append(_polygon(raw_out, field="keep_out"))
        else:
            for i, item in enumerate(raw_out):
                keep_out.append(_polygon(item, field=f"keep_out[{i}]"))
    home_raw = data.get("home") or {}
    if not isinstance(home_raw, dict):
        raise ProfileError("home must be {x, y, theta}")
    trail = _polygon(data.get("trail") or [], field="trail")
    return YardProfile(
        schema=YARD_PROFILE_SCHEMA,
        name=str(data.get("name") or "taught"),
        width_m=float(data.get("width_m") or 12.0),
        height_m=float(data.get("height_m") or 12.0),
        resolution_m=float(data.get("resolution_m") or 0.20),
        keep_in=keep_in,
        keep_out=keep_out,
        home={
            "x": float(home_raw.get("x", 1.0)),
            "y": float(home_raw.get("y", 1.0)),
            "theta": float(home_raw.get("theta", 0.0)),
        },
        mesh=str(data.get("mesh") or "yard.glb"),
        trail=trail,
        inflate_m=float(data.get("inflate_m") or 0.30),
        not_a_benchmark=True,
    )


def load_yard_profile(source: Union[str, Path, dict]) -> YardProfile:
    if isinstance(source, dict):
        return parse_yard_profile(source)
    path = Path(source)
    if not path.is_file():
        raise ProfileError(f"YardProfile not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return parse_yard_profile(data)


def write_yard_profile(path: Union[str, Path], profile: YardProfile) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(profile.as_dict(), indent=2), encoding="utf-8")
    return dest


def apply_profile_to_scenario(scenario: Any, profile: YardProfile) -> Any:
    """Mutate a Scenario's geofence / keep-out / name from a taught profile."""
    scenario.geofence = list(profile.keep_in)
    scenario.keepout = [list(p) for p in profile.keep_out]
    if not scenario.name:
        scenario.name = profile.name
    if profile.width_m > 0.0:
        scenario.config.world.width_m = float(profile.width_m)
    if profile.height_m > 0.0:
        scenario.config.world.height_m = float(profile.height_m)
    if profile.resolution_m > 0.0:
        scenario.config.world.resolution_m = float(profile.resolution_m)
    return scenario


def profile_to_scenario(profile: YardProfile, *, base: str = "default") -> Any:
    from jims_mower.scenarios import parse_scenario

    payload: dict[str, Any] = {
        "kind": "scenario",
        "name": profile.name or "taught",
        "description": "Taught yard profile (WAVE UX-A)",
        "base": base,
        "geofence": {
            "keep_in": [list(p) for p in profile.keep_in],
            "keep_out": [[list(p) for p in poly] for poly in profile.keep_out],
        },
        "world": {
            "width_m": float(profile.width_m),
            "height_m": float(profile.height_m),
            "resolution_m": float(profile.resolution_m),
            "n_people": 0,
            "n_dogs": 0,
            "n_cats": 0,
            "n_birds": 0,
            "n_trees": 0,
            "n_furniture": 0,
            "n_toys": 0,
            "terrain": {"enabled": True, "n_drains": 1, "n_banks": 0},
        },
    }
    return parse_scenario(payload)

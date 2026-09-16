"""YardProfile: taught geofence + home pose + mesh ref (WAVE UX-A)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import re

from jims_mower.constants import (
    RADIO_LINKS,
    SCHEDULE_DAYS,
    SURVEY_ORIGIN_FRAME,
    SURVEY_SCHEMA,
    YARD_PROFILE_SCHEMA,
)
from jims_mower.geofence import GeofenceSpec
from jims_mower.schedule import DEFAULT_NOTE, ScheduleSpec, resolve_timezone
from jims_mower.types import Pose

ORIGIN_NOTE = (
    "not a WGS84 field survey — tape/RTK still required for tape-stop acceptance"
)

# Older name — same document, now a real weekly window.
ScheduleStub = ScheduleSpec


@dataclass(frozen=True)
class SurveyOrigin:
    """Surveyed / local-ENU anchor for keep-in metres (MAP-5).

    Gym default is the world SW corner ``(e_m, n_m, u_m) = (0, 0, 0)``.
    ``lat_deg`` / ``lon_deg`` / ``alt_m`` are optional WGS84 labels for the
    rig peg. This repo has **no** field survey — ``surveyed`` stays false
    until a human tapes/RTKs the peg and commits real numbers.
    """

    lat_deg: Optional[float] = None
    lon_deg: Optional[float] = None
    alt_m: Optional[float] = None
    e_m: float = 0.0
    n_m: float = 0.0
    u_m: float = 0.0
    frame: str = SURVEY_ORIGIN_FRAME
    surveyed: bool = False
    note: str = ORIGIN_NOTE

    def has_wgs84(self) -> bool:
        return self.lat_deg is not None and self.lon_deg is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "lat_deg": None if self.lat_deg is None else float(self.lat_deg),
            "lon_deg": None if self.lon_deg is None else float(self.lon_deg),
            "alt_m": None if self.alt_m is None else float(self.alt_m),
            "e_m": float(self.e_m),
            "n_m": float(self.n_m),
            "u_m": float(self.u_m),
            "frame": str(self.frame or SURVEY_ORIGIN_FRAME),
            "surveyed": bool(self.surveyed),
            "note": str(self.note or ORIGIN_NOTE),
        }


def parse_survey_origin(raw: Any) -> SurveyOrigin:
    if raw is None:
        return SurveyOrigin()
    if isinstance(raw, SurveyOrigin):
        return raw
    if not isinstance(raw, dict):
        raise ProfileError("origin must be a mapping")
    frame = str(raw.get("frame") or SURVEY_ORIGIN_FRAME).strip() or SURVEY_ORIGIN_FRAME
    if frame not in {SURVEY_ORIGIN_FRAME, "enu", "local"}:
        raise ProfileError(f"origin.frame must be {SURVEY_ORIGIN_FRAME!r} (or enu/local)")
    if frame in {"enu", "local"}:
        frame = SURVEY_ORIGIN_FRAME

    def _opt_float(key: str) -> Optional[float]:
        val = raw.get(key)
        if val is None or val == "":
            return None
        try:
            return float(val)
        except (TypeError, ValueError) as exc:
            raise ProfileError(f"origin.{key} must be a number or null") from exc

    def _req_float(key: str, default: float = 0.0) -> float:
        val = raw.get(key, default)
        if val is None or val == "":
            return float(default)
        try:
            return float(val)
        except (TypeError, ValueError) as exc:
            raise ProfileError(f"origin.{key} must be a number") from exc

    lat = _opt_float("lat_deg") if "lat_deg" in raw else _opt_float("lat")
    lon = _opt_float("lon_deg") if "lon_deg" in raw else _opt_float("lon")
    alt = _opt_float("alt_m") if "alt_m" in raw else _opt_float("alt")
    note = str(raw.get("note") or ORIGIN_NOTE).strip() or ORIGIN_NOTE
    surveyed = bool(raw.get("surveyed", False))
    if surveyed and (lat is None or lon is None):
        raise ProfileError("origin.surveyed requires lat_deg and lon_deg")
    if "e_m" in raw:
        e_m = _req_float("e_m")
    elif "east_m" in raw:
        e_m = _req_float("east_m")
    else:
        e_m = 0.0
    if "n_m" in raw:
        n_m = _req_float("n_m")
    elif "north_m" in raw:
        n_m = _req_float("north_m")
    else:
        n_m = 0.0
    if "u_m" in raw:
        u_m = _req_float("u_m")
    elif "up_m" in raw:
        u_m = _req_float("up_m")
    else:
        u_m = 0.0
    return SurveyOrigin(
        lat_deg=lat,
        lon_deg=lon,
        alt_m=alt,
        e_m=e_m,
        n_m=n_m,
        u_m=u_m,
        frame=frame,
        surveyed=surveyed,
        note=note,
    )


def world_xy_from_enu(e_m: float, n_m: float, origin: Optional[SurveyOrigin] = None) -> tuple[float, float]:
    """Local ENU metres → gym world XY. Identity when origin is the SW corner."""
    peg = origin or SurveyOrigin()
    return float(peg.e_m) + float(e_m), float(peg.n_m) + float(n_m)


def enu_from_world_xy(x: float, y: float, origin: Optional[SurveyOrigin] = None) -> tuple[float, float]:
    peg = origin or SurveyOrigin()
    return float(x) - float(peg.e_m), float(y) - float(peg.n_m)


def shift_polygon(
    poly: Iterable[tuple[float, float]],
    origin: Optional[SurveyOrigin],
    *,
    to_world: bool = True,
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for item in poly:
        e, n = float(item[0]), float(item[1])
        if to_world:
            out.append(world_xy_from_enu(e, n, origin))
        else:
            out.append(enu_from_world_xy(e, n, origin))
    return out

_TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class ProfileError(ValueError):
    """Invalid YardProfile JSON."""


YardProfileError = ProfileError


@dataclass(frozen=True)
class RadioPrefs:
    """BT pair required, Wi-Fi optional, LoRa long-range default (UX-C)."""

    bluetooth: bool = True
    wifi_enabled: bool = False
    wifi_ssid: str = ""
    lora_enabled: bool = True
    lora_channel: int = 1
    primary: str = "lora"

    def as_dict(self) -> dict[str, Any]:
        return {
            "bluetooth": bool(self.bluetooth),
            "wifi": {"enabled": bool(self.wifi_enabled), "ssid": str(self.wifi_ssid)},
            "lora": {"enabled": bool(self.lora_enabled), "channel": int(self.lora_channel)},
            "primary": str(self.primary),
        }


def _safe_relpath(raw: Any, *, field: str, default: str = "yard.glb") -> str:
    path = str(raw or default).strip() or default
    if Path(path).is_absolute() or path.startswith("~"):
        raise ProfileError(f"{field} must be a relative path")
    if ".." in Path(path).parts:
        raise ProfileError(f"{field} must not traverse parent directories")
    return path.replace("\\", "/")


def _parse_radio(raw: Any) -> dict[str, Any]:
    prefs = RadioPrefs() if raw is None else None
    if prefs is not None:
        return prefs.as_dict()
    if not isinstance(raw, dict):
        raise ProfileError("radio must be a mapping")
    wifi = raw.get("wifi") if isinstance(raw.get("wifi"), dict) else {}
    lora = raw.get("lora") if isinstance(raw.get("lora"), dict) else {}
    primary = str(raw.get("primary") or "lora").strip().lower()
    if primary not in RADIO_LINKS:
        raise ProfileError(f"radio.primary must be one of {tuple(RADIO_LINKS)}")
    bluetooth = raw.get("bluetooth", True)
    if isinstance(bluetooth, dict):
        bluetooth = bluetooth.get("enabled", bluetooth.get("paired", True))
    try:
        channel_i = int(lora.get("channel", raw.get("lora_channel", 1)))
    except (TypeError, ValueError) as exc:
        raise ProfileError("radio.lora.channel must be an integer") from exc
    if channel_i < 0 or channel_i > 64:
        raise ProfileError("radio.lora.channel must be in 0..64")
    return RadioPrefs(
        bluetooth=bool(bluetooth),
        wifi_enabled=bool(wifi.get("enabled", raw.get("wifi_enabled", False))),
        wifi_ssid=str(wifi.get("ssid") or raw.get("wifi_ssid") or ""),
        lora_enabled=bool(lora.get("enabled", raw.get("lora_enabled", True))),
        lora_channel=channel_i,
        primary=primary,
    ).as_dict()


def _parse_schedule(raw: Any) -> dict[str, Any]:
    if raw is None:
        return ScheduleSpec().as_dict()
    if not isinstance(raw, dict):
        raise ProfileError("schedule must be a mapping")
    days_raw = raw.get("days") or []
    if not isinstance(days_raw, list):
        raise ProfileError("schedule.days must be a list")
    days: list[str] = []
    for i, day in enumerate(days_raw):
        key = str(day).strip().lower()[:3]
        if key not in SCHEDULE_DAYS:
            raise ProfileError(f"schedule.days[{i}] must be one of {tuple(SCHEDULE_DAYS)}")
        if key not in days:
            days.append(key)
    start = str(raw.get("start_local") or "09:00").strip()
    if not _TIME_RE.match(start):
        raise ProfileError("schedule.start_local must be HH:MM (24h)")
    try:
        duration = int(raw.get("duration_min", 60))
    except (TypeError, ValueError) as exc:
        raise ProfileError("schedule.duration_min must be an integer") from exc
    if duration < 0 or duration > 24 * 60:
        raise ProfileError("schedule.duration_min must be in 0..1440")
    timezone = str(raw.get("timezone") or "local").strip() or "local"
    try:
        resolve_timezone(timezone)
    except ValueError as exc:
        raise ProfileError(str(exc)) from exc
    try:
        min_soc = float(raw.get("min_soc", 0.25))
    except (TypeError, ValueError) as exc:
        raise ProfileError("schedule.min_soc must be a number") from exc
    if min_soc < 0.0 or min_soc > 1.0:
        raise ProfileError("schedule.min_soc must be in [0, 1]")
    try:
        arm_window = int(raw.get("arm_window_min", 15))
    except (TypeError, ValueError) as exc:
        raise ProfileError("schedule.arm_window_min must be an integer") from exc
    if arm_window < 1 or arm_window > 180:
        raise ProfileError("schedule.arm_window_min must be in 1..180")
    note = str(raw.get("note") or DEFAULT_NOTE).strip() or DEFAULT_NOTE
    if note == "stub — not a scheduler":
        note = DEFAULT_NOTE
    return ScheduleSpec(
        enabled=bool(raw.get("enabled", False)),
        days=tuple(days),
        start_local=start,
        duration_min=duration,
        timezone=timezone,
        min_soc=min_soc,
        skip_rain=bool(raw.get("skip_rain", True)),
        arm_window_min=arm_window,
        note=note,
    ).as_dict()


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


def keep_in_metrics(keep_in: Iterable[tuple[float, float]]) -> dict[str, float]:
    """BBox / area / edge stats for a keep-in ring (open or closed)."""
    pts = [(float(p[0]), float(p[1])) for p in keep_in]
    if len(pts) >= 2 and math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) <= 1e-6:
        pts = pts[:-1]
    if not pts:
        return {
            "span_x": 0.0,
            "span_y": 0.0,
            "area": 0.0,
            "min_edge": 0.0,
            "n": 0.0,
            "unique": 0.0,
        }
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    area = 0.0
    min_edge = float("inf")
    unique = 1
    for i, (x, y) in enumerate(pts):
        nx, ny = pts[(i + 1) % len(pts)]
        area += x * ny - nx * y
        edge = math.hypot(nx - x, ny - y)
        if edge > 1e-6:
            min_edge = min(min_edge, edge)
        if i > 0 and math.hypot(x - pts[i - 1][0], y - pts[i - 1][1]) >= 0.25:
            unique += 1
    if min_edge == float("inf"):
        min_edge = 0.0
    return {
        "span_x": float(max(xs) - min(xs)),
        "span_y": float(max(ys) - min(ys)),
        "area": abs(area) * 0.5,
        "min_edge": float(min_edge),
        "n": float(len(pts)),
        "unique": float(unique),
    }


def keep_in_usable(
    keep_in: Iterable[tuple[float, float]],
    width_m: float,
    height_m: float,
    *,
    min_span_frac: float = 0.18,
    min_area_frac: float = 0.05,
    min_span_m: float = 1.5,
    min_area_m2: float = 2.0,
    min_edge_m: float = 0.20,
) -> bool:
    """True when the ring is a yard-scale fence, not a centre scribble.

    Thresholds are relative to the physics world so ``mission_tiny`` (6×5 m)
    still accepts a 4×3 m teach box while ``acre_yard_demo`` (70×58 m)
    rejects a 4 m trail near the lot centre.
    """
    pts = [(float(p[0]), float(p[1])) for p in keep_in]
    if len(pts) < 3:
        return False
    stats = keep_in_metrics(pts)
    world_short = min(max(float(width_m), 0.5), max(float(height_m), 0.5))
    need_span = max(float(min_span_m), float(min_span_frac) * world_short)
    need_area = max(float(min_area_m2), float(min_area_frac) * float(width_m) * float(height_m))
    if stats["span_x"] < need_span * 0.35 or stats["span_y"] < need_span * 0.35:
        return False
    if max(stats["span_x"], stats["span_y"]) < need_span:
        return False
    if stats["area"] < need_area:
        return False
    if stats["min_edge"] < float(min_edge_m) and stats["n"] > 8:
        return False
    return True


def inflate_keep_in(
    keep_in: Iterable[tuple[float, float]],
    inflate_m: float,
) -> list[tuple[float, float]]:
    """Push vertices outward from the centroid. Open ring stays open."""
    pts = [(float(p[0]), float(p[1])) for p in keep_in]
    if len(pts) < 3 or inflate_m <= 0.0:
        return pts
    cx = sum(p[0] for p in pts) / float(len(pts))
    cy = sum(p[1] for p in pts) / float(len(pts))
    out: list[tuple[float, float]] = []
    for x, y in pts:
        dx, dy = x - cx, y - cy
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            out.append((x + float(inflate_m), y))
            continue
        out.append((x + float(inflate_m) * dx / dist, y + float(inflate_m) * dy / dist))
    return out


def starter_keep_in(
    width_m: float,
    height_m: float,
    *,
    margin_m: float = 0.80,
) -> list[tuple[float, float]]:
    """Inset rectangle — the editable first-run starter fence."""
    ring = perimeter_waypoints(width_m, height_m, margin_m=margin_m)
    if len(ring) >= 2 and math.hypot(ring[0][0] - ring[-1][0], ring[0][1] - ring[-1][1]) < 1e-6:
        ring = ring[:-1]
    return ring


def repair_keep_in(
    keep_in: Iterable[tuple[float, float]],
    *,
    width_m: float,
    height_m: float,
    fallback: Optional[list[tuple[float, float]]] = None,
    inflate_m: float = 0.0,
) -> tuple[list[tuple[float, float]], str]:
    """Return a usable ring plus how it was obtained.

    ``source`` is ``taught``, ``inflated``, ``fallback``, ``starter``, or
    ``unusable``.
    """
    ring = [(float(p[0]), float(p[1])) for p in keep_in]
    if keep_in_usable(ring, width_m, height_m):
        return ring, "taught"
    if inflate_m > 0.0 and len(ring) >= 3:
        grown = inflate_keep_in(ring, inflate_m)
        if keep_in_usable(grown, width_m, height_m):
            return grown, "inflated"
    if fallback is not None:
        fb = [(float(p[0]), float(p[1])) for p in fallback]
        if keep_in_usable(fb, width_m, height_m):
            return fb, "fallback"
    starter = starter_keep_in(width_m, height_m)
    if keep_in_usable(starter, width_m, height_m):
        return starter, "starter"
    return ring, "unusable"


def trail_to_polygon(
    trail: Iterable[tuple[float, float]],
    *,
    epsilon_m: float = 0.35,
    min_vertices: int = 3,
    fallback: Optional[list[tuple[float, float]]] = None,
    width_m: Optional[float] = None,
    height_m: Optional[float] = None,
) -> list[tuple[float, float]]:
    """Smooth a pose trail into an open keep-in ring.

    Closes the loop before RDP when the last point is near the first, then
    drops the repeated closer so ``keep_in`` stays an open ring.

    A short drive that RDP-collapses (or stays a centre scribble) uses
    ``fallback`` — the raw 90-point trail is not a yard fence.
    """
    pts = _dedupe_trail(trail)
    simple: list[tuple[float, float]] = []
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
            ring = [(float(x), float(y)) for x, y in simple]
            if width_m is None or height_m is None or keep_in_usable(ring, width_m, height_m):
                return ring
    if fallback is not None and len(fallback) >= min_vertices:
        return [(float(x), float(y)) for x, y in fallback]
    if len(simple) >= min_vertices:
        return [(float(x), float(y)) for x, y in simple]
    if len(pts) >= min_vertices and (width_m is None or height_m is None):
        return [(float(x), float(y)) for x, y in pts]
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
    radio: dict[str, Any] = field(default_factory=lambda: RadioPrefs().as_dict())
    schedule: dict[str, Any] = field(default_factory=lambda: ScheduleStub().as_dict())
    description: str = ""
    not_a_benchmark: bool = True
    origin: SurveyOrigin = field(default_factory=SurveyOrigin)

    def keep_in_world(self) -> list[tuple[float, float]]:
        """Keep-in vertices in gym world metres (origin + local ENU)."""
        return shift_polygon(self.keep_in, self.origin, to_world=True)

    def keep_out_world(self) -> list[list[tuple[float, float]]]:
        return [shift_polygon(poly, self.origin, to_world=True) for poly in self.keep_out]

    def to_geofence_spec(self, inflate_m: Optional[float] = None) -> GeofenceSpec:
        spec = self.geofence_spec()
        if inflate_m is None:
            return spec
        return GeofenceSpec(
            keep_in=list(spec.keep_in),
            keep_out=[list(p) for p in spec.keep_out],
            inflate_m=float(inflate_m),
            origin=spec.origin,
        )

    def geofence_spec(self) -> GeofenceSpec:
        return GeofenceSpec(
            keep_in=self.keep_in_world(),
            keep_out=self.keep_out_world(),
            inflate_m=float(self.inflate_m),
            origin=self.origin.as_dict(),
        )

    def home_pose(self) -> Pose:
        h = self.home or {}
        x, y = world_xy_from_enu(
            float(h.get("x", 1.0)),
            float(h.get("y", 1.0)),
            self.origin,
        )
        return Pose(x, y, float(h.get("theta", 0.0)))

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
            "mesh_path": self.mesh,
            "trail": [list(p) for p in self.trail],
            "inflate_m": float(self.inflate_m),
            "radio": dict(self.radio or RadioPrefs().as_dict()),
            "schedule": dict(self.schedule or ScheduleStub().as_dict()),
            "description": self.description,
            "origin": (self.origin if isinstance(self.origin, SurveyOrigin) else parse_survey_origin(self.origin)).as_dict(),
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
    if keep_in and len(keep_in) < 3:
        raise ProfileError("keep_in needs at least 3 vertices")
    raw_out = data.get("keep_out") or data.get("keepout") or []
    keep_out: list[list[tuple[float, float]]] = []
    if raw_out:
        if not isinstance(raw_out, list):
            raise ProfileError("keep_out must be a list of polygons")
        if raw_out and isinstance(raw_out[0], (list, tuple)) and len(raw_out[0]) == 2:
            keep_out.append(_polygon(raw_out, field="keep_out"))
        else:
            for i, item in enumerate(raw_out):
                poly = _polygon(item, field=f"keep_out[{i}]")
                if poly and len(poly) < 3:
                    raise ProfileError(f"keep_out[{i}] needs at least 3 vertices")
                keep_out.append(poly)
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
        mesh=_safe_relpath(data.get("mesh") or data.get("mesh_path"), field="mesh"),
        trail=trail,
        inflate_m=float(data.get("inflate_m") or 0.30),
        radio=_parse_radio(data.get("radio")),
        schedule=_parse_schedule(data.get("schedule")),
        description=str(data.get("description") or ""),
        not_a_benchmark=True,
        origin=parse_survey_origin(data.get("origin")),
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


def apply_profile_to_scenario(
    scenario: Any,
    profile: YardProfile,
    *,
    resize_world: bool = True,
) -> Any:
    """Mutate a Scenario's geofence / keep-out / name from a taught profile.

    Live first-run keeps the acre-scale physics world and only overlays the
    taught keep-in / keep-out (``resize_world=False``). Demo ``--profile``
    still resizes the gym to the document.
    """
    scenario.geofence = list(profile.keep_in_world())
    scenario.keepout = [list(p) for p in profile.keep_out_world()]
    if not scenario.name:
        scenario.name = profile.name
    if resize_world:
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
            "keep_in": [list(p) for p in profile.keep_in_world()],
            "keep_out": [[list(p) for p in poly] for poly in profile.keep_out_world()],
        },
        "origin": profile.origin.as_dict() if isinstance(profile.origin, SurveyOrigin) else parse_survey_origin(profile.origin).as_dict(),
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

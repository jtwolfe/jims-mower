"""YardProfile JSON: home pose, keep-in/out, mesh path, radio, schedule stub.

Owner-app source of truth for a single yard. This is not a GIS / SLAM
document and not a claimed mapping quality score.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from jims_mower.constants import (
    RADIO_LINKS,
    SCHEDULE_DAYS,
    SURVEY_SCHEMA,
    YARD_PROFILE_SCHEMA,
)
from jims_mower.geofence import GeofenceSpec

_TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class YardProfileError(ValueError):
    """Invalid YardProfile JSON."""


def _finite(value: Any, *, field: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise YardProfileError(f"{field} must be a number") from exc
    if not math.isfinite(out):
        raise YardProfileError(f"{field} must be finite")
    return out


def _xy(item: Any, *, field: str) -> tuple[float, float]:
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return _finite(item[0], field=f"{field}.x"), _finite(item[1], field=f"{field}.y")
    if isinstance(item, dict) and "x" in item and "y" in item:
        return _finite(item["x"], field=f"{field}.x"), _finite(item["y"], field=f"{field}.y")
    raise YardProfileError(f"{field} entries must be [x, y] or {{x, y}}")


def _polygon(raw: Any, *, field: str, allow_empty: bool = True) -> list[tuple[float, float]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise YardProfileError(f"{field} must be a list of vertices")
    poly = [_xy(p, field=f"{field}[{i}]") for i, p in enumerate(raw)]
    if poly and len(poly) < 3:
        raise YardProfileError(f"{field} needs at least 3 vertices")
    if not poly and not allow_empty:
        raise YardProfileError(f"{field} needs at least 3 vertices")
    return poly


def _safe_relpath(raw: Any, *, field: str) -> str:
    if raw is None:
        return ""
    path = str(raw).strip()
    if not path:
        return ""
    if Path(path).is_absolute() or path.startswith("~"):
        raise YardProfileError(f"{field} must be a relative path")
    parts = Path(path).parts
    if ".." in parts:
        raise YardProfileError(f"{field} must not traverse parent directories")
    return path.replace("\\", "/")


@dataclass(frozen=True)
class HomePose:
    x: float = 1.0
    y: float = 1.0
    theta: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {"x": float(self.x), "y": float(self.y), "theta": float(self.theta)}


@dataclass(frozen=True)
class RadioPrefs:
    """BT pair required, Wi-Fi optional, LoRa long-range default."""

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


@dataclass(frozen=True)
class ScheduleStub:
    """Placeholder weekly window — not a running scheduler."""

    enabled: bool = False
    days: tuple[str, ...] = ()
    start_local: str = "09:00"
    duration_min: int = 60
    note: str = "stub — not a scheduler"

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "days": list(self.days),
            "start_local": str(self.start_local),
            "duration_min": int(self.duration_min),
            "note": str(self.note),
        }


@dataclass
class YardProfile:
    name: str = "yard"
    home: HomePose = field(default_factory=HomePose)
    keep_in: list[tuple[float, float]] = field(default_factory=list)
    keep_out: list[list[tuple[float, float]]] = field(default_factory=list)
    mesh_path: str = ""
    radio: RadioPrefs = field(default_factory=RadioPrefs)
    schedule: ScheduleStub = field(default_factory=ScheduleStub)
    width_m: float = 16.0
    height_m: float = 12.0
    resolution_m: float = 0.20
    description: str = ""
    schema: str = YARD_PROFILE_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema or YARD_PROFILE_SCHEMA,
            "name": self.name,
            "description": self.description,
            "width_m": float(self.width_m),
            "height_m": float(self.height_m),
            "resolution_m": float(self.resolution_m),
            "home": self.home.as_dict(),
            "keep_in": [list(p) for p in self.keep_in],
            "keep_out": [[list(p) for p in poly] for poly in self.keep_out],
            "mesh_path": self.mesh_path,
            "radio": self.radio.as_dict(),
            "schedule": self.schedule.as_dict(),
        }

    def to_geofence_spec(self, inflate_m: float = 0.30) -> GeofenceSpec:
        return GeofenceSpec(
            keep_in=list(self.keep_in),
            keep_out=[list(p) for p in self.keep_out],
            inflate_m=float(inflate_m),
        )


def _parse_home(raw: Any) -> HomePose:
    if raw is None:
        return HomePose()
    if not isinstance(raw, dict):
        raise YardProfileError("home must be a mapping with x, y, theta")
    return HomePose(
        x=_finite(raw.get("x", 1.0), field="home.x"),
        y=_finite(raw.get("y", 1.0), field="home.y"),
        theta=_finite(raw.get("theta", 0.0), field="home.theta"),
    )


def _parse_radio(raw: Any) -> RadioPrefs:
    if raw is None:
        return RadioPrefs()
    if not isinstance(raw, dict):
        raise YardProfileError("radio must be a mapping")
    wifi = raw.get("wifi") if isinstance(raw.get("wifi"), dict) else {}
    lora = raw.get("lora") if isinstance(raw.get("lora"), dict) else {}
    primary = str(raw.get("primary") or "lora").strip().lower()
    if primary not in RADIO_LINKS:
        raise YardProfileError(f"radio.primary must be one of {tuple(RADIO_LINKS)}")
    bluetooth = raw.get("bluetooth", True)
    if isinstance(bluetooth, dict):
        bluetooth = bluetooth.get("enabled", bluetooth.get("paired", True))
    wifi_enabled = wifi.get("enabled", raw.get("wifi_enabled", False))
    lora_enabled = lora.get("enabled", raw.get("lora_enabled", True))
    channel = lora.get("channel", raw.get("lora_channel", 1))
    try:
        channel_i = int(channel)
    except (TypeError, ValueError) as exc:
        raise YardProfileError("radio.lora.channel must be an integer") from exc
    if channel_i < 0 or channel_i > 64:
        raise YardProfileError("radio.lora.channel must be in 0..64")
    return RadioPrefs(
        bluetooth=bool(bluetooth),
        wifi_enabled=bool(wifi_enabled),
        wifi_ssid=str(wifi.get("ssid") or raw.get("wifi_ssid") or ""),
        lora_enabled=bool(lora_enabled),
        lora_channel=channel_i,
        primary=primary,
    )


def _parse_schedule(raw: Any) -> ScheduleStub:
    if raw is None:
        return ScheduleStub()
    if not isinstance(raw, dict):
        raise YardProfileError("schedule must be a mapping")
    days_raw = raw.get("days") or []
    if not isinstance(days_raw, list):
        raise YardProfileError("schedule.days must be a list")
    days: list[str] = []
    for i, day in enumerate(days_raw):
        key = str(day).strip().lower()[:3]
        if key not in SCHEDULE_DAYS:
            raise YardProfileError(f"schedule.days[{i}] must be one of {tuple(SCHEDULE_DAYS)}")
        if key not in days:
            days.append(key)
    start = str(raw.get("start_local") or "09:00").strip()
    if not _TIME_RE.match(start):
        raise YardProfileError("schedule.start_local must be HH:MM (24h)")
    try:
        duration = int(raw.get("duration_min", 60))
    except (TypeError, ValueError) as exc:
        raise YardProfileError("schedule.duration_min must be an integer") from exc
    if duration < 0 or duration > 24 * 60:
        raise YardProfileError("schedule.duration_min must be in 0..1440")
    return ScheduleStub(
        enabled=bool(raw.get("enabled", False)),
        days=tuple(days),
        start_local=start,
        duration_min=duration,
        note=str(raw.get("note") or "stub — not a scheduler"),
    )


def validate_yard_profile(data: Any) -> dict[str, Any]:
    """Return a normalised dict or raise ``YardProfileError``."""
    profile = yard_profile_from_dict(data)
    return profile.as_dict()


def yard_profile_from_dict(data: Any) -> YardProfile:
    if not isinstance(data, dict):
        raise YardProfileError("YardProfile JSON must be a mapping")
    schema = str(data.get("schema") or "").strip()
    if not schema:
        raise YardProfileError(f"YardProfile missing required 'schema' field ({YARD_PROFILE_SCHEMA})")
    if schema != YARD_PROFILE_SCHEMA:
        raise YardProfileError(f"unsupported YardProfile schema {schema!r}; expected {YARD_PROFILE_SCHEMA}")
    keep_in = _polygon(data.get("keep_in") or data.get("geofence"), field="keep_in")
    keep_out_raw = data.get("keep_out") or data.get("keepout") or []
    if not isinstance(keep_out_raw, list):
        raise YardProfileError("keep_out must be a list of polygons")
    keep_out = [
        _polygon(poly, field=f"keep_out[{i}]", allow_empty=False) for i, poly in enumerate(keep_out_raw)
    ]
    width = _finite(data.get("width_m", 16.0), field="width_m")
    height = _finite(data.get("height_m", 12.0), field="height_m")
    resolution = _finite(data.get("resolution_m", 0.20), field="resolution_m")
    if width <= 0.0 or height <= 0.0:
        raise YardProfileError("width_m and height_m must be positive")
    if resolution <= 0.0:
        raise YardProfileError("resolution_m must be positive")
    name = str(data.get("name") or "yard").strip() or "yard"
    return YardProfile(
        name=name,
        description=str(data.get("description") or ""),
        width_m=width,
        height_m=height,
        resolution_m=resolution,
        home=_parse_home(data.get("home")),
        keep_in=keep_in,
        keep_out=keep_out,
        mesh_path=_safe_relpath(data.get("mesh_path"), field="mesh_path"),
        radio=_parse_radio(data.get("radio")),
        schedule=_parse_schedule(data.get("schedule")),
        schema=YARD_PROFILE_SCHEMA,
    )


def load_yard_profile(source: Union[str, Path, dict]) -> YardProfile:
    if isinstance(source, dict):
        return yard_profile_from_dict(source)
    path = Path(source)
    if not path.is_file():
        raise YardProfileError(f"YardProfile JSON not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise YardProfileError(f"YardProfile JSON is not valid JSON: {exc}") from exc
    return yard_profile_from_dict(data)


def save_yard_profile(profile: Union[YardProfile, dict], dest: Union[str, Path]) -> Path:
    if isinstance(profile, dict):
        payload = yard_profile_from_dict(profile).as_dict()
    else:
        payload = yard_profile_from_dict(profile.as_dict()).as_dict()
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def default_yard_profile(*, name: str = "example_yard") -> YardProfile:
    return YardProfile(
        name=name,
        description="Default suburban-scale keep-in for the owner app.",
        width_m=16.0,
        height_m=12.0,
        resolution_m=0.20,
        home=HomePose(x=2.0, y=2.0, theta=0.0),
        keep_in=[(1.0, 1.0), (15.0, 1.0), (15.0, 11.0), (1.0, 11.0)],
        keep_out=[[(7.0, 5.0), (9.0, 5.0), (9.0, 7.0), (7.0, 7.0)]],
        mesh_path="maps/example_yard.mesh.json",
        radio=RadioPrefs(),
        schedule=ScheduleStub(days=("mon", "wed", "fri")),
    )


def yard_profile_from_geofence(
    spec: GeofenceSpec,
    *,
    name: str = "yard",
    width_m: float = 16.0,
    height_m: float = 12.0,
    resolution_m: float = 0.20,
    home: Optional[HomePose] = None,
) -> YardProfile:
    keep_in = list(spec.keep_in)
    if home is None and keep_in:
        home = HomePose(x=keep_in[0][0], y=keep_in[0][1], theta=0.0)
    return YardProfile(
        name=name,
        width_m=float(width_m),
        height_m=float(height_m),
        resolution_m=float(resolution_m),
        home=home or HomePose(),
        keep_in=keep_in,
        keep_out=[list(p) for p in spec.keep_out],
    )


def yard_profile_from_survey(data: dict[str, Any]) -> YardProfile:
    """Lift a WAVE 4 survey JSON into a YardProfile (no drain physics)."""
    if not isinstance(data, dict):
        raise YardProfileError("survey JSON must be a mapping")
    schema = str(data.get("schema") or "").strip()
    if schema and schema != SURVEY_SCHEMA:
        raise YardProfileError(f"unsupported survey schema {schema!r}")
    keep_in = _polygon(data.get("geofence") or data.get("keep_in"), field="keep_in")
    keep_out_raw = data.get("keep_out") or []
    keep_out = [
        _polygon(poly, field=f"keep_out[{i}]", allow_empty=False)
        for i, poly in enumerate(keep_out_raw)
    ] if isinstance(keep_out_raw, list) else []
    width = float(data.get("width_m") or 0.0)
    height = float(data.get("height_m") or 0.0)
    if keep_in and (width <= 0.0 or height <= 0.0):
        width = max(p[0] for p in keep_in) + 0.8
        height = max(p[1] for p in keep_in) + 0.8
    home_raw = data.get("home")
    return YardProfile(
        name=str(data.get("name") or "imported").strip() or "imported",
        description=str(data.get("description") or "Imported survey polygon"),
        width_m=width or 16.0,
        height_m=height or 12.0,
        resolution_m=float(data.get("resolution_m") or 0.20),
        home=_parse_home(home_raw) if home_raw is not None else (
            HomePose(x=keep_in[0][0], y=keep_in[0][1], theta=0.0) if keep_in else HomePose()
        ),
        keep_in=keep_in,
        keep_out=keep_out,
        mesh_path=_safe_relpath(data.get("mesh_path"), field="mesh_path"),
        radio=_parse_radio(data.get("radio")),
        schedule=_parse_schedule(data.get("schedule")),
    )

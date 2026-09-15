"""Owner-app helpers on top of the UX-A ``YardProfile`` (`jims_mower.yard.v1`).

Radio prefs and the schedule stub live on the same document. Do not invent
a second schema or a second mesh stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from jims_mower.constants import SURVEY_SCHEMA, YARD_PROFILE_SCHEMA
from jims_mower.geofence import GeofenceSpec
from jims_mower.profile import (
    ProfileError,
    RadioPrefs,
    ScheduleStub,
    YardProfile,
    YardProfileError,
    load_yard_profile,
    parse_yard_profile,
    write_yard_profile,
)

def save_yard_profile(profile: YardProfile, dest: Union[str, Path]) -> Path:
    """Owner-app order: profile, then path (UX-A write_yard_profile is reversed)."""
    return write_yard_profile(dest, profile)


def yard_profile_from_dict(data: Any) -> YardProfile:
    if not isinstance(data, dict) or not str(data.get("schema") or "").strip():
        raise YardProfileError(f"YardProfile missing required 'schema' field ({YARD_PROFILE_SCHEMA})")
    return parse_yard_profile(data)


def validate_yard_profile(data: Any) -> dict[str, Any]:
    return yard_profile_from_dict(data).as_dict()


def default_yard_profile(*, name: str = "example_yard") -> YardProfile:
    return YardProfile(
        name=name,
        description="Default suburban-scale keep-in for the owner app.",
        width_m=16.0,
        height_m=12.0,
        resolution_m=0.20,
        home={"x": 2.0, "y": 2.0, "theta": 0.0},
        keep_in=[(1.0, 1.0), (15.0, 1.0), (15.0, 11.0), (1.0, 11.0)],
        keep_out=[[(7.0, 5.0), (9.0, 5.0), (9.0, 7.0), (7.0, 7.0)]],
        mesh="yard.glb",
        radio=RadioPrefs().as_dict(),
        schedule=ScheduleStub(days=("mon", "wed", "fri")).as_dict(),
    )


def yard_profile_from_geofence(
    spec: GeofenceSpec,
    *,
    name: str = "yard",
    width_m: float = 16.0,
    height_m: float = 12.0,
    resolution_m: float = 0.20,
    home: Optional[dict[str, float]] = None,
) -> YardProfile:
    keep_in = list(spec.keep_in)
    if home is None and keep_in:
        home = {"x": float(keep_in[0][0]), "y": float(keep_in[0][1]), "theta": 0.0}
    return YardProfile(
        name=name,
        width_m=float(width_m),
        height_m=float(height_m),
        resolution_m=float(resolution_m),
        home=home or {"x": 1.0, "y": 1.0, "theta": 0.0},
        keep_in=keep_in,
        keep_out=[list(p) for p in spec.keep_out],
        mesh="yard.glb",
    )


def yard_profile_from_survey(data: dict[str, Any]) -> YardProfile:
    if not isinstance(data, dict):
        raise ProfileError("survey JSON must be a mapping")
    schema = str(data.get("schema") or "").strip()
    if schema and schema != SURVEY_SCHEMA:
        raise ProfileError(f"unsupported survey schema {schema!r}")
    payload = dict(data)
    payload["schema"] = YARD_PROFILE_SCHEMA
    if "keep_in" not in payload and "geofence" in payload:
        payload["keep_in"] = payload.get("geofence")
    return parse_yard_profile(payload)


def radio_prefs(profile: YardProfile) -> RadioPrefs:
    raw = profile.radio or {}
    wifi = raw.get("wifi") if isinstance(raw.get("wifi"), dict) else {}
    lora = raw.get("lora") if isinstance(raw.get("lora"), dict) else {}
    return RadioPrefs(
        bluetooth=bool(raw.get("bluetooth", True)),
        wifi_enabled=bool(wifi.get("enabled", False)),
        wifi_ssid=str(wifi.get("ssid") or ""),
        lora_enabled=bool(lora.get("enabled", True)),
        lora_channel=int(lora.get("channel", 1)),
        primary=str(raw.get("primary") or "lora"),
    )

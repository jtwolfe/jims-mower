"""Fused owner terrain decision — one state, one line.

Costmap / explore / IMU / look-ahead / blockage all feed this. The
phone must not show *Tip risk — reversing* next to *Seeking frontier*.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

from jims_mower.planning.grade_tip import KIND_GRADE, KIND_TIP

TERRAIN_OK = "ok"
TERRAIN_CONTOUR = "contour"
TERRAIN_TIP_REVERSE = "tip_reverse"
TERRAIN_RETRACE = "retrace"
TERRAIN_BLOCKED_NOGO = "blocked_nogo"
TERRAIN_IMMOBILISED = "immobilised"

TERRAIN_STATES = (
    TERRAIN_OK,
    TERRAIN_CONTOUR,
    TERRAIN_TIP_REVERSE,
    TERRAIN_RETRACE,
    TERRAIN_BLOCKED_NOGO,
    TERRAIN_IMMOBILISED,
)

OWNER_RETRACE = "Retracing last metres"
OWNER_BLOCKED_NOGO = "Blocked — remapping around obstacle"

# Explore-reason codes that match the fused state (no seeking+tip mix).
TERRAIN_REASON_CODE = {
    TERRAIN_OK: "seeking_frontier",
    TERRAIN_CONTOUR: "steep_grade",
    TERRAIN_TIP_REVERSE: "tip_recovery",
    TERRAIN_RETRACE: "retrace",
    TERRAIN_BLOCKED_NOGO: "remapping",
    TERRAIN_IMMOBILISED: "tip_recovery",
}


def fuse_terrain_state(
    *,
    chassis_tipped: bool = False,
    tipover: bool = False,
    retracing: bool = False,
    tilt_kind: str = "ok",
    advice: str = "ok",
    stamped_recent: bool = False,
    blockage_recovering: bool = False,
) -> str:
    """Single owner terrain state. Immobilise wins; then retrace; then tip."""
    if chassis_tipped or tipover:
        return TERRAIN_IMMOBILISED
    if retracing:
        return TERRAIN_RETRACE
    kind = str(tilt_kind or "ok")
    if kind == KIND_TIP and advice in {"stop", "reroute", ""}:
        return TERRAIN_TIP_REVERSE
    if stamped_recent or blockage_recovering:
        return TERRAIN_BLOCKED_NOGO
    if kind == KIND_GRADE:
        return TERRAIN_CONTOUR
    return TERRAIN_OK


def owner_copy_for_terrain(state: str) -> str:
    """Owner words that belong to the fused state (empty = use phase copy)."""
    if state == TERRAIN_IMMOBILISED:
        return "SOS — immobilised. Retrieve the mower."
    if state == TERRAIN_TIP_REVERSE:
        return "Tip risk — reversing"
    if state == TERRAIN_RETRACE:
        return OWNER_RETRACE
    if state == TERRAIN_BLOCKED_NOGO:
        return OWNER_BLOCKED_NOGO
    if state == TERRAIN_CONTOUR:
        return "Steep grade — contouring"
    return ""


def climbable_grade(
    *,
    tilt_kind: str = "ok",
    advice: str = "ok",
    look_ahead_kind: str = "",
    chassis_tipped: bool = False,
    tipover: bool = False,
) -> bool:
    """True when the face is a climbable / contour grade, not a no-go."""
    if chassis_tipped or tipover:
        return False
    if str(tilt_kind or "") == KIND_GRADE:
        return True
    if str(look_ahead_kind or "") == KIND_GRADE:
        return True
    if str(tilt_kind or "") == KIND_TIP:
        return False
    return advice in {"slow", "reroute"}


def may_stamp_blockage(
    *,
    reason: str,
    tilt_kind: str = "ok",
    advice: str = "ok",
    look_ahead_kind: str = "",
    collision: bool = False,
    drain_drop: bool = False,
    tipover: bool = False,
    chassis_tipped: bool = False,
    hard_structure: bool = False,
    drain_lip: bool = False,
) -> bool:
    """Stamp only collision, static-α tip, drain/lip, hard structure, or
    repeated zero-travel with **non-grade** advice.
    """
    if collision or reason == "collision":
        return True
    if drain_drop or drain_lip or reason in {"drain", "lip"}:
        return True
    if chassis_tipped or tipover:
        return True
    if hard_structure:
        return True
    if climbable_grade(
        tilt_kind=tilt_kind,
        advice=advice,
        look_ahead_kind=look_ahead_kind,
        chassis_tipped=chassis_tipped,
        tipover=tipover,
    ):
        return False
    # Software tip-risk (under static α) reverses / retraces — do not
    # paint the climbable face as learned no-go.
    if str(tilt_kind or "") == KIND_TIP and not (chassis_tipped or tipover):
        if reason in {"tip_collision", "no_progress", "mow_no_progress", "path_blocked"}:
            return False
    if reason in {"no_progress", "mow_no_progress", "path_blocked", "tip_collision"}:
        return True
    return False


def stamp_on_cooldown(
    *,
    cool_left: int,
    last_xy: Optional[tuple[float, float]],
    stamp_xy: tuple[float, float],
    step: int,
    last_step: int,
    cooldown_steps: int,
    min_sep_m: float,
) -> bool:
    """True when a stamp should be dropped (rate-limit / spatial cool)."""
    if int(cool_left) > 0:
        return True
    if last_xy is None:
        return False
    dist = math.hypot(stamp_xy[0] - last_xy[0], stamp_xy[1] - last_xy[1])
    since = int(step) - int(last_step)
    return dist < float(min_sep_m) and since < max(1, int(cooldown_steps))


def retrace_waypoints(
    trail: Sequence[tuple[float, ...]],
    pose_xy: tuple[float, float],
    *,
    length_m: float = 2.8,
    skip_near_m: float = 0.08,
) -> list[tuple[float, float]]:
    """Walk breadcrumbs backwards for ``length_m``. Oldest-behind first.

    ``trail`` is chronological (first = oldest). Return waypoints from
    the pose back along the driven path — downhill if the robot climbed.
    """
    if length_m <= 0.0 or len(trail) < 1:
        return []
    wps: list[tuple[float, float]] = []
    acc = 0.0
    prev = (float(pose_xy[0]), float(pose_xy[1]))
    for item in reversed(list(trail)):
        x, y = float(item[0]), float(item[1])
        dist = math.hypot(x - prev[0], y - prev[1])
        if dist < float(skip_near_m):
            continue
        acc += dist
        wps.append((x, y))
        prev = (x, y)
        if acc >= float(length_m):
            break
    return wps


def look_ahead_reason_blob(
    *,
    kind: str = "",
    advice: str = "",
    pitch: float = 0.0,
    roll: float = 0.0,
    note: str = "",
) -> dict[str, Any]:
    if not kind and not note:
        return {}
    blob: dict[str, Any] = {
        "kind": str(kind or ""),
        "advice": str(advice or ""),
        "pitch": float(pitch),
        "roll": float(roll),
    }
    if note:
        blob["note"] = note
    return blob


def near_level_attitude(pitch: float, roll: float, *, eps: float = 0.06) -> bool:
    return abs(float(pitch)) + abs(float(roll)) < float(eps)

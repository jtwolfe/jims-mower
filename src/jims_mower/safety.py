"""Trimmer interlock, body-collision, and terrain-hazard queries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Optional

from jims_mower.constants import (
    BIRD_COLLISION_Z_M,
    LIVING_KINDS,
    TERRAIN_DRAIN,
    TERRAIN_DRAIN_EDGE,
)
from jims_mower.kinematics import sit_on_terrain, trimmer_xy, wheel_positions
from jims_mower.types import Obstacle, Pose

if TYPE_CHECKING:
    from jims_mower.terrain import HeightField


@dataclass(frozen=True)
class SafetyDecision:
    trimmer_enabled: bool
    requested: bool
    nearest_living_m: float
    nearest_kind: Optional[str]
    blocked_reason: Optional[str]


def is_living(kind: str) -> bool:
    return kind in LIVING_KINDS


def hypot2(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def nearest_living(
    xy: tuple[float, float],
    obstacles: Iterable[Obstacle],
) -> tuple[float, Optional[Obstacle]]:
    """Center-to-center distance to the closest person or animal."""
    best = math.inf
    chosen: Optional[Obstacle] = None
    for obst in obstacles:
        if not is_living(obst.kind):
            continue
        d = hypot2(xy[0], xy[1], obst.x, obst.y)
        if d < best:
            best = d
            chosen = obst
    return best, chosen


def trimmer_interlock(
    requested: bool,
    pose: Pose,
    obstacles: Iterable[Obstacle],
    *,
    offset_m: float,
    safety_radius_m: float,
) -> SafetyDecision:
    """Enable the string trimmer only when requested and no living thing is near.

    Distance is measured from the front-mounted trimmer hub, not the body
    center — the tool is the hazard.
    """
    if safety_radius_m <= 0:
        raise ValueError("safety_radius_m must be positive")
    hub = trimmer_xy(pose, offset_m)
    dist, obst = nearest_living(hub, obstacles)
    kind = obst.kind if obst is not None else None
    if not requested:
        return SafetyDecision(False, False, dist, kind, None)
    if dist < safety_radius_m:
        return SafetyDecision(
            False,
            True,
            dist,
            kind,
            f"{kind} within {safety_radius_m:.2f} m of trimmer",
        )
    return SafetyDecision(True, True, dist, kind, None)


def is_body_collision(
    pose: Pose,
    obstacle: Obstacle,
    collision_radius_m: float,
) -> bool:
    """True when the robot body circle overlaps an obstacle that can strike it."""
    if obstacle.kind == "bird" and obstacle.z >= BIRD_COLLISION_Z_M:
        return False
    gap = hypot2(pose.x, pose.y, obstacle.x, obstacle.y)
    return gap < (collision_radius_m + obstacle.radius)


def first_collision(
    pose: Pose,
    obstacles: Iterable[Obstacle],
    collision_radius_m: float,
) -> Optional[Obstacle]:
    for obst in obstacles:
        if is_body_collision(pose, obst, collision_radius_m):
            return obst
    return None


def in_yard(
    pose: Pose,
    width_m: float,
    height_m: float,
    collision_radius_m: float,
) -> bool:
    return (
        collision_radius_m <= pose.x <= width_m - collision_radius_m
        and collision_radius_m <= pose.y <= height_m - collision_radius_m
    )


@dataclass(frozen=True)
class TerrainSafety:
    """Tip-over, drain-drop, and steep-slope assessment from the height field."""

    tipover: bool
    drain_drop: bool
    steep: bool
    roll: float
    pitch: float
    max_slope: float
    wheels_in_drain: tuple[bool, bool, bool, bool]
    advice: str
    reason: Optional[str]


def _look_ahead_xy(pose: Pose, dist_m: float) -> tuple[float, float]:
    return (
        pose.x + dist_m * math.cos(pose.theta),
        pose.y + dist_m * math.sin(pose.theta),
    )


def terrain_hazards(
    pose: Pose,
    height_field: Optional["HeightField"],
    *,
    length_m: float,
    track_m: float,
    tip_roll_rad: float,
    tip_pitch_rad: float,
    wheel_drop_m: float,
    steep_slope_rad: float,
    look_ahead_m: float = 0.55,
) -> TerrainSafety:
    """Physics-side terrain safety (true height field, not the observer maps).

    Recommended policy behaviour (also returned as ``advice``):
    - ``ok`` — continue
    - ``slow`` — high slope under the chassis; cut speed
    - ``reroute`` — drain lip ahead or under a wheel; do not straddle
    - ``stop`` — tip-over risk or a wheel already in the channel
    """
    if height_field is None:
        return TerrainSafety(
            False, False, False, pose.roll, pose.pitch, 0.0,
            (False, False, False, False), "ok", None,
        )
    seated = sit_on_terrain(pose, height_field, length_m, track_m)
    wheels = wheel_positions(seated, length_m, track_m)
    zs = [height_field.sample(wx, wy) for wx, wy in wheels]
    labels = [height_field.sample_label(wx, wy) for wx, wy in wheels]
    chassis_z = seated.z
    in_channel = []
    for z, lab in zip(zs, labels):
        dropped = (chassis_z - z) >= wheel_drop_m
        in_channel.append(bool(lab == TERRAIN_DRAIN or dropped and lab in {TERRAIN_DRAIN, TERRAIN_DRAIN_EDGE}))
    wheels_in = (in_channel[0], in_channel[1], in_channel[2], in_channel[3])
    drain_drop = any(in_channel)
    tipover = abs(seated.roll) >= tip_roll_rad or abs(seated.pitch) >= tip_pitch_rad
    slope_here = height_field.sample_slope(seated.x, seated.y)
    steep = (not tipover) and (
        slope_here >= steep_slope_rad
        or abs(seated.roll) >= 0.6 * tip_roll_rad
        or abs(seated.pitch) >= 0.6 * tip_pitch_rad
    )
    ahead = _look_ahead_xy(seated, look_ahead_m)
    ahead_label = height_field.sample_label(*ahead)
    lip_ahead = ahead_label in {TERRAIN_DRAIN, TERRAIN_DRAIN_EDGE}
    lip_under = any(lab == TERRAIN_DRAIN_EDGE for lab in labels)

    if tipover:
        advice, reason = "stop", "tip-over risk from pitch/roll"
    elif drain_drop:
        advice, reason = "stop", "wheel in earth drain / channel"
    elif lip_ahead or lip_under:
        advice, reason = "reroute", "drain edge — do not drop a wheel in"
    elif steep:
        advice, reason = "slow", "steep slope — reduce speed"
    else:
        advice, reason = "ok", None
    return TerrainSafety(
        tipover=tipover,
        drain_drop=drain_drop,
        steep=steep,
        roll=seated.roll,
        pitch=seated.pitch,
        max_slope=slope_here,
        wheels_in_drain=wheels_in,
        advice=advice,
        reason=reason,
    )

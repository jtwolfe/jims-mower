"""Trimmer interlock and body-collision queries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

from jims_mower.constants import BIRD_COLLISION_Z_M, LIVING_KINDS
from jims_mower.kinematics import trimmer_xy
from jims_mower.types import Obstacle, Pose


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

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
    TERRAIN_POND,
    TERRAIN_PUDDLE,
)
from jims_mower.geofence import (
    GeofenceSpec,
    allowed_xy,
    geofence_advice,
    point_in_polygon,
)
from jims_mower.world import point_to_segment_distance
from jims_mower.kinematics import sit_on_terrain, trimmer_xy, wheel_positions
from jims_mower.types import Detection, Obstacle, Pose

if TYPE_CHECKING:
    from jims_mower.terrain import HeightField

# Forward cameras used by the dets-from-camera living interlock (PLN-4).
# Rear / side blobs do not trip the trimmer — a person behind the robot
# is out of the cutting view.
FORWARD_CAMERAS = frozenset(
    {"front", "front_left", "front_right", "stereo_left", "stereo_right"}
)


@dataclass(frozen=True)
class SafetyDecision:
    trimmer_enabled: bool
    requested: bool
    nearest_living_m: float
    nearest_kind: Optional[str]
    blocked_reason: Optional[str]
    advice: str = "ok"


@dataclass(frozen=True)
class LivingSafety:
    """Planner-side extension of the trimmer interlock for moving people/animals."""

    advice: str
    nearest_living_m: float
    nearest_kind: Optional[str]
    reason: Optional[str]


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
    living = living_advice(
        dist,
        kind,
        slow_m=max(3.0, safety_radius_m * 2.0),
        reroute_m=max(1.6, safety_radius_m),
        stop_m=max(0.85, safety_radius_m * 0.55),
    )
    if not requested:
        return SafetyDecision(False, False, dist, kind, None, living.advice)
    if dist < safety_radius_m:
        return SafetyDecision(
            False,
            True,
            dist,
            kind,
            f"{kind} within {safety_radius_m:.2f} m of trimmer",
            living.advice if living.advice != "ok" else "reroute",
        )
    return SafetyDecision(True, True, dist, kind, None, living.advice)


def _is_living_detection(det: Detection) -> bool:
    if is_living(det.label):
        return True
    return str(det.category or "") in {"person", "animal"}


def living_from_detections(
    detections: Iterable[Detection],
    *,
    cameras: Optional[Iterable[str]] = None,
) -> tuple[float, Optional[Detection]]:
    """Nearest living appearance det. No world range — in-view is the trip.

    Appearance blobs have no metric depth. A living det in a forward
    camera is treated as inside the tool radius (``0.0`` m). Out of
    those cameras → no trip (``inf``). Does not read the oracle list.
    """
    allowed = {str(c) for c in cameras} if cameras is not None else set(FORWARD_CAMERAS)
    chosen: Optional[Detection] = None
    for det in detections:
        if not _is_living_detection(det):
            continue
        if allowed and det.camera not in allowed:
            continue
        chosen = det
        break
    if chosen is None:
        return math.inf, None
    return 0.0, chosen


def trimmer_interlock_from_detections(
    requested: bool,
    detections: Iterable[Detection],
    *,
    safety_radius_m: float,
    cameras: Optional[Iterable[str]] = None,
) -> SafetyDecision:
    """Trimmer interlock from camera dets — not ``context.obstacles``.

    Gym trip: a painted living blob in a forward camera. A person on
    the oracle list who is behind the robot and out of those cameras
    does **not** fire. No invented metres.
    """
    if safety_radius_m <= 0:
        raise ValueError("safety_radius_m must be positive")
    dist, det = living_from_detections(detections, cameras=cameras)
    kind = det.label if det is not None else None
    living = living_advice(
        dist,
        kind,
        slow_m=max(3.0, safety_radius_m * 2.0),
        reroute_m=max(1.6, safety_radius_m),
        stop_m=max(0.85, safety_radius_m * 0.55),
    )
    if not requested:
        return SafetyDecision(False, False, dist, kind, None, living.advice)
    if det is not None:
        cam = det.camera
        return SafetyDecision(
            False,
            True,
            dist,
            kind,
            f"{kind} in {cam} camera (appearance)",
            living.advice if living.advice != "ok" else "stop",
        )
    return SafetyDecision(True, True, dist, kind, None, living.advice)


def living_advice(
    nearest_m: float,
    kind: Optional[str],
    *,
    slow_m: float = 3.0,
    reroute_m: float = 1.8,
    stop_m: float = 0.90,
) -> LivingSafety:
    """Map range-to-living-thing onto the same advice ladder as terrain.

    Extends the trimmer interlock: the controller must slow, temporarily
    block / replan, or stop — not only disable the string head.
    """
    if kind is None or not math.isfinite(nearest_m):
        return LivingSafety("ok", nearest_m, kind, None)
    if nearest_m <= stop_m:
        return LivingSafety("stop", nearest_m, kind, f"{kind} too close — hold")
    if nearest_m <= reroute_m:
        return LivingSafety("reroute", nearest_m, kind, f"{kind} nearby — go around")
    if nearest_m <= slow_m:
        return LivingSafety("slow", nearest_m, kind, f"{kind} in the yard — cut speed")
    return LivingSafety("ok", nearest_m, kind, None)


def is_body_collision(
    pose: Pose,
    obstacle: Obstacle,
    collision_radius_m: float,
) -> bool:
    """True when the robot body circle overlaps an obstacle that can strike it."""
    if obstacle.is_soft:
        return False
    if obstacle.kind == "bird" and obstacle.z >= BIRD_COLLISION_Z_M:
        return False
    gap = hypot2(pose.x, pose.y, obstacle.x, obstacle.y)
    return gap < (collision_radius_m + obstacle.radius)


def cutter_risk_hit(
    hub_xy: tuple[float, float],
    obstacles: Iterable[Obstacle],
    radius_m: float,
) -> Optional[Obstacle]:
    """First hose/cord whose footprint overlaps the spinning trimmer disk."""
    hx, hy = hub_xy
    for obst in obstacles:
        if not obst.is_cutter_risk:
            continue
        if obst.length_m > 0.15:
            x0, y0, x1, y1 = obst.segment_ends()
            dist = point_to_segment_distance(hx, hy, x0, y0, x1, y1)
        else:
            dist = hypot2(hx, hy, obst.x, obst.y)
        if dist < (radius_m + obst.radius):
            return obst
    return None


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
    geofence: Optional[list[tuple[float, float]]] = None,
    keepout: Optional[list[list[tuple[float, float]]]] = None,
    spec: Optional[GeofenceSpec] = None,
) -> bool:
    """True when the body center is inside the yard (and geofence, if given)."""
    in_rect = (
        collision_radius_m <= pose.x <= width_m - collision_radius_m
        and collision_radius_m <= pose.y <= height_m - collision_radius_m
    )
    if not in_rect:
        return False
    fence = spec
    if fence is None and (geofence or keepout):
        fence = GeofenceSpec(keep_in=list(geofence or []), keep_out=list(keepout or []))
    if fence is not None and fence.has_polygons():
        return allowed_xy(pose.x, pose.y, fence)
    return True


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
    look_ahead_m: float = 1.10,
    look_ahead_samples: int = 4,
    max_climb_slope_rad: Optional[float] = None,
    static_tip_roll_rad: Optional[float] = None,
    static_tip_pitch_rad: Optional[float] = None,
) -> TerrainSafety:
    """Physics-side terrain safety (true height field, not the observer maps).

    Recommended policy behaviour (also returned as ``advice``):
    - ``ok`` — continue
    - ``slow`` — high slope under the chassis or a climbable face ahead; cut speed
    - ``reroute`` — drain lip or tip-risk face ahead; do not drive in at speed
    - ``stop`` — tip-over risk or a wheel already in the channel

    Look-ahead uses the same kinematic sit as ``sit_on_terrain``. A face
    ahead is slow / reroute — it is not a seated physics tip-over.
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
    static_r = float(static_tip_roll_rad) if static_tip_roll_rad is not None else float(tip_roll_rad)
    static_p = float(static_tip_pitch_rad) if static_tip_pitch_rad is not None else float(tip_pitch_rad)
    # True tip-over is the geometric static α. Software trips stay earlier.
    tipover = abs(seated.roll) >= static_r or abs(seated.pitch) >= static_p
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
    climb = float(max_climb_slope_rad) if max_climb_slope_rad is not None else float(steep_slope_rad)
    # Late import: safety is a leaf used by terrain/structures.
    from jims_mower.planning.grade_tip import KIND_GRADE, KIND_TIP, probe_forward_grade

    ahead_grade = probe_forward_grade(
        seated,
        height_field.sample,
        length_m=length_m,
        track_m=track_m,
        look_ahead_m=look_ahead_m,
        n_samples=look_ahead_samples,
        tip_roll_rad=tip_roll_rad,
        tip_pitch_rad=tip_pitch_rad,
        max_climb_slope_rad=climb,
        static_tip_roll_rad=static_r,
        static_tip_pitch_rad=static_p,
    )

    if tipover:
        advice, reason = "stop", "tip-over risk from pitch/roll"
    elif drain_drop:
        advice, reason = "stop", "wheel in earth drain / channel"
    elif ahead_grade.past_tip:
        advice, reason = "stop", "grade look-ahead — sit would exceed tip"
        steep = True
    elif lip_ahead or lip_under:
        advice, reason = "reroute", "drain edge — do not drop a wheel in"
    elif ahead_grade.kind == KIND_TIP:
        advice, reason = "reroute", "grade look-ahead — tip-risk face"
        steep = True
    elif ahead_grade.kind == KIND_GRADE or steep:
        advice, reason = "slow", "steep slope — reduce speed"
        steep = True
    elif any(lab == TERRAIN_POND for lab in labels):
        advice, reason = "stop", "pond keep-out — not mowable water"
        steep = True
    elif any(lab == TERRAIN_PUDDLE for lab in labels):
        advice, reason = "slow", "rain puddle — wet / temporary hazard"
        steep = True
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

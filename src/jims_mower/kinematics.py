"""Zero-turn differential-drive kinematics (no lateral slip)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from jims_mower.types import Pose

if TYPE_CHECKING:
    from jims_mower.terrain import HeightField


def wrap_angle(theta: float) -> float:
    """Wrap radians to (-pi, pi]."""
    wrapped = (float(theta) + math.pi) % (2.0 * math.pi) - math.pi
    if wrapped <= -math.pi:
        wrapped += 2.0 * math.pi
    return wrapped


def clip_wheel_speeds(
    v_left: float, v_right: float, max_speed: float
) -> tuple[float, float]:
    if max_speed <= 0:
        raise ValueError("max_speed must be positive")
    return (
        float(np.clip(v_left, -max_speed, max_speed)),
        float(np.clip(v_right, -max_speed, max_speed)),
    )


def unicycle_from_wheels(
    v_left: float, v_right: float, wheelbase: float
) -> tuple[float, float]:
    """Map wheel linear speeds to (forward v, yaw rate omega)."""
    if wheelbase <= 0:
        raise ValueError("wheelbase must be positive")
    v = 0.5 * (v_right + v_left)
    omega = (v_right - v_left) / wheelbase
    return float(v), float(omega)


def wheels_from_unicycle(
    v: float, omega: float, wheelbase: float
) -> tuple[float, float]:
    """Inverse of :func:`unicycle_from_wheels`."""
    if wheelbase <= 0:
        raise ValueError("wheelbase must be positive")
    v_left = v - 0.5 * omega * wheelbase
    v_right = v + 0.5 * omega * wheelbase
    return float(v_left), float(v_right)


def integrate_pose(
    pose: Pose,
    v_left: float,
    v_right: float,
    wheelbase: float,
    dt: float,
    max_speed: float,
) -> Pose:
    """Closed-form integration of a differential-drive pose over ``dt``.

    Equal wheel speeds drive straight. Equal-and-opposite speeds are a
    zero-radius pivot (zero-turn). There is no holonomic strafe term.
    """
    if dt < 0:
        raise ValueError("dt must be non-negative")
    if dt == 0.0:
        return Pose(pose.x, pose.y, wrap_angle(pose.theta))
    vl, vr = clip_wheel_speeds(v_left, v_right, max_speed)
    v, omega = unicycle_from_wheels(vl, vr, wheelbase)
    theta = pose.theta
    if abs(omega) < 1e-9:
        return Pose(
            pose.x + v * math.cos(theta) * dt,
            pose.y + v * math.sin(theta) * dt,
            wrap_angle(theta),
        )
    ratio = v / omega
    new_theta = wrap_angle(theta + omega * dt)
    return Pose(
        pose.x + ratio * (math.sin(new_theta) - math.sin(theta)),
        pose.y - ratio * (math.cos(new_theta) - math.cos(theta)),
        new_theta,
    )


def trimmer_xy(pose: Pose, offset_m: float) -> tuple[float, float]:
    """World XY of the front-mounted trimmer hub."""
    return (
        pose.x + offset_m * math.cos(pose.theta),
        pose.y + offset_m * math.sin(pose.theta),
    )


def trimmer_xyz(
    pose: Pose,
    offset_m: float,
    height_field: Optional["HeightField"] = None,
    hover_m: float = 0.12,
) -> tuple[float, float, float]:
    """Trimmer hub XY plus a height that follows the local ground."""
    x, y = trimmer_xy(pose, offset_m)
    ground = height_field.sample(x, y) if height_field is not None else pose.z
    return x, y, float(ground + hover_m)


def heading_vector(theta: float) -> tuple[float, float]:
    return (math.cos(theta), math.sin(theta))


def wheel_positions(
    pose: Pose, length_m: float, track_m: float
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
    """FL, FR, RL, RR wheel contact XY (body +y is left)."""
    c = math.cos(pose.theta)
    s = math.sin(pose.theta)
    hl = 0.5 * length_m
    ht = 0.5 * track_m
    # body (x_fwd, y_left) → world
    corners = ((hl, ht), (hl, -ht), (-hl, ht), (-hl, -ht))
    out = []
    for bx, by in corners:
        out.append((pose.x + bx * c - by * s, pose.y + bx * s + by * c))
    fl, fr, rl, rr = out
    return fl, fr, rl, rr


def contact_midpoints(
    pose: Pose, length_m: float, track_m: float
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Front, rear, left, right mid-axle points used for pitch / roll."""
    c = math.cos(pose.theta)
    s = math.sin(pose.theta)
    hl = 0.5 * length_m
    ht = 0.5 * track_m
    front = (pose.x + hl * c, pose.y + hl * s)
    rear = (pose.x - hl * c, pose.y - hl * s)
    left = (pose.x - ht * s, pose.y + ht * c)
    right = (pose.x + ht * s, pose.y - ht * c)
    return front, rear, left, right


def sit_on_height_fn(
    pose: Pose,
    sample_z: Callable[[float, float], float],
    length_m: float,
    track_m: float,
) -> Pose:
    """Kinematic seating: four contact heights → z / pitch / roll.

    This is a ground-polygon tangent, not rigid-body rolling. Pitch and
    roll are ``atan2`` of the front–rear and left–right height differences.
    """
    front, rear, left, right = contact_midpoints(pose, length_m, track_m)
    z_f = float(sample_z(*front))
    z_r = float(sample_z(*rear))
    z_l = float(sample_z(*left))
    z_ri = float(sample_z(*right))
    z = 0.25 * (z_f + z_r + z_l + z_ri)
    pitch = math.atan2(z_f - z_r, max(length_m, 1e-6))
    roll = math.atan2(z_l - z_ri, max(track_m, 1e-6))
    return Pose(pose.x, pose.y, pose.theta, float(z), float(pitch), float(roll))


def sit_on_terrain(
    pose: Pose,
    height_field: Optional["HeightField"],
    length_m: float,
    track_m: float,
) -> Pose:
    """Lift the planar pose onto the height field and set pitch / roll.

    Four-wheel samples drive tip / drop checks; pitch and roll come from
    the front–rear and left–right height differences. Not inertia / CG
    tip-moment physics — see ``docs/CHASSIS_PHYSICS.md``. Seating will
    happily report |roll| / |pitch| past the software tip. Immobilise
    only when the sit crosses static α(t, b, h_cg) — see
    ``docs/CHASSIS_PHYSICS.md``.
    """
    if height_field is None:
        return Pose(pose.x, pose.y, pose.theta, 0.0, 0.0, 0.0)
    return sit_on_height_fn(pose, height_field.sample, length_m, track_m)


def static_tip_angle_rad(half_support_m: float, h_cg_m: float) -> float:
    """Geometric static tip: gravity through the CG leaves a t×b patch.

    ``α = atan((support/2) / h_cg)``. Hang-measure ``h_cg`` before treating
    this as a field number. Not a rolling rigid-body moment.
    """
    if float(h_cg_m) <= 0.0:
        raise ValueError("h_cg_m must be positive")
    if float(half_support_m) <= 0.0:
        raise ValueError("half_support_m must be positive")
    return math.atan(float(half_support_m) / float(h_cg_m))


def static_tip_angles_rad(
    *,
    track_m: float,
    wheelbase_m: float,
    h_cg_m: float,
) -> tuple[float, float]:
    """Return ``(α_roll, α_pitch)`` from track, wheelbase, and CG height.

    Iso-stable when ``track_m ≈ wheelbase_m`` (square support). Software
    trips must stay *below* these angles.
    """
    return (
        static_tip_angle_rad(0.5 * float(track_m), h_cg_m),
        static_tip_angle_rad(0.5 * float(wheelbase_m), h_cg_m),
    )


def attitude_past_tip(
    roll: float,
    pitch: float,
    tip_roll_rad: float,
    tip_pitch_rad: float,
) -> bool:
    """True when seated / IMU attitude is at or past the given trips."""
    return abs(float(roll)) >= float(tip_roll_rad) or abs(float(pitch)) >= float(tip_pitch_rad)


def static_tip_latch(
    pose: Pose,
    *,
    tip_roll_rad: float,
    tip_pitch_rad: float,
    latched: bool = False,
) -> bool:
    """Latch once seated |roll| / |pitch| reaches the *static* α(t, b, h_cg).

    Callers must pass the geometric static angles, not the earlier
    software tip-stop. Not a full CG tip-moment / rolling rigid-body
    engine — just do not treat a past-static sit as a driveable re-seat.
    """
    return bool(latched) or attitude_past_tip(
        pose.roll, pose.pitch, tip_roll_rad, tip_pitch_rad
    )


def wheel_clearances(
    pose: Pose,
    height_field: Optional["HeightField"],
    length_m: float,
    track_m: float,
    chassis_hover_m: float = 0.06,
) -> np.ndarray:
    """Downward gap from chassis plane to ground at each wheel (FL FR RL RR)."""
    wheels = wheel_positions(pose, length_m, track_m)
    out = np.zeros(4, dtype=np.float32)
    if height_field is None:
        out.fill(chassis_hover_m)
        return out
    for i, (wx, wy) in enumerate(wheels):
        ground = height_field.sample(wx, wy)
        out[i] = float(pose.z + chassis_hover_m - ground)
    return out

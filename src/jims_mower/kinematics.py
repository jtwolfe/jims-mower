"""Zero-turn differential-drive kinematics (no lateral slip)."""

from __future__ import annotations

import math

import numpy as np

from jims_mower.types import Pose


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


def heading_vector(theta: float) -> tuple[float, float]:
    return (math.cos(theta), math.sin(theta))

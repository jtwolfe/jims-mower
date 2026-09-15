"""Zero-turn differential-drive kinematics."""

from __future__ import annotations

import math

import pytest

from jims_mower.kinematics import (
    clip_wheel_speeds,
    heading_vector,
    integrate_pose,
    trimmer_xy,
    unicycle_from_wheels,
    wheels_from_unicycle,
    wrap_angle,
)
from jims_mower.types import Pose


def test_wrap_angle_identity() -> None:
    assert wrap_angle(0.0) == pytest.approx(0.0)
    assert wrap_angle(0.5) == pytest.approx(0.5)


def test_wrap_angle_two_pi() -> None:
    assert wrap_angle(2.0 * math.pi) == pytest.approx(0.0, abs=1e-9)
    assert wrap_angle(-2.0 * math.pi) == pytest.approx(0.0, abs=1e-9)


def test_wrap_angle_three_pi() -> None:
    # 3π maps onto ±π; we canonicalize −π to +π.
    assert wrap_angle(3.0 * math.pi) == pytest.approx(math.pi, abs=1e-9)


def test_wrap_angle_range() -> None:
    for th in (-8.0, -math.pi, -0.1, 0.0, math.pi, 7.4, 20.0):
        w = wrap_angle(th)
        assert -math.pi < w <= math.pi + 1e-12


def test_clip_wheel_speeds() -> None:
    lo, hi = clip_wheel_speeds(-4.0, 3.0, 1.2)
    assert lo == pytest.approx(-1.2)
    assert hi == pytest.approx(1.2)


def test_clip_rejects_non_positive_max() -> None:
    with pytest.raises(ValueError):
        clip_wheel_speeds(0.0, 0.0, 0.0)


def test_unicycle_straight() -> None:
    v, omega = unicycle_from_wheels(1.0, 1.0, 0.4)
    assert v == pytest.approx(1.0)
    assert omega == pytest.approx(0.0)


def test_unicycle_zero_turn() -> None:
    v, omega = unicycle_from_wheels(-1.0, 1.0, 0.4)
    assert v == pytest.approx(0.0)
    assert omega == pytest.approx(5.0)


def test_unicycle_rejects_zero_wheelbase() -> None:
    with pytest.raises(ValueError):
        unicycle_from_wheels(0.1, 0.1, 0.0)


def test_wheels_roundtrip() -> None:
    v_l, v_r = wheels_from_unicycle(0.6, -0.4, 0.4)
    v, omega = unicycle_from_wheels(v_l, v_r, 0.4)
    assert v == pytest.approx(0.6)
    assert omega == pytest.approx(-0.4)


def test_integrate_zero_action_stays_put() -> None:
    pose = Pose(1.0, 2.0, 0.3)
    nxt = integrate_pose(pose, 0.0, 0.0, 0.4, 0.1, 1.2)
    assert nxt.x == pytest.approx(pose.x)
    assert nxt.y == pytest.approx(pose.y)
    assert nxt.theta == pytest.approx(pose.theta)


def test_integrate_straight_along_x() -> None:
    pose = Pose(0.0, 0.0, 0.0)
    nxt = integrate_pose(pose, 1.0, 1.0, 0.4, 0.2, 1.2)
    assert nxt.x == pytest.approx(0.2)
    assert nxt.y == pytest.approx(0.0)
    assert nxt.theta == pytest.approx(0.0)


def test_integrate_backward() -> None:
    pose = Pose(1.0, 0.0, 0.0)
    nxt = integrate_pose(pose, -1.0, -1.0, 0.4, 0.2, 1.2)
    assert nxt.x == pytest.approx(0.8)
    assert nxt.y == pytest.approx(0.0)


def test_integrate_zero_turn_holds_xy() -> None:
    pose = Pose(3.0, 4.0, 0.25)
    nxt = integrate_pose(pose, -0.8, 0.8, 0.4, 0.3, 1.2)
    assert nxt.x == pytest.approx(pose.x, abs=1e-9)
    assert nxt.y == pytest.approx(pose.y, abs=1e-9)
    assert nxt.theta != pytest.approx(pose.theta)


def test_integrate_no_lateral_slip() -> None:
    pose = Pose(0.0, 0.0, 0.0)
    nxt = integrate_pose(pose, 0.7, 0.7, 0.4, 0.5, 1.2)
    assert nxt.y == pytest.approx(0.0, abs=1e-12)


def test_integrate_dt_zero_is_noop() -> None:
    pose = Pose(1.0, 1.0, 1.0)
    nxt = integrate_pose(pose, 1.0, 0.2, 0.4, 0.0, 1.2)
    assert nxt.x == pytest.approx(1.0)
    assert nxt.y == pytest.approx(1.0)


def test_integrate_rejects_negative_dt() -> None:
    with pytest.raises(ValueError):
        integrate_pose(Pose(0, 0, 0), 0.0, 0.0, 0.4, -0.1, 1.2)


def test_wider_wheelbase_slower_yaw() -> None:
    pose = Pose(0.0, 0.0, 0.0)
    narrow = integrate_pose(pose, -0.5, 0.5, 0.3, 0.2, 1.2)
    wide = integrate_pose(pose, -0.5, 0.5, 0.6, 0.2, 1.2)
    assert abs(narrow.theta) > abs(wide.theta)


def test_speeds_are_clipped_to_max() -> None:
    pose = Pose(0.0, 0.0, 0.0)
    nxt = integrate_pose(pose, 50.0, 50.0, 0.4, 0.1, 1.0)
    assert nxt.x == pytest.approx(0.1)


def test_forward_then_back_returns() -> None:
    pose = Pose(2.0, 3.0, 0.4)
    mid = integrate_pose(pose, 0.8, 0.8, 0.4, 0.25, 1.2)
    back = integrate_pose(mid, -0.8, -0.8, 0.4, 0.25, 1.2)
    assert back.x == pytest.approx(pose.x, abs=1e-9)
    assert back.y == pytest.approx(pose.y, abs=1e-9)


def test_trimmer_is_in_front() -> None:
    pose = Pose(0.0, 0.0, 0.0)
    x, y = trimmer_xy(pose, 0.32)
    assert x == pytest.approx(0.32)
    assert y == pytest.approx(0.0)
    pose = Pose(0.0, 0.0, math.pi / 2)
    x, y = trimmer_xy(pose, 0.32)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(0.32)


def test_heading_vector() -> None:
    hx, hy = heading_vector(0.0)
    assert hx == pytest.approx(1.0)
    assert hy == pytest.approx(0.0)

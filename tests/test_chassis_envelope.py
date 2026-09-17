"""70×70×40 envelope, iso static α, software trip below static."""

from __future__ import annotations

import math

import pytest

from jims_mower.config import EnvConfig, load_config
from jims_mower.kinematics import (
    attitude_past_tip,
    sit_on_terrain,
    static_tip_angle_rad,
    static_tip_angles_rad,
    static_tip_latch,
)
from jims_mower.planning.grade_tip import KIND_TIP, classify_tilt
from jims_mower.safety import terrain_hazards
from jims_mower.terrain import HeightField
from jims_mower.types import Pose


def test_default_robot_is_70_by_70_by_40_belly_class() -> None:
    cfg = load_config()
    robot = cfg.robot
    assert robot.length_m == pytest.approx(0.70)
    assert robot.width_m == pytest.approx(0.70)
    assert robot.height_m == pytest.approx(0.40)
    assert robot.track_m == pytest.approx(0.55)
    assert robot.wheelbase_m == pytest.approx(0.55)
    assert robot.h_cg_m == pytest.approx(0.14)
    assert robot.collision_radius_m == pytest.approx(0.40)
    assert robot.trimmer.offset_m == pytest.approx(0.42)
    assert robot.trimmer.height_m == pytest.approx(0.12)
    assert robot.software_tip_frac == pytest.approx(0.50)
    for cam in cfg.resolved_cameras():
        assert 0.0 < cam.z < robot.height_m


def test_iso_static_alpha_when_track_equals_wheelbase() -> None:
    roll, pitch = static_tip_angles_rad(track_m=0.55, wheelbase_m=0.55, h_cg_m=0.14)
    assert roll == pytest.approx(pitch, abs=1e-12)
    expected = math.atan(0.275 / 0.14)
    assert roll == pytest.approx(expected, abs=1e-9)
    assert math.degrees(roll) == pytest.approx(63.0, abs=0.1)
    robot = EnvConfig().robot
    assert robot.static_tip_roll_rad() == pytest.approx(robot.static_tip_pitch_rad())
    assert robot.static_tip_roll_rad() == pytest.approx(expected, abs=1e-9)


def test_software_trip_is_half_static_and_below() -> None:
    robot = EnvConfig().robot
    static_r = robot.static_tip_roll_rad()
    static_p = robot.static_tip_pitch_rad()
    assert robot.tip_roll_rad == pytest.approx(0.50 * static_r, abs=0.02)
    assert robot.tip_pitch_rad == pytest.approx(0.50 * static_p, abs=0.02)
    assert robot.tip_roll_rad < static_r
    assert robot.tip_pitch_rad < static_p
    assert robot.derived_software_tip_roll_rad() == pytest.approx(0.50 * static_r)


def test_tipover_at_static_alpha_not_software() -> None:
    robot = EnvConfig().robot
    static_r = robot.static_tip_roll_rad()
    # Planar side slope just under / over static α. Sit roll = atan(k).
    under = HeightField.from_function(
        8.0, 6.0, 0.10, lambda x, y: math.tan(static_r - 0.04) * y
    )
    over = HeightField.from_function(
        8.0, 6.0, 0.10, lambda x, y: math.tan(static_r + 0.04) * y
    )
    seated_under = sit_on_terrain(Pose(4.0, 3.0, 0.0), under, 0.70, 0.55)
    seated_over = sit_on_terrain(Pose(4.0, 3.0, 0.0), over, 0.70, 0.55)
    ev_under = terrain_hazards(
        seated_under,
        under,
        length_m=0.70,
        track_m=0.55,
        tip_roll_rad=robot.tip_roll_rad,
        tip_pitch_rad=robot.tip_pitch_rad,
        wheel_drop_m=0.08,
        steep_slope_rad=0.30,
        static_tip_roll_rad=static_r,
        static_tip_pitch_rad=robot.static_tip_pitch_rad(),
    )
    ev_over = terrain_hazards(
        seated_over,
        over,
        length_m=0.70,
        track_m=0.55,
        tip_roll_rad=robot.tip_roll_rad,
        tip_pitch_rad=robot.tip_pitch_rad,
        wheel_drop_m=0.08,
        steep_slope_rad=0.30,
        static_tip_roll_rad=static_r,
        static_tip_pitch_rad=robot.static_tip_pitch_rad(),
    )
    assert ev_under.tipover is False
    assert ev_over.tipover is True
    assert static_tip_latch(
        seated_over,
        tip_roll_rad=static_r,
        tip_pitch_rad=robot.static_tip_pitch_rad(),
    )
    assert not static_tip_latch(
        seated_under,
        tip_roll_rad=static_r,
        tip_pitch_rad=robot.static_tip_pitch_rad(),
    )


def test_software_trip_below_static_does_not_immobilise() -> None:
    robot = EnvConfig().robot
    # ~0.65 rad ≈ 37° — past software 0.55, under static ~1.10.
    roll = 0.65
    assert roll > robot.tip_roll_rad
    assert roll < robot.static_tip_roll_rad()
    got = classify_tilt(
        roll,
        0.0,
        tip_roll_rad=robot.tip_roll_rad,
        tip_pitch_rad=robot.tip_pitch_rad,
        slow_frac=0.55,
        stop_frac=0.85,
        max_climb_slope_rad=0.32,
        static_tip_roll_rad=robot.static_tip_roll_rad(),
        static_tip_pitch_rad=robot.static_tip_pitch_rad(),
    )
    assert got.kind == KIND_TIP
    assert got.advice == "stop"
    assert got.past_tip is False
    assert not attitude_past_tip(
        roll, 0.0, robot.static_tip_roll_rad(), robot.static_tip_pitch_rad()
    )


def test_static_tip_angle_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        static_tip_angle_rad(0.275, 0.0)

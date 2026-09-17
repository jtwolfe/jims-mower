"""Kinematic sit, acre mesh, explore speed, and look-ahead — not rolling CG."""

from __future__ import annotations

import math

import numpy as np
import pytest

from jims_mower.config import EnvConfig, load_config
from jims_mower.kinematics import integrate_pose, sit_on_height_fn, sit_on_terrain
from jims_mower.planning.grade_tip import (
    KIND_GRADE,
    KIND_OK,
    KIND_TIP,
    OWNER_STEEP_GRADE,
    OWNER_TIP_RISK,
    grade_aware_cruise,
    look_ahead_advice,
    look_ahead_from_elevation,
    merge_look_ahead_kind,
    probe_forward_grade,
)
from jims_mower.safety import terrain_hazards
from jims_mower.scenarios import load_source
from jims_mower.terrain import HeightField
from jims_mower.types import Pose


def _ramp_fn(x0: float = 3.30, width: float = 0.40, rise: float = 0.18):
    """0.40 m face: one 0.50 m cell (aliased cliff), two-plus 0.25 m cells."""

    def fn(x, y):
        _ = y
        return np.clip((np.asarray(x) - x0) / width, 0.0, 1.0) * rise

    return fn


def _drive_ramp(
    resolution_m: float,
    cruise: float,
    *,
    dt: float = 0.10,
    vmax: float = 1.2,
    length_m: float = 0.50,
    track_m: float = 0.40,
) -> dict[str, float]:
    hf = HeightField.from_function(10.0, 6.0, resolution_m, _ramp_fn())
    pose = sit_on_terrain(Pose(2.20, 3.0, 0.0), hf, length_m, track_m)
    pitches = [pose.pitch]
    for _ in range(55):
        pose = integrate_pose(pose, cruise * vmax, cruise * vmax, track_m, dt, vmax)
        pose = sit_on_terrain(pose, hf, length_m, track_m)
        pitches.append(pose.pitch)
        if pose.x >= 7.0:
            break
    deltas = [abs(pitches[i + 1] - pitches[i]) for i in range(len(pitches) - 1)]
    return {
        "max_delta": max(deltas) if deltas else 0.0,
        "max_abs_pitch": max(abs(p) for p in pitches),
        "tip": 1.0 if any(abs(p) >= 0.45 for p in pitches) else 0.0,
        "steps": float(len(pitches)),
    }


def test_sit_on_height_fn_matches_height_field() -> None:
    hf = HeightField.from_function(6.0, 4.0, 0.20, lambda x, y: 0.12 * x)
    pose = Pose(3.0, 2.0, 0.15)
    a = sit_on_terrain(pose, hf, 0.50, 0.40)
    b = sit_on_height_fn(pose, hf.sample, 0.50, 0.40)
    assert a.z == pytest.approx(b.z)
    assert a.pitch == pytest.approx(b.pitch)
    assert a.roll == pytest.approx(b.roll)


def test_finer_mesh_reduces_aliased_ridge_peak_pitch() -> None:
    # Same cruise: 0.50 m one-cell cliff overstates sit pitch. 0.25 m
    # resolves the face so peak pitch is closer to the authored ramp.
    coarse = _drive_ramp(0.50, 0.98)
    fine = _drive_ramp(0.25, 0.98)
    assert fine["max_abs_pitch"] < coarse["max_abs_pitch"]
    assert fine["tip"] == 0.0


def test_slower_explore_reduces_ridge_pitch_jumps() -> None:
    fast = _drive_ramp(0.25, 0.98)
    slow = _drive_ramp(0.25, 0.45)
    assert slow["max_delta"] <= fast["max_delta"] + 1e-9
    assert slow["max_delta"] < fast["max_delta"] or slow["max_delta"] < 0.08
    assert slow["tip"] == 0.0


def test_finer_mesh_and_slower_explore_beat_old_acre_demo() -> None:
    old = _drive_ramp(0.50, 0.98)
    new = _drive_ramp(0.25, 0.45)
    assert new["max_delta"] < old["max_delta"]
    assert new["max_abs_pitch"] < old["max_abs_pitch"]
    assert new["tip"] == 0.0


def test_climbable_face_does_not_hard_tip() -> None:
    # ~0.25 rad face — under acre climb (0.34) and tip pitch (0.45).
    hf = HeightField.from_function(
        8.0, 5.0, 0.25, lambda x, y: np.clip((x - 2.0) / 0.70, 0.0, 1.0) * 0.18
    )
    pose = sit_on_terrain(Pose(2.4, 2.5, 0.0), hf, 0.50, 0.40)
    vmax = 1.2
    tipped = False
    for _ in range(40):
        pose = integrate_pose(pose, 0.45 * vmax, 0.45 * vmax, 0.40, 0.10, vmax)
        pose = sit_on_terrain(pose, hf, 0.50, 0.40)
        ev = terrain_hazards(
            pose,
            hf,
            length_m=0.50,
            track_m=0.40,
            tip_roll_rad=0.40,
            tip_pitch_rad=0.45,
            wheel_drop_m=0.08,
            steep_slope_rad=0.30,
            look_ahead_m=1.10,
            max_climb_slope_rad=0.34,
        )
        if ev.tipover:
            tipped = True
            break
        if pose.x >= 6.2:
            break
    assert tipped is False
    assert abs(pose.pitch) < 0.45


def test_look_ahead_fires_before_seated_pitch_crosses_climb() -> None:
    # Flat pad, then a climbable face ~1.0 m ahead (under physics tip).
    hf = HeightField.from_function(
        10.0, 6.0, 0.25, lambda x, y: np.clip((x - 3.60) / 0.45, 0.0, 1.0) * 0.15
    )
    pose = sit_on_terrain(Pose(2.70, 3.0, 0.0), hf, 0.50, 0.40)
    climb = 0.32
    assert abs(pose.pitch) < climb
    ev = terrain_hazards(
        pose,
        hf,
        length_m=0.50,
        track_m=0.40,
        tip_roll_rad=0.40,
        tip_pitch_rad=0.45,
        wheel_drop_m=0.08,
        steep_slope_rad=0.30,
        look_ahead_m=1.10,
        max_climb_slope_rad=climb,
    )
    assert ev.tipover is False
    assert ev.advice in {"slow", "reroute"}
    ahead = probe_forward_grade(
        pose,
        hf.sample,
        length_m=0.50,
        track_m=0.40,
        look_ahead_m=1.10,
        n_samples=6,
        max_climb_slope_rad=climb,
    )
    assert ahead.kind in {KIND_GRADE, KIND_TIP}
    assert ahead.past_tip is False
    assert look_ahead_advice(ahead) in {"slow", "reroute"}
    assert look_ahead_advice(ahead) != "stop"


def test_observer_elevation_look_ahead_matches_sit_probe() -> None:
    hf = HeightField.from_function(
        6.0, 4.0, 0.20, lambda x, y: np.clip((x - 2.4) / 0.20, 0.0, 1.0) * 0.22
    )
    pose = sit_on_terrain(Pose(1.4, 2.0, 0.0), hf, 0.50, 0.40)
    ahead = look_ahead_from_elevation(
        pose,
        hf.elevation,
        resolution_m=0.20,
        length_m=0.50,
        track_m=0.40,
        look_ahead_m=1.10,
        max_climb_slope_rad=0.32,
    )
    assert ahead is not None
    assert ahead.kind in {KIND_GRADE, KIND_TIP}
    if ahead.past_tip:
        assert merge_look_ahead_kind(KIND_OK, ahead) == KIND_TIP
    else:
        assert merge_look_ahead_kind(KIND_OK, ahead) == KIND_GRADE


def test_look_ahead_tip_does_not_rewrite_owner_tip_copy() -> None:
    # Ahead tip-risk is a contour, not "Tip risk — reversing".
    assert merge_look_ahead_kind(KIND_OK, None) == KIND_OK
    fake = probe_forward_grade(
        Pose(0.0, 0.0, 0.0),
        lambda x, y: 0.0,
        length_m=0.50,
        track_m=0.40,
    )
    assert fake.kind == KIND_OK
    assert OWNER_STEEP_GRADE != OWNER_TIP_RISK


def test_grade_aware_cruise_slows_explore_and_grade() -> None:
    base = 0.45
    ok = grade_aware_cruise(
        base,
        advice="ok",
        tilt_kind=KIND_OK,
        slow_speed_factor=0.35,
        grade_speed_factor=0.50,
    )
    grade = grade_aware_cruise(
        base,
        advice="slow",
        tilt_kind=KIND_GRADE,
        slow_speed_factor=0.35,
        grade_speed_factor=0.50,
    )
    other_slow = grade_aware_cruise(
        base,
        advice="slow",
        tilt_kind=KIND_OK,
        slow_speed_factor=0.35,
        grade_speed_factor=0.50,
    )
    assert ok == pytest.approx(0.45)
    assert grade == pytest.approx(0.225)
    assert other_slow == pytest.approx(0.45 * 0.35)
    assert grade < ok
    # 0.45 × 1.2 m/s ≈ 0.54 m/s explore cap
    assert 0.40 <= base * 1.2 <= 0.60


def test_acre_and_default_speed_caps() -> None:
    cfg = load_config()
    assert cfg.mission.explore_cruise == pytest.approx(0.45)
    assert 0.0 < cfg.planner.grade_speed_factor <= 1.0
    assert cfg.planner.grade_look_ahead_m >= 0.80
    acre, _ = load_source("acre_yard_demo")
    assert acre.world.resolution_m == pytest.approx(0.25)
    assert acre.mission.explore_cruise <= 0.60
    assert acre.planner.grade_look_ahead_m >= 0.80
    robot = EnvConfig().robot
    assert robot.tip_roll_rad == pytest.approx(0.55)
    assert robot.tip_pitch_rad == pytest.approx(0.55)
    assert robot.tip_roll_rad < robot.static_tip_roll_rad()
    assert robot.tip_pitch_rad < robot.static_tip_pitch_rad()


def test_gentle_hill_bank_drain_fixtures_do_not_hard_tip() -> None:
    gentle = HeightField.from_function(8.0, 6.0, 0.25, lambda x, y: math.tan(0.18) * x)
    pose = sit_on_terrain(Pose(4.0, 3.0, 0.0), gentle, 0.50, 0.40)
    ev = terrain_hazards(
        pose,
        gentle,
        length_m=0.50,
        track_m=0.40,
        tip_roll_rad=0.40,
        tip_pitch_rad=0.45,
        wheel_drop_m=0.08,
        steep_slope_rad=0.30,
        look_ahead_m=1.10,
        max_climb_slope_rad=0.32,
    )
    assert ev.tipover is False
    assert ev.advice in {"ok", "slow", "reroute"}
    assert ev.advice != "stop"

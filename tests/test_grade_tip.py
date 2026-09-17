"""Hill vs tip-stop: climbable grade must not thrash reverse; true tip must stop."""

from __future__ import annotations

import math

import numpy as np
import pytest

from jims_mower.config import EnvConfig
from jims_mower.constants import TERRAIN_DRAIN_EDGE
from jims_mower.env import MowerEnv
from jims_mower.kinematics import sit_on_terrain
from jims_mower.live import owner_copy_for
from jims_mower.mission_flow import MissionPhase, MissionPolicy
from jims_mower.planning.controller import TerrainPolicy, imu_advice
from jims_mower.planning.costmap import CONTOUR_COST, build_costmap
from jims_mower.planning.grade_tip import (
    KIND_GRADE,
    KIND_TIP,
    OWNER_STEEP_GRADE,
    OWNER_TIP_RISK,
    TipHoldFilter,
    classify_tilt,
    look_ahead_advice,
    probe_forward_grade,
    read_tilt,
    tip_lethal_slope_rad,
)
from jims_mower.profile import YardProfile
from jims_mower.safety import terrain_hazards
from jims_mower.scenarios import load_source
from jims_mower.terrain import HeightField
from jims_mower.types import Pose


def _cls(
    roll: float,
    pitch: float,
    *,
    max_climb: float = 0.32,
    tip_roll: float = 0.40,
    tip_pitch: float = 0.45,
) -> object:
    return classify_tilt(
        roll,
        pitch,
        tip_roll_rad=tip_roll,
        tip_pitch_rad=tip_pitch,
        slow_frac=0.55,
        stop_frac=0.85,
        max_climb_slope_rad=max_climb,
        tip_lethal_frac=0.95,
    )


def test_legacy_imu_advice_still_stops_at_stop_frac() -> None:
    imu = np.array([-8.0, 0.0, 5.6, 0.0, 0.0, 0.0], dtype=np.float32)
    assert imu_advice(
        imu,
        tip_roll_rad=0.40,
        tip_pitch_rad=0.45,
        slow_frac=0.55,
        stop_frac=0.85,
    ) == "stop"


def test_gentle_hill_is_grade_slow_not_tip() -> None:
    # ~15° pitch on an 18° climb cap — camera is tilted, chassis is not tipping.
    got = _cls(0.0, 0.26)
    assert got.kind == KIND_GRADE
    assert got.advice == "slow"
    assert got.owner_copy == OWNER_STEEP_GRADE


def test_max_climb_boundary_is_not_tip_stop() -> None:
    # acre_yard_demo max_climb 0.34 == old imu_stop_frac * tip_roll.
    got = _cls(0.0, 0.34, max_climb=0.34)
    assert got.kind == KIND_GRADE
    assert got.advice in {"slow", "reroute"}
    assert got.advice != "stop"


def test_past_climb_cap_is_tip_stop() -> None:
    got = _cls(0.0, 0.36, max_climb=0.32)
    assert got.kind == KIND_TIP
    assert got.advice == "stop"


def test_ridge_side_roll_near_tip_is_tip() -> None:
    got = _cls(0.39, 0.05, max_climb=0.32)
    assert got.kind == KIND_TIP
    assert got.advice == "stop"
    assert got.owner_copy == OWNER_TIP_RISK


def test_full_tip_trip_is_immediate_stop() -> None:
    got = _cls(0.41, 0.0)
    assert got.kind == KIND_TIP
    assert got.advice == "stop"


def test_spike_filter_does_not_emit_stop_on_one_frame() -> None:
    filt = TipHoldFilter(window=5, hold_steps=3)
    spike = classify_tilt(
        0.36,
        0.0,
        tip_roll_rad=0.40,
        tip_pitch_rad=0.45,
        slow_frac=0.55,
        stop_frac=0.85,
    )
    first = filt.update(spike, tip_roll_rad=0.40, tip_pitch_rad=0.45)
    assert first.advice != "stop"
    assert first.kind == KIND_GRADE


def test_filter_emits_stop_after_hold() -> None:
    filt = TipHoldFilter(window=3, hold_steps=3)
    spike = classify_tilt(
        0.36,
        0.0,
        tip_roll_rad=0.40,
        tip_pitch_rad=0.45,
        slow_frac=0.55,
        stop_frac=0.85,
    )
    last = None
    for _ in range(3):
        last = filt.update(spike, tip_roll_rad=0.40, tip_pitch_rad=0.45)
    assert last is not None
    assert last.advice == "stop"
    assert last.kind == KIND_TIP


def test_filter_full_tip_after_flat_frames_is_immediate() -> None:
    filt = TipHoldFilter(window=5, hold_steps=3)
    flat = classify_tilt(
        0.0,
        0.0,
        tip_roll_rad=0.40,
        tip_pitch_rad=0.45,
        slow_frac=0.55,
        stop_frac=0.85,
    )
    for _ in range(4):
        filt.update(flat, tip_roll_rad=0.40, tip_pitch_rad=0.45)
    hard = classify_tilt(
        0.45,
        0.0,
        tip_roll_rad=0.40,
        tip_pitch_rad=0.45,
        slow_frac=0.55,
        stop_frac=0.85,
    )
    got = filt.update(hard, tip_roll_rad=0.40, tip_pitch_rad=0.45)
    assert got.advice == "stop"
    assert got.kind == KIND_TIP


def test_filter_full_tip_is_immediate() -> None:
    filt = TipHoldFilter(window=5, hold_steps=8)
    hard = classify_tilt(
        0.42,
        0.0,
        tip_roll_rad=0.40,
        tip_pitch_rad=0.45,
        slow_frac=0.55,
        stop_frac=0.85,
    )
    got = filt.update(hard, tip_roll_rad=0.40, tip_pitch_rad=0.45)
    assert got.advice == "stop"
    assert got.kind == KIND_TIP


def test_bank_fixture_physics_slow_or_ok_not_tipover() -> None:
    grade = 0.20
    hf = HeightField.from_function(8.0, 6.0, 0.10, lambda x, y: math.tan(grade) * x)
    pose = sit_on_terrain(Pose(4.0, 3.0, 0.0), hf, 0.50, 0.40)
    ev = terrain_hazards(
        pose,
        hf,
        length_m=0.50,
        track_m=0.40,
        tip_roll_rad=0.40,
        tip_pitch_rad=0.45,
        wheel_drop_m=0.08,
        steep_slope_rad=0.30,
    )
    assert ev.tipover is False
    assert ev.advice in {"ok", "slow"}
    got = _cls(pose.roll, pose.pitch, max_climb=0.32)
    assert got.advice != "stop"
    assert got.kind != KIND_TIP


def test_ridge_fixture_high_roll_is_tip_risk() -> None:
    hf = HeightField.from_function(8.0, 6.0, 0.10, lambda x, y: 0.55 * y)
    pose = sit_on_terrain(Pose(4.0, 3.0, 0.0), hf, 0.50, 0.40)
    ev = terrain_hazards(
        pose,
        hf,
        length_m=0.50,
        track_m=0.40,
        tip_roll_rad=0.40,
        tip_pitch_rad=0.45,
        wheel_drop_m=0.08,
        steep_slope_rad=0.30,
    )
    assert ev.tipover is True
    assert ev.advice == "stop"
    got = _cls(pose.roll, pose.pitch)
    assert got.kind == KIND_TIP


def test_drain_lip_is_reroute_not_tip() -> None:
    hf = HeightField.empty(6.0, 6.0, 0.10)
    for row in range(hf.rows):
        for col in range(hf.cols):
            x = (col + 0.5) * 0.10
            if abs(x - 2.55) < 0.12:
                hf.labels[row, col] = TERRAIN_DRAIN_EDGE
    hf.recompute_slope()
    ev = terrain_hazards(
        Pose(2.0, 3.0, 0.0),
        hf,
        length_m=0.50,
        track_m=0.40,
        tip_roll_rad=0.40,
        tip_pitch_rad=0.45,
        wheel_drop_m=0.08,
        steep_slope_rad=0.30,
        look_ahead_m=0.55,
    )
    assert ev.advice == "reroute"
    assert ev.tipover is False


def test_tip_lethal_margin_matches_config() -> None:
    lethal = tip_lethal_slope_rad(tip_roll_rad=0.40, tip_pitch_rad=0.45, tip_lethal_frac=0.95)
    assert lethal == pytest.approx(0.38)
    hazard = np.zeros((12, 12), dtype=np.float32)
    slope = np.full((12, 12), 0.36, dtype=np.float32)
    cm = build_costmap(
        hazard,
        slope,
        resolution_m=0.2,
        width_m=2.4,
        height_m=2.4,
        max_climb_slope_rad=0.32,
        tip_lethal_slope_rad=lethal,
        drain_clearance_m=0.0,
        margin_m=0.0,
    )
    assert not bool(cm.blocked[6, 6])
    assert float(cm.cost[6, 6]) >= CONTOUR_COST


def test_owner_copy_distinguishes_tip_and_grade() -> None:
    assert owner_copy_for("running", "mow", tilt_kind="tip") == OWNER_TIP_RISK
    assert owner_copy_for("running", "mow", tilt_kind="grade") == OWNER_STEEP_GRADE


def test_terrain_policy_gentle_hill_does_not_recovery_loop() -> None:
    cfg = EnvConfig()
    policy = TerrainPolicy(cfg)
    policy.fusion.reset(0.0, 0.0, 0.0)
    from jims_mower.planning.coverage import CoveragePlan

    policy.plan = CoveragePlan(waypoints=[(4.0, 0.0)])
    policy.index = 0
    # Pitch at the old IMU stop (0.34) but under tip and at max_climb demo.
    pitch = 0.30
    imu = np.array(
        [-math.sin(pitch) * 9.81, 0.0, math.cos(pitch) * 9.81, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    obs = {
        "pose": np.array([0.0, 0.0, 0.0, 0.0, pitch, 0.0], dtype=np.float32),
        "gps": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "imu": imu,
        "hazard": np.zeros((16, 16), dtype=np.float32),
        "slope": np.full((16, 16), 0.22, dtype=np.float32),
        "occupancy": np.zeros((16, 16), dtype=np.float32),
        "coverage": np.zeros((16, 16), dtype=np.float32),
    }
    info = {
        "terrain_advice": "slow",
        "pose": {"x": 0.0, "y": 0.0, "theta": 0.0, "z": 0.0, "pitch": pitch, "roll": 0.0},
    }
    stops = 0
    for _ in range(12):
        action = policy.act(obs, info)
        if policy.last_advice == "stop":
            stops += 1
        assert policy.last_recovery != "help"
        assert abs(float(action[0])) < 1.1
    assert stops == 0
    assert policy.last_tilt_kind == KIND_GRADE
    assert policy.recoveries == 0


def test_mission_gentle_hill_no_tip_stamp_or_loop() -> None:
    cfg, scenario = load_source("mission_tiny")
    cfg.sensors.width = 32
    cfg.sensors.height = 24
    cfg.sensors.camera_count = 4
    cfg.sensors.cameras = []
    env = MowerEnv(config=cfg, scenario=scenario, render_mode=None)
    profile = YardProfile(
        name="grade_tiny",
        width_m=env.cfg.world.width_m,
        height_m=env.cfg.world.height_m,
        resolution_m=env.cfg.world.resolution_m,
        keep_in=[(0.7, 0.7), (5.0, 0.7), (5.0, 4.0), (0.7, 4.0)],
        home={"x": 1.2, "y": 1.2, "theta": 0.0},
    )
    obs, info = env.reset(seed=2, options={"yard_profile": profile, "resize_world": False})
    policy = MissionPolicy(env.cfg, fast=True)
    policy.reset(obs, info, profile=profile)
    pitch = 0.28
    imu = np.array(
        [-math.sin(pitch) * 9.81, 0.0, math.cos(pitch) * 9.81, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    tip_stops = 0
    stamps = 0
    for _ in range(20):
        info = dict(info)
        info["terrain_advice"] = "slow"
        info["pose"] = {**(info.get("pose") or {}), "pitch": pitch, "roll": 0.0}
        info["tipover"] = False
        obs = dict(obs)
        obs["imu"] = imu
        policy.act(obs, info)
        if policy.last_advice == "stop" and policy.last_tilt_kind == KIND_TIP:
            tip_stops += 1
        stamps = int(policy._blockage_events)
        if policy.phase != MissionPhase.EXPLORE:
            break
    env.close()
    assert tip_stops == 0
    assert stamps == 0
    assert policy.last_tilt_kind != KIND_TIP


def test_extreme_ramp_still_tips() -> None:
    hf = HeightField.from_function(6.0, 6.0, 0.1, lambda x, y: 1.2 * y)
    pose = sit_on_terrain(Pose(3.0, 3.0, 0.0), hf, 0.50, 0.40)
    ev = terrain_hazards(
        pose,
        hf,
        length_m=0.50,
        track_m=0.40,
        tip_roll_rad=0.40,
        tip_pitch_rad=0.45,
        wheel_drop_m=0.08,
        steep_slope_rad=0.30,
    )
    assert ev.tipover is True
    assert abs(pose.roll) >= 0.40


def test_read_tilt_prefers_worse_accel() -> None:
    imu = np.array([-3.2, 0.0, 9.2, 0.0, 0.0, 0.0], dtype=np.float32)
    roll, pitch = read_tilt(imu, pose_pitch=0.05, pose_roll=0.0)
    assert abs(pitch) > 0.20


def test_forward_probe_is_grade_or_reroute_not_seated_tip() -> None:
    hf = HeightField.from_function(
        8.0, 5.0, 0.25, lambda x, y: np.clip((x - 3.8) / 0.16, 0.0, 1.0) * 0.22
    )
    pose = sit_on_terrain(Pose(2.6, 2.5, 0.0), hf, 0.50, 0.40)
    assert abs(pose.pitch) < 0.20
    ahead = probe_forward_grade(
        pose,
        hf.sample,
        length_m=0.50,
        track_m=0.40,
        look_ahead_m=1.10,
        max_climb_slope_rad=0.32,
    )
    assert ahead.kind in {KIND_GRADE, KIND_TIP}
    assert look_ahead_advice(ahead) in {"slow", "reroute"}
    assert ahead.owner_copy in {OWNER_STEEP_GRADE, OWNER_TIP_RISK}

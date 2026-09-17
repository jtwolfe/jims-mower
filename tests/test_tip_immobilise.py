"""Past-tip chassis must latch SOS — never Idle Ready with tilt_kind=ok."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from jims_mower.env import MowerEnv
from jims_mower.kinematics import attitude_past_tip, sit_on_terrain, static_tip_latch
from jims_mower.scenarios import load_source
from jims_mower.live import (
    OWNER_COPY,
    LiveSession,
    owner_copy_for,
    snapshot_tilt_kind,
)
from jims_mower.planning.grade_tip import (
    KIND_GRADE,
    KIND_OK,
    KIND_TIP,
    look_ahead_advice,
    probe_forward_grade,
)
from jims_mower.safety import terrain_hazards
from jims_mower.terrain import HeightField
from jims_mower.types import Pose


def _ridge_field() -> HeightField:
    # Side bank: sit roll = atan(0.55) ≈ 0.50 > tip_roll 0.40.
    return HeightField.from_function(8.0, 6.0, 0.10, lambda x, y: 0.55 * y)


def _climbable_field() -> HeightField:
    return HeightField.from_function(
        8.0, 5.0, 0.25, lambda x, y: np.clip((x - 2.0) / 0.70, 0.0, 1.0) * 0.18
    )


def test_sit_on_ridge_exceeds_tip_roll() -> None:
    hf = _ridge_field()
    pose = sit_on_terrain(Pose(4.0, 3.0, 0.0), hf, 0.50, 0.40)
    assert abs(pose.roll) >= 0.40
    assert attitude_past_tip(pose.roll, pose.pitch, 0.40, 0.45)
    assert static_tip_latch(pose, tip_roll_rad=0.40, tip_pitch_rad=0.45) is True


def test_snapshot_tilt_kind_cannot_be_ok_when_past_tip() -> None:
    pose = {"x": 4.0, "y": 3.0, "theta": 0.0, "z": 0.0, "pitch": -0.275, "roll": -0.420}
    assert snapshot_tilt_kind(pose, last_kind="ok") == KIND_TIP
    assert snapshot_tilt_kind(pose, last_kind="grade") == KIND_TIP
    assert owner_copy_for("idle", "explore", tilt_kind="ok", tipped=True) == OWNER_COPY["immobilised"]
    assert owner_copy_for("idle", "explore", tilt_kind="tip") == OWNER_COPY["tip_risk"]
    assert "Ready" not in owner_copy_for("idle", "idle", tilt_kind="tip")
    assert "Ready" not in owner_copy_for("idle", "explore", tipped=True)


def test_live_snapshot_ridge_is_tip_not_ok_ready(tmp_path: Path) -> None:
    session = LiveSession(
        config="mission_tiny",
        fast=True,
        speed="max",
        steps=20,
        seed=3,
        cameras=4,
        out_dir=tmp_path / "tip-snap",
        cam_stride=80,
        map_stride=8,
        require_pair=False,
    )
    session.reset()
    session.job_state = "idle"
    assert session.policy is not None
    session.policy.last_tilt_kind = "ok"
    hf = _ridge_field()
    session.env._terrain = hf
    session.env._pose = sit_on_terrain(Pose(4.0, 3.0, 0.0), hf, 0.50, 0.40)
    session.obs, session.info = session.env._observe()
    session._record_pose()
    snap = session.snapshot()
    assert abs(float(snap["pose"]["roll"])) >= 0.40
    assert snap["tilt_kind"] == "tip"
    assert snap["tilt_kind"] != "ok"
    assert snap.get("chassis_tipped") is True
    assert "Ready" not in str(snap["mode_banner"]["label"])
    assert "immobilised" in str(snap["owner_copy"]).lower() or "sos" in str(snap["owner_copy"]).lower()
    assert "Idle ·" not in str(snap.get("owner_copy") or "")
    assert not (snap.get("explore_reason") or {}).get("label")
    assert any(f.get("code") == "FAULT_IMMOBILISED" for f in snap.get("faults") or [])
    assert snap["phase_label"] == "SOS"
    start = session.control("start")
    assert start.get("ok") is False
    assert "tipped" in str(start.get("reason") or start.get("error") or "").lower()
    session.close()


def test_env_tipover_latches_immobilised() -> None:
    cfg, scenario = load_source("mission_tiny")
    cfg.sensors.width = 32
    cfg.sensors.height = 24
    cfg.sensors.camera_count = 4
    cfg.sensors.cameras = []
    cfg_env = MowerEnv(config=cfg, scenario=scenario, render_mode=None)
    obs, info = cfg_env.reset(seed=1)
    hf = _ridge_field()
    cfg_env._terrain = hf
    cfg_env._pose = sit_on_terrain(Pose(4.0, 3.0, 0.0), hf, 0.50, 0.40)
    obs, _rew, terminated, _trunc, info = cfg_env.step(np.array([0.6, 0.6, 0.0], dtype=np.float32))
    assert info["tipover"] is True
    assert info["chassis_tipped"] is True
    assert info["fault"]["code"] == "FAULT_IMMOBILISED"
    assert info["fault"]["retrieve"] is True
    assert cfg_env.fault_bus.immobilised is True
    assert terminated is True
    start = (float(info["pose"]["x"]), float(info["pose"]["y"]))
    obs, _rew, _t, _tr, info2 = cfg_env.step(np.array([0.9, 0.9, 1.0], dtype=np.float32))
    end = (float(info2["pose"]["x"]), float(info2["pose"]["y"]))
    assert math.hypot(end[0] - start[0], end[1] - start[1]) < 1e-5
    assert info2["chassis_tipped"] is True
    cfg_env.close()


def test_look_ahead_stops_before_tip_bank() -> None:
    # Sharp face: sit pitch ≈ atan(0.40 / 0.50) = 0.67 > tip_pitch 0.45.
    hf = HeightField.from_function(
        10.0, 6.0, 0.25, lambda x, y: np.clip((x - 4.20) / 0.16, 0.0, 1.0) * 0.40
    )
    pose = sit_on_terrain(Pose(3.00, 3.0, 0.0), hf, 0.50, 0.40)
    assert abs(pose.pitch) < 0.32
    ahead = probe_forward_grade(
        pose,
        hf.sample,
        length_m=0.50,
        track_m=0.40,
        look_ahead_m=1.10,
        n_samples=6,
        max_climb_slope_rad=0.32,
    )
    assert ahead.past_tip is True
    assert look_ahead_advice(ahead, physics=True) == "stop"
    assert look_ahead_advice(ahead) == "reroute"
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
        max_climb_slope_rad=0.32,
    )
    assert ev.tipover is False
    assert ev.advice == "stop"


def test_look_ahead_refuses_forward_before_crossing() -> None:
    hf = HeightField.from_function(
        10.0, 6.0, 0.25, lambda x, y: np.clip((x - 4.20) / 0.16, 0.0, 1.0) * 0.40
    )
    pose = sit_on_terrain(Pose(3.00, 3.0, 0.0), hf, 0.50, 0.40)
    vmax = 1.2
    crossed = False
    for _ in range(30):
        ahead = probe_forward_grade(
            pose,
            hf.sample,
            length_m=0.50,
            track_m=0.40,
            look_ahead_m=1.10,
            n_samples=6,
            max_climb_slope_rad=0.32,
        )
        if look_ahead_advice(ahead, physics=True) == "stop" or ahead.past_tip:
            break
        from jims_mower.kinematics import integrate_pose

        pose = integrate_pose(pose, 0.45 * vmax, 0.45 * vmax, 0.40, 0.10, vmax)
        pose = sit_on_terrain(pose, hf, 0.50, 0.40)
        if abs(pose.pitch) >= 0.45 or abs(pose.roll) >= 0.40:
            crossed = True
            break
    assert crossed is False
    assert ahead.past_tip is True
    assert look_ahead_advice(ahead, physics=True) == "stop"


def test_mission_past_tip_holds_not_reverse() -> None:
    from jims_mower.constants import GRAVITY_MPS2
    from jims_mower.mission_flow import MissionPhase, MissionPolicy
    from jims_mower.profile import YardProfile

    cfg, scenario = load_source("mission_tiny")
    cfg.sensors.width = 32
    cfg.sensors.height = 24
    cfg.sensors.camera_count = 4
    cfg.sensors.cameras = []
    env = MowerEnv(config=cfg, scenario=scenario, render_mode=None)
    profile = YardProfile(
        name="tip_hold",
        width_m=env.cfg.world.width_m,
        height_m=env.cfg.world.height_m,
        resolution_m=env.cfg.world.resolution_m,
        keep_in=[(0.7, 0.7), (5.0, 0.7), (5.0, 4.0), (0.7, 4.0)],
        home={"x": 1.2, "y": 1.2, "theta": 0.0},
    )
    obs, info = env.reset(seed=2, options={"yard_profile": profile, "resize_world": False})
    policy = MissionPolicy(env.cfg, fast=True)
    policy.reset(obs, info, profile=profile)
    roll = 0.42
    g = float(GRAVITY_MPS2)
    info = dict(info)
    info["terrain_advice"] = "stop"
    info["pose"] = {**(info.get("pose") or {}), "roll": roll, "pitch": 0.0}
    info["tipover"] = False
    obs = dict(obs)
    obs["imu"] = np.array(
        [0.0, g * math.sin(roll), g * math.cos(roll), 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    action = policy.act(obs, info)
    env.close()
    assert float(action[0]) == 0.0 and float(action[1]) == 0.0
    assert policy.last_tilt_kind == KIND_TIP
    assert policy.help_requested is True
    assert policy.phase == MissionPhase.FAULT


def test_climbable_grade_still_contours() -> None:
    hf = _climbable_field()
    pose = sit_on_terrain(Pose(2.4, 2.5, 0.0), hf, 0.50, 0.40)
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
    assert ev.tipover is False
    assert ev.advice in {"ok", "slow", "reroute"}
    assert ev.advice != "stop"
    ahead = probe_forward_grade(
        pose,
        hf.sample,
        length_m=0.50,
        track_m=0.40,
        look_ahead_m=1.10,
        max_climb_slope_rad=0.34,
    )
    assert ahead.past_tip is False
    assert look_ahead_advice(ahead) != "stop"
    assert ahead.kind in {KIND_GRADE, KIND_TIP, KIND_OK}
    vmax = 1.2
    from jims_mower.kinematics import integrate_pose

    tipped = False
    for _ in range(40):
        pose = integrate_pose(pose, 0.45 * vmax, 0.45 * vmax, 0.40, 0.10, vmax)
        pose = sit_on_terrain(pose, hf, 0.50, 0.40)
        if abs(pose.pitch) >= 0.45 or abs(pose.roll) >= 0.40:
            tipped = True
            break
        if pose.x >= 6.2:
            break
    assert tipped is False

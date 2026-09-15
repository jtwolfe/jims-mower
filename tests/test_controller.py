"""Zero-turn controller maps terrain_advice and IMU tilt onto wheel speeds."""

from __future__ import annotations

import numpy as np
import pytest

from jims_mower.config import EnvConfig, load_config
from jims_mower.planning.controller import (
    TerrainPolicy,
    combine_advice,
    imu_advice,
    tracking_action,
)
from jims_mower.planning.coverage import CoveragePlan
from jims_mower.types import Pose


def test_combine_advice_priority() -> None:
    assert combine_advice("ok", "slow") == "slow"
    assert combine_advice("slow", "reroute") == "reroute"
    assert combine_advice("reroute", "stop") == "stop"
    assert combine_advice("ok") == "ok"


def test_tracking_turns_in_place_for_large_heading_error() -> None:
    pose = Pose(0.0, 0.0, 0.0)
    wheels, dist, err = tracking_action(
        pose,
        (0.0, 2.0),
        cruise=0.5,
        wheelbase_m=0.4,
        turn_in_place_rad=0.6,
    )
    assert dist == pytest.approx(2.0)
    assert abs(err) > 0.6
    assert wheels[0] * wheels[1] < 0.0


def test_controller_slows_on_steep_advice() -> None:
    cfg = EnvConfig()
    policy = TerrainPolicy(cfg)
    policy.fusion.reset(0.0, 0.0, 0.0)
    policy.plan = CoveragePlan(waypoints=[(3.0, 0.0)])
    policy.index = 0
    obs = {
        "pose": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "gps": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "imu": np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0], dtype=np.float32),
        "hazard": np.zeros((12, 12), dtype=np.float32),
        "slope": np.zeros((12, 12), dtype=np.float32),
        "occupancy": np.zeros((12, 12), dtype=np.float32),
        "coverage": np.zeros((12, 12), dtype=np.float32),
    }
    info_ok = {
        "terrain_advice": "ok",
        "pose": {"x": 0.0, "y": 0.0, "theta": 0.0, "z": 0.0, "pitch": 0.0, "roll": 0.0},
    }
    a_ok = policy.act(obs, info_ok)
    policy.fusion.reset(0.0, 0.0, 0.0)
    policy.index = 0
    policy._last_v = 0.0
    policy._last_omega = 0.0
    info_slow = dict(info_ok)
    info_slow["terrain_advice"] = "slow"
    a_slow = policy.act(obs, info_slow)
    assert max(abs(float(a_ok[0])), abs(float(a_ok[1]))) > 0.2
    assert max(abs(float(a_slow[0])), abs(float(a_slow[1]))) < max(
        abs(float(a_ok[0])), abs(float(a_ok[1]))
    ) * 0.7
    assert float(a_slow[2]) == pytest.approx(1.0)


def test_controller_stops_on_stop_advice() -> None:
    cfg = EnvConfig()
    policy = TerrainPolicy(cfg)
    policy.fusion.reset(0.0, 0.0, 0.0)
    policy.plan = CoveragePlan(waypoints=[(2.0, 0.0)])
    policy.index = 0
    obs = {
        "pose": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "gps": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "imu": np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0], dtype=np.float32),
    }
    info = {
        "terrain_advice": "stop",
        "pose": {"x": 0.0, "y": 0.0, "theta": 0.0, "z": 0.0, "pitch": 0.0, "roll": 0.0},
    }
    action = policy.act(obs, info)
    assert action[0] == pytest.approx(0.0)
    assert action[1] == pytest.approx(0.0)
    assert action[2] == pytest.approx(0.0)


def test_imu_cross_check_stop_on_tip_angle() -> None:
    # Large pitch from accel (specific force along -x).
    imu = np.array([-8.0, 0.0, 5.6, 0.0, 0.0, 0.0], dtype=np.float32)
    assert imu_advice(
        imu,
        tip_roll_rad=0.40,
        tip_pitch_rad=0.45,
        slow_frac=0.55,
        stop_frac=0.85,
    ) == "stop"


def test_imu_cross_check_slow_on_moderate_tilt() -> None:
    imu = np.array([-3.2, 0.0, 9.2, 0.0, 0.0, 0.0], dtype=np.float32)
    advice = imu_advice(
        imu,
        tip_roll_rad=0.40,
        tip_pitch_rad=0.45,
        slow_frac=0.55,
        stop_frac=0.85,
    )
    assert advice in {"slow", "stop"}


def test_controller_imu_stop_overrides_ok() -> None:
    cfg = load_config()
    policy = TerrainPolicy(cfg)
    policy.fusion.reset(0.0, 0.0, 0.0)
    policy.plan = CoveragePlan(waypoints=[(2.0, 0.0)])
    policy.index = 0
    imu = np.array([-8.0, 0.0, 5.6, 0.0, 0.0, 0.0], dtype=np.float32)
    obs = {
        "pose": np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0], dtype=np.float32),
        "gps": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "imu": imu,
    }
    info = {
        "terrain_advice": "ok",
        "pose": {"x": 0.0, "y": 0.0, "theta": 0.0, "z": 0.0, "pitch": 0.50, "roll": 0.0},
    }
    action = policy.act(obs, info)
    assert action[0] == pytest.approx(0.0)
    assert action[1] == pytest.approx(0.0)
    assert policy.last_advice == "stop"

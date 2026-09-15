"""Recovery: reverse off a lip, pivot, then call-for-help on repeated stop."""

from __future__ import annotations

import numpy as np
import pytest

from jims_mower.config import EnvConfig
from jims_mower.planning.controller import TerrainPolicy
from jims_mower.planning.coverage import CoveragePlan


def _obs() -> dict:
    return {
        "pose": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "gps": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "imu": np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0], dtype=np.float32),
        "hazard": np.zeros((12, 12), dtype=np.float32),
        "slope": np.zeros((12, 12), dtype=np.float32),
        "occupancy": np.zeros((12, 12), dtype=np.float32),
        "coverage": np.zeros((12, 12), dtype=np.float32),
        "hand_signal": 0,
    }


def _info(advice: str = "stop") -> dict:
    return {
        "terrain_advice": advice,
        "living_advice": "ok",
        "geofence_advice": "ok",
        "pose": {"x": 0.0, "y": 0.0, "theta": 0.0, "z": 0.0, "pitch": 0.0, "roll": 0.0},
    }


def _policy() -> TerrainPolicy:
    cfg = EnvConfig()
    cfg.planner.recovery_trigger = 3
    cfg.planner.recovery_reverse_steps = 2
    cfg.planner.recovery_pivot_steps = 2
    cfg.planner.max_recoveries = 1
    policy = TerrainPolicy(cfg)
    policy.fusion.reset(0.0, 0.0, 0.0)
    policy.plan = CoveragePlan(waypoints=[(3.0, 0.0)])
    policy.index = 0
    return policy


def test_first_stop_still_holds() -> None:
    policy = _policy()
    action = policy.act(_obs(), _info("stop"))
    assert action[0] == pytest.approx(0.0)
    assert action[1] == pytest.approx(0.0)
    assert policy.last_recovery == "idle"


def test_repeated_stop_reverses_then_pivots() -> None:
    policy = _policy()
    obs = _obs()
    modes = []
    wheels = []
    for _ in range(8):
        action = policy.act(obs, _info("stop"))
        modes.append(policy.last_recovery)
        wheels.append((float(action[0]), float(action[1])))
    assert "reverse" in modes
    assert "pivot" in modes
    rev = next(w for w, m in zip(wheels, modes) if m == "reverse")
    assert rev[0] < 0.0 and rev[1] < 0.0
    piv = next(w for w, m in zip(wheels, modes) if m == "pivot")
    assert piv[0] * piv[1] < 0.0


def test_exhausted_recoveries_call_for_help() -> None:
    policy = _policy()
    obs = _obs()
    for _ in range(16):
        policy.act(obs, _info("stop"))
    assert policy.help_requested is True
    assert policy.last_recovery == "help"
    hold = policy.act(obs, _info("ok"))
    assert hold[0] == pytest.approx(0.0)
    assert hold[1] == pytest.approx(0.0)

"""ESTOP / limp / safe-state machine and controller wiring."""

from __future__ import annotations

import numpy as np
import pytest

from jims_mower.config import EnvConfig
from jims_mower.planning.controller import TerrainPolicy
from jims_mower.planning.coverage import CoveragePlan
from jims_mower.safe_state import SafeStateMachine, estop_requested


def test_estop_latches_until_clear() -> None:
    sm = SafeStateMachine(limp_after_stops=8)
    sm.tick(advice="ok", estop=True)
    cmd = sm.command()
    assert cmd.mode == "estop"
    assert cmd.hold is True
    sm.tick(advice="ok", estop=False)
    assert sm.mode == "estop"
    sm.clear()
    assert sm.mode == "run"


def test_repeated_stop_enters_limp_then_safe() -> None:
    sm = SafeStateMachine(limp_after_stops=3, safe_after_limp_steps=2, recover_ok_steps=4)
    for _ in range(3):
        sm.tick(advice="stop")
    assert sm.mode == "limp"
    out = sm.apply(np.array([0.8, 0.8, 1.0], dtype=np.float32))
    assert out[2] == pytest.approx(0.0)
    assert max(abs(float(out[0])), abs(float(out[1]))) < 0.8
    sm.tick(advice="stop")
    sm.tick(advice="stop")
    assert sm.mode == "safe"
    hold = sm.apply(np.array([0.5, 0.5, 1.0], dtype=np.float32))
    assert hold[0] == pytest.approx(0.0)
    assert hold[1] == pytest.approx(0.0)


def test_limp_recovers_after_ok_ticks() -> None:
    sm = SafeStateMachine(limp_after_stops=2, recover_ok_steps=2)
    sm.tick(advice="stop")
    sm.tick(advice="stop")
    assert sm.mode == "limp"
    sm.tick(advice="ok")
    sm.tick(advice="ok")
    assert sm.mode == "run"


def test_go_does_not_clear_estop() -> None:
    sm = SafeStateMachine()
    sm.request_estop("button")
    sm.clear_if_not_estop()
    assert sm.mode == "estop"


def test_estop_requested_from_info() -> None:
    assert estop_requested(None, {"estop": True})
    assert not estop_requested({}, {"estop": False})


def test_controller_estop_zeros_wheels() -> None:
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
        "hand_signal": 0,
    }
    info = {
        "terrain_advice": "ok",
        "living_advice": "ok",
        "geofence_advice": "ok",
        "estop": True,
        "pose": {"x": 0.0, "y": 0.0, "theta": 0.0, "z": 0.0, "pitch": 0.0, "roll": 0.0},
    }
    action = policy.act(obs, info)
    assert action[0] == pytest.approx(0.0)
    assert action[1] == pytest.approx(0.0)
    assert action[2] == pytest.approx(0.0)
    assert policy.last_safe_mode == "estop"
    info2 = dict(info)
    info2["estop"] = False
    still = policy.act(obs, info2)
    assert still[0] == pytest.approx(0.0)
    assert policy.safe.mode == "estop"

"""Hand-signal curriculum maps onto controller overrides (mock/oracle labels)."""

from __future__ import annotations

import numpy as np
import pytest

from jims_mower.config import EnvConfig
from jims_mower.constants import SIGNAL_TO_ID
from jims_mower.planning.controller import TerrainPolicy, observed_hand_signal
from jims_mower.planning.coverage import CoveragePlan


def _obs(signal: int = 0) -> dict:
    return {
        "pose": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "gps": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "imu": np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0], dtype=np.float32),
        "hazard": np.zeros((12, 12), dtype=np.float32),
        "slope": np.zeros((12, 12), dtype=np.float32),
        "occupancy": np.zeros((12, 12), dtype=np.float32),
        "coverage": np.zeros((12, 12), dtype=np.float32),
        "hand_signal": signal,
    }


def _info() -> dict:
    return {
        "terrain_advice": "ok",
        "living_advice": "ok",
        "geofence_advice": "ok",
        "pose": {"x": 0.0, "y": 0.0, "theta": 0.0, "z": 0.0, "pitch": 0.0, "roll": 0.0},
        "detections": [
            {"label": "person", "world_xy": [2.0, 0.5], "camera": "front", "bbox": [1, 1, 4, 4]}
        ],
    }


def _policy(*, enabled: bool = True) -> TerrainPolicy:
    cfg = EnvConfig()
    cfg.curriculum.hand_signals = enabled
    policy = TerrainPolicy(cfg)
    policy.fusion.reset(0.0, 0.0, 0.0)
    policy.plan = CoveragePlan(waypoints=[(4.0, 0.0)])
    policy.index = 0
    return policy


def test_observed_signal_respects_curriculum_flag() -> None:
    obs = _obs(SIGNAL_TO_ID["stop"])
    assert observed_hand_signal(obs, True) == "stop"
    assert observed_hand_signal(obs, False) is None


def test_stop_signal_zeros_wheels() -> None:
    policy = _policy()
    action = policy.act(_obs(SIGNAL_TO_ID["stop"]), _info())
    assert action[0] == pytest.approx(0.0)
    assert action[1] == pytest.approx(0.0)
    assert action[2] == pytest.approx(0.0)
    assert policy.last_signal == "stop"


def test_back_signal_reverses() -> None:
    policy = _policy()
    action = policy.act(_obs(SIGNAL_TO_ID["back"]), _info())
    assert float(action[0]) < 0.0
    assert float(action[1]) < 0.0
    assert action[2] == pytest.approx(0.0)
    assert policy.last_signal == "back"


def test_follow_tracks_person_xy() -> None:
    policy = _policy()
    action = policy.act(_obs(SIGNAL_TO_ID["follow"]), _info())
    # Person is ahead and slightly left — both wheels forward-ish, not a hold.
    assert max(abs(float(action[0])), abs(float(action[1]))) > 0.05
    assert policy.last_signal == "follow"


def test_go_clears_help_and_resumes() -> None:
    policy = _policy()
    policy.help_requested = True
    action = policy.act(_obs(SIGNAL_TO_ID["go"]), _info())
    assert policy.help_requested is False
    assert max(abs(float(action[0])), abs(float(action[1]))) > 0.05


def test_disabled_curriculum_ignores_signal_id() -> None:
    policy = _policy(enabled=False)
    action = policy.act(_obs(SIGNAL_TO_ID["stop"]), _info())
    # Coverage continues toward (4, 0).
    assert max(abs(float(action[0])), abs(float(action[1]))) > 0.05
    assert policy.last_signal is None

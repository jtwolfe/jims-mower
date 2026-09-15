"""Orin-class battery / thermal stub limps the terrain policy."""

from __future__ import annotations

import numpy as np
import pytest

from jims_mower.config import EnvConfig, load_config
from jims_mower.env import MowerEnv
from jims_mower.planning.controller import TerrainPolicy
from jims_mower.planning.coverage import CoveragePlan
from jims_mower.runtime.budget import OrinBudget, budget_from_config


def _obs() -> dict:
    return {
        "pose": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "gps": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "imu": np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0], dtype=np.float32),
        "hazard": np.zeros((12, 12), dtype=np.float32),
        "slope": np.zeros((12, 12), dtype=np.float32),
        "occupancy": np.zeros((12, 12), dtype=np.float32),
        "coverage": np.zeros((12, 12), dtype=np.float32),
    }


def _info(budget: str = "ok") -> dict:
    return {
        "terrain_advice": "ok",
        "budget_advice": budget,
        "pose": {"x": 0.0, "y": 0.0, "theta": 0.0, "z": 0.0, "pitch": 0.0, "roll": 0.0},
    }


def test_budget_limps_when_hot_or_low() -> None:
    hot = OrinBudget(t_c=80.0, t_hot_c=75.0, t_crit_c=90.0, soc=1.0)
    assert hot.advice() == "slow"
    assert "thermal_hot" in hot.reason()
    empty = OrinBudget(soc=0.04, stop_soc=0.05, limp_soc=0.15, t_c=40.0)
    assert empty.advice() == "stop"
    assert "battery_empty" in empty.reason()
    ok = OrinBudget(soc=0.9, t_c=40.0)
    assert ok.advice() == "ok"


def test_budget_drains_on_drive() -> None:
    b = OrinBudget(capacity_wh=0.5, soc=1.0, t_c=40.0, idle_w=8.0, drive_w=40.0)
    before = b.soc
    b.step(1.0, np.array([1.0, 1.0, 1.0], dtype=np.float32), n_cameras=6)
    assert b.soc < before
    assert b.t_c > 40.0


def test_policy_limps_and_stops_on_budget() -> None:
    cfg = EnvConfig()
    policy = TerrainPolicy(cfg)
    policy.fusion.reset(0.0, 0.0, 0.0)
    policy.plan = CoveragePlan(waypoints=[(3.0, 0.0)])
    policy.index = 0
    obs = _obs()
    a_ok = policy.act(obs, _info("ok"))
    policy.fusion.reset(0.0, 0.0, 0.0)
    policy.index = 0
    policy._last_v = 0.0
    policy._last_omega = 0.0
    a_limp = policy.act(obs, _info("slow"))
    assert policy.last_budget == "slow"
    assert max(abs(float(a_limp[0])), abs(float(a_limp[1]))) < max(
        abs(float(a_ok[0])), abs(float(a_ok[1]))
    ) * 0.7
    policy.fusion.reset(0.0, 0.0, 0.0)
    policy.index = 0
    a_stop = policy.act(obs, _info("stop"))
    assert float(a_stop[0]) == pytest.approx(0.0)
    assert float(a_stop[1]) == pytest.approx(0.0)


def test_env_budget_info_when_enabled() -> None:
    cfg = {
        "sensors": {"width": 16, "height": 12, "camera_count": 4},
        "world": {
            "width_m": 8.0,
            "height_m": 8.0,
            "resolution_m": 0.25,
            "n_people": 0,
            "n_dogs": 0,
            "n_cats": 0,
            "n_birds": 0,
            "n_trees": 0,
            "n_furniture": 0,
            "n_toys": 0,
            "terrain": {"enabled": False},
        },
        "runtime": {
            "enabled": True,
            "battery": {"soc": 0.04, "stop_soc": 0.05, "limp_soc": 0.15},
            "thermal": {"t_c": 40.0},
        },
        "perception": {"terrain_mode": "blind"},
    }
    env = MowerEnv(config=cfg, render_mode=None)
    _obs_i, info = env.reset(seed=1)
    assert info["budget_enabled"] is True
    assert info["not_a_power_trace"] is True
    assert info["budget_advice"] == "stop"
    assert 0.0 <= info["battery_soc"] <= 1.0
    env.close()


def test_budget_from_default_config_disabled() -> None:
    cfg = load_config()
    assert cfg.runtime.enabled is False
    b = budget_from_config(cfg)
    assert b.enabled is False
    assert b.advice() == "ok"

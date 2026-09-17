"""FaultBus: dead motor → immobilised + SOS; stuck still recovers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jims_mower.config import EnvConfig, load_config
from jims_mower.env import MowerEnv
from jims_mower.episode import record_episode
from jims_mower.faults import CODE_IMMOBILISED, CODE_STUCK, FaultBus
from jims_mower.incident import write_incident_viewer
from jims_mower.planning.controller import TerrainPolicy
from jims_mower.planning.coverage import CoveragePlan
from jims_mower.telemetry import summarize_telemetry, telemetry_from_episode


def _tiny() -> dict:
    return {
        "dt": 0.1,
        "max_steps": 30,
        "sensors": {
            "width": 32,
            "height": 24,
            "camera_count": 4,
            "fov_deg": 70.0,
            "gps": {"dropout_prob": 0.0},
        },
        "world": {
            "width_m": 8.0,
            "height_m": 8.0,
            "resolution_m": 0.20,
            "n_people": 0,
            "n_dogs": 0,
            "n_cats": 0,
            "n_birds": 0,
            "n_trees": 0,
            "n_furniture": 0,
            "n_toys": 0,
            "terrain": {"enabled": False},
        },
        "perception": {"terrain_mode": "oracle"},
        "robot": {
            "trimmer": {"safety_radius_m": 1.2, "offset_m": 0.32, "radius_m": 0.16}
        },
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


def _info(advice: str = "stop", fault: dict | None = None) -> dict:
    blob = {
        "terrain_advice": advice,
        "living_advice": "ok",
        "geofence_advice": "ok",
        "pose": {"x": 0.0, "y": 0.0, "theta": 0.0, "z": 0.0, "pitch": 0.0, "roll": 0.0},
    }
    if fault is not None:
        blob["fault"] = fault
    return blob


@pytest.mark.parametrize("side", ["motor_left", "motor_right"])
@pytest.mark.parametrize("mode", ["open_circuit", "cmd_ignored", "encoder_stuck"])
def test_dead_motor_immobilised_and_sos(side: str, mode: str) -> None:
    env = MowerEnv(config=_tiny())
    obs, info = env.reset(seed=2)
    start = (float(info["pose"]["x"]), float(info["pose"]["y"]))
    env.fault_bus.inject(side, mode=mode)
    obs, _rew, _term, _trunc, info = env.step(np.array([0.85, 0.85, 1.0], dtype=np.float32))
    fault = info["fault"]
    assert fault["code"] == CODE_IMMOBILISED
    assert fault["retrieve"] is True
    assert fault["component"] in {"drive_left", "drive_right"}
    assert fault["mode"] == mode
    assert "x" in fault["pose"]
    assert info["trimmer_enabled"] is False
    end = (float(info["pose"]["x"]), float(info["pose"]["y"]))
    assert math_hypot(end, start) < 1e-5
    env.close()


def math_hypot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def test_dead_motor_does_not_run_recovery() -> None:
    policy = _policy()
    fault = {
        "code": CODE_IMMOBILISED,
        "component": "drive_left",
        "pose": {"x": 0.0, "y": 0.0, "theta": 0.0},
        "retrieve": True,
        "immobilised": True,
    }
    modes = []
    for _ in range(8):
        action = policy.act(_obs(), _info("stop", fault))
        modes.append(policy.last_recovery)
        assert action[0] == pytest.approx(0.0)
        assert action[1] == pytest.approx(0.0)
        assert action[2] == pytest.approx(0.0)
    assert "reverse" not in modes
    assert "pivot" not in modes
    assert policy.help_requested is True


def test_stuck_still_uses_recovery() -> None:
    policy = _policy()
    fault = {
        "code": CODE_STUCK,
        "component": "chassis",
        "pose": {"x": 0.0, "y": 0.0, "theta": 0.0},
        "retrieve": False,
        "immobilised": False,
        "stuck": True,
    }
    modes = []
    wheels = []
    for _ in range(8):
        action = policy.act(_obs(), _info("stop", fault))
        modes.append(policy.last_recovery)
        wheels.append((float(action[0]), float(action[1])))
    assert "reverse" in modes
    assert "pivot" in modes
    rev = next(w for w, m in zip(wheels, modes) if m == "reverse")
    assert rev[0] < 0.0 and rev[1] < 0.0


def test_fault_bus_unifies_sensor_faults() -> None:
    env = MowerEnv(config=_tiny())
    env.reset(seed=1)
    env.fault_bus.inject("cam_blind")
    obs, _r, _t, _tr, info = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    for frame in obs["cameras"].values():
        assert int(np.asarray(frame).max()) == 0
    assert info["fault"]["code"] == "CAM_BLIND"

    env.reset(seed=1)
    env.fault_bus.inject("imu_freeze")
    obs_a, *_ = env.step(np.array([0.4, 0.4, 0.0], dtype=np.float32))
    imu_a = np.asarray(obs_a["imu"]).copy()
    obs_b, *_rest, info_b = env.step(np.array([-0.4, 0.4, 0.0], dtype=np.float32))
    assert np.allclose(obs_b["imu"], imu_a)
    assert info_b["fault"]["code"] == "IMU_FREEZE"

    env.reset(seed=1)
    env.fault_bus.inject("gnss_dropout")
    obs, *_rest, info = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert float(obs["gps"][3]) == pytest.approx(0.0)
    assert info["fault"]["code"] == "GNSS_DROPOUT"

    env.reset(seed=1)
    env.fault_bus.inject("trimmer_jam")
    _obs2, *_rest, info = env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
    assert info["trimmer_enabled"] is False
    assert info["fault"]["code"] == "TRIMMER_JAM"
    env.close()


def test_scheduled_inject_from_config() -> None:
    cfg = _tiny()
    cfg["faults"] = {
        "enabled": True,
        "inject": [{"at_step": 1, "kind": "motor_right", "mode": "cmd_ignored"}],
    }
    env = MowerEnv(config=cfg)
    env.reset(seed=0)
    _obs, _r, _t, _tr, info0 = env.step(np.array([0.5, 0.5, 0.0], dtype=np.float32))
    assert info0["fault"]["code"] == "ok"
    _obs, _r, _t, _tr, info1 = env.step(np.array([0.5, 0.5, 0.0], dtype=np.float32))
    assert info1["fault"]["code"] == CODE_IMMOBILISED
    assert info1["fault"]["retrieve"] is True
    env.close()


def test_telemetry_includes_sos(tmp_path: Path) -> None:
    steps = [
        {
            "info": {
                "terrain_advice": "ok",
                "coverage_fraction": 0.1,
                "fault": {
                    "code": CODE_IMMOBILISED,
                    "component": "drive_left",
                    "pose": {"x": 1.0, "y": 2.0, "theta": 0.1},
                    "retrieve": True,
                },
            }
        }
    ]
    tel = summarize_telemetry(steps=steps, seed=0, scenario="flat", policy="scripted")
    assert tel["sos"]["active"] is True
    assert tel["sos"]["retrieve"] is True
    assert tel["sos"]["code"] == CODE_IMMOBILISED
    assert tel["sos_steps"] == 1
    assert tel["not_a_benchmark"] is True


def test_incident_scrubber_fault_banner(tmp_path: Path) -> None:
    cfg = _tiny()
    cfg["faults"] = {
        "enabled": True,
        "inject": [{"at_step": 0, "kind": "motor_left", "mode": "open_circuit"}],
    }
    ep = tmp_path / "ep"
    record_episode(ep, steps=2, seed=2, config=cfg, cameras=4, policy="scripted")
    view = tmp_path / "view"
    index = write_incident_viewer(ep, view)
    html = (view / "index.html").read_text(encoding="utf-8")
    assert "fault-banner" in html
    assert any(rec.get("sos") for rec in index["frames"])
    tel = telemetry_from_episode(ep)
    assert tel["sos"]["retrieve"] is True
    payload = json.loads((view / "index.json").read_text(encoding="utf-8"))
    assert payload["not_a_benchmark"] is True


def test_fault_defaults_off() -> None:
    cfg = load_config()
    assert cfg.faults.enabled is False
    assert cfg.faults.inject == []
    bus = FaultBus.from_config(cfg)
    assert bus.immobilised is False
    report = bus.report()
    assert report.code == "ok"
    assert report.retrieve is False


def test_chassis_tipover_latches_immobilised() -> None:
    bus = FaultBus()
    bus.latch_tipover()
    assert bus.immobilised is True
    assert bus.chassis_tipped is True
    blob = bus.as_info()
    assert blob["code"] == CODE_IMMOBILISED
    assert blob["retrieve"] is True
    assert blob["component"] == "chassis"
    assert blob["chassis_tipped"] is True
    left, right, trim = bus.apply_drive(0.8, 0.8, True)
    assert left == 0.0 and right == 0.0 and trim is False
    bus.clear_tipover()
    assert bus.immobilised is False
    assert bus.as_info()["code"] == "ok"

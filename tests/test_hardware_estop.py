"""Hardware ESTOP rail latch + bench watchdog (gym). Not a field paddle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jims_mower.config import load_config
from jims_mower.env import MowerEnv
from jims_mower.hardware_estop import HardwareEstop, estop_kind
from jims_mower.live import LiveSession
from jims_mower.planning.controller import TerrainPolicy
from jims_mower.planning.coverage import CoveragePlan
from jims_mower.runtime.watchdog import SensorWatchdog
from jims_mower.safe_state import SafeStateMachine


def _tiny() -> dict:
    return {
        "dt": 0.1,
        "max_steps": 40,
        "sensors": {
            "width": 32,
            "height": 24,
            "camera_count": 4,
            "fov_deg": 70.0,
            "gps": {"dropout_prob": 0.0},
            "imu": {"accel_noise_std": 0.0, "gyro_noise_std": 0.0, "accel_bias_std": 0.0},
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
    }


def _pose_xy(info: dict) -> tuple[float, float]:
    return float(info["pose"]["x"]), float(info["pose"]["y"])


def _moved(a: tuple[float, float], b: tuple[float, float], thresh: float = 0.02) -> bool:
    return float(np.hypot(a[0] - b[0], a[1] - b[1])) > thresh


def test_hardware_estop_unit_zeros_and_needs_explicit_reset() -> None:
    hw = HardwareEstop()
    left, right, trim = hw.apply(0.8, -0.4, True)
    assert left == pytest.approx(0.8)
    assert trim is True
    hw.hit("paddle")
    left, right, trim = hw.apply(0.8, -0.4, True)
    assert (left, right, trim) == (0.0, 0.0, False)
    held = hw.filter_action(np.array([0.9, 0.9, 1.0], dtype=np.float32))
    assert float(np.max(np.abs(held))) == 0.0
    sm = SafeStateMachine()
    sm.request_estop("owner")
    sm.clear()
    assert sm.mode == "run"
    still = hw.filter_action(np.array([0.5, 0.5, 1.0], dtype=np.float32))
    assert float(np.max(np.abs(still))) == 0.0
    assert hw.latched is True
    hw.reset()
    live = hw.filter_action(np.array([0.5, 0.5, 1.0], dtype=np.float32))
    assert live[0] == pytest.approx(0.5)
    assert estop_kind(software=True, hardware=True) == "both"
    assert estop_kind(software=False, hardware=True) == "hardware"


def test_dummy_load_hw_estop_zeros_rails_software_clear_does_not_restore() -> None:
    env = MowerEnv(config=_tiny())
    _, info = env.reset(seed=2)
    start = _pose_xy(info)
    for _ in range(4):
        _, _, _, _, info = env.step(np.array([0.85, 0.85, 1.0], dtype=np.float32))
    mid = _pose_xy(info)
    assert _moved(start, mid), "dummy load should move before the paddle"

    env.hit_hw_estop("bench paddle")
    latched = _pose_xy(info)
    for _ in range(5):
        _, _, _, _, info = env.step(np.array([0.95, 0.95, 1.0], dtype=np.float32))
    after = _pose_xy(info)
    assert not _moved(latched, after, thresh=1e-4)
    assert info["hw_estop"] is True
    assert info["hw_estop_rails"] == "dead"
    assert info["trimmer_enabled"] is False
    assert info["hw_estop_applied"] == [0.0, 0.0, 0.0]

    # Software ESTOP latch + clear must not restore rails.
    sm = SafeStateMachine()
    sm.request_estop("owner")
    sm.clear()
    assert sm.mode == "run"
    held = _pose_xy(info)
    for _ in range(4):
        _, _, _, _, info = env.step(np.array([0.95, 0.95, 1.0], dtype=np.float32))
    assert not _moved(held, _pose_xy(info), thresh=1e-4)
    assert env.hw_estop.latched is True

    env.reset_hw_estop()
    freed = _pose_xy(info)
    for _ in range(5):
        _, _, _, _, info = env.step(np.array([0.85, 0.85, 1.0], dtype=np.float32))
    assert _moved(freed, _pose_xy(info))
    assert info["hw_estop"] is False
    env.close()


def test_policy_output_cannot_spin_motors_while_hw_latched() -> None:
    env = MowerEnv(config=_tiny())
    obs, info = env.reset(seed=3)
    policy = TerrainPolicy(env.cfg)
    policy.fusion.reset(0.0, 0.0, 0.0)
    policy.plan = CoveragePlan(waypoints=[(3.0, 0.0)])
    policy.index = 0
    env.hit_hw_estop("policy test")
    start = _pose_xy(info)
    for _ in range(4):
        action = policy.act(obs, info)
        if policy.safe.mode == "estop":
            policy.safe.clear()
            action = policy.act(obs, info)
        if max(abs(float(action[0])), abs(float(action[1]))) < 0.05:
            action = np.array([0.85, 0.85, 1.0], dtype=np.float32)
        rails = env.hw_estop.filter_action(action)
        assert float(np.max(np.abs(rails))) == 0.0
        obs, _, _, _, info = env.step(action)
    assert not _moved(start, _pose_xy(info), thresh=1e-4)
    assert info["hw_estop"] is True
    env.close()


def test_inject_paddle_and_reset_mode() -> None:
    env = MowerEnv(config=_tiny())
    env.reset(seed=4)
    env.inject_fault("paddle")
    _, _, _, _, info = env.step(np.array([0.7, 0.7, 1.0], dtype=np.float32))
    assert info["hw_estop"] is True
    env.inject_fault("hw_estop", mode="reset")
    assert env.hw_estop.latched is False
    env.close()


def test_watchdog_frozen_imu_stamp_zeros_wheels_in_gym() -> None:
    cfg = dict(_tiny())
    cfg["runtime"] = {
        "watchdog": {"enabled": True, "imu_stall_s": 0.20, "vision_stall_s": 9.0}
    }
    env = MowerEnv(config=cfg)
    _, info = env.reset(seed=5)
    assert info["watchdog_enabled"] is True
    start = _pose_xy(info)
    for _ in range(3):
        _, _, _, _, info = env.step(np.array([0.8, 0.8, 0.0], dtype=np.float32))
    assert _moved(start, _pose_xy(info))
    env.fault_bus.inject("imu")
    # First observe after freeze still has the last live stamp; then it holds.
    for _ in range(6):
        _, _, _, _, info = env.step(np.array([0.8, 0.8, 1.0], dtype=np.float32))
    assert info["watchdog_stalled"] is True
    assert info["watchdog_reason"] == "imu_stall"
    held = _pose_xy(info)
    for _ in range(3):
        _, _, _, _, info = env.step(np.array([0.9, 0.9, 1.0], dtype=np.float32))
    assert not _moved(held, _pose_xy(info), thresh=1e-4)
    env.close()


def test_watchdog_frozen_camera_stamp_zeros_wheels_in_gym() -> None:
    cfg = dict(_tiny())
    cfg["runtime"] = {
        "watchdog": {"enabled": True, "imu_stall_s": 9.0, "vision_stall_s": 0.20}
    }
    env = MowerEnv(config=cfg)
    _, info = env.reset(seed=6)
    env.fault_bus.inject("cam_blind")
    for _ in range(6):
        _, _, _, _, info = env.step(np.array([0.8, 0.8, 1.0], dtype=np.float32))
    assert info["watchdog_stalled"] is True
    assert info["watchdog_reason"] == "vision_stall"
    held = _pose_xy(info)
    for _ in range(3):
        _, _, _, _, info = env.step(np.array([0.9, 0.9, 1.0], dtype=np.float32))
    assert not _moved(held, _pose_xy(info), thresh=1e-4)
    env.close()


def test_watchdog_stamp_api_without_fingerprint() -> None:
    wd = SensorWatchdog(enabled=True, imu_stall_s=0.15, vision_stall_s=0.15, dt=0.10)
    imu = np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.1], dtype=np.float32)
    cams = {"front": np.zeros((4, 4, 3), dtype=np.uint8)}
    wd.observe(imu, cams, imu_stamp_s=0.0, vision_stamp_s=0.0)
    wd.observe(imu + 1.0, cams, imu_stamp_s=0.0, vision_stamp_s=0.1)
    wd.observe(imu + 2.0, cams, imu_stamp_s=0.0, vision_stamp_s=0.2)
    assert wd.stalled
    assert wd.reason == "imu_stall"
    held = wd.filter_action(np.array([0.4, 0.4, 1.0], dtype=np.float32))
    assert float(np.max(np.abs(held))) == 0.0


def test_live_distinguishes_software_and_hardware_and_start_does_not_reset_paddle(
    tmp_path: Path,
) -> None:
    session = LiveSession(
        config="mission_tiny",
        fast=True,
        speed="max",
        steps=20,
        seed=2,
        cameras=4,
        out_dir=tmp_path / "hw",
        cam_stride=80,
        map_stride=8,
    )
    session.reset()
    hit = session.control("hw_estop")
    assert hit["ok"] is True
    assert hit["hw_estop"] is True
    assert hit["estop_kind"] == "hardware"
    assert any(f.get("code") == "HW_ESTOP" for f in hit.get("faults") or [])
    session.control("estop")
    both = session.snapshot()
    assert both["estop"] is True
    assert both["hw_estop"] is True
    assert both["estop_kind"] == "both"
    session.control("clear")
    cleared = session.snapshot()
    assert cleared["estop"] is False
    assert cleared["hw_estop"] is True
    assert session.env is not None and session.env.hw_estop.latched
    session.control("hw_reset")
    assert session.snapshot()["hw_estop"] is False
    session.close()


def test_bench_yaml_enables_watchdog() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "orin" / "bench.yaml")
    assert cfg.runtime.watchdog.enabled is True
    assert cfg.runtime.watchdog.imu_stall_s == pytest.approx(0.40)
    assert cfg.runtime.watchdog.vision_stall_s == pytest.approx(0.40)

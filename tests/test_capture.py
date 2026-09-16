"""RT-1: Fake CSI / Gst path fills named obs['cameras'] at the ICD size."""

from __future__ import annotations

import numpy as np
import pytest

from jims_mower.config import field_stereo_cameras, load_config
from jims_mower.constants import GRAVITY_MPS2
from jims_mower.env import MowerEnv
from jims_mower.runtime.capture import (
    DEFAULT_CONTRACT_HEIGHT,
    DEFAULT_CONTRACT_WIDTH,
    DOWNSAMPLE_NOTE,
    FIELD_RIG_NAMES,
    FIELD_STEREO_NAMES,
    downsample_rgb,
    load_field_camera_specs,
    names_from_specs,
)
from jims_mower.runtime.drivers import FakeCsiDriver, level_rest_imu
from jims_mower.runtime.gstreamer import (
    FakeGstAdapter,
    GstNotAvailable,
    GstNvmmAdapter,
    gstreamer_available,
)
from jims_mower.runtime.watchdog import SensorWatchdog


def _tiny_bench() -> dict:
    return {
        "dt": 0.1,
        "max_steps": 30,
        "sensors": {
            "width": 16,
            "height": 12,
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
        "perception": {"terrain_mode": "blind"},
        "runtime": {
            "cameras": {"adapter": "fake_csi"},
            "watchdog": {"enabled": True, "imu_stall_s": 9.0, "vision_stall_s": 0.20},
        },
    }


def test_field_names_match_stereo_yaml() -> None:
    specs = load_field_camera_specs()
    names = names_from_specs(specs)
    assert names[:2] == list(FIELD_STEREO_NAMES)
    assert "stereo_left" in names and "stereo_right" in names
    yaml_names = [c.name for c in field_stereo_cameras()]
    assert names == yaml_names
    assert set(FIELD_RIG_NAMES) <= set(names)


def test_gst_raises_without_gstreamer() -> None:
    assert gstreamer_available() is False
    real = GstNvmmAdapter()
    assert real.names[:2] == list(FIELD_STEREO_NAMES)
    assert real.fps_claim is None
    assert "downsample" in real.downsample_note.lower()
    with pytest.raises(GstNotAvailable):
        real.capture()
    with pytest.raises(GstNotAvailable):
        real.fill_obs({})
    with pytest.raises(GstNotAvailable):
        real.grab()


def test_fake_gst_fills_named_contract_frames() -> None:
    adapter = FakeGstAdapter()
    assert adapter.names == list(FIELD_RIG_NAMES)
    assert adapter.fps_claim is None
    obs: dict = {}
    filled = adapter.fill_obs(obs)
    assert filled is obs
    assert set(obs["cameras"]) == set(FIELD_RIG_NAMES)
    for name, frame in obs["cameras"].items():
        assert frame.shape == (DEFAULT_CONTRACT_HEIGHT, DEFAULT_CONTRACT_WIDTH, 3)
        assert frame.dtype == np.uint8
        assert frame[0, 0].sum() > 0 or name  # synthetic live, not all-zero
    assert obs["stamp_s"] == pytest.approx(adapter.last_stamp_s)


def test_downsample_happens_before_obs() -> None:
    raw = {"stereo_left": np.ones((240, 320, 3), dtype=np.uint8) * 200}
    adapter = FakeGstAdapter(["stereo_left", "stereo_right"])
    frames = adapter.capture(raw)
    assert frames["stereo_left"].shape == (DEFAULT_CONTRACT_HEIGHT, DEFAULT_CONTRACT_WIDTH, 3)
    assert frames["stereo_right"].shape == (DEFAULT_CONTRACT_HEIGHT, DEFAULT_CONTRACT_WIDTH, 3)
    tiny = downsample_rgb(np.arange(16, dtype=np.uint8).reshape(4, 4), width=2, height=2)
    assert tiny.shape == (2, 2, 3)
    assert "obs['cameras']" in DOWNSAMPLE_NOTE or "obs[\"cameras\"]" in DOWNSAMPLE_NOTE


def test_fake_csi_named_live_frames() -> None:
    csi = FakeCsiDriver(names=["stereo_left", "stereo_right", "rear"])
    first = csi.grab()
    second = csi.grab()
    assert set(first.cameras) == {"stereo_left", "stereo_right", "rear"}
    assert first.cameras["stereo_left"].shape == (DEFAULT_CONTRACT_HEIGHT, DEFAULT_CONTRACT_WIDTH, 3)
    assert second.stamp_s > first.stamp_s
    obs = csi.fill_obs()
    assert set(obs["cameras"]) == set(csi.names)
    msgs = csi.publish({"cameras": first.cameras, "stamp_s": first.stamp_s}, stamp_s=first.stamp_s)
    assert len(msgs) == 3
    assert {m.payload["name"] for m in msgs} == set(csi.names)
    assert all(m.payload["width"] == DEFAULT_CONTRACT_WIDTH for m in msgs)
    assert all(m.payload["height"] == DEFAULT_CONTRACT_HEIGHT for m in msgs)
    assert all(m.bus == "csi" for m in msgs)


def test_fresh_stamps_keep_watchdog_happy() -> None:
    adapter = FakeGstAdapter(list(FIELD_STEREO_NAMES) + ["rear"])
    wd = SensorWatchdog(enabled=True, imu_stall_s=9.0, vision_stall_s=0.40, dt=0.10)
    imu = level_rest_imu()
    assert imu[2] == pytest.approx(GRAVITY_MPS2, abs=1e-5)
    for i in range(8):
        result = adapter.grab()
        wd.observe(
            imu,
            result.cameras,
            imu_stamp_s=float(i) * 0.10,
            vision_stamp_s=result.stamp_s,
        )
    assert wd.stalled is False
    assert wd.reason == "ok"
    held = wd.filter_action(np.array([0.5, 0.5, 1.0], dtype=np.float32))
    assert held[0] == pytest.approx(0.5)


def test_frozen_vision_stamp_zeros_wheels() -> None:
    adapter = FakeGstAdapter(["stereo_left", "stereo_right"])
    wd = SensorWatchdog(enabled=True, imu_stall_s=9.0, vision_stall_s=0.20, dt=0.10)
    imu = level_rest_imu()
    live = adapter.grab()
    wd.observe(imu, live.cameras, imu_stamp_s=0.0, vision_stamp_s=live.stamp_s)
    frozen = live.stamp_s
    for i in range(1, 6):
        wd.observe(
            imu,
            live.cameras,
            imu_stamp_s=float(i) * 0.10,
            vision_stamp_s=frozen,
        )
    assert wd.stalled is True
    assert wd.reason == "vision_stall"
    held = wd.filter_action(np.array([0.8, 0.8, 1.0], dtype=np.float32))
    assert float(np.max(np.abs(held))) == 0.0


def test_gym_lookaround_names_still_ok() -> None:
    adapter = FakeGstAdapter(["front", "front_left", "front_right", "rear"])
    frames = adapter.capture()
    assert set(frames) == {"front", "front_left", "front_right", "rear"}
    for frame in frames.values():
        assert frame.shape == (DEFAULT_CONTRACT_HEIGHT, DEFAULT_CONTRACT_WIDTH, 3)


def test_env_fake_csi_watchdog_fresh_then_freeze() -> None:
    env = MowerEnv(config=_tiny_bench())
    obs, info = env.reset(seed=8)
    assert set(obs["cameras"]) == set(env.camera_index)
    for frame in obs["cameras"].values():
        assert frame.shape == (12, 16, 3)
        assert frame.dtype == np.uint8
    assert info["watchdog_enabled"] is True
    start = (float(info["pose"]["x"]), float(info["pose"]["y"]))
    for _ in range(4):
        obs, _, _, _, info = env.step(np.array([0.85, 0.85, 0.0], dtype=np.float32))
    assert info["watchdog_stalled"] is False
    moved = (
        abs(float(info["pose"]["x"]) - start[0]) + abs(float(info["pose"]["y"]) - start[1])
    )
    assert moved > 0.02
    env.fault_bus.inject("cam_blind")
    for _ in range(6):
        _, _, _, _, info = env.step(np.array([0.9, 0.9, 1.0], dtype=np.float32))
    assert info["watchdog_stalled"] is True
    assert info["watchdog_reason"] == "vision_stall"
    held = (float(info["pose"]["x"]), float(info["pose"]["y"]))
    for _ in range(3):
        _, _, _, _, info = env.step(np.array([0.9, 0.9, 1.0], dtype=np.float32))
    assert abs(float(info["pose"]["x"]) - held[0]) < 1e-4
    assert abs(float(info["pose"]["y"]) - held[1]) < 1e-4
    env.close()


def test_bench_yaml_prefers_fake_csi_adapter() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "orin" / "bench.yaml")
    assert cfg.runtime.cameras.adapter == "fake_csi"
    assert cfg.runtime.watchdog.enabled is True

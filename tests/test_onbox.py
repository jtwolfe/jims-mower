"""RT-7: on-box loop must not import the gym renderer."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from jims_mower.config import load_config
from jims_mower.onbox import OnboxLoop


def _bench(tmp_path: Path, **runtime_extra):
    runtime = {
        "enabled": True,
        "cameras": {"adapter": "fake_csi"},
        "watchdog": {"enabled": True, "imu_stall_s": 0.40, "vision_stall_s": 0.40},
    }
    runtime.update(runtime_extra)
    return load_config(
        {
            "dt": 0.1,
            "sensors": {"width": 32, "height": 24, "camera_count": 4},
            "world": {
                "width_m": 8.0,
                "height_m": 8.0,
                "resolution_m": 0.20,
                "n_people": 0,
                "terrain": {"enabled": False},
            },
            "runtime": runtime,
        }
    )


def test_onbox_subprocess_does_not_import_renderer(tmp_path: Path) -> None:
    script = r"""
import sys
from jims_mower.onbox import OnboxLoop, assert_no_renderer, renderer_loaded
from jims_mower.config import load_config
assert renderer_loaded() is False
cfg = load_config({
    "dt": 0.1,
    "sensors": {"width": 16, "height": 12, "camera_count": 4},
    "world": {"width_m": 6.0, "height_m": 6.0, "resolution_m": 0.25, "n_people": 0,
              "terrain": {"enabled": False}},
    "runtime": {
        "enabled": True,
        "cameras": {"adapter": "fake_csi"},
        "watchdog": {"enabled": True, "imu_stall_s": 0.40, "vision_stall_s": 0.40},
    },
})
loop = OnboxLoop(cfg, blackbox=%r)
info = loop.step([0.4, 0.4, 0.0])
assert_no_renderer()
assert info["renderer_imported"] is False
assert "jims_mower.renderer" not in sys.modules
assert info["cameras"]
print("ONBOX_OK")
"""
    bb = tmp_path / "bb.jsonl"
    proc = subprocess.run(
        [sys.executable, "-c", script % str(bb)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ONBOX_OK" in proc.stdout
    assert "jims_mower.renderer" not in proc.stdout


def test_onbox_loop_steps_with_fake_sensors(tmp_path: Path) -> None:
    loop = OnboxLoop(_bench(tmp_path), blackbox=tmp_path / "box.jsonl")
    info = loop.step(np.array([0.5, 0.5, 0.0], dtype=np.float32))
    assert info["watchdog_stalled"] is False
    assert info["hw_estop"] is False
    assert float(info["cmd"][0]) > 0.0
    assert "front" in info["cameras"] or len(info["cameras"]) >= 1
    assert info["fps_claim"] is None
    assert (tmp_path / "box.jsonl").is_file()


def test_onbox_hw_estop_and_watchdog_zero_cmd(tmp_path: Path) -> None:
    loop = OnboxLoop(_bench(tmp_path), blackbox=tmp_path / "box.jsonl")
    loop.hit_hw_estop("test")
    info = loop.step(np.array([0.9, 0.9, 1.0], dtype=np.float32))
    assert info["hw_estop"] is True
    assert float(np.max(np.abs(info["cmd"]))) == 0.0
    loop.reset_hw_estop()
    live = loop.step(np.array([0.6, 0.6, 0.0], dtype=np.float32))
    assert live["hw_estop"] is False
    assert float(live["cmd"][0]) > 0.0
    loop.freeze_vision_stamp()
    stalled = False
    for _ in range(8):
        frozen = loop.step(np.array([0.6, 0.6, 0.0], dtype=np.float32))
        if frozen["watchdog_stalled"] and float(np.max(np.abs(frozen["cmd"][:2]))) == 0.0:
            stalled = True
            break
    assert stalled


def test_onbox_cli_help() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "jims_mower.onbox", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "renderer" in proc.stdout.lower() or "ESTOP" in proc.stdout

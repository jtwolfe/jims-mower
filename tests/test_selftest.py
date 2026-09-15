"""Software self-test CLI: wheel spin, IMU still, camera entropy."""

from __future__ import annotations

import json
from pathlib import Path

from jims_mower.selftest import frame_entropy, main, run_selftest
import numpy as np


def test_black_frame_entropy_near_zero() -> None:
    black = np.zeros((24, 32, 3), dtype=np.uint8)
    assert frame_entropy(black) == 0.0


def test_run_selftest_passes() -> None:
    report = run_selftest()
    assert report["schema"].startswith("jims_mower.selftest")
    assert report["not_a_benchmark"] is True
    assert report["ok"] is True
    names = {c["name"] for c in report["checks"]}
    assert "unloaded_wheel_spin" in names
    assert "imu_still" in names
    assert "cam_frame_entropy" in names
    assert "dead_motor_no_drag" in names


def test_selftest_cli(tmp_path: Path) -> None:
    out = tmp_path / "selftest.json"
    rc = main(["--out", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["not_hardware_ate"] is True

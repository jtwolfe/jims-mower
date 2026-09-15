"""Headless demo dumps camera frames and detections."""

from __future__ import annotations

import json
from pathlib import Path

from jims_mower.demo import run_demo


def test_demo_writes_cameras_and_summary(tmp_path: Path) -> None:
    summary = run_demo(tmp_path, steps=3, seed=1, cameras=4)
    assert summary["steps_run"] >= 1
    assert len(summary["cameras"]) == 4
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "coverage_final.png").is_file()
    step0 = tmp_path / "step_000"
    assert step0.is_dir()
    frames = list(step0.glob("cam_*.png"))
    assert len(frames) == 4
    dets = json.loads((step0 / "detections.json").read_text(encoding="utf-8"))
    assert isinstance(dets, list)
    montage = step0 / "montage.png"
    assert montage.is_file()


def test_demo_six_cameras(tmp_path: Path) -> None:
    summary = run_demo(tmp_path, steps=2, seed=2, cameras=6)
    assert len(summary["cameras"]) == 6
    assert (tmp_path / "step_000" / "cam_front.png").is_file()
    assert (tmp_path / "step_000" / "cam_front_left.png").is_file()

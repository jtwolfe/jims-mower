"""Headless demo dumps camera frames and detections."""

from __future__ import annotations

import json
from pathlib import Path

from jims_mower.demo import run_demo


def test_demo_writes_cameras_and_summary(tmp_path: Path) -> None:
    summary = run_demo(tmp_path, steps=3, seed=1, cameras=4)
    assert summary["steps_run"] >= 1
    assert summary["policy"] == "terrain"
    assert len(summary["cameras"]) == 4
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "coverage_final.png").is_file()
    assert (tmp_path / "plan.json").is_file()
    assert (tmp_path / "plan_overlay_final.png").is_file()
    plan = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
    assert plan["policy"] == "terrain"
    assert isinstance(plan["waypoints"], list)
    step0 = tmp_path / "step_000"
    assert step0.is_dir()
    frames = list(step0.glob("cam_*.png"))
    assert len(frames) == 4
    dets = json.loads((step0 / "detections.json").read_text(encoding="utf-8"))
    assert isinstance(dets, list)
    montage = step0 / "montage.png"
    assert montage.is_file()
    assert (step0 / "elevation.png").is_file()
    assert (step0 / "slope.png").is_file()
    assert (step0 / "hazard.png").is_file()
    assert (step0 / "plan_overlay.png").is_file()
    assert (step0 / "sensors.json").is_file()
    sensors = json.loads((step0 / "sensors.json").read_text(encoding="utf-8"))
    assert len(sensors["imu"]) == 6
    assert len(sensors["gps"]) == 4
    assert sensors.get("n_drains", 0) >= 1
    assert sensors.get("policy") == "terrain"
    assert (step0 / "drain_view.png").is_file()


def test_demo_six_cameras(tmp_path: Path) -> None:
    summary = run_demo(tmp_path, steps=2, seed=2, cameras=6)
    assert len(summary["cameras"]) == 6
    assert (tmp_path / "step_000" / "cam_front.png").is_file()
    assert (tmp_path / "step_000" / "cam_front_left.png").is_file()


def test_demo_scripted_policy_still_works(tmp_path: Path) -> None:
    summary = run_demo(tmp_path, steps=2, seed=3, cameras=4, policy="scripted")
    assert summary["policy"] == "scripted"
    assert summary["n_waypoints"] == 0
    assert (tmp_path / "plan_overlay_final.png").is_file()


def test_demo_rejects_unknown_policy(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError):
        run_demo(tmp_path, steps=1, policy="fly")

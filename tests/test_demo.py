"""Headless demo dumps camera frames and detections."""

from __future__ import annotations

import json
from pathlib import Path

from jims_mower.demo import camera_dump_indices, camera_dump_stride, run_demo


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
    assert (step0 / "bev.png").is_file()
    assert (tmp_path / "bev_final.png").is_file()
    assert (step0 / "sensors.json").is_file()
    sensors = json.loads((step0 / "sensors.json").read_text(encoding="utf-8"))
    assert len(sensors["imu"]) == 6
    assert len(sensors["gps"]) == 4
    assert sensors.get("n_drains", 0) >= 1
    assert sensors.get("policy") == "terrain"
    assert sensors.get("terrain_source") == "heuristic"
    assert summary.get("terrain_source") == "heuristic"
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


def test_demo_scenario_yaml(tmp_path: Path) -> None:
    summary = run_demo(
        tmp_path,
        steps=2,
        seed=4,
        cameras=4,
        policy="scripted",
        config="configs/scenarios/suburban.yaml",
    )
    assert summary["steps_run"] >= 1
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "step_000" / "cam_front.png").is_file()


def test_demo_rejects_unknown_policy(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError):
        run_demo(tmp_path, steps=1, policy="fly")


def test_demo_terrain_observer_override(tmp_path: Path) -> None:
    summary = run_demo(tmp_path, steps=2, seed=4, cameras=4, terrain_observer="oracle")
    assert summary["terrain_source"] == "oracle"
    assert summary["terrain_mode"] == "oracle"


def test_camera_dump_stride_long_runs() -> None:
    assert camera_dump_stride(12) == 10
    assert camera_dump_stride(160) == 10
    assert camera_dump_stride(160, requested=8) == 8
    picks = camera_dump_indices(160)
    assert 0 in picks
    assert 159 in picks
    assert 10 in picks
    assert 7 not in picks
    # Every 10 steps plus last — not 3 frames, not every step.
    assert 15 <= len(picks) <= 20


def test_demo_cam_stride_writes_interval_folders(tmp_path: Path) -> None:
    run_demo(tmp_path, steps=12, seed=1, cameras=4, policy="scripted", cam_stride=5)
    assert (tmp_path / "step_000").is_dir()
    assert (tmp_path / "step_005").is_dir()
    assert (tmp_path / "step_010").is_dir()
    assert (tmp_path / "step_011").is_dir()
    assert not (tmp_path / "step_001").is_dir()

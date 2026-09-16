"""JS-free unit tests for the owner phone path overlay builder."""

from __future__ import annotations

from pathlib import Path

from jims_mower.path_overlay import (
    FRONTIER_MAX_POINTS,
    PLAN_MAX_POINTS,
    TRAIL_MAX_POINTS,
    build_path_overlay,
    downsample_xy,
    mode_banner_for,
    progress_kind_for,
    trail_from_poses,
)


def test_path_overlay_module_stays_stdlib() -> None:
    """Keep overlay cheap so mission inspect cannot circular-import."""
    src = Path(build_path_overlay.__code__.co_filename).read_text(encoding="utf-8")
    assert "from jims_mower.live" not in src
    assert "from jims_mower.mission" not in src
    assert "from jims_mower.viewer" not in src
    assert "from jims_mower.app" not in src


def test_downsample_xy_keeps_first_last_and_caps() -> None:
    pts = [[float(i), float(i * 2)] for i in range(200)]
    out = downsample_xy(pts, 20)
    assert out[0] == [0.0, 0.0]
    assert out[-1] == [199.0, 398.0]
    assert len(out) <= 20
    assert downsample_xy([], 10) == []
    assert downsample_xy([[1, 2], {"x": 1, "y": 2}, (3, 4)], 8) == [[1.0, 2.0], [3.0, 4.0]]


def test_mode_banner_mapping_vs_mowing() -> None:
    explore = mode_banner_for("explore", "running")
    assert explore["label"] == "Mapping yard"
    assert explore["tone"] == "map"
    assert explore["kind"] == "mapping"
    assert explore["hold"] is None

    mow = mode_banner_for("mow", "running")
    assert mow["label"] == "Mowing"
    assert mow["tone"] == "mow"
    assert mow["kind"] == "mowing"

    teach = mode_banner_for("calibrate_boundary", "running")
    assert teach["label"] == "Teaching boundary"
    assert teach["tone"] == "teach"

    review = mode_banner_for("review", "running")
    assert review["label"] == "Map ready — review"

    home = mode_banner_for("return_home", "running")
    assert home["label"] == "Returning home"

    done = mode_banner_for("complete", "idle")
    assert done["label"] == "Done"

    paused = mode_banner_for("explore", "paused")
    assert paused["label"] == "Mapping yard"
    assert paused["hold"] == "Paused"

    estop = mode_banner_for("mow", "estop")
    assert estop["label"] == "Mowing"
    assert estop["hold"] == "E-STOP"

    hold = mode_banner_for("review", "hold")
    assert hold["label"] == "Map ready — review"
    assert hold["hold"] == "Hold"

    idle = mode_banner_for("calibrate_boundary", "idle")
    assert idle["label"] == "Ready"
    assert idle["kind"] == "idle"
    assert mode_banner_for("complete", "idle")["label"] == "Done"


def test_progress_kind_swaps_map_vs_cut() -> None:
    assert progress_kind_for("explore") == "mapping"
    assert progress_kind_for("calibrate_boundary") == "mapping"
    assert progress_kind_for("mow") == "mowing"
    assert progress_kind_for("return_home") == "mowing"
    assert progress_kind_for("complete") == "done"


def test_build_path_overlay_explore_hides_mow_plan() -> None:
    overlay = build_path_overlay(
        phase="explore",
        job_state="running",
        pose={"x": 3.0, "y": 4.0, "theta": 0.2},
        poses=[{"x": i * 0.5, "y": 1.0} for i in range(12)],
        plan=[[10.0, 10.0], [11.0, 10.0], [12.0, 11.0]],
        explore=[[1.0, 1.0], [2.0, 1.5], [3.0, 2.0]],
        frontiers=[{"x": 8.0, "y": 7.0}, [8.5, 7.2]],
        n_waypoints=20,
        waypoint_index=5,
    )
    assert overlay["phase"] == "explore"
    assert overlay["mode"]["label"] == "Mapping yard"
    assert overlay["progress_kind"] == "mapping"
    assert overlay["plan"] == []
    assert overlay["explore"][0] == [1.0, 1.0]
    assert overlay["frontiers"]
    assert overlay["trail"][0] == [0.0, 1.0]
    assert overlay["pose"]["x"] == 3.0
    assert overlay["path_remaining"] == 15
    assert overlay["colors"]["trail"] == "#aa88ff"
    assert overlay["colors"]["plan"] == "#2ad4e6"


def test_build_path_overlay_mow_shows_plan_not_frontiers() -> None:
    overlay = build_path_overlay(
        phase="mow",
        job_state="paused",
        pose={"x": 2.0, "y": 2.0, "theta": 1.1},
        trail=[[0.0, 0.0], [1.0, 0.2], [2.0, 0.4]],
        plan=[{"x": 2.0, "y": 2.0}, {"x": 4.0, "y": 2.0}, {"x": 4.0, "y": 6.0}],
        explore=[[0.0, 0.0], [1.0, 1.0]],
        frontiers=[[9.0, 9.0]],
        n_waypoints=40,
        waypoint_index=12,
    )
    assert overlay["mode"]["label"] == "Mowing"
    assert overlay["mode"]["hold"] == "Paused"
    assert overlay["progress_kind"] == "mowing"
    assert overlay["plan"][0] == [2.0, 2.0]
    assert overlay["explore"] == []
    assert overlay["frontiers"] == []
    assert overlay["trail"][-1] == [2.0, 2.0]
    assert overlay["path_remaining"] == 28


def test_overlay_caps_long_arrays() -> None:
    trail = [[float(i), 0.0] for i in range(400)]
    plan = [[float(i), 1.0] for i in range(300)]
    fronts = [[float(i), 2.0] for i in range(120)]
    overlay = build_path_overlay(
        phase="explore",
        poses=trail,
        plan=plan,
        frontiers=fronts,
        explore=plan,
    )
    assert len(overlay["trail"]) <= TRAIL_MAX_POINTS
    assert len(overlay["explore"]) <= PLAN_MAX_POINTS
    assert len(overlay["frontiers"]) <= FRONTIER_MAX_POINTS
    assert trail_from_poses(trail)[0] == [0.0, 0.0]
    assert trail_from_poses(trail)[-1] == [399.0, 0.0]

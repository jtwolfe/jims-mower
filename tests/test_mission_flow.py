"""Mission phases, unknown-space mapping, and global coverage correctness."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jims_mower.constants import (
    HAZARD_DRAIN,
    STRUCTURE_BUILDING,
    STRUCTURE_PATH,
)
from jims_mower.env import MowerEnv
from jims_mower.mission_flow import MissionPhase, MissionPolicy
from jims_mower.planning.costmap import build_costmap
from jims_mower.planning.coverage import plan_coverage
from jims_mower.planning.explore import plan_explore
from jims_mower.planning.observed import ObservedMap, frontiers
from jims_mower.scenarios import load_source


def _tiny_env() -> MowerEnv:
    cfg, scenario = load_source("mission_tiny")
    cfg.sensors.width = 32
    cfg.sensors.height = 24
    cfg.sensors.camera_count = 4
    cfg.sensors.cameras = []
    cfg.max_steps = 800
    return MowerEnv(config=cfg, scenario=scenario, render_mode=None)


def test_frontier_exploration_expands_observed_mask() -> None:
    omap = ObservedMap.empty(4.0, 4.0, 0.25)
    # A known-free strip along the south edge; interior is unknown.
    omap.observed[1, :] = True
    omap.free[1, :] = True
    omap.confidence[1, :] = 0.6
    cells = frontiers(omap.observed, omap.free)
    assert cells, "expected a frontier between the known strip and unknown"
    before = int(omap.observed.sum())
    omap.stamp_disk(2.0, 1.4, 0.6)
    assert int(omap.observed.sum()) > before


def test_unknown_is_not_mowable_or_safe() -> None:
    omap = ObservedMap.empty(3.0, 3.0, 0.25)
    omap.stamp_disk(0.6, 0.6, 0.4, explored=True)
    omap.refresh_free()
    mowable = omap.mowable_mask()
    assert np.any(mowable)
    assert not np.any(mowable & ~omap.observed)
    blocked = omap.unknown_blocked()
    assert np.all(blocked[~omap.observed])


def test_explore_path_stays_in_known_safe() -> None:
    omap = ObservedMap.empty(5.0, 5.0, 0.25)
    omap.observed[2:8, 2:16] = True
    omap.free[2:8, 2:16] = True
    omap.confidence[2:8, 2:16] = 0.7
    plan = plan_explore(omap, (1.0, 1.0))
    assert plan.frontier_cells
    for row, col in plan.cells:
        assert omap.observed[row, col]
        assert omap.free[row, col]


def test_plan_reports_unreachable_instead_of_dropping() -> None:
    n = 20
    res = 0.25
    hazard = np.zeros((n, n), dtype=np.float32)
    hazard[10, :] = HAZARD_DRAIN
    cm = build_costmap(
        hazard,
        np.zeros_like(hazard),
        resolution_m=res,
        width_m=n * res,
        height_m=n * res,
        max_climb_slope_rad=0.3,
        drain_clearance_m=0.25,
        margin_m=0.25,
    )
    mowable = ~cm.blocked
    plan = plan_coverage(cm, (0.8, 0.8), strip_spacing_m=0.5, waypoint_stride_m=0.5, mowable=mowable)
    assert plan.planned_mowable_cells > 0
    assert plan.unreachable_mowable_cells > 0
    assert plan.reachable_mowable_cells + plan.unreachable_mowable_cells == plan.planned_mowable_cells
    ys = [y for _x, y in plan.waypoints]
    assert ys
    assert max(ys) < 10 * res + 0.6


def test_global_plan_avoids_path_building_and_reports_metrics() -> None:
    n = 24
    res = 0.25
    hazard = np.zeros((n, n), dtype=np.float32)
    structure = np.zeros((n, n), dtype=np.uint8)
    structure[8:11, :] = STRUCTURE_PATH
    structure[16:20, 16:20] = STRUCTURE_BUILDING
    cm = build_costmap(
        hazard,
        np.zeros_like(hazard),
        resolution_m=res,
        width_m=n * res,
        height_m=n * res,
        max_climb_slope_rad=0.3,
        drain_clearance_m=0.0,
        margin_m=0.25,
        structure=structure,
    )
    mowable = (structure == 0) & ~cm.blocked
    plan = plan_coverage(cm, (1.0, 1.0), strip_spacing_m=0.5, waypoint_stride_m=0.5, mowable=mowable)
    assert plan.planned_mowable_cells == int(mowable.sum())
    for row, col in plan.cells:
        assert structure[row, col] != STRUCTURE_BUILDING
        assert not cm.blocked[row, col]
    xs = [x for x, _y in plan.waypoints]
    ys = [y for _x, y in plan.waypoints]
    assert max(xs) - min(xs) > 2.0
    assert max(ys) - min(ys) > 1.5


def test_phase_order_and_trimmer_off_before_mow() -> None:
    env = _tiny_env()
    obs, info = env.reset(seed=3)
    policy = MissionPolicy(env.cfg, fast=True)
    policy.reset(obs, info)
    seen: list[str] = []
    trimmer_before_mow = True
    for _ in range(500):
        if policy.phase.value not in seen:
            seen.append(policy.phase.value)
        action = policy.act(obs, info)
        if policy.phase in {MissionPhase.CALIBRATE_BOUNDARY, MissionPhase.EXPLORE, MissionPhase.REVIEW}:
            assert float(action[2]) == 0.0
            if info.get("trimmer_enabled"):
                trimmer_before_mow = False
        obs, _reward, terminated, truncated, info = env.step(action)
        if policy.phase in {MissionPhase.CALIBRATE_BOUNDARY, MissionPhase.EXPLORE, MissionPhase.REVIEW}:
            assert not info.get("trimmer_enabled")
        if terminated or truncated or policy.done:
            break
    env.close()
    assert seen[0] == MissionPhase.CALIBRATE_BOUNDARY.value
    assert "explore" in seen
    assert "review" in seen
    assert "mow" in seen
    order = [p for p in seen if p in {"calibrate_boundary", "explore", "review", "mow"}]
    assert order == sorted(order, key=["calibrate_boundary", "explore", "review", "mow"].index)
    assert trimmer_before_mow


def test_tiny_mission_reaches_mow_or_complete(tmp_path: Path) -> None:
    from jims_mower.mission_demo import run_mission_demo

    out = tmp_path / "mission_fast"
    summary = run_mission_demo(out, fast=True, seed=3, cameras=4, steps=360, cam_stride=80, snapshot_stride=30)
    assert summary.get("scenario") == "mission_tiny"
    phases = [r["phase"] for r in summary.get("phase_ranges") or []]
    assert "calibrate_boundary" in phases
    assert "explore" in phases
    assert "review" in phases
    assert "mow" in phases or summary["final_phase"] in {"mow", "return_home", "complete"}
    assert (out / "mission.json").is_file()
    assert (out / "viewer.json").is_file()
    metrics = summary["mission"]
    assert metrics["map_completion"] > 0.2
    if summary.get("n_waypoints", 0) > 4:
        assert metrics["planned_mowable_cells"] >= metrics["reachable_mowable_cells"]
    events = {e["event"] for e in summary.get("events") or []}
    assert "phase_enter" in events
    assert "map_ready" in events or "explore_complete" in events
    # Mow must actually track the global plan — not sit on a self-ToF replan.
    if "mow" in phases:
        assert metrics.get("waypoint_index", 0) > 2 or float(summary.get("final_coverage_fraction") or 0.0) > 0.0
        assert int(metrics.get("replans") or 0) < int(summary.get("steps_run") or 1) // 2


def test_mission_cli_help() -> None:
    from jims_mower.mission_demo import build_parser, resolve_mission_config

    text = build_parser().format_help()
    assert "--fast" in text
    assert "golf_rough" in text
    assert resolve_mission_config(None, fast=True) == "mission_tiny"
    assert resolve_mission_config("golf_rough", fast=True) == "mission_tiny"
    assert resolve_mission_config("golf_rough", fast=False) == "golf_rough"


def test_scale_mission_budget_shrinks_caps() -> None:
    from jims_mower.config import MissionConfig
    from jims_mower.mission_flow import scale_mission_budget

    base = MissionConfig(max_calibrate_steps=1000, max_explore_steps=2000)
    scaled = scale_mission_budget(base, 0.4)
    assert scaled.max_calibrate_steps == 400
    assert scaled.max_explore_steps == 800
    assert scaled.phase_budget_scale == pytest.approx(0.4)


def test_calibrate_confirm_closes_on_authored_keep_in() -> None:
    from jims_mower.config import MissionConfig

    env = _tiny_env()
    obs, info = env.reset(seed=2)
    settings = MissionConfig(
        stamp_radius_m=2.4,
        camera_range_m=5.0,
        max_calibrate_steps=400,
        max_explore_steps=80,
        calibrate_confirm_m=1.6,
        calibrate_cruise=1.0,
        calibrate_arrive_m=0.55,
        calibrate_stride_m=0.70,
    )
    policy = MissionPolicy(env.cfg, settings=settings)
    policy.reset(obs, info)
    phases = {policy.phase.value}
    for _ in range(80):
        action = policy.act(obs, info)
        obs, _reward, terminated, truncated, info = env.step(action)
        phases.add(policy.phase.value)
        if policy.phase != MissionPhase.CALIBRATE_BOUNDARY:
            break
        if terminated or truncated:
            break
    env.close()
    assert policy.phase != MissionPhase.CALIBRATE_BOUNDARY
    assert policy.step < 80
    assert policy.profile is not None
    assert len(policy.profile.keep_in) >= 3


def test_acre_yard_demo_reaches_explore_without_full_lap() -> None:
    cfg, scenario = load_source("acre_yard_demo")
    cfg.sensors.width = 16
    cfg.sensors.height = 12
    cfg.sensors.camera_count = 4
    cfg.sensors.cameras = []
    cfg.max_steps = 360
    env = MowerEnv(config=cfg, scenario=scenario, render_mode=None)
    obs, info = env.reset(seed=3)
    policy = MissionPolicy(env.cfg)
    policy.reset(obs, info)
    assert policy.settings.calibrate_confirm_m >= 20.0
    for _ in range(320):
        action = policy.act(obs, info)
        obs, _reward, terminated, truncated, info = env.step(action)
        if policy.phase != MissionPhase.CALIBRATE_BOUNDARY:
            break
        if terminated or truncated:
            break
    phase = policy.phase.value
    step = policy.step
    events = {e.event for e in policy.events}
    env.close()
    assert phase in {"explore", "review", "mow", "return_home", "complete"}
    assert step < 300
    assert "boundary_recorded" in events


def test_review_hold_waits_until_start_mow() -> None:
    from jims_mower.config import MissionConfig

    env = _tiny_env()
    obs, info = env.reset(seed=5)
    settings = MissionConfig(
        stamp_radius_m=2.4,
        camera_range_m=5.0,
        max_calibrate_steps=8,
        max_explore_steps=8,
        explore_complete=0.01,
        explore_no_frontier=0.01,
        review_hold_steps=80,
        calibrate_confirm_m=0.4,
    )
    policy = MissionPolicy(env.cfg, settings=settings)
    policy.reset(obs, info)
    for _ in range(80):
        action = policy.act(obs, info)
        obs, _reward, terminated, truncated, info = env.step(action)
        if policy.phase == MissionPhase.REVIEW:
            break
        if terminated or truncated:
            break
    assert policy.phase == MissionPhase.REVIEW
    for _ in range(4):
        action = policy.act(obs, info)
        obs, _reward, _term, _trunc, info = env.step(action)
        assert policy.phase == MissionPhase.REVIEW
    if policy.global_plan is None or len(getattr(policy.global_plan, "waypoints", [])) < 2:
        from jims_mower.planning.coverage import CoveragePlan

        policy.global_plan = CoveragePlan(waypoints=[(1.2, 1.2), (2.4, 1.8), (3.0, 2.2)])
        policy.plan = policy.global_plan
    assert policy.request_start_mow() is True
    policy.act(obs, info)
    env.close()
    assert policy.phase == MissionPhase.MOW


def test_mow_stop_reverse_then_skip() -> None:
    env = _tiny_env()
    obs, info = env.reset(seed=4)
    policy = MissionPolicy(env.cfg, fast=True)
    policy.reset(obs, info)
    policy.phase = MissionPhase.MOW
    policy.global_plan = type("P", (), {"waypoints": [(2.0, 2.0), (3.0, 2.0), (4.0, 2.0)]})()
    policy.index = 0
    pose = type("Z", (), {"x": 1.0, "y": 1.0, "theta": 0.0, "z": 0.0, "pitch": 0.0, "roll": 0.0})()
    first = policy._tick_mow(obs, info, pose, "stop")
    assert float(first[0]) < 0.0 and float(first[1]) < 0.0
    policy._tick_mow(obs, info, pose, "stop")
    policy._tick_mow(obs, info, pose, "stop")
    assert policy.index >= 1
    assert policy._skipped_global
    env.close()

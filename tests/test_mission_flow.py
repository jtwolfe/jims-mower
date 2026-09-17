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
from jims_mower.live import owner_copy_for
from jims_mower.mission_flow import MissionPhase, MissionPolicy
from jims_mower.profile import YardProfile
from jims_mower.planning.costmap import build_costmap
from jims_mower.planning.coverage import plan_coverage
from jims_mower.planning.explore import plan_explore
from jims_mower.perception.grade import PlanarGradeModel
from jims_mower.planning.observed import ObservedMap, frontiers
from jims_mower.scenarios import load_source
from jims_mower.types import Pose


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


def test_taught_profile_skips_calibrate_and_starts_explore() -> None:
    from jims_mower.profile import YardProfile

    env = _tiny_env()
    keep_in = [(0.7, 0.7), (5.0, 0.7), (5.0, 4.0), (0.7, 4.0)]
    profile = YardProfile(
        name="taught_tiny",
        width_m=env.cfg.world.width_m,
        height_m=env.cfg.world.height_m,
        resolution_m=env.cfg.world.resolution_m,
        keep_in=keep_in,
        home={"x": 1.2, "y": 1.2, "theta": 0.0},
    )
    obs, info = env.reset(seed=3, options={"yard_profile": profile, "resize_world": False})
    policy = MissionPolicy(env.cfg, fast=True)
    policy.reset(obs, info, profile=profile)
    assert policy.phase == MissionPhase.EXPLORE
    assert policy.profile is not None
    assert len(policy.profile.keep_in) == 4
    events = {e.event for e in policy.events}
    assert "boundary_taught" in events
    seen = {policy.phase.value}
    for _ in range(80):
        action = policy.act(obs, info)
        assert float(action[2]) == 0.0 or policy.phase == MissionPhase.MOW
        obs, _reward, terminated, truncated, info = env.step(action)
        seen.add(policy.phase.value)
        if policy.phase in {MissionPhase.REVIEW, MissionPhase.MOW, MissionPhase.COMPLETE}:
            break
        if terminated or truncated or policy.done:
            break
    env.close()
    assert "calibrate_boundary" not in seen
    assert "explore" in seen


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


def test_explore_ready_accepts_near_full_keep_in() -> None:
    """1.0 target must not hang on leftover unreachable cells / float noise."""
    env = _tiny_env()
    policy = MissionPolicy(env.cfg)
    policy.settings.explore_complete = 1.0
    policy.settings.explore_no_frontier = 0.90
    policy.phase_step = 12
    assert policy._explore_ready(0.995, True) is True
    assert policy._explore_ready(0.92, False) is True
    assert policy._explore_ready(0.40, True) is False
    env.close()


def test_acre_yard_demo_reaches_map_ready_then_mow() -> None:
    """Small taught keep-in on the acre demo: MAP READY then MOW. Not full acre."""
    from jims_mower.profile import YardProfile

    cfg, scenario = load_source("acre_yard_demo")
    cfg.sensors.width = 16
    cfg.sensors.height = 12
    cfg.sensors.camera_count = 4
    cfg.sensors.cameras = []
    cfg.max_steps = 760
    env = MowerEnv(config=cfg, scenario=scenario, render_mode=None)
    profile = YardProfile(
        name="acre_ci_pocket",
        width_m=cfg.world.width_m,
        height_m=cfg.world.height_m,
        resolution_m=cfg.world.resolution_m,
        keep_in=[(8.0, 8.0), (16.0, 8.0), (16.0, 14.0), (8.0, 14.0)],
        home={"x": 10.0, "y": 10.0, "theta": 0.0},
    )
    obs, info = env.reset(seed=3, options={"yard_profile": profile, "resize_world": False})
    policy = MissionPolicy(env.cfg)
    policy.reset(obs, info, profile=profile)
    seen: list[str] = []
    cut = 0.0
    for _ in range(720):
        if policy.phase.value not in seen:
            seen.append(policy.phase.value)
        if policy.phase == MissionPhase.REVIEW:
            policy.request_start_mow()
        action = policy.act(obs, info)
        obs, _reward, terminated, truncated, info = env.step(action)
        cut = float(info.get("coverage_fraction") or 0.0)
        if policy.phase == MissionPhase.MOW and (cut > 0.0 or policy.phase_step >= 80):
            break
        if terminated or truncated or policy.done:
            break
    status = policy.status(info)
    env.close()
    assert "explore" in seen
    assert "review" in seen
    assert "mow" in seen or policy.phase.value == "mow"
    assert policy.settings.explore_complete >= 0.95
    assert status["map_completion"] >= 0.70
    assert policy.step < 740
    assert cut > 0.0


def test_scribble_keep_in_review_is_reteach_not_safe() -> None:
    """Tiny centre fence must not sit on Hold-safe after an empty mow plan."""
    env = _tiny_env()
    profile = YardProfile(
        name="scribble",
        width_m=env.cfg.world.width_m,
        height_m=env.cfg.world.height_m,
        resolution_m=env.cfg.world.resolution_m,
        keep_in=[(2.0, 2.0), (2.25, 2.0), (2.25, 2.18), (2.0, 2.18)],
        home={"x": 2.08, "y": 2.06, "theta": 0.0},
    )
    obs, info = env.reset(seed=3, options={"yard_profile": profile, "resize_world": False})
    policy = MissionPolicy(env.cfg, fast=True)
    policy.reset(obs, info, profile=profile)
    assert policy.phase == MissionPhase.EXPLORE
    for _ in range(40):
        action = policy.act(obs, info)
        obs, _reward, terminated, truncated, info = env.step(action)
        assert policy.phase != MissionPhase.SAFE
        copy = owner_copy_for("running", policy.phase.value, fence_unusable=policy.fence_unusable)
        assert "hold — safe" not in copy.lower()
        if policy.phase == MissionPhase.REVIEW:
            break
        if terminated or truncated or policy.done:
            break
    assert policy.phase == MissionPhase.REVIEW
    for _ in range(3):
        action = policy.act(obs, info)
        obs, _reward, _term, _trunc, info = env.step(action)
        assert policy.phase == MissionPhase.REVIEW
        assert policy.phase != MissionPhase.SAFE
    assert policy.fence_unusable is True
    assert policy.request_start_mow() is False
    copy = owner_copy_for("running", policy.phase.value, fence_unusable=True)
    assert "re-teach" in copy.lower()
    env.close()


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


def test_fog_islands_not_counted_unreachable() -> None:
    """Unknown corridors are unmapped leftover, not unreachable grass."""
    n = 20
    res = 0.25
    hazard = np.zeros((n, n), dtype=np.float32)
    extra = np.zeros((n, n), dtype=bool)
    extra[:, 9:11] = True
    cm = build_costmap(
        hazard,
        np.zeros_like(hazard),
        resolution_m=res,
        width_m=n * res,
        height_m=n * res,
        max_climb_slope_rad=0.3,
        drain_clearance_m=0.0,
        margin_m=0.25,
        extra_blocked=extra,
    )
    mowable = ~cm.blocked
    mowable[:, 9:11] = False
    mowable[:, 12:] = True
    fog = extra.copy()
    plan = plan_coverage(
        cm,
        (0.8, 0.8),
        strip_spacing_m=0.5,
        waypoint_stride_m=0.5,
        mowable=mowable,
        soft_transit=fog,
    )
    assert plan.unmapped_mowable_cells > 0
    assert plan.unreachable_mowable_cells < plan.unmapped_mowable_cells
    xs = [x for x, _y in plan.waypoints]
    assert xs
    assert max(xs) < 11 * res + 0.4


def test_taught_tiny_mows_then_completes() -> None:
    from jims_mower.profile import YardProfile

    env = _tiny_env()
    keep_in = [(0.7, 0.7), (5.0, 0.7), (5.0, 4.0), (0.7, 4.0)]
    profile = YardProfile(
        name="taught_tiny",
        width_m=env.cfg.world.width_m,
        height_m=env.cfg.world.height_m,
        resolution_m=env.cfg.world.resolution_m,
        keep_in=keep_in,
        home={"x": 1.2, "y": 1.2, "theta": 0.0},
    )
    obs, info = env.reset(seed=3, options={"yard_profile": profile, "resize_world": False})
    policy = MissionPolicy(env.cfg, fast=True)
    policy.reset(obs, info, profile=profile)
    seen: set[str] = {policy.phase.value}
    peak_cut = 0.0
    for _ in range(520):
        action = policy.act(obs, info)
        obs, _reward, terminated, truncated, info = env.step(action)
        seen.add(policy.phase.value)
        peak_cut = max(peak_cut, float(policy.status(info)["actual_coverage_fraction"]))
        if policy.phase == MissionPhase.COMPLETE:
            break
        if terminated or truncated or policy.done:
            break
    status = policy.status(info)
    env.close()
    assert "explore" in seen
    assert "mow" in seen
    assert peak_cut > 0.04
    assert "return_home" in seen or "complete" in seen or policy.phase.value in {
        "return_home",
        "complete",
    }
    assert status["unreachable_mowable_cells"] <= status["planned_mowable_cells"]


def test_mow_complete_frac_homes() -> None:
    from jims_mower.planning.coverage import CoveragePlan
    from jims_mower.types import Pose

    env = _tiny_env()
    obs, info = env.reset(seed=4)
    policy = MissionPolicy(env.cfg, fast=True)
    policy.reset(obs, info)
    policy.phase = MissionPhase.MOW
    policy.settings.mow_complete_frac = 0.05
    policy.global_plan = CoveragePlan(
        waypoints=[(2.0, 2.0), (3.0, 2.0), (4.0, 2.0)],
        planned_mowable_cells=20,
        reachable_mowable_cells=20,
    )
    info = dict(info)
    info["coverage_cut_cells"] = 4
    pose = Pose(1.0, 1.0, 0.0)
    policy._tick_mow(obs, info, pose, "ok")
    env.close()
    assert policy.phase == MissionPhase.RETURN_HOME


def test_mow_stop_reverse_then_skip() -> None:
    env = _tiny_env()
    obs, info = env.reset(seed=4)
    policy = MissionPolicy(env.cfg, fast=True)
    policy.reset(obs, info)
    policy.phase = MissionPhase.MOW
    from jims_mower.planning.coverage import CoveragePlan

    policy.global_plan = CoveragePlan(
        waypoints=[(2.0, 2.0), (3.0, 2.0), (4.0, 2.0), (5.0, 2.2), (5.5, 2.8)],
        planned_mowable_cells=40,
        reachable_mowable_cells=40,
    )
    policy.index = 0
    pose = type("Z", (), {"x": 1.0, "y": 1.0, "theta": 0.0, "z": 0.0, "pitch": 0.0, "roll": 0.0})()
    first = policy._tick_mow(obs, info, pose, "stop")
    assert float(first[0]) < 0.0 and float(first[1]) < 0.0
    policy._tick_mow(obs, info, pose, "stop")
    third = policy._tick_mow(obs, info, pose, "stop")
    assert policy.index >= 1
    assert policy._skipped_global
    assert policy._stop_cool > 0
    # After the cluster skip, keep driving — do not sit in hold.
    assert abs(float(third[0])) + abs(float(third[1])) > 0.0 or policy.phase == MissionPhase.RETURN_HOME
    env.close()


def test_observed_elevation_stable_under_pitch_roll() -> None:
    """Already-mapped heights stay put when the chassis tips on a ridge."""
    omap = ObservedMap.empty(8.0, 8.0, 0.25)
    model = PlanarGradeModel()
    pose = Pose(2.0, 2.0, 0.0, z=0.08, pitch=0.06, roll=0.0)
    model.update(pose, None, None)
    prior, _ = model.raster(omap.elevation.shape, omap.resolution_m)
    omap.stamp_disk(2.0, 2.0, 1.6, explored=True)
    omap.ingest_observer(
        {"elevation": prior, "elevation_prior": prior},
        pose=pose,
        grade_radius_m=2.2,
    )
    locked = omap.elevation_set.copy()
    frozen = omap.elevation.copy()
    assert int(locked.sum()) > 10
    # Violent synthetic pitch/roll while creeping forward — old code
    # recopied the IMU plane onto every seen cell.
    for i in range(14):
        tip = Pose(2.1 + 0.04 * i, 2.0, 0.15, z=0.10, pitch=0.42 * ((-1) ** i), roll=-0.30)
        model.update(tip, None, None)
        swung, _ = model.raster(omap.elevation.shape, omap.resolution_m)
        omap.stamp_disk(tip.x, tip.y, 1.6)
        omap.ingest_observer(
            {"elevation": swung, "elevation_prior": swung},
            pose=tip,
            grade_radius_m=2.2,
        )
    err = np.abs(omap.elevation[locked] - frozen[locked])
    assert float(err.max()) < 1e-4
    # Already-mapped patch does not flip sign across the yard.
    zs = frozen[locked]
    assert float(np.ptp(zs)) < 0.45


def test_growing_edge_follows_neighbors_not_swung_plane() -> None:
    """New cells after a tip stay near the locked patch, not the IMU plane."""
    omap = ObservedMap.empty(8.0, 8.0, 0.25)
    pose = Pose(2.0, 2.0, 0.0, z=0.10, pitch=0.0, roll=0.0)
    flat = np.full(omap.elevation.shape, 0.10, dtype=np.float32)
    omap.stamp_disk(2.0, 2.0, 1.2, explored=True)
    omap.ingest_observer({"elevation": flat}, pose=pose, grade_radius_m=1.4)
    locked = omap.elevation_set.copy()
    assert int(locked.sum()) > 8
    # Swung plane would put the forward edge ~0.5 m higher.
    gx, gy = np.meshgrid(
        (np.arange(omap.cols) + 0.5) * 0.25,
        (np.arange(omap.rows) + 0.5) * 0.25,
    )
    swung = (0.10 + 0.45 * (gx - 2.0)).astype(np.float32)
    tip = Pose(2.6, 2.0, 0.0, z=0.12, pitch=0.40, roll=0.0)
    omap.stamp_disk(2.6, 2.0, 1.2)
    omap.ingest_observer({"elevation": swung, "elevation_prior": swung}, pose=tip, grade_radius_m=1.4)
    fresh = omap.elevation_set & ~locked
    assert int(fresh.sum()) > 0
    # Neighbour inherit + 15% pose.z — not the 0.45 grade rewrite.
    assert float(np.abs(omap.elevation[fresh] - 0.10).max()) < 0.08
    assert float(np.abs(omap.elevation[locked] - 0.10).max()) < 1e-4


def test_observed_elevation_ignores_far_plane_copy() -> None:
    """Camera-seen far cells must not inherit the current IMU plane."""
    omap = ObservedMap.empty(10.0, 10.0, 0.25)
    pose = Pose(1.2, 1.2, 0.0, z=0.05, pitch=0.0, roll=0.0)
    prior = np.zeros(omap.elevation.shape, dtype=np.float32)
    yy = (np.arange(omap.rows) + 0.5) * 0.25
    xx = (np.arange(omap.cols) + 0.5) * 0.25
    gx, gy = np.meshgrid(xx, yy)
    prior[:, :] = (0.40 * (gx - 1.2)).astype(np.float32)
    omap.observed[:, :] = True
    omap.ingest_observer(
        {"elevation": prior, "elevation_prior": prior},
        pose=pose,
        grade_radius_m=2.0,
    )
    far = omap.world_to_cell(8.5, 8.5)
    near = omap.world_to_cell(1.3, 1.3)
    assert far is not None and near is not None
    assert omap.elevation_set[near]
    assert not omap.elevation_set[far]
    z = omap.observed_elevation()
    assert np.isnan(z[far])
    assert np.isfinite(z[near])


def test_mission_mapped_elev_stable_when_heuristic_tips() -> None:
    """Headless: heuristic + MissionPolicy must not flop mapped elevation."""
    env = _tiny_env()
    env.cfg.perception.terrain_mode = "heuristic"
    obs, info = env.reset(seed=5)
    policy = MissionPolicy(env.cfg, fast=True)
    policy.reset(obs, info)
    pose = env._pose
    policy._stamp(obs, info, pose, explored=True)
    assert policy.observed is not None
    locked = policy.observed.elevation_set.copy()
    if not np.any(locked):
        env.close()
        pytest.skip("tiny seed did not lock a local height sample")
    frozen = policy.observed.elevation.copy()
    # Drive a few steps, then overwrite pose attitude in the next obs
    # the way a ridge tip would (pitch/roll swing, same seen mask).
    for _ in range(6):
        action = policy.act(obs, info)
        obs, _reward, terminated, truncated, info = env.step(action)
        policy._stamp(obs, info, env._pose, explored=True)
        if terminated or truncated:
            break
    still = policy.observed.elevation_set & locked
    if np.any(still):
        err = np.abs(policy.observed.elevation[still] - frozen[still])
        assert float(err.max()) < 0.10
    env.close()

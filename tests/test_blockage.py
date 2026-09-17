"""Learned no-go: no-progress stamps a blockage and explore skips that lip."""

from __future__ import annotations

import numpy as np

from jims_mower.config import MissionConfig
from jims_mower.env import MowerEnv
from jims_mower.mission_flow import MissionPhase, MissionPolicy
from jims_mower.planning.explore import plan_explore
from jims_mower.planning.observed import ObservedMap, frontiers
from jims_mower.profile import YardProfile
from jims_mower.scenarios import load_source
from jims_mower.planning.grade_tip import KIND_GRADE
from jims_mower.types import Obstacle, Pose


def test_mission_explore_collision_does_not_end_episode() -> None:
    env, profile = _occlude_env()
    obs, info = env.reset(seed=1, options={"yard_profile": profile, "resize_world": False})
    policy = MissionPolicy(env.cfg)
    policy.reset(obs, info, profile=profile)
    policy.attach_to_env(env)
    env._mission_phase = "explore"
    tree = Obstacle(kind="tree", x=env._pose.x + 0.12, y=env._pose.y, radius=0.35)
    env._yard.obstacles.append(tree)
    obs, _reward, terminated, _trunc, info = env.step(
        policy.act(obs, info)
    )
    env.close()
    assert info.get("collision") == "tree"
    assert terminated is False


def test_stamp_blockage_is_known_nogo_not_free() -> None:
    omap = ObservedMap.empty(4.0, 4.0, 0.25)
    omap.observed[2:6, 2:10] = True
    omap.refresh_free()
    before_free = int(omap.free.sum())
    added = omap.stamp_blockage(1.5, 1.0, 0.45)
    assert added > 0
    assert omap.blockage_count() == added
    assert not np.any(omap.free & omap.blockage)
    assert np.all(omap.observed[omap.blockage])
    blocked = omap.unknown_blocked()
    assert np.all(blocked[omap.blockage])
    assert int(omap.free.sum()) < before_free
    rgb = omap.as_rgb()
    assert rgb.shape[2] == 3
    areas = omap.areas_rgb()
    assert int((areas == (196, 76, 122)).any()) or int(areas.sum()) > 0
    ids = {row["id"] for row in ObservedMap.area_legend()}
    assert "blocked" in ids


def test_no_progress_skips_frontier_and_counts_unreachable() -> None:
    omap = ObservedMap.empty(5.0, 5.0, 0.25)
    omap.observed[2:8, 2:16] = True
    omap.free[2:8, 2:16] = True
    omap.confidence[2:8, 2:16] = 0.7
    first = plan_explore(omap, (1.0, 1.0))
    assert first.target is not None
    assert first.waypoints
    assert first.unreachable_frontiers == 0

    omap.stamp_blockage(*omap.cell_to_world(*first.target), 0.55)
    second = plan_explore(
        omap,
        (1.0, 1.0),
        skip_cells=[first.target],
        avoid_xy=omap.cell_to_world(*first.target),
        cluster_cells=3,
    )
    assert second.unreachable_frontiers > 0
    assert second.skipped_frontiers >= 1
    if second.target is not None:
        assert second.target != first.target
    lips = frontiers(
        omap.observed,
        omap.free,
        ignore_unknown=omap.blockage,
    )
    assert first.target not in lips or omap.blockage[first.target]


def test_policy_no_progress_stamps_and_invalidates_frontier() -> None:
    cfg, scenario = load_source("mission_tiny")
    cfg.sensors.width = 32
    cfg.sensors.height = 24
    cfg.sensors.camera_count = 4
    cfg.sensors.cameras = []
    env = MowerEnv(config=cfg, scenario=scenario, render_mode=None)
    profile = YardProfile(
        name="block_unit",
        width_m=env.cfg.world.width_m,
        height_m=env.cfg.world.height_m,
        resolution_m=env.cfg.world.resolution_m,
        keep_in=[(0.6, 0.6), (5.2, 0.6), (5.2, 4.2), (0.6, 4.2)],
        home={"x": 1.1, "y": 1.1, "theta": 0.0},
    )
    obs, info = env.reset(seed=2, options={"yard_profile": profile, "resize_world": False})
    policy = MissionPolicy(env.cfg, settings=MissionConfig(blockage_no_progress_steps=6))
    policy.reset(obs, info, profile=profile)
    policy.phase = MissionPhase.EXPLORE
    assert policy.observed is not None
    pose = Pose(1.2, 1.2, 0.0)
    policy.explore_plan = plan_explore(policy.observed, (pose.x, pose.y))
    if policy.explore_plan is None or policy.explore_plan.target is None:
        policy.observed.stamp_disk(1.2, 1.2, 1.4, explored=True)
        policy.observed.refresh_free()
        policy.explore_plan = plan_explore(policy.observed, (pose.x, pose.y))
    assert policy.explore_plan is not None
    target = policy.explore_plan.target
    policy._last_v = 0.35
    policy._progress_pose = (pose.x, pose.y)
    policy._progress_best = 0.01
    policy._progress_map = 0.20
    stalled = False
    for _ in range(12):
        if policy._explore_no_progress(pose, info, "ok", 0.12):
            stalled = True
            break
    assert stalled
    added = policy._stamp_learned_blockage(pose, info, reason="no_progress")
    env.close()
    assert added > 0
    assert policy.observed.blockage_count() > 0
    assert policy._blocked_frontier_count >= 1
    assert policy._blockage_events >= 1
    reason = policy._build_explore_reason(0.12, info, n_frontiers=8, code="blockage_stamped")
    assert reason["unreachable_frontiers"] >= 1
    assert "Blocked" in reason["label"] or "unreachable" in reason["label"]
    if target is not None:
        assert target in policy._skipped_frontiers
        again = plan_explore(
            policy.observed,
            (pose.x, pose.y),
            skip_cells=policy._skipped_frontiers,
        )
        assert again.unreachable_frontiers >= 1


def _occlude_env() -> tuple[MowerEnv, YardProfile]:
    cfg, scenario = load_source("mission_tiny")
    cfg.sensors.width = 32
    cfg.sensors.height = 24
    cfg.sensors.camera_count = 4
    cfg.sensors.cameras = []
    cfg.max_steps = 700
    cfg.mission.full_explore = True
    cfg.mission.explore_complete = 0.80
    cfg.mission.explore_no_frontier = 0.70
    cfg.mission.max_explore_steps = 400
    cfg.mission.min_explore_steps = 40
    cfg.mission.stamp_radius_m = 0.55
    cfg.mission.camera_range_m = 2.2
    cfg.mission.blockage_no_progress_steps = 8
    cfg.mission.blockage_radius_m = 0.45
    env = MowerEnv(config=cfg, scenario=scenario, render_mode=None)
    profile = YardProfile(
        name="occlude_tiny",
        width_m=env.cfg.world.width_m,
        height_m=env.cfg.world.height_m,
        resolution_m=env.cfg.world.resolution_m,
        keep_in=[(0.55, 0.55), (5.4, 0.55), (5.4, 4.35), (0.55, 4.35)],
        home={"x": 1.15, "y": 1.15, "theta": 0.0},
    )
    return env, profile


def test_explore_occlusion_does_not_loop_same_frontier() -> None:
    env, profile = _occlude_env()
    obs, info = env.reset(seed=7, options={"yard_profile": profile, "resize_world": False})
    policy = MissionPolicy(env.cfg)
    policy.reset(obs, info, profile=profile)
    policy.attach_to_env(env)
    policy.apply_full_explore_mode(True)
    policy.settings.blockage_no_progress_steps = 8
    assert policy.phase == MissionPhase.EXPLORE
    targets: list[tuple[int, int]] = []
    start_pct = 0.0
    mid_pct = 0.0
    for step in range(260):
        action = policy.act(obs, info)
        obs, _reward, terminated, truncated, info = env.step(action)
        if policy.explore_plan is not None and policy.explore_plan.target is not None:
            targets.append(policy.explore_plan.target)
        if step == 24:
            start_pct = float(policy.status(info)["map_completion"])
        if step == 180:
            mid_pct = float(policy.status(info)["map_completion"])
        if policy.phase in {MissionPhase.REVIEW, MissionPhase.MOW, MissionPhase.COMPLETE}:
            break
        if terminated or truncated or policy.done:
            break
    status = policy.status(info)
    env.close()
    end_pct = float(status["map_completion"])
    # Past the live stuck band (~0.21) on this occluding tiny yard.
    assert end_pct >= 0.28 or status["phase"] in {"review", "mow", "return_home", "complete"}
    assert end_pct + 1e-6 >= start_pct
    unique = {t for t in targets}
    still_mapping = status["phase"] == "explore"
    if still_mapping and len(targets) >= 20:
        assert len(unique) >= 2 or int(status.get("blocked_frontiers") or 0) >= 1
    events = {e.event for e in policy.events}
    reason = status["explore_reason"]
    if "blockage_stamped" in events or int(status.get("n_blockages") or 0) > 0:
        assert int(reason.get("unreachable_frontiers") or 0) >= 1
        assert int(status.get("blocked_cells") or 0) >= 1
    if mid_pct > 0.0:
        assert end_pct + 0.01 >= mid_pct or int(status.get("n_blockages") or 0) >= 1


def test_full_explore_occlusion_climbs_past_stuck_band() -> None:
    env, profile = _occlude_env()
    obs, info = env.reset(seed=3, options={"yard_profile": profile, "resize_world": False})
    policy = MissionPolicy(env.cfg, fast=True)
    policy.reset(obs, info, profile=profile)
    policy.attach_to_env(env)
    policy.apply_full_explore_mode(True)
    policy.settings.blockage_no_progress_steps = 8
    policy.settings.explore_complete = 0.70
    seen: set[str] = {policy.phase.value}
    for _ in range(420):
        action = policy.act(obs, info)
        obs, _reward, terminated, truncated, info = env.step(action)
        seen.add(policy.phase.value)
        if policy.phase in {MissionPhase.REVIEW, MissionPhase.MOW, MissionPhase.COMPLETE}:
            break
        if terminated or truncated or policy.done:
            break
    status = policy.status(info)
    env.close()
    assert "explore" in seen
    pct = float(status["map_completion"])
    # Prove we left the ~20% thrash band. Tiny + shed can still MAP READY
    # below the acre 80% gate; the point is growth + an honest unreachable count.
    assert pct >= 0.32 or int(status.get("n_frontiers") or 0) == 0
    if status["phase"] in {"review", "mow", "return_home", "complete"}:
        assert pct >= 0.32 or int(status.get("n_frontiers") or 0) == 0
    if int(status.get("n_blockages") or 0) > 0:
        assert int(status["explore_reason"].get("unreachable_frontiers") or 0) >= 1


def test_gentle_grade_no_progress_does_not_stamp() -> None:
    cfg, scenario = load_source("mission_tiny")
    cfg.sensors.width = 32
    cfg.sensors.height = 24
    cfg.sensors.camera_count = 4
    cfg.sensors.cameras = []
    env = MowerEnv(config=cfg, scenario=scenario, render_mode=None)
    profile = YardProfile(
        name="grade_stamp",
        width_m=env.cfg.world.width_m,
        height_m=env.cfg.world.height_m,
        resolution_m=env.cfg.world.resolution_m,
        keep_in=[(0.7, 0.7), (5.0, 0.7), (5.0, 4.0), (0.7, 4.0)],
        home={"x": 1.2, "y": 1.2, "theta": 0.0},
    )
    obs, info = env.reset(seed=2, options={"yard_profile": profile, "resize_world": False})
    policy = MissionPolicy(env.cfg, settings=MissionConfig(blockage_no_progress_steps=6))
    policy.reset(obs, info, profile=profile)
    policy.phase = MissionPhase.EXPLORE
    policy.last_tilt_kind = KIND_GRADE
    policy.last_advice = "slow"
    pose = Pose(1.4, 1.4, 0.0)
    added = policy._stamp_learned_blockage(pose, info, reason="no_progress")
    env.close()
    assert added == 0
    assert policy._blockage_events == 0


def test_collision_and_lip_still_stamp() -> None:
    cfg, scenario = load_source("mission_tiny")
    cfg.sensors.width = 32
    cfg.sensors.height = 24
    cfg.sensors.camera_count = 4
    cfg.sensors.cameras = []
    env = MowerEnv(config=cfg, scenario=scenario, render_mode=None)
    profile = YardProfile(
        name="lip_stamp",
        width_m=env.cfg.world.width_m,
        height_m=env.cfg.world.height_m,
        resolution_m=env.cfg.world.resolution_m,
        keep_in=[(0.7, 0.7), (5.0, 0.7), (5.0, 4.0), (0.7, 4.0)],
        home={"x": 1.2, "y": 1.2, "theta": 0.0},
    )
    obs, info = env.reset(seed=2, options={"yard_profile": profile, "resize_world": False})
    policy = MissionPolicy(env.cfg)
    policy.reset(obs, info, profile=profile)
    policy.phase = MissionPhase.EXPLORE
    policy.last_tilt_kind = KIND_GRADE
    pose = Pose(1.5, 1.5, 0.0)
    added = policy._stamp_learned_blockage(
        pose, {**info, "collision": "tree"}, reason="collision"
    )
    first_events = policy._blockage_events
    # Cooldown: a second stamp on the same lip must not explode.
    added_again = policy._stamp_learned_blockage(
        pose, {**info, "collision": "tree"}, reason="collision"
    )
    env.close()
    assert added > 0
    assert first_events == 1
    assert added_again == 0
    assert policy._blockage_events == 1


def test_drain_lip_stamps_on_grade() -> None:
    cfg, scenario = load_source("mission_tiny")
    cfg.sensors.width = 32
    cfg.sensors.height = 24
    cfg.sensors.camera_count = 4
    cfg.sensors.cameras = []
    env = MowerEnv(config=cfg, scenario=scenario, render_mode=None)
    profile = YardProfile(
        name="drain_stamp",
        width_m=env.cfg.world.width_m,
        height_m=env.cfg.world.height_m,
        resolution_m=env.cfg.world.resolution_m,
        keep_in=[(0.7, 0.7), (5.0, 0.7), (5.0, 4.0), (0.7, 4.0)],
        home={"x": 1.2, "y": 1.2, "theta": 0.0},
    )
    obs, info = env.reset(seed=2, options={"yard_profile": profile, "resize_world": False})
    policy = MissionPolicy(env.cfg)
    policy.reset(obs, info, profile=profile)
    policy.last_tilt_kind = KIND_GRADE
    pose = Pose(1.6, 1.6, 0.0)
    added = policy._stamp_learned_blockage(
        pose,
        {**info, "terrain_reason": "drain edge — do not drop a wheel"},
        reason="drain",
    )
    env.close()
    assert added > 0
    assert policy._blockage_events == 1

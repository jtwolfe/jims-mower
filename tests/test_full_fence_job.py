"""Real gym explore→mow on a taught keep-in until the fence is actually cut.

Acre wall-clock at 0.25 m is not CI. This pocket is the honest CI proof.
Local acre long-run (not CI)::

    jims-mower-live --config acre_yard --speed max --prepare-only --steps 20000 --out live_acre_full
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from jims_mower.env import MowerEnv
from jims_mower.live import LiveSession
from jims_mower.mission_flow import MissionPhase, MissionPolicy, apply_full_explore
from jims_mower.planning.coverage import CoveragePlan
from jims_mower.planning.grade_tip import KIND_GRADE
from jims_mower.profile import YardProfile
from jims_mower.scenarios import load_source
from jims_mower.types import Pose


def _taught_tiny(env: MowerEnv) -> YardProfile:
    return YardProfile(
        name="full_fence_ci",
        width_m=env.cfg.world.width_m,
        height_m=env.cfg.world.height_m,
        resolution_m=env.cfg.world.resolution_m,
        keep_in=[(0.8, 0.8), (5.0, 0.8), (5.0, 4.0), (0.8, 4.0)],
        home={"x": 1.4, "y": 1.4, "theta": 0.0},
    )


def _tiny_job_env() -> tuple[MowerEnv, YardProfile]:
    cfg, scenario = load_source("mission_tiny")
    cfg.sensors.width = 32
    cfg.sensors.height = 24
    cfg.sensors.camera_count = 4
    cfg.sensors.cameras = []
    cfg.max_steps = 2400
    # Trimmer radius is 0.16 m. 0.40 m lanes leave half the keep-in uncut.
    cfg.planner.strip_spacing_m = 0.20
    cfg.planner.waypoint_stride_m = 0.28
    env = MowerEnv(config=cfg, scenario=scenario, render_mode=None)
    return env, _taught_tiny(env)


def _run_taught_tiny_job() -> dict[str, object]:
    env, profile = _tiny_job_env()
    obs, info = env.reset(seed=4, options={"yard_profile": profile, "resize_world": False})
    policy = MissionPolicy(env.cfg)
    policy.reset(obs, info, profile=profile)
    policy.attach_to_env(env)
    apply_full_explore(policy.settings, world_width_m=float(env.cfg.world.width_m))
    policy.settings.stamp_radius_m = 1.35
    policy.settings.explore_complete = 1.0
    policy.settings.explore_no_frontier = 0.90
    policy.settings.max_explore_steps = 480
    policy.settings.max_mow_steps = 1600
    policy.settings.max_return_steps = 120
    policy.settings.mow_complete_frac = 0.0
    policy.settings.review_hold_steps = 1
    seen: set[str] = {policy.phase.value}
    last_status: dict[str, object] = {}
    term = False
    trunc = False
    for _ in range(2200):
        if policy.phase == MissionPhase.REVIEW:
            policy.request_start_mow()
        action = policy.act(obs, info)
        obs, _reward, terminated, truncated, info = env.step(action)
        seen.add(policy.phase.value)
        last_status = policy.status(info)
        if policy.phase == MissionPhase.COMPLETE:
            break
        if terminated or truncated or policy.done:
            term, trunc = bool(terminated), bool(truncated)
            break
    leftover = [
        ev.detail
        for ev in policy.events
        if ev.event == "mow_leftover"
    ]
    summary = {
        "map_pct": float(last_status.get("map_completion") or 0.0),
        "planned_pct": float(last_status.get("planned_coverage_fraction") or 0.0),
        "cut_pct": float(last_status.get("actual_coverage_fraction") or 0.0),
        "phase": policy.phase.value,
        "seen": sorted(seen),
        "step": int(policy.step),
        "phase_step": int(policy.phase_step),
        "planned_mowable_cells": last_status.get("planned_mowable_cells"),
        "reachable_mowable_cells": last_status.get("reachable_mowable_cells"),
        "skipped_global": last_status.get("skipped_global"),
        "leftover_passes": int(policy._leftover_replans),
        "leftover_events": leftover,
        "tilt_kind": last_status.get("tilt_kind"),
        "terrain_state": last_status.get("terrain_state"),
        "n_blockages": int(last_status.get("n_blockages") or 0),
        "blocked_cells": int(last_status.get("blocked_cells") or 0),
        "chassis_tipped": bool(last_status.get("chassis_tipped")),
        "terminated": term,
        "truncated": trunc,
        "done": bool(policy.done),
    }
    env.close()
    return summary


def test_taught_tiny_explore_mow_near_complete() -> None:
    summary = _run_taught_tiny_job()
    assert "explore" in summary["seen"], summary
    assert "mow" in summary["seen"], summary
    assert summary["map_pct"] >= 0.99, summary
    assert summary["planned_pct"] >= 0.99, summary
    assert float(summary["cut_pct"]) >= 0.95, summary
    assert summary["phase"] in {"return_home", "complete"}, summary
    assert "return_home" in summary["seen"] or summary["phase"] == "complete", summary
    assert summary["chassis_tipped"] is not True, summary
    assert summary["terminated"] is not True, summary
    assert int(summary.get("n_blockages") or 0) < 80, summary


def test_mow_grade_stop_does_not_skip_plan() -> None:
    """A climbable / look-ahead face must contour, not eat the coverage plan."""
    env, profile = _tiny_job_env()
    obs, info = env.reset(seed=4, options={"yard_profile": profile, "resize_world": False})
    policy = MissionPolicy(env.cfg)
    policy.reset(obs, info, profile=profile)
    policy.phase = MissionPhase.MOW
    policy.last_tilt_kind = KIND_GRADE
    policy.global_plan = CoveragePlan(
        waypoints=[(2.0, 2.0), (3.0, 2.0), (4.0, 2.0), (5.0, 2.2)],
        planned_mowable_cells=40,
        reachable_mowable_cells=40,
        planned_coverage_fraction=1.0,
    )
    policy.index = 0
    pose = Pose(1.0, 1.0, 0.0)
    policy._tick_mow(obs, info, pose, "stop")
    env.close()
    assert policy.phase == MissionPhase.MOW
    assert policy.index == 0
    assert policy._skipped_global == []


def test_mow_blockage_stamp_does_not_reset_coverage_index() -> None:
    env, profile = _tiny_job_env()
    obs, info = env.reset(seed=4, options={"yard_profile": profile, "resize_world": False})
    policy = MissionPolicy(env.cfg)
    policy.reset(obs, info, profile=profile)
    policy.phase = MissionPhase.MOW
    policy.global_plan = CoveragePlan(
        waypoints=[(2.0, 2.0), (3.0, 2.0), (4.0, 2.0), (5.0, 2.0)],
        planned_mowable_cells=20,
        reachable_mowable_cells=20,
    )
    policy.index = 2
    pose = Pose(3.0, 2.0, 0.0)
    policy._stamp_learned_blockage(pose, info, reason="mow_no_progress")
    env.close()
    assert policy.index == 2
    assert policy.phase == MissionPhase.MOW


def test_mow_leftover_replans_when_waypoints_done() -> None:
    env, profile = _tiny_job_env()
    obs, info = env.reset(seed=4, options={"yard_profile": profile, "resize_world": False})
    policy = MissionPolicy(env.cfg)
    policy.reset(obs, info, profile=profile)
    policy.attach_to_env(env)
    policy.phase = MissionPhase.MOW
    omap = policy.observed
    assert omap is not None
    omap.observed[:, :] = True
    omap.free[:, :] = True
    omap.confidence[:, :] = 0.85
    omap.refresh_free()
    policy.keep_in_mask = np.ones(omap.observed.shape, dtype=bool)
    policy.global_plan = CoveragePlan(
        waypoints=[(1.4, 1.4)],
        planned_mowable_cells=40,
        reachable_mowable_cells=40,
        planned_coverage_fraction=1.0,
    )
    policy.index = 1
    cut = np.zeros_like(omap.observed, dtype=bool)
    info = dict(info)
    info["coverage_cut"] = cut
    info["coverage_cut_cells"] = 0
    policy._last_info = info
    policy._tick_mow(obs, info, Pose(1.4, 1.4, 0.0), "ok")
    env.close()
    assert policy.phase == MissionPhase.MOW
    assert policy._leftover_replans >= 1
    assert len(policy.global_plan.waypoints) >= 2


def test_mission_mow_tipover_does_not_end_episode() -> None:
    env, profile = _tiny_job_env()
    obs, info = env.reset(seed=1, options={"yard_profile": profile, "resize_world": False})
    env._mission_phase = "mow"
    start = Pose(env._pose.x, env._pose.y, env._pose.theta)
    env._prev_pose = start
    from jims_mower.kinematics import sit_on_terrain
    from jims_mower.terrain import HeightField

    hf = HeightField.from_function(8.0, 6.0, 0.10, lambda x, y: 2.2 * y)
    env._terrain = hf
    env._pose = sit_on_terrain(Pose(4.0, 3.0, 0.0), hf, 0.70, 0.55)
    obs, _reward, terminated, _trunc, info = env.step(np.array([0.4, 0.4, 0.0], dtype=np.float32))
    env.close()
    assert info.get("tipover") is True
    assert terminated is False


def test_live_reset_lands_idle_ready_at_1x(tmp_path: Path) -> None:
    session = LiveSession(
        config="mission_tiny",
        fast=True,
        speed="5",
        steps=40,
        seed=2,
        cameras=4,
        out_dir=tmp_path / "full-fence-reset",
        cam_stride=80,
        map_stride=8,
        require_pair=False,
    )
    session.reset()
    session.control("pair")
    saved = session.control(
        "save_yard",
        keep_in=[[0.8, 0.8], [5.0, 0.8], [5.0, 4.0], [0.8, 4.0]],
    )
    assert saved.get("ok") is True
    session.control("speed", speed="5")
    assert session.speed == 5.0
    out = session.control("reset")
    session.close()
    assert out["ok"] is True
    assert out["job_state"] == "idle"
    assert out.get("speed_label") == "1"
    assert out.get("kept_fence") is True
    assert out.get("kept_blockages") is False
    assert "Ready" in str((out.get("mode_banner") or {}).get("label") or "")
    assert out.get("chassis_tipped") is not True

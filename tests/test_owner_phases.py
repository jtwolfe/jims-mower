"""Manual phases, explore reason, full explore, area legend, dock-resume."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from jims_mower.constants import (
    STRUCTURE_BUILDING,
    STRUCTURE_BUNKER,
    STRUCTURE_GARDEN,
    STRUCTURE_PATH,
    STRUCTURE_POND,
)
from jims_mower.env import MowerEnv
from jims_mower.live import LiveSession, owner_copy_for
from jims_mower.mission_flow import MissionPhase, MissionPolicy, apply_full_explore
from jims_mower.planning.coverage import CoveragePlan
from jims_mower.planning.observed import ObservedMap
from jims_mower.profile import YardProfile
from jims_mower.scenarios import load_source


def _tiny_env() -> MowerEnv:
    cfg, scenario = load_source("mission_tiny")
    cfg.sensors.width = 32
    cfg.sensors.height = 24
    cfg.sensors.camera_count = 4
    cfg.sensors.cameras = []
    cfg.max_steps = 900
    return MowerEnv(config=cfg, scenario=scenario, render_mode=None)


def _halt_live(session: LiveSession) -> None:
    session._stop.set()
    if session._thread is not None:
        session._thread.join(timeout=2.0)
        session._thread = None


def _taught(env: MowerEnv) -> YardProfile:
    return YardProfile(
        name="taught_tiny",
        width_m=env.cfg.world.width_m,
        height_m=env.cfg.world.height_m,
        resolution_m=env.cfg.world.resolution_m,
        keep_in=[(0.7, 0.7), (5.0, 0.7), (5.0, 4.0), (0.7, 4.0)],
        home={"x": 1.2, "y": 1.2, "theta": 0.0},
        schedule={"enabled": False, "min_soc": 0.25, "timezone": "Australia/Brisbane"},
    )


def test_explore_reason_field_on_status() -> None:
    env = _tiny_env()
    obs, info = env.reset(seed=3, options={"yard_profile": _taught(env), "resize_world": False})
    policy = MissionPolicy(env.cfg, fast=True)
    policy.reset(obs, info, profile=_taught(env))
    policy.attach_to_env(env)
    assert policy.phase == MissionPhase.EXPLORE
    for _ in range(16):
        action = policy.act(obs, info)
        obs, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    status = policy.status(info)
    env.close()
    reason = status["explore_reason"]
    assert reason["code"] in {
        "seeking_frontier",
        "path_blocked",
        "tip_recovery",
        "map_progress",
        "waiting_cap",
        "no_frontier",
        "stalled",
        "idle",
    }
    assert "map" in reason["label"]
    assert "step" in reason["label"]
    assert 0.0 <= float(reason["map_pct"]) <= 1.0
    assert float(reason["target_pct"]) > 0.0
    assert int(reason["max_steps"]) >= int(reason["phase_step"])


def test_manual_explore_mow_return_controls(tmp_path: Path) -> None:
    session = LiveSession(
        config="mission_tiny",
        fast=True,
        speed="max",
        steps=200,
        seed=2,
        cameras=4,
        out_dir=tmp_path / "manual-phases",
        require_pair=False,
    )
    session.reset()
    session.pairing.force_paired()
    explored = session.control("explore")
    assert explored["ok"] is True
    assert session.policy is not None
    _halt_live(session)
    session.job_state = "running"
    for _ in range(10):
        session.step_once()
    assert session.policy.phase.value in {
        "calibrate_boundary",
        "explore",
        "review",
        "mow",
    }
    returned = session.control("return")
    assert returned["ok"] is True
    _halt_live(session)
    assert session.policy.phase.value == "return_home"
    session.control("explore")
    _halt_live(session)
    session.step_once()
    mowed = session.control("mow")
    assert mowed["ok"] is True
    _halt_live(session)
    snap = session.snapshot()
    assert "explore_reason" in snap
    assert snap.get("can_explore") is True
    session.close()


def test_full_explore_reaches_map_ready_with_mowable_plan() -> None:
    env = _tiny_env()
    profile = _taught(env)
    obs, info = env.reset(seed=4, options={"yard_profile": profile, "resize_world": False})
    policy = MissionPolicy(env.cfg, fast=True)
    policy.reset(obs, info, profile=profile)
    policy.attach_to_env(env)
    policy.apply_full_explore_mode(True)
    assert policy.settings.full_explore is True
    assert policy.settings.explore_complete >= 0.70
    assert policy.settings.mow_complete_frac == 0.0
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
    if policy.phase == MissionPhase.REVIEW and (
        policy.global_plan is None or len(getattr(policy.global_plan, "waypoints", [])) < 2
    ):
        policy.request_start_mow()
    env.close()
    assert "explore" in seen
    assert policy.phase.value in {"review", "mow", "return_home", "complete"}
    assert float(status["map_completion"]) >= 0.62 or int(status["n_frontiers"]) == 0
    assert int(status.get("planned_mowable_cells") or 0) > 0 or (
        policy.global_plan is not None and len(policy.global_plan.waypoints) >= 2
    )


def test_apply_full_explore_raises_acre_demo_caps() -> None:
    from jims_mower.config import MissionConfig

    demo = MissionConfig(explore_complete=0.30, max_explore_steps=420, mow_complete_frac=0.10)
    apply_full_explore(demo, world_width_m=70.0)
    assert demo.full_explore is True
    assert demo.explore_complete >= 0.80
    assert demo.max_explore_steps >= 4000
    assert demo.mow_complete_frac == 0.0


def test_area_type_overlay_and_legend() -> None:
    omap = ObservedMap.empty(4.0, 4.0, 0.25)
    omap.observed[:, :] = True
    omap.explored[:, :] = True
    omap.free[:, :] = True
    omap.structure[2:4, 2:6] = STRUCTURE_PATH
    omap.structure[8:12, 8:12] = STRUCTURE_BUILDING
    omap.structure[4:6, 10:14] = STRUCTURE_POND
    omap.structure[10:12, 2:5] = STRUCTURE_BUNKER
    omap.structure[1:3, 10:13] = STRUCTURE_GARDEN
    omap.refresh_free()
    rgb = omap.areas_rgb()
    assert rgb.shape[2] == 3
    assert rgb.dtype == np.uint8
    assert int(rgb.sum()) > 0
    painted = omap.as_rgb()
    assert painted.shape == rgb.shape
    legend = ObservedMap.area_legend()
    ids = {row["id"] for row in legend}
    assert {
        "grass",
        "mowable",
        "path",
        "sand",
        "building",
        "water",
        "drain",
        "beds",
        "keepout",
        "fog",
    } <= ids
    assert all(row.get("label") and row.get("color") for row in legend)


def test_low_soc_inject_return_charge_resume() -> None:
    env = _tiny_env()
    profile = _taught(env)
    obs, info = env.reset(seed=5, options={"yard_profile": profile, "resize_world": False})
    policy = MissionPolicy(env.cfg, fast=True)
    policy.reset(obs, info, profile=profile)
    policy.attach_to_env(env)
    policy.settings.min_soc_return = 0.25
    policy.settings.charge_resume_soc = 0.70
    policy.settings.gym_charge_soc_per_step = 0.20
    env._charge_delta = 0.20
    seen: list[str] = []
    for _ in range(280):
        if policy.phase.value not in seen:
            seen.append(policy.phase.value)
        action = policy.act(obs, info)
        obs, _reward, terminated, truncated, info = env.step(action)
        if policy.phase == MissionPhase.MOW and policy.phase_step >= 6:
            break
        if policy.phase == MissionPhase.REVIEW:
            if policy.global_plan is None or len(getattr(policy.global_plan, "waypoints", [])) < 2:
                policy.global_plan = CoveragePlan(
                    waypoints=[(1.3, 1.3), (2.2, 1.6), (3.1, 2.0), (3.8, 2.6)]
                )
                policy.plan = policy.global_plan
                policy.index = 0
            policy.request_start_mow()
        if terminated or truncated or policy.done:
            break
    if policy.phase != MissionPhase.MOW:
        policy.global_plan = CoveragePlan(
            waypoints=[(1.3, 1.3), (2.2, 1.6), (3.1, 2.0), (3.8, 2.6), (4.2, 3.0)]
        )
        policy.plan = policy.global_plan
        policy.index = 1
        policy._mow_requested = True
        policy._transition(MissionPhase.MOW)
    leftover = int(policy.index)
    env.budget.set_soc(0.10)
    info = dict(info or {})
    info.update(env.budget.as_info())
    charged = False
    resumed = False
    for _ in range(220):
        if policy.phase.value not in seen:
            seen.append(policy.phase.value)
        action = policy.act(obs, info)
        obs, _reward, terminated, truncated, info = env.step(action)
        if policy.phase == MissionPhase.CHARGING:
            charged = True
        if charged and policy.phase == MissionPhase.MOW:
            resumed = True
            break
        if terminated or truncated or policy.done:
            break
    env.close()
    events = {e.event for e in policy.events}
    assert "battery_return" in events
    assert "return_home" in seen or "charging" in seen or MissionPhase.CHARGING.value in seen
    assert charged
    assert resumed
    assert policy.global_plan is not None
    assert policy.index >= leftover
    copy = owner_copy_for("running", "return_home", return_kind="battery")
    assert copy == "Low battery — returning to charge"
    assert owner_copy_for("running", "charging") == "Charging…"
    assert owner_copy_for("running", "mow", charge_state="resuming") == "Resuming mow"


def test_live_low_soc_inject(tmp_path: Path) -> None:
    session = LiveSession(
        config="mission_tiny",
        fast=True,
        speed="max",
        steps=80,
        seed=2,
        cameras=4,
        out_dir=tmp_path / "soc-inject",
        require_pair=False,
    )
    session.reset()
    session.pairing.force_paired()
    session.control("start")
    _halt_live(session)
    session.control("inject", kind="low_soc", soc=0.11)
    assert session.env is not None
    assert session.env.budget.soc <= 0.12
    snap = session.snapshot()
    assert float((session.info or {}).get("battery_soc") or 1.0) <= 0.12
    session.close()
    assert snap["schema"]

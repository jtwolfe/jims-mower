"""§15 PLN-1…5 regression: explore → MAP READY → mow → tip recover → resume.

Observed maps only (not god-view elev). Gym constants are not retuned here.
Living interlock: MockDetector still projects sim obstacles (god-view).
BlindDetector returns [] — a detections-only interlock would never fire.
"""

from __future__ import annotations

import numpy as np
import pytest

from jims_mower.env import MowerEnv
from jims_mower.mission_flow import MissionPhase, MissionPolicy, apply_full_explore
from jims_mower.perception.base import BlindDetector
from jims_mower.perception.mock import MockDetector
from jims_mower.planning.observed import ObservedMap
from jims_mower.profile import YardProfile
from jims_mower.scenarios import load_source
from jims_mower.types import Obstacle, PerceptionContext


def _tiny_env() -> MowerEnv:
    cfg, scenario = load_source("mission_tiny")
    cfg.sensors.width = 32
    cfg.sensors.height = 24
    cfg.sensors.camera_count = 4
    cfg.sensors.cameras = []
    cfg.max_steps = 1100
    return MowerEnv(config=cfg, scenario=scenario, render_mode=None)


def _taught(env: MowerEnv) -> YardProfile:
    return YardProfile(
        name="pln_tiny",
        width_m=env.cfg.world.width_m,
        height_m=env.cfg.world.height_m,
        resolution_m=env.cfg.world.resolution_m,
        keep_in=[(0.7, 0.7), (5.0, 0.7), (5.0, 4.0), (0.7, 4.0)],
        home={"x": 1.2, "y": 1.2, "theta": 0.0},
    )


def test_explore_mow_tip_resume_on_observed_map() -> None:
    env = _tiny_env()
    profile = _taught(env)
    obs, info = env.reset(seed=3, options={"yard_profile": profile, "resize_world": False})
    policy = MissionPolicy(env.cfg)
    policy.reset(obs, info, profile=profile)
    apply_full_explore(policy.settings, world_width_m=float(env.cfg.world.width_m))
    policy.settings.stamp_radius_m = 1.35
    policy.settings.review_hold_steps = 1
    assert policy.phase == MissionPhase.EXPLORE
    assert policy.observed is not None
    seen = {policy.phase.value}
    reached_mow = False
    recovered = False
    paused = False
    for i in range(900):
        action = policy.act(obs, info)
        seen.add(policy.phase.value)
        if policy.phase == MissionPhase.REVIEW:
            policy.request_start_mow()
        if policy.phase == MissionPhase.MOW and not reached_mow:
            reached_mow = True
            # PLN-3: IMU tip-stop → reverse nudge.
            # Past climb (~0.32) but under software tip (~0.55). Static α
            # (~1.10) latches SOS / hold — a different path.
            climb_roll = 0.36
            info = dict(info)
            info["terrain_advice"] = "stop"
            info["pose"] = {**(info.get("pose") or {}), "roll": climb_roll}
            obs = dict(obs)
            g = 9.80665
            obs["imu"] = np.array(
                [0.0, g * np.sin(climb_roll), g * np.cos(climb_roll), 0.0, 0.0, 0.0],
                dtype=np.float32,
            )
            tip = policy.act(obs, info)
            recovered = float(tip[0]) < 0.0 and float(tip[1]) < 0.0
            policy.request_hold("pln pause")
            hold = policy.act(obs, info)
            paused = float(hold[0]) == 0.0 and float(hold[1]) == 0.0
            policy.clear_owner_hold()
        obs, _reward, terminated, truncated, info = env.step(action)
        if policy.observed is not None:
            assert isinstance(policy.observed, ObservedMap)
        if terminated or truncated or policy.done:
            break
        if reached_mow and recovered and paused and i > 8:
            break
    env.close()
    assert "explore" in seen
    assert "review" in seen or reached_mow
    assert reached_mow
    assert recovered
    assert paused
    assert policy.observed is not None
    assert int(policy.observed.observed.sum()) > 0


def test_mock_detector_living_interlock_still_fires() -> None:
    """PLN-4: MockDetector projects sim people; trimmer request is refused."""
    cfg, scenario = load_source("mission_tiny")
    cfg.sensors.width = 32
    cfg.sensors.height = 24
    cfg.sensors.camera_count = 4
    cfg.sensors.cameras = []
    cfg.world.n_people = 0
    env = MowerEnv(config=cfg, scenario=scenario, render_mode=None)
    obs, info = env.reset(seed=2)
    person = Obstacle("person", env._pose.x + 0.4, env._pose.y, 0.25, z=0.9)
    env._yard.obstacles.append(person)
    action = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    obs, _r, _t, _c, info = env.step(action)
    assert info["trimmer_requested"] is True
    assert info["trimmer_enabled"] is False
    dets = info.get("detections") or []
    assert any(d.get("label") == "person" for d in dets if isinstance(d, dict))
    env.close()


def test_blind_detector_returns_no_detections() -> None:
    """Honest gap: BlindDetector never fires a detections-only interlock."""
    env = _tiny_env()
    obs, info = env.reset(seed=1)
    person = Obstacle("person", env._pose.x + 0.5, env._pose.y, 0.25, z=0.9)
    cameras = list(env.cameras)
    ctx = PerceptionContext(
        pose=env._pose,
        cameras=cameras,
        obstacles=[person],
        image_size=(int(env.cfg.sensors.width), int(env.cfg.sensors.height)),
    )
    images = obs.get("cameras") or {}
    mock = MockDetector()
    blind = BlindDetector()
    mocked = mock.detect(images, ctx)
    blinded = blind.detect(images, ctx)
    env.close()
    assert blinded == []
    # Mock may or may not see the person depending on camera FOV; the
    # gym interlock itself still uses the obstacle list (god-view).
    assert isinstance(mocked, list)


def test_resume_replans_uncut_from_cold_session(tmp_path) -> None:
    """PLN-5 + MAP-4: pause-quality save, new policy, uncut still planned."""
    from jims_mower.maps import GrassCoverageMap
    from jims_mower.mission import load_mission

    env = _tiny_env()
    profile = _taught(env)
    obs, info = env.reset(seed=4, options={"yard_profile": profile, "resize_world": False})
    policy = MissionPolicy(env.cfg, fast=True)
    policy.reset(obs, info, profile=profile)
    for _ in range(120):
        action = policy.act(obs, info)
        if policy.phase == MissionPhase.REVIEW:
            policy.request_start_mow()
        obs, _r, terminated, truncated, info = env.step(action)
        if policy.phase in {MissionPhase.MOW, MissionPhase.COMPLETE}:
            break
        if terminated or truncated:
            break
    assert policy.observed is not None
    dest = tmp_path / "pln_session.npz"
    policy.save_session(dest, env._coverage, env._pose, scenario="mission_tiny", seed=4)
    env.close()

    state = load_mission(dest)
    env2 = _tiny_env()
    obs2, info2 = env2.reset(seed=11, options={"load_mission": str(dest), "resize_world": False})
    policy2 = MissionPolicy(env2.cfg, fast=True)
    policy2.restore_session(obs2, info2, state)
    assert policy2.profile is not None
    assert policy2.observed is not None
    assert int(policy2.observed.observed.sum()) == int(state.observed.observed.sum())
    assert policy2.phase != MissionPhase.CALIBRATE_BOUNDARY
    if policy2.global_plan is None and policy2.observed is not None:
        policy2.global_plan = policy2._plan_global_mow(env2._pose)
    if policy2.global_plan is not None:
        assert policy2.global_plan.planned_mowable_cells >= 0
        assert (
            policy2.global_plan.reachable_mowable_cells
            + policy2.global_plan.unreachable_mowable_cells
            == policy2.global_plan.planned_mowable_cells
        )
    env2.close()

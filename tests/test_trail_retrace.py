"""Trail retrace: follow breadcrumbs downhill, never reverse into SOS."""

from __future__ import annotations

from jims_mower.env import MowerEnv
from jims_mower.mission_flow import MissionPhase, MissionPolicy
from jims_mower.planning.terrain_decision import TERRAIN_RETRACE, retrace_waypoints
from jims_mower.profile import YardProfile
from jims_mower.scenarios import load_source
from jims_mower.types import Pose


def _tiny() -> tuple[MowerEnv, YardProfile]:
    cfg, scenario = load_source("mission_tiny")
    cfg.sensors.width = 32
    cfg.sensors.height = 24
    cfg.sensors.camera_count = 4
    cfg.sensors.cameras = []
    env = MowerEnv(config=cfg, scenario=scenario, render_mode=None)
    profile = YardProfile(
        name="retrace_tiny",
        width_m=env.cfg.world.width_m,
        height_m=env.cfg.world.height_m,
        resolution_m=env.cfg.world.resolution_m,
        keep_in=[(0.7, 0.7), (5.0, 0.7), (5.0, 4.0), (0.7, 4.0)],
        home={"x": 1.2, "y": 1.2, "theta": 0.0},
    )
    return env, profile


def test_retrace_waypoints_follow_trail_order() -> None:
    trail = [(0.0, 0.0, 0.0), (0.4, 0.1, 0.2), (0.8, 0.2, 0.2), (1.2, 0.4, 0.3)]
    wps = retrace_waypoints(trail, (1.25, 0.42), length_m=1.2)
    assert len(wps) >= 2
    # First retrace point is the most recent crumb behind the pose.
    assert wps[0][0] >= wps[-1][0]


def test_uphill_dead_end_retraces_downhill_not_stamp() -> None:
    env, profile = _tiny()
    obs, info = env.reset(seed=3, options={"yard_profile": profile, "resize_world": False})
    policy = MissionPolicy(env.cfg, fast=True)
    policy.reset(obs, info, profile=profile)
    policy.phase = MissionPhase.EXPLORE
    # Climbed +y. Pose at the dead-end top.
    policy._pose_trail = [
        (1.4, 1.2, 1.57),
        (1.4, 1.8, 1.57),
        (1.4, 2.4, 1.57),
        (1.4, 3.0, 1.57),
    ]
    policy.last_tilt_kind = "grade"
    policy.last_advice = "slow"
    pose = Pose(1.42, 3.05, 1.57)
    before = policy._blockage_events
    added = policy._stamp_learned_blockage(pose, info, reason="no_progress")
    action = policy._retrace_or_nudge(pose, reason="no_progress")
    env.close()
    assert added == 0
    assert policy._blockage_events == before
    assert policy._retrace_wps
    assert policy.terrain_state == TERRAIN_RETRACE
    ys = [p[1] for p in policy._retrace_wps]
    assert ys == sorted(ys, reverse=True)
    assert ys[-1] < 2.5
    assert float(action[0]) != 0.0 or float(action[1]) != 0.0


def test_immobilised_does_not_retrace() -> None:
    env, profile = _tiny()
    obs, info = env.reset(seed=1, options={"yard_profile": profile, "resize_world": False})
    policy = MissionPolicy(env.cfg, fast=True)
    policy.reset(obs, info, profile=profile)
    policy._pose_trail = [(1.2, 1.2, 0.0), (1.8, 1.4, 0.2), (2.4, 1.6, 0.2)]
    policy.latch_chassis_tip()
    pose = Pose(2.4, 1.6, 0.2)
    started = policy._start_retrace(pose, reason="tip_risk")
    action = policy._retrace_or_nudge(pose, reason="tip_risk")
    env.close()
    assert started is False
    assert policy._retrace_wps == []
    assert float(action[0]) == 0.0
    assert float(action[1]) == 0.0

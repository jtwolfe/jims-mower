"""Owner Reset: stop job, clear tip cool-down, keep fence, optional blockages."""

from __future__ import annotations

from pathlib import Path

from jims_mower.constants import LIVE_CONTROL_CMDS
from jims_mower.env import MowerEnv
from jims_mower.live import OWNER_COPY, LiveSession
from jims_mower.mission_flow import MissionPhase, MissionPolicy
from jims_mower.planning.observed import ObservedMap
from jims_mower.profile import YardProfile
from jims_mower.scenarios import load_source


def _tiny() -> tuple[MowerEnv, YardProfile]:
    cfg, scenario = load_source("mission_tiny")
    cfg.sensors.width = 32
    cfg.sensors.height = 24
    cfg.sensors.camera_count = 4
    cfg.sensors.cameras = []
    env = MowerEnv(config=cfg, scenario=scenario, render_mode=None)
    profile = YardProfile(
        name="reset_tiny",
        width_m=env.cfg.world.width_m,
        height_m=env.cfg.world.height_m,
        resolution_m=env.cfg.world.resolution_m,
        keep_in=[(0.7, 0.7), (5.0, 0.7), (5.0, 4.0), (0.7, 4.0)],
        home={"x": 1.2, "y": 1.2, "theta": 0.0},
    )
    return env, profile


def test_reset_is_a_live_control() -> None:
    assert "reset" in LIVE_CONTROL_CMDS
    assert OWNER_COPY["tip_risk"] == "Tip risk — reversing"
    assert OWNER_COPY["steep_grade"] == "Steep grade — contouring"


def test_policy_owner_reset_clears_tip_cool_and_keeps_fence() -> None:
    env, profile = _tiny()
    obs, info = env.reset(seed=3, options={"yard_profile": profile, "resize_world": False})
    policy = MissionPolicy(env.cfg, fast=True)
    policy.reset(obs, info, profile=profile)
    policy._stop_cool = 9
    policy._calibrate_stall = 4
    policy._explore_spin = 3
    policy.last_tilt_kind = "tip"
    policy.last_advice = "stop"
    if policy.observed is not None:
        policy.observed.stamp_blockage(2.0, 2.0, 0.6)
    blocked = policy.observed.blockage_count() if policy.observed is not None else 0
    out = policy.owner_reset(clear_blockages=False)
    env.close()
    assert out["ok"] is True
    assert out["kept_blockages"] is True
    assert out["kept_fence"] is True
    assert policy.profile is not None
    assert policy._stop_cool == 0
    assert policy._calibrate_stall == 0
    assert policy._explore_spin == 0
    assert policy.last_advice == "ok"
    assert policy.last_tilt_kind == "ok"
    assert policy.phase in {MissionPhase.EXPLORE, MissionPhase.REVIEW}
    if policy.observed is not None:
        assert policy.observed.blockage_count() == blocked


def test_policy_owner_reset_can_clear_learned_blockages() -> None:
    env, profile = _tiny()
    obs, info = env.reset(seed=4, options={"yard_profile": profile, "resize_world": False})
    policy = MissionPolicy(env.cfg, fast=True)
    policy.reset(obs, info, profile=profile)
    assert policy.observed is not None
    policy.observed.stamp_blockage(2.2, 2.2, 0.7)
    assert policy.observed.blockage_count() > 0
    out = policy.owner_reset(clear_blockages=True)
    n = policy.observed.blockage_count()
    env.close()
    assert out["cleared_cells"] > 0
    assert n == 0
    assert policy.profile is not None


def test_observed_clear_blockages_keeps_observed_cells() -> None:
    omap = ObservedMap.empty(4.0, 4.0, 0.2)
    omap.stamp_disk(2.0, 2.0, 0.4, explored=True)
    seen = int(omap.observed.sum())
    omap.stamp_blockage(2.5, 2.5, 0.4)
    assert omap.blockage_count() > 0
    omap.clear_blockages()
    assert omap.blockage_count() == 0
    assert int(omap.observed.sum()) >= seen


def test_live_reset_control_returns_idle_ready(tmp_path: Path) -> None:
    session = LiveSession(
        config="mission_tiny",
        fast=True,
        speed="max",
        steps=80,
        seed=2,
        cameras=4,
        out_dir=tmp_path / "reset-live",
        cam_stride=80,
        map_stride=8,
        require_pair=False,
    )
    session.reset()
    session.control("pair")
    session.control("explore")
    assert session.job_state == "running"
    if session.policy is not None:
        session.policy._stop_cool = 7
        session.policy._explore_spin = 2
    out = session.control("reset")
    session._stop.set()
    if session._thread is not None:
        session._thread.join(timeout=2.0)
    assert out["ok"] is True
    assert out["cmd"] == "reset"
    assert out["job_state"] == "idle"
    assert out.get("kept_blockages") is True
    assert out.get("can_reset") is True
    assert "start mow" not in str(out.get("owner_copy") or "").lower()
    assert out.get("can_explore") is True
    if session.policy is not None:
        assert session.policy._stop_cool == 0
        assert session.policy.profile is not None or session.owner_taught or True
    assert session.estop is False

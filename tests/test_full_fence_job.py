"""Real gym explore→mow on a taught keep-in until near-full fence coverage.

Acre wall-clock at 0.25 m is not CI. This pocket is the honest CI proof.
Local acre long-run: ``jims-mower-live --config acre_yard --speed max --prepare-only --steps 20000``.
"""

from __future__ import annotations

from pathlib import Path

from jims_mower.env import MowerEnv
from jims_mower.live import LiveSession
from jims_mower.mission_flow import MissionPhase, MissionPolicy, apply_full_explore
from jims_mower.profile import YardProfile
from jims_mower.scenarios import load_source


def _taught_tiny(env: MowerEnv) -> YardProfile:
    return YardProfile(
        name="full_fence_ci",
        width_m=env.cfg.world.width_m,
        height_m=env.cfg.world.height_m,
        resolution_m=env.cfg.world.resolution_m,
        keep_in=[(0.8, 0.8), (5.0, 0.8), (5.0, 4.0), (0.8, 4.0)],
        home={"x": 1.4, "y": 1.4, "theta": 0.0},
    )


def test_taught_tiny_explore_mow_near_complete() -> None:
    cfg, scenario = load_source("mission_tiny")
    cfg.sensors.width = 32
    cfg.sensors.height = 24
    cfg.sensors.camera_count = 4
    cfg.sensors.cameras = []
    cfg.max_steps = 900
    env = MowerEnv(config=cfg, scenario=scenario, render_mode=None)
    profile = _taught_tiny(env)
    obs, info = env.reset(seed=4, options={"yard_profile": profile, "resize_world": False})
    policy = MissionPolicy(env.cfg)
    policy.reset(obs, info, profile=profile)
    policy.attach_to_env(env)
    apply_full_explore(policy.settings, world_width_m=float(cfg.world.width_m))
    policy.settings.explore_complete = 0.95
    policy.settings.explore_no_frontier = 0.80
    policy.settings.max_explore_steps = 420
    policy.settings.max_mow_steps = 480
    policy.settings.mow_complete_frac = 0.0
    policy.settings.review_hold_steps = 1
    seen: set[str] = {policy.phase.value}
    last_status = {}
    for _ in range(820):
        if policy.phase == MissionPhase.REVIEW:
            policy.request_start_mow()
        action = policy.act(obs, info)
        obs, _reward, terminated, truncated, info = env.step(action)
        seen.add(policy.phase.value)
        last_status = policy.status(info)
        if policy.phase in {MissionPhase.COMPLETE, MissionPhase.RETURN_HOME}:
            break
        if policy.phase == MissionPhase.MOW:
            planned = float(last_status.get("planned_coverage_fraction") or 0.0)
            actual = float(last_status.get("actual_coverage_fraction") or 0.0)
            if planned >= 0.90 or actual >= 0.85:
                break
        if terminated or truncated or policy.done:
            break
    summary = {
        "map_pct": float(last_status.get("map_completion") or 0.0),
        "planned_pct": float(last_status.get("planned_coverage_fraction") or 0.0),
        "cut_pct": float(last_status.get("actual_coverage_fraction") or 0.0),
        "phase": policy.phase.value,
        "seen": sorted(seen),
        "step": int(policy.step),
    }
    env.close()
    assert "explore" in seen, summary
    assert "mow" in seen or policy.phase == MissionPhase.MOW, summary
    assert summary["map_pct"] >= 0.80, summary
    assert summary["planned_pct"] >= 0.70 or summary["cut_pct"] >= 0.70, summary
    assert policy.phase.value in {"mow", "return_home", "complete", "review"}, summary


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

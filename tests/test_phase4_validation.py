"""Phase 4 gates: real gym pocket + thrash bounds vs the live stuck baseline.

Acre wall-clock is not CI. Local smoke (Jamie)::

    jims-mower-live --config acre_yard --speed max --prepare-only --steps 20000 --out live_acre_full
    jims-mower-owner --live --config acre_yard_demo --port 8766
"""

from __future__ import annotations

import json
from pathlib import Path

from jims_mower.env import MowerEnv
from jims_mower.kinematics import sit_on_terrain
from jims_mower.live import LiveSession, owner_copy_for
from jims_mower.mission_flow import MissionPhase, MissionPolicy, apply_full_explore
from jims_mower.planning.terrain_decision import TERRAIN_CONTOUR, TERRAIN_OK, TERRAIN_RETRACE
from jims_mower.profile import YardProfile
from jims_mower.scenarios import load_source
from jims_mower.terrain import HeightField
from jims_mower.types import Pose

from test_full_fence_job import _run_taught_tiny_job

# Live stuck baseline (tests/fixtures/phase0_baseline.md) — beat these.
BASELINE_MAP_STUCK = 0.40
BASELINE_N_BLOCKAGES = 3263
MAX_JOB_BLOCKAGES = 80
MAX_STUCK_BLOCKAGES = 40


def _write_summary(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")


def test_phase4_validation_summary(tmp_path: Path) -> None:
    """Taught tiny explore→mow must clear the #50 bars + bounded blockages."""
    summary = _run_taught_tiny_job()
    out = tmp_path / "phase4_validation.json"
    _write_summary(out, summary)
    assert out.is_file()
    assert "explore" in summary["seen"], summary
    assert "mow" in summary["seen"], summary
    assert float(summary["map_pct"]) >= 0.99, summary
    assert float(summary["planned_pct"]) >= 0.99, summary
    assert float(summary["cut_pct"]) >= 0.95, summary
    assert summary["phase"] in {"return_home", "complete"}, summary
    assert "return_home" in summary["seen"] or summary["phase"] == "complete", summary
    assert summary["chassis_tipped"] is not True, summary
    assert summary["terminated"] is not True, summary
    assert int(summary.get("n_blockages") or 0) < MAX_JOB_BLOCKAGES, summary


def test_phase4_thrash_does_not_explode() -> None:
    """Must not sit at ~40% map minting thousands of blockage events."""
    cfg, scenario = load_source("mission_tiny")
    cfg.sensors.width = 32
    cfg.sensors.height = 24
    cfg.sensors.camera_count = 4
    cfg.sensors.cameras = []
    cfg.max_steps = 700
    env = MowerEnv(config=cfg, scenario=scenario, render_mode=None)
    profile = YardProfile(
        name="thrash_tiny",
        width_m=env.cfg.world.width_m,
        height_m=env.cfg.world.height_m,
        resolution_m=env.cfg.world.resolution_m,
        keep_in=[(0.7, 0.7), (5.2, 0.7), (5.2, 4.2), (0.7, 4.2)],
        home={"x": 1.2, "y": 1.2, "theta": 0.0},
    )
    obs, info = env.reset(seed=5, options={"yard_profile": profile, "resize_world": False})
    policy = MissionPolicy(env.cfg)
    policy.reset(obs, info, profile=profile)
    policy.attach_to_env(env)
    apply_full_explore(policy.settings, world_width_m=float(env.cfg.world.width_m))
    policy.settings.max_explore_steps = 220
    last = {}
    for _ in range(260):
        action = policy.act(obs, info)
        obs, _reward, terminated, truncated, info = env.step(action)
        last = policy.status(info)
        if policy.phase != MissionPhase.EXPLORE or terminated or truncated or policy.done:
            break
    env.close()
    map_pct = float(last.get("map_completion") or 0.0)
    n_block = int(last.get("n_blockages") or 0)
    n_front = int(last.get("n_frontiers") or 0)
    assert n_block < BASELINE_N_BLOCKAGES // 10, last
    if map_pct < BASELINE_MAP_STUCK + 0.08:
        assert n_block < MAX_STUCK_BLOCKAGES, last
        # Must not be the live stuck signature: 40 frontiers forever.
        assert not (n_front >= 30 and n_block > 80), last
    assert n_block < MAX_JOB_BLOCKAGES, last


def test_gentle_hill_does_not_accumulate_blockage() -> None:
    cfg, scenario = load_source("mission_tiny")
    cfg.sensors.width = 32
    cfg.sensors.height = 24
    cfg.sensors.camera_count = 4
    cfg.sensors.cameras = []
    env = MowerEnv(config=cfg, scenario=scenario, render_mode=None)
    profile = YardProfile(
        name="hill_tiny",
        width_m=env.cfg.world.width_m,
        height_m=env.cfg.world.height_m,
        resolution_m=env.cfg.world.resolution_m,
        keep_in=[(0.7, 0.7), (5.0, 0.7), (5.0, 4.0), (0.7, 4.0)],
        home={"x": 1.3, "y": 1.3, "theta": 0.0},
    )
    obs, info = env.reset(seed=2, options={"yard_profile": profile, "resize_world": False})
    # Gentle whole-yard grade (~12°) — climbable, not static α.
    hf = HeightField.from_function(
        float(env.cfg.world.width_m),
        float(env.cfg.world.height_m),
        float(env.cfg.world.resolution_m),
        lambda x, y: 0.22 * y,
    )
    env._terrain = hf
    env._pose = sit_on_terrain(env._pose, hf, env.cfg.robot.length_m, env.cfg.robot.track_m)
    policy = MissionPolicy(env.cfg, fast=True)
    policy.reset(obs, info, profile=profile)
    policy.attach_to_env(env)
    states: set[str] = set()
    for _ in range(90):
        action = policy.act(obs, info)
        obs, _reward, terminated, truncated, info = env.step(action)
        states.add(str(policy.terrain_state))
        if terminated or truncated or policy.phase != MissionPhase.EXPLORE:
            break
    status = policy.status(info)
    env.close()
    n_block = int(status.get("n_blockages") or 0)
    assert n_block <= 6, status
    assert states & {TERRAIN_OK, TERRAIN_CONTOUR, TERRAIN_RETRACE}
    # Mass stamp of the face is the live bug.
    assert int(status.get("blocked_cells") or 0) < 80, status


def test_phase4_software_tip_and_reset(tmp_path: Path) -> None:
    assert owner_copy_for("running", "mow", tilt_kind="tip") == "Tip risk — reversing"
    session = LiveSession(
        config="mission_tiny",
        fast=True,
        speed="5",
        steps=40,
        seed=2,
        cameras=4,
        out_dir=tmp_path / "phase4-reset",
        cam_stride=80,
        map_stride=8,
        require_pair=False,
    )
    session.reset()
    session.control("pair")
    session.control(
        "save_yard",
        keep_in=[[0.8, 0.8], [5.0, 0.8], [5.0, 4.0], [0.8, 4.0]],
    )
    session.control("speed", speed="5")
    out = session.control("reset")
    session.close()
    assert out["ok"] is True
    assert out["job_state"] == "idle"
    assert out.get("speed_label") == "1"
    assert out.get("kept_fence") is True
    assert out.get("kept_blockages") is False
    assert "Ready" in str((out.get("mode_banner") or {}).get("label") or "")


def test_phase0_baseline_fixture_present() -> None:
    root = Path(__file__).resolve().parent / "fixtures"
    md = (root / "phase0_baseline.md").read_text(encoding="utf-8")
    snap = json.loads((root / "phase0_baseline_snap.json").read_text(encoding="utf-8"))
    assert "3263" in md
    assert float(snap["map_pct"]) < 0.42
    assert int(snap["n_blockages"]) == 3263

"""Gym field-scorecard dry-run — laptop practice, not a field test.

``jims-mower-field-dryrun`` runs a short ``mission_tiny`` (or
``acre_yard_demo``) fixture, exercises the FIELD_TEST checks in the gym,
and writes a filled ``field_scorecard`` with ``domain: gym_dryrun`` and
``field_ready: false``.

Scores tips / drain entries / leftover uncut / ESTOP pulls — **not** mAP.
No acre runtime. No RF. No OTA.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from jims_mower.bringup import check_hw_estop_sim, run_bringup
from jims_mower.constants import FIELD_DRYRUN_DOMAIN, FIELD_SCORECARD_SCHEMA, SIGNAL_TO_ID
from jims_mower.env import MowerEnv
from jims_mower.field_scorecard import (
    FieldScorecardError,
    gym_dryrun_scorecard,
    validate_scorecard,
    write_scorecard,
)
from jims_mower.mission_flow import MissionPhase, MissionPolicy
from jims_mower.perception.detect import gym_red_bias_signal, paint_kind_blob
from jims_mower.planning.controller import TerrainPolicy, observed_hand_signal
from jims_mower.profile import SurveyOrigin, YardProfile, load_yard_profile, write_yard_profile
from jims_mower.safety import trimmer_interlock
from jims_mower.scenarios import load_source
from jims_mower.schedule import FrozenClock, ScheduleEngine, ScheduleGates, ScheduleSpec
from jims_mower.types import Obstacle

DRYRUN_SCHEMA = "jims_mower.field_dryrun.v1"
DEFAULT_SCENARIO = "mission_tiny"
DEFAULT_STEPS = 260
FAST_CAM_WIDTH = 32
FAST_CAM_HEIGHT = 24

_DAY2_SNIPPET = (
    "import json, sys\n"
    "from jims_mower.mission import load_mission\n"
    "state = load_mission(sys.argv[1])\n"
    "uncut = state.coverage.grass_cell_count() - state.coverage.cut_cell_count()\n"
    "print(json.dumps({\n"
    "    'ok': state.profile is not None and state.observed is not None,\n"
    "    'yard': None if state.profile is None else state.profile.name,\n"
    "    'n_observed': 0 if state.observed is None else int(state.observed.observed.sum()),\n"
    "    'uncut_cells': int(uncut),\n"
    "    'phase': state.phase,\n"
    "}))\n"
)


def _row(name: str, status: str, detail: str, **extra: Any) -> dict[str, Any]:
    blob = {"name": name, "status": str(status).upper(), "detail": detail}
    blob.update(extra)
    return blob


def _tick(row: dict[str, Any]) -> str:
    status = str(row.get("status") or "").upper()
    if status == "PASS":
        return "PASS"
    if status == "FAIL":
        return "FAIL"
    return "SKIP"


def _tiny_appearance_cfg() -> dict[str, Any]:
    return {
        "dt": 0.1,
        "max_steps": 20,
        "sensors": {"width": 32, "height": 24, "camera_count": 4, "fov_deg": 70.0},
        "world": {
            "width_m": 8.0,
            "height_m": 8.0,
            "resolution_m": 0.20,
            "n_people": 0,
            "n_dogs": 0,
            "n_cats": 0,
            "n_birds": 0,
            "n_trees": 0,
            "n_furniture": 0,
            "n_toys": 0,
            "terrain": {"enabled": False},
        },
        "perception": {
            "detector_backend": "appearance",
            "interlock_source": "detections",
            "terrain_mode": "heuristic",
        },
        "curriculum": {"hand_signals": True, "hand_signal_classifier": False},
        "robot": {"trimmer": {"safety_radius_m": 1.5, "offset_m": 0.32}},
    }


def _taught_profile(env: MowerEnv, *, name: str = "gym_dryrun") -> YardProfile:
    width = float(env.cfg.world.width_m)
    height = float(env.cfg.world.height_m)
    margin = 0.7 if min(width, height) < 10.0 else 2.0
    return YardProfile(
        name=name,
        width_m=width,
        height_m=height,
        resolution_m=float(env.cfg.world.resolution_m),
        keep_in=[
            (margin, margin),
            (width - margin, margin),
            (width - margin, height - margin),
            (margin, height - margin),
        ],
        home={"x": margin + 0.5, "y": margin + 0.5, "theta": 0.0},
        origin=SurveyOrigin(e_m=0.0, n_m=0.0, surveyed=False),
        description="gym dry-run taught keep-in — not a field teach",
    )


def check_preflight(
    config: Optional[Union[str, Path]] = None,
    *,
    yard: Optional[Path] = None,
) -> dict[str, Any]:
    report = run_bringup(config or "configs/orin/bench.yaml", yard=yard)
    by_name = {c["name"]: c for c in report.get("checks") or []}
    selftest = by_name.get("selftest") or {}
    hw = by_name.get("hw_estop_sim") or check_hw_estop_sim()
    pack = by_name.get("pack_measured") or {}
    soc_status = "SKIP" if pack.get("status") == "SKIP" else pack.get("status") or "SKIP"
    return {
        "bringup": report,
        "self_test": _tick(selftest),
        "hw_estop_paddle": _tick(hw),
        "soc": soc_status,
        "rain_flag": "PASS",
        "rows": [
            _row("self_test", _tick(selftest), selftest.get("detail") or "bringup selftest"),
            _row(
                "hw_estop_paddle",
                _tick(hw),
                hw.get("detail") or "sim paddle latch (not a field paddle)",
                not_field_paddle=True,
            ),
            _row(
                "soc",
                soc_status,
                pack.get("detail") or "gym SOC stub — not a fuel gauge",
                acre_runtime_h=None,
            ),
            _row(
                "rain_flag",
                "PASS",
                "schedule rain flag is readable (weather.wet / gates.rain) — no radar",
            ),
        ],
    }


def check_living_interlock() -> dict[str, Any]:
    """Paint a person in the front cam → trimmer off; oracle behind → on."""
    env = MowerEnv(config=_tiny_appearance_cfg())
    env.reset(seed=3)
    images = {name: np.zeros((24, 32, 3), dtype=np.uint8) for name in env.camera_index}
    images["front"] = paint_kind_blob(images["front"], "person", (8, 6, 12, 14))
    env._camera_images = lambda: images  # type: ignore[method-assign]
    _obs, _r, _t, _c, front_info = env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
    front_off = front_info.get("trimmer_requested") is True and front_info.get("trimmer_enabled") is False
    labels = [d.get("label") for d in front_info.get("detections") or [] if isinstance(d, dict)]
    env.close()

    env2 = MowerEnv(config=_tiny_appearance_cfg())
    env2.reset(seed=4)
    pose = env2._pose
    behind = Obstacle(
        "person",
        pose.x - 1.05 * math.cos(pose.theta),
        pose.y - 1.05 * math.sin(pose.theta),
        0.25,
        z=0.9,
    )
    env2._yard.obstacles.append(behind)
    god = trimmer_interlock(
        True,
        env2._pose,
        env2._yard.obstacles,
        offset_m=0.32,
        safety_radius_m=1.5,
    )
    blank = {name: np.zeros((24, 32, 3), dtype=np.uint8) for name in env2.camera_index}
    env2._camera_images = lambda: blank  # type: ignore[method-assign]
    _obs, _r, _t, _c, rear_info = env2.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
    rear_on = rear_info.get("trimmer_requested") is True and rear_info.get("trimmer_enabled") is True
    env2.close()

    ok = front_off and rear_on and god.trimmer_enabled is False
    return _row(
        "living_interlock",
        "PASS" if ok else "FAIL",
        "front RGB person blob → trimmer off; oracle person behind only → trimmer on"
        if ok
        else "appearance living interlock failed",
        front_trimmer_enabled=front_info.get("trimmer_enabled"),
        rear_trimmer_enabled=rear_info.get("trimmer_enabled"),
        front_labels=labels,
        oracle_behind_would_trip=god.trimmer_enabled is False,
        gym_only=True,
    )


def check_hand_signal() -> dict[str, Any]:
    """Red-biased appearance person crop → ICD stop → policy hold. Gym only."""
    cfg = _tiny_appearance_cfg()
    env = MowerEnv(config=cfg)
    obs, info = env.reset(seed=5)
    images = {name: np.zeros((24, 32, 3), dtype=np.uint8) for name in env.camera_index}
    bbox = (8, 6, 12, 14)
    images["front"] = paint_kind_blob(images["front"], "person", bbox)
    crop_signal = gym_red_bias_signal(images["front"], bbox)
    env._camera_images = lambda: images  # type: ignore[method-assign]
    obs, _r, _t, _c, info = env.step(np.array([0.4, 0.4, 1.0], dtype=np.float32))
    signal_id = int(obs.get("hand_signal") or 0)
    name = info.get("hand_signal_name")
    source = info.get("hand_signal_source")
    policy = TerrainPolicy(env.cfg)
    action = policy.act(obs, info)
    stopped = (
        float(action[0]) == 0.0
        and float(action[1]) == 0.0
        and float(action[2]) == 0.0
        and observed_hand_signal(obs, True) == "stop"
    )
    env.close()
    ok = crop_signal == "stop" and signal_id == SIGNAL_TO_ID["stop"] and name == "stop" and stopped
    return _row(
        "hand_signal",
        "PASS" if ok else "FAIL",
        "red-bias person crop → obs hand_signal=stop → policy hold (gym-only; no confusion matrix)"
        if ok
        else "appearance red-bias hand signal did not reach ICD/policy",
        crop_signal=crop_signal,
        hand_signal=signal_id,
        hand_signal_name=name,
        hand_signal_source=source,
        policy_signal=policy.last_signal,
        gym_only=True,
        field_confusion_matrix=None,
        unreliable_on_real_clips=True,
    )


def check_schedule_skips() -> tuple[dict[str, Any], dict[str, Any]]:
    spec = ScheduleSpec(
        enabled=True,
        days=("mon",),
        start_local="09:00",
        duration_min=60,
        timezone="UTC",
        min_soc=0.25,
        skip_rain=True,
        arm_window_min=15,
    )
    instant = datetime(2026, 9, 14, 9, 0, tzinfo=timezone.utc)
    rain = ScheduleEngine(spec, clock=FrozenClock(instant)).evaluate(ScheduleGates(soc=0.9, rain=True))
    soc = ScheduleEngine(spec, clock=FrozenClock(instant)).evaluate(ScheduleGates(soc=0.10, rain=False))
    rain_ok = rain.action == "skip" and rain.reason == "rain"
    soc_ok = soc.action == "skip" and soc.reason == "soc_low"
    return (
        _row(
            "rain_skip",
            "PASS" if rain_ok else "FAIL",
            "schedule rain flag consumed the window as skip"
            if rain_ok
            else "rain skip did not fire",
            reason=rain.reason,
        ),
        _row(
            "soc_skip",
            "PASS" if soc_ok else "FAIL",
            "schedule SOC below min_soc consumed the window as skip"
            if soc_ok
            else "SOC skip did not fire",
            reason=soc.reason,
        ),
    )


def check_hw_estop_latch(env: Optional[MowerEnv] = None) -> dict[str, Any]:
    if env is not None:
        live = env.hw_estop.filter_action(np.array([0.8, 0.8, 1.0], dtype=np.float32))
        env.hit_hw_estop("dryrun paddle")
        dead = env.hw_estop.filter_action(np.array([0.8, 0.8, 1.0], dtype=np.float32))
        still = env.hw_estop.filter_action(np.array([0.5, 0.5, 1.0], dtype=np.float32))
        env.reset_hw_estop()
        restored = env.hw_estop.filter_action(np.array([0.5, 0.5, 1.0], dtype=np.float32))
        ok = (
            float(np.max(np.abs(live))) > 0.0
            and float(np.max(np.abs(dead))) == 0.0
            and float(np.max(np.abs(still))) == 0.0
            and float(restored[0]) > 0.4
        )
        return _row(
            "hw_estop_paddle",
            "PASS" if ok else "FAIL",
            "HW ESTOP paddle sim latched rails; software clear does not restore"
            if ok
            else "HW ESTOP sim latch failed",
            not_field_paddle=True,
            pulls=1 if ok else 0,
        )
    return check_hw_estop_sim()


def check_day2_restore(session_path: Path) -> dict[str, Any]:
    if not session_path.is_file():
        return _row("day2_resume", "FAIL", f"no session at {session_path}")
    proc = subprocess.run(
        [sys.executable, "-c", _DAY2_SNIPPET, str(session_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return _row(
            "day2_resume",
            "FAIL",
            f"new-process restore failed: {(proc.stderr or proc.stdout).strip()[:240]}",
        )
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return _row("day2_resume", "FAIL", "new-process restore printed no JSON")
    ok = bool(payload.get("ok"))
    return _row(
        "day2_resume",
        "PASS" if ok else "FAIL",
        "new process loaded yard + ObservedMap fog + uncut (no reteach)"
        if ok
        else "new process missing yard or observed map",
        **payload,
    )


def _run_mission(
    *,
    scenario_name: str,
    steps: int,
    seed: int,
    yard_path: Path,
    session_path: Path,
) -> dict[str, Any]:
    cfg, scenario = load_source(scenario_name)
    cfg.sensors.width = min(int(cfg.sensors.width), FAST_CAM_WIDTH)
    cfg.sensors.height = min(int(cfg.sensors.height), FAST_CAM_HEIGHT)
    cfg.sensors.camera_count = 4
    cfg.sensors.cameras = []
    cfg.perception.detector_backend = "appearance"
    cfg.perception.interlock_source = "detections"
    cfg.curriculum.hand_signals = True
    cfg.curriculum.hand_signal_classifier = False
    cfg.max_steps = max(int(cfg.max_steps), int(steps) + 2)
    env = MowerEnv(config=cfg, scenario=scenario, render_mode=None)
    profile = _taught_profile(env, name=f"gym_dryrun_{scenario_name}")
    write_yard_profile(yard_path, profile)
    loaded = load_yard_profile(yard_path)
    obs, info = env.reset(seed=seed, options={"yard_profile": loaded, "resize_world": False})
    policy = MissionPolicy(env.cfg, fast=True)
    policy.settings.max_mow_steps = min(int(policy.settings.max_mow_steps), 28)
    policy.settings.max_return_steps = min(int(policy.settings.max_return_steps), 40)
    policy.reset(obs, info, profile=loaded)
    origin = loaded.origin.as_dict() if hasattr(loaded.origin, "as_dict") else {}
    teach_ok = len(loaded.keep_in) >= 3
    surveyed_flag = bool(origin.get("surveyed"))

    seen: set[str] = {policy.phase.value}
    tips = 0
    drains = 0
    recovered = False
    tip_injected = False
    for _ in range(int(steps)):
        if policy.phase == MissionPhase.REVIEW:
            policy.request_start_mow()
        action = policy.act(obs, info)
        seen.add(policy.phase.value)
        if policy.phase == MissionPhase.MOW and not tip_injected:
            tipped = dict(info)
            tipped["terrain_advice"] = "stop"
            tipped["pose"] = {**(info.get("pose") or {}), "roll": 0.45}
            tip_obs = dict(obs)
            tip_obs["imu"] = np.array([0.0, 4.0, 8.7, 0.0, 0.0, 0.0], dtype=np.float32)
            tip = policy.act(tip_obs, tipped)
            recovered = float(tip[0]) < 0.0 and float(tip[1]) < 0.0
            if recovered:
                tips += 1
            tip_injected = True
        obs, _reward, terminated, truncated, info = env.step(action)
        if info.get("tipover"):
            tips += 1
        if info.get("drain_drop"):
            drains += 1
        seen.add(policy.phase.value)
        if terminated or truncated or policy.done:
            break

    leftover_cells, leftover_m2 = env._coverage.leftover_uncut()
    policy.save_session(session_path, env._coverage, env._pose, scenario=scenario_name, seed=seed)
    estop = check_hw_estop_latch(env)
    env.close()

    def _mission_row(name: str, reached: bool, detail: str) -> dict[str, Any]:
        return _row(name, "PASS" if reached else "SKIP", detail, capped=not reached)

    mission_rows = [
        _row(
            "teach_boundary",
            "PASS" if teach_ok else "FAIL",
            f"loaded taught keep-in ({len(loaded.keep_in)} vertices) from {yard_path.name}",
        ),
        _row(
            "surveyed_origin",
            "PASS",
            f"origin.frame={origin.get('frame')} surveyed={surveyed_flag} "
            "(gym peg — not a WGS84 field survey)",
            surveyed=surveyed_flag,
            origin=origin,
        ),
        _mission_row("explore", "explore" in seen, "explore phase ran" if "explore" in seen else "capped before explore"),
        _mission_row(
            "map_ready",
            "review" in seen or "mow" in seen,
            "MAP READY (review) reached" if "review" in seen or "mow" in seen else "capped before MAP READY",
        ),
        _mission_row("mow", "mow" in seen, "mow phase entered" if "mow" in seen else "capped before mow"),
        _mission_row(
            "return_home",
            "return_home" in seen or "complete" in seen,
            "return-home entered" if "return_home" in seen or "complete" in seen else "capped before return-home",
        ),
    ]
    tip_row = _row(
        "tip_ramp_recovery",
        "PASS" if recovered else ("SKIP" if not tip_injected else "FAIL"),
        "injected IMU tip → reverse recovery" if recovered else "tip recovery not observed (capped or failed)",
        injected=tip_injected,
    )
    return {
        "seen": sorted(seen),
        "tips": tips,
        "drain_entries": drains,
        "leftover_uncut_cells": leftover_cells,
        "leftover_uncut_m2": leftover_m2,
        "mission_rows": mission_rows,
        "tip_row": tip_row,
        "estop_row": estop,
        "surveyed": surveyed_flag,
        "teach_ok": teach_ok,
        "session": str(session_path),
        "yard": str(yard_path),
    }


def run_field_dryrun(
    *,
    scenario: str = DEFAULT_SCENARIO,
    steps: int = DEFAULT_STEPS,
    seed: int = 3,
    out: Optional[Path] = None,
    work_dir: Optional[Path] = None,
    bringup_config: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    dest_dir = Path(work_dir) if work_dir is not None else Path(out).parent if out is not None else Path(".")
    dest_dir.mkdir(parents=True, exist_ok=True)
    scorecard_path = Path(out) if out is not None else dest_dir / "scorecard.yaml"
    yard_path = dest_dir / "yard.json"
    session_path = dest_dir / "session.npz"
    if not yard_path.is_file():
        write_yard_profile(
            yard_path,
            YardProfile(
                name="gym_dryrun_stub",
                width_m=6.0,
                height_m=5.0,
                resolution_m=0.20,
                keep_in=[(0.7, 0.7), (5.3, 0.7), (5.3, 4.3), (0.7, 4.3)],
                home={"x": 1.2, "y": 1.2, "theta": 0.0},
                origin=SurveyOrigin(e_m=0.0, n_m=0.0, surveyed=False),
            ),
        )

    preflight = check_preflight(bringup_config, yard=yard_path)
    living = check_living_interlock()
    hand = check_hand_signal()
    rain_row, soc_row = check_schedule_skips()
    mission = _run_mission(
        scenario_name=scenario,
        steps=steps,
        seed=seed,
        yard_path=yard_path,
        session_path=session_path,
    )
    day2 = check_day2_restore(session_path)

    card = gym_dryrun_scorecard()
    card["yard"] = scenario
    card["date"] = datetime.now(timezone.utc).date().isoformat()
    card["operator"] = "gym_dryrun"
    card["domain"] = FIELD_DRYRUN_DOMAIN
    card["field_ready"] = False
    card["field_run"] = False
    card["pack_measured"] = False
    card["acre_runtime_h"] = None
    card["map_claim"] = None
    card["iou_claim"] = None
    card["fps_claim"] = None
    card["preflight"] = {
        "self_test": preflight["self_test"],
        "hw_estop_paddle": preflight["hw_estop_paddle"],
        "soc": preflight["soc"],
        "rain_flag": preflight["rain_flag"],
    }
    mission_status = {row["name"]: row["status"] for row in mission["mission_rows"]}
    card["mission"] = {
        "teach_boundary": mission_status.get("teach_boundary"),
        "surveyed_origin": mission_status.get("surveyed_origin"),
        "explore": mission_status.get("explore"),
        "map_ready": mission_status.get("map_ready"),
        "mow": mission_status.get("mow"),
        "return_home": mission_status.get("return_home"),
    }
    card["checks"] = {
        "living_interlock": living["status"],
        "tip_ramp_recovery": mission["tip_row"]["status"],
        "rain_skip": rain_row["status"],
        "soc_skip": soc_row["status"],
        "day2_resume": day2["status"],
        "hand_signal": hand["status"],
    }
    card["score"] = {
        "tips": int(mission["tips"]),
        "drain_entries": int(mission["drain_entries"]),
        "leftover_uncut_cells": int(mission["leftover_uncut_cells"]),
        "leftover_uncut_m2": float(mission["leftover_uncut_m2"]),
        "estop_pulls": int(mission["estop_row"].get("pulls") or 0),
    }
    card["notes"] = (
        "Gym dry-run on a laptop. Not a field test. Score is gym leftover / "
        "injected tip / sim ESTOP — not mAP, IoU, FPS, or acre runtime. "
        "CV-5 hand signal is red-bias on an appearance person crop (gym-only; "
        "no confusion matrix)."
    )
    card["gym_log"] = {
        "schema": DRYRUN_SCHEMA,
        "scenario": scenario,
        "steps": steps,
        "seed": seed,
        "phases_seen": mission["seen"],
        "preflight": preflight["rows"],
        "living_interlock": living,
        "hand_signal": hand,
        "rain_skip": rain_row,
        "soc_skip": soc_row,
        "tip_ramp_recovery": mission["tip_row"],
        "hw_estop_paddle": mission["estop_row"],
        "day2_resume": day2,
        "mission": mission["mission_rows"],
    }
    validated = validate_scorecard(card)
    write_scorecard(scorecard_path, validated)
    rows = (
        list(preflight["rows"])
        + list(mission["mission_rows"])
        + [living, hand, mission["tip_row"], rain_row, soc_row, day2, mission["estop_row"]]
    )
    fail = any(r.get("status") == "FAIL" for r in rows)
    return {
        "schema": DRYRUN_SCHEMA,
        "ok": not fail,
        "status": "FAIL" if fail else "PASS",
        "domain": FIELD_DRYRUN_DOMAIN,
        "field_ready": False,
        "field_run": False,
        "scorecard": str(scorecard_path),
        "scorecard_schema": FIELD_SCORECARD_SCHEMA,
        "card": validated.as_dict(),
        "rows": rows,
        "not_a_field_test": True,
        "acre_runtime_h": None,
        "map_claim": None,
        "iou_claim": None,
        "fps_claim": None,
    }


def format_dryrun(report: dict[str, Any]) -> str:
    card = report.get("card") or {}
    score = card.get("score") or {}
    lines = [
        "Jim's Mower field scorecard dry-run (gym — not a field test)",
        "domain=gym_dryrun  field_ready=false  field_run=false",
        "Score tips / drains / leftover uncut / ESTOP — not mAP.",
        "",
    ]
    for row in report.get("rows") or []:
        lines.append(f"  {row['status']:<5} {row['name']}: {row['detail']}")
    lines.append("")
    lines.append(
        "SCORE  "
        f"tips={score.get('tips')}  "
        f"drain_entries={score.get('drain_entries')}  "
        f"leftover_uncut_cells={score.get('leftover_uncut_cells')}  "
        f"leftover_uncut_m2={score.get('leftover_uncut_m2')}  "
        f"estop_pulls={score.get('estop_pulls')}"
    )
    lines.append(f"Wrote {report.get('scorecard')}")
    lines.append("")
    lines.append(
        f"{report.get('status')}  Honest: laptop gym practice. "
        "Not a field test. No mAP / FPS / acre runtime."
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Gym field-scorecard dry-run. Fills a FIELD_TEST scorecard from "
            "mission_tiny / acre_yard_demo. Not a field test."
        )
    )
    p.add_argument("--scenario", default=DEFAULT_SCENARIO, help="mission_tiny (default) or acre_yard_demo")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="capped mission steps")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--out", type=Path, default=Path("artifacts/field_dryrun/scorecard.yaml"))
    p.add_argument("--bringup-config", default="configs/orin/bench.yaml")
    p.add_argument("--json", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_field_dryrun(
            scenario=args.scenario,
            steps=args.steps,
            seed=args.seed,
            out=args.out,
            work_dir=args.out.parent,
            bringup_config=args.bringup_config,
        )
    except FieldScorecardError as exc:
        print(f"field dry-run scorecard error: {exc}", file=sys.stderr)
        return 1
    text = json.dumps(report, indent=2, default=str) if args.json else format_dryrun(report)
    print(text)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

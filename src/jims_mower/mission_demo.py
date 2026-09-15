"""End-to-end mission demo: calibrate → explore → review → mow → home."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from jims_mower.bev import render_bev
from jims_mower.demo import camera_dump_indices, camera_dump_stride
from jims_mower.env import MowerEnv
from jims_mower.kinematics import trimmer_xy
from jims_mower.mesh import write_map_png
from jims_mower.mission_flow import (
    PHASE_LABELS,
    MissionPolicy,
    mission_timeline,
)
from jims_mower.profile import write_yard_profile
from jims_mower.renderer import render_topdown
from jims_mower.scenarios import load_source
from jims_mower.teach import keepouts_from_env
from jims_mower.viewer import write_viewer_bundle

DEFAULT_STEPS = 6000
FAST_STEPS = 420
DEFAULT_CAM_STRIDE = 20


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(path)


def _plan_overlay(env: MowerEnv, policy: MissionPolicy) -> np.ndarray:
    return render_topdown(
        env._pose,
        env._coverage,
        env._yard.obstacles,
        trimmer_xy=trimmer_xy(env._pose, env.cfg.robot.trimmer.offset_m),
        trimmer_on=env._trimmer_on,
        terrain=env._terrain,
        waypoints=policy.waypoints,
        waypoint_index=policy.index,
    )


def _snapshot_row(
    policy: MissionPolicy,
    info: dict[str, Any],
    rel: str,
) -> dict[str, Any]:
    status = policy.status(info)
    return {
        "step": policy.step,
        "phase": policy.phase.value,
        "phase_label": PHASE_LABELS.get(policy.phase.value, policy.phase.value.upper()),
        "file": rel,
        "map_completion": status["map_completion"],
        "actual_coverage_fraction": status["actual_coverage_fraction"],
        "reachable_mowable_cells": status["reachable_mowable_cells"],
        "unreachable_mowable_cells": status["unreachable_mowable_cells"],
        "n_frontiers": status["n_frontiers"],
        "trimmer_allowed": status["trimmer_allowed"],
    }


def run_mission_demo(
    out_dir: Path,
    *,
    steps: Optional[int] = None,
    seed: int = 7,
    cameras: Optional[int] = 4,
    config: Optional[Any] = None,
    fast: bool = False,
    cam_stride: Optional[int] = None,
    snapshot_stride: Optional[int] = None,
) -> dict[str, Any]:
    cfg, scenario = load_source(config)
    if cameras is not None:
        cfg.sensors.camera_count = cameras
        cfg.sensors.cameras = []
    n_steps = int(steps if steps is not None else (FAST_STEPS if fast else DEFAULT_STEPS))
    cfg.max_steps = max(int(cfg.max_steps), n_steps + 2)
    env = MowerEnv(config=cfg, scenario=scenario, render_mode="rgb_array")
    obs, info = env.reset(seed=seed)
    policy = MissionPolicy(env.cfg, fast=fast)
    policy.reset(obs, info)

    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    snap_dir = dest / "maps" / "snap"
    snap_dir.mkdir(parents=True, exist_ok=True)
    names = list(obs["cameras"].keys())
    dump_steps = camera_dump_indices(n_steps, cam_stride if cam_stride is not None else DEFAULT_CAM_STRIDE)
    stride = int(snapshot_stride or policy.settings.snapshot_stride)
    poses: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []

    def _record_pose(info_row: dict[str, Any]) -> None:
        pose = dict(info_row.get("pose") or {})
        status = policy.status(info_row)
        pose["phase"] = status["phase"]
        pose["phase_label"] = status["phase_label"]
        pose["map_completion"] = status["map_completion"]
        pose["actual_coverage_fraction"] = status["actual_coverage_fraction"]
        poses.append(pose)

    def _maybe_snapshot(force: bool = False) -> None:
        if policy.observed is None:
            return
        if not force and policy.step % stride != 0:
            return
        name = f"{policy.step:04d}.png"
        rgb = policy.observed.as_rgb(
            frontiers=[
                policy.observed.world_to_cell(x, y)
                for x, y in policy._frontier_xy
                if policy.observed.world_to_cell(x, y) is not None
            ]
        )
        write_map_png(snap_dir / name, rgb)
        snapshots.append(_snapshot_row(policy, info, f"maps/snap/{name}"))

    _record_pose(info)
    _maybe_snapshot(force=True)
    if 0 in dump_steps:
        step_dir = dest / "step_000"
        step_dir.mkdir(parents=True, exist_ok=True)
        for cam_name, frame in obs["cameras"].items():
            _save_rgb(step_dir / f"cam_{cam_name}.png", frame)
        topdown = env.render()
        if topdown is not None:
            _save_rgb(step_dir / "topdown.png", topdown)
        _save_rgb(step_dir / "plan_overlay.png", _plan_overlay(env, policy))

    for t in range(n_steps):
        if t in dump_steps and t > 0:
            step_dir = dest / f"step_{t:03d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            for cam_name, frame in obs["cameras"].items():
                _save_rgb(step_dir / f"cam_{cam_name}.png", frame)
            topdown = env.render()
            if topdown is not None:
                _save_rgb(step_dir / "topdown.png", topdown)
            _save_rgb(step_dir / "plan_overlay.png", _plan_overlay(env, policy))
            _save_rgb(
                step_dir / "bev.png",
                render_bev(
                    env,
                    obs,
                    waypoints=policy.waypoints,
                    waypoint_index=policy.index,
                    costmap=policy.global_plan.costmap if policy.global_plan else None,
                    cameras=obs["cameras"],
                ),
            )
            if policy.observed is not None:
                _save_rgb(step_dir / "observed.png", policy.observed.as_rgb())

        action = policy.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        status = policy.status(info)
        records.append(
            {
                "step": t,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "phase": status["phase"],
                "phase_label": status["phase_label"],
                "coverage_fraction": info.get("coverage_fraction"),
                "trimmer_enabled": info.get("trimmer_enabled"),
                "trimmer_allowed": status["trimmer_allowed"],
                "map_completion": status["map_completion"],
                "collision": info.get("collision"),
                "tipover": info.get("tipover"),
                "drain_drop": info.get("drain_drop"),
                "pose": info.get("pose"),
            }
        )
        _record_pose(info)
        _maybe_snapshot()
        if terminated or truncated or policy.done:
            break

    _maybe_snapshot(force=True)
    if policy.profile is not None:
        if not policy.profile.keep_out:
            policy.profile.keep_out = keepouts_from_env(env)
        write_yard_profile(dest / "profile.json", policy.profile)
    timeline = mission_timeline(
        policy,
        actual_coverage=float(info.get("coverage_fraction") or 0.0),
    )
    timeline["snapshots"] = snapshots
    (dest / "mission.json").write_text(json.dumps(timeline, indent=2), encoding="utf-8")

    plan_payload: dict[str, Any] = {
        "policy": "mission",
        "phase": policy.phase.value,
        "waypoints": [
            {"x": x, "y": y, "z": float(env._terrain.sample(x, y))}
            for x, y in (policy.global_plan.waypoints if policy.global_plan else policy.waypoints)
        ],
        "explore": [{"x": x, "y": y} for x, y in (policy.explore_plan.waypoints if policy.explore_plan else [])],
        "index": policy.index,
        "n_segments": policy.global_plan.n_segments if policy.global_plan else 0,
        "replans": policy.replans,
        **(policy.global_plan.as_metrics() if policy.global_plan else {}),
    }
    (dest / "plan.json").write_text(json.dumps(plan_payload, indent=2), encoding="utf-8")

    topdown = env.render()
    if topdown is not None:
        _save_rgb(dest / "coverage_final.png", topdown)
    _save_rgb(dest / "plan_overlay_final.png", _plan_overlay(env, policy))
    if policy.observed is not None:
        _save_rgb(dest / "observed_final.png", policy.observed.as_rgb())
        write_map_png(dest / "maps" / "observed.png", policy.observed.as_rgb())

    summary = {
        "steps_run": len(records),
        "cameras": names,
        "policy": "mission",
        "fast": bool(fast),
        "scenario": scenario.name if scenario else "",
        "final_phase": policy.phase.value,
        "phase_ranges": policy.close_phase_ranges(),
        "final_coverage_fraction": records[-1]["coverage_fraction"] if records else 0.0,
        "final_pose": info.get("pose"),
        "n_waypoints": len(plan_payload["waypoints"]),
        "terrain_source": info.get("terrain_source"),
        "terrain_mode": env.cfg.perception.terrain_mode,
        "world": {
            "width_m": env.cfg.world.width_m,
            "height_m": env.cfg.world.height_m,
            "resolution_m": env.cfg.world.resolution_m,
        },
        "mission": timeline["metrics"],
        "events": timeline["events"],
        "not_a_benchmark": True,
        "log": records,
    }
    (dest / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_viewer_bundle(
        dest,
        env=env,
        poses=poses,
        plan=plan_payload,
        profile=policy.profile.as_dict() if policy.profile is not None else None,
        cameras=names,
        policy="mission",
        mission=timeline,
    )
    env.close()
    summary["out"] = str(dest)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Jim's Mower mission demo: calibrate → explore → map-ready → mow"
    )
    p.add_argument("--out", type=Path, default=Path("mission_out"))
    p.add_argument("--config", type=str, default="golf_rough")
    p.add_argument("--steps", type=int, default=None, help="episode budget (default 6000, or 420 with --fast)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cameras", type=int, default=4)
    p.add_argument(
        "--fast",
        action="store_true",
        help="tiny-yard CI smoke: mission_tiny + short phase budgets",
    )
    p.add_argument("--cam-stride", type=int, default=DEFAULT_CAM_STRIDE)
    p.add_argument("--snapshot-stride", type=int, default=None)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    config = args.config
    if args.fast and (config is None or config == "golf_rough"):
        config = "mission_tiny"
    summary = run_mission_demo(
        args.out,
        steps=args.steps,
        seed=args.seed,
        cameras=args.cameras,
        config=config,
        fast=args.fast,
        cam_stride=args.cam_stride,
        snapshot_stride=args.snapshot_stride,
    )
    metrics = summary.get("mission") or {}
    print(
        f"Mission {summary['final_phase']} after {summary['steps_run']} steps "
        f"({summary.get('scenario') or 'default'}) → {args.out}"
    )
    print(
        f"map {metrics.get('map_completion', 0):.3f}  "
        f"planned {metrics.get('planned_coverage_fraction', 0):.3f}  "
        f"cut {metrics.get('actual_coverage_fraction', 0):.3f}  "
        f"unreachable {metrics.get('unreachable_mowable_cells', 0)} cells"
    )
    print(f"phases: {[r['phase'] for r in summary.get('phase_ranges') or []]}")
    _ = camera_dump_stride  # keep helper imported for callers / tests


if __name__ == "__main__":
    main()

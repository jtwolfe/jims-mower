"""Headless demo: dump multi-camera frames, detections, and a coverage map."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from jims_mower.bev import render_bev
from jims_mower.env import MowerEnv
from jims_mower.kinematics import sit_on_terrain, trimmer_xy
from jims_mower.planning import TerrainPolicy
from jims_mower.renderer import render_camera, render_topdown
from jims_mower.scenarios import load_source
from jims_mower.types import Pose

POLICIES = ("terrain", "scripted", "random")


def _save_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image, mode="RGB").save(path)


def _drain_camera_view(env: MowerEnv) -> Optional[np.ndarray]:
    """Front-camera view aimed at a drain so the ditch is visible in the dump."""
    if not env._terrain.drains:
        return None
    drain = env._terrain.drains[0]
    mx = 0.5 * (drain.x0 + drain.x1)
    my = 0.5 * (drain.y0 + drain.y1)
    heading = drain.heading
    # Stand off along the perpendicular so the channel crosses the image.
    nx, ny = -math.sin(heading), math.cos(heading)
    pose = Pose(mx - 1.6 * nx, my - 1.6 * ny, math.atan2(ny, nx))
    pose = sit_on_terrain(pose, env._terrain, env.cfg.robot.length_m, env.cfg.robot.track_m)
    cam = next((c for c in env.cameras if c.name == "front"), env.cameras[0])
    return render_camera(
        pose,
        cam,
        env._coverage,
        env._yard.obstacles,
        env.cfg.sensors.width,
        env.cfg.sensors.height,
        (env.cfg.world.width_m, env.cfg.world.height_m),
        terrain=env._terrain,
        appearance=env._appearance,
    )


def _montage(images: dict[str, np.ndarray], order: list[str]) -> np.ndarray:
    frames = [images[name] for name in order if name in images]
    if not frames:
        raise ValueError("no camera frames to montage")
    h, w = frames[0].shape[:2]
    cols = min(3, len(frames))
    rows = int(np.ceil(len(frames) / cols))
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        r, c = divmod(i, cols)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = frame
    return canvas


def _plan_overlay(env: MowerEnv, policy: Optional[TerrainPolicy]) -> np.ndarray:
    waypoints = policy.waypoints if policy is not None else []
    index = policy.index if policy is not None else 0
    return render_topdown(
        env._pose,
        env._coverage,
        env._yard.obstacles,
        trimmer_xy=trimmer_xy(env._pose, env.cfg.robot.trimmer.offset_m),
        trimmer_on=env._trimmer_on,
        terrain=env._terrain,
        waypoints=waypoints,
        waypoint_index=index,
    )


def _scripted_action() -> np.ndarray:
    return np.array([0.55, 0.50, 1.0], dtype=np.float32)


def _random_action(env: MowerEnv, rng: np.random.Generator) -> np.ndarray:
    action = rng.uniform(
        env.action_space.low,
        env.action_space.high,
    ).astype(np.float32)
    action[2] = 1.0
    return action


def run_demo(
    out_dir: Path,
    *,
    steps: int = 40,
    seed: int = 7,
    cameras: Optional[int] = None,
    hand_signals: bool = False,
    config: Optional[str] = None,
    policy: str = "terrain",
    terrain_observer: Optional[str] = None,
    load_mission: Optional[str] = None,
    save_mission: Optional[str] = None,
) -> dict:
    name = (policy or "terrain").strip().lower()
    if name not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}; got {policy!r}")
    cfg, scenario = load_source(config)
    if cameras is not None:
        cfg.sensors.camera_count = cameras
        cfg.sensors.cameras = []
    if terrain_observer:
        key = terrain_observer.strip().lower()
        if key not in {"oracle", "heuristic", "blind", "learned"}:
            raise ValueError(
                f"terrain_observer must be oracle|heuristic|blind|learned; got {terrain_observer!r}"
            )
        cfg.perception.terrain_mode = key
    env = MowerEnv(
        config=cfg,
        scenario=scenario,
        render_mode="rgb_array",
        hand_signals=hand_signals,
    )
    reset_opts: dict = {}
    if load_mission:
        reset_opts["load_mission"] = load_mission
    if save_mission:
        reset_opts["save_mission"] = save_mission
    obs, info = env.reset(seed=seed, options=reset_opts or None)
    out_dir.mkdir(parents=True, exist_ok=True)

    terrain_policy: Optional[TerrainPolicy] = None
    rng = np.random.default_rng(seed)
    if name == "terrain":
        terrain_policy = TerrainPolicy(env.cfg)
        terrain_policy.reset(obs, info)

    names = list(obs["cameras"].keys())
    records: list[dict] = []
    dump_steps = {0, max(0, steps // 2), max(0, steps - 1)}

    def _dump_step(step_dir: Path, obs: dict, info: dict) -> None:
        step_dir.mkdir(parents=True, exist_ok=True)
        for cam_name, frame in obs["cameras"].items():
            _save_rgb(step_dir / f"cam_{cam_name}.png", frame)
        _save_rgb(step_dir / "montage.png", _montage(obs["cameras"], names))
        topdown = env.render()
        if topdown is not None:
            _save_rgb(step_dir / "topdown.png", topdown)
        _save_rgb(step_dir / "plan_overlay.png", _plan_overlay(env, terrain_policy))
        _save_rgb(
            step_dir / "bev.png",
            render_bev(
                env,
                obs,
                waypoints=terrain_policy.waypoints if terrain_policy else [],
                waypoint_index=terrain_policy.index if terrain_policy else 0,
                costmap=terrain_policy.plan.costmap if terrain_policy and terrain_policy.plan else None,
                cameras=obs["cameras"],
            ),
        )
        for layer, img in env.terrain_layer_images().items():
            _save_rgb(step_dir / f"{layer}.png", img)
        drain_view = _drain_camera_view(env)
        if drain_view is not None:
            _save_rgb(step_dir / "drain_view.png", drain_view)
        (step_dir / "detections.json").write_text(
            json.dumps(info["detections"], indent=2),
            encoding="utf-8",
        )
        fused = None
        if terrain_policy is not None:
            fp = terrain_policy.fusion.pose()
            fused = {
                "x": fp.x,
                "y": fp.y,
                "theta": fp.theta,
                "z": fp.z,
                "pitch": fp.pitch,
                "roll": fp.roll,
            }
        (step_dir / "sensors.json").write_text(
            json.dumps(
                {
                    "imu": info.get("imu"),
                    "gps": info.get("gps"),
                    "tof": info.get("tof"),
                    "pose": info.get("pose"),
                    "fused_pose": fused,
                    "n_drains": info.get("n_drains"),
                    "n_banks": info.get("n_banks"),
                    "terrain_source": info.get("terrain_source"),
                    "terrain_mode": env.cfg.perception.terrain_mode,
                    "terrain_advice": info.get("terrain_advice"),
                    "policy": name,
                    "waypoint_index": terrain_policy.index if terrain_policy else None,
                    "n_waypoints": len(terrain_policy.waypoints) if terrain_policy else 0,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    for t in range(steps):
        if t in dump_steps:
            _dump_step(out_dir / f"step_{t:03d}", obs, info)

        if name == "scripted":
            action = _scripted_action()
        elif name == "random":
            action = _random_action(env, rng)
        else:
            assert terrain_policy is not None
            action = terrain_policy.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        records.append(
            {
                "step": t,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "coverage_fraction": info["coverage_fraction"],
                "trimmer_enabled": info["trimmer_enabled"],
                "n_detections": len(info["detections"]),
                "collision": info.get("collision"),
                "tipover": info.get("tipover"),
                "drain_drop": info.get("drain_drop"),
                "steep": info.get("steep"),
                "terrain_advice": info.get("terrain_advice"),
                "policy_advice": terrain_policy.last_advice if terrain_policy else None,
                "imu": info.get("imu"),
                "gps": info.get("gps"),
            }
        )
        if terminated or truncated:
            break

    topdown = env.render()
    if topdown is not None:
        _save_rgb(out_dir / "coverage_final.png", topdown)
    _save_rgb(out_dir / "plan_overlay_final.png", _plan_overlay(env, terrain_policy))
    _save_rgb(
        out_dir / "bev_final.png",
        render_bev(
            env,
            obs,
            waypoints=terrain_policy.waypoints if terrain_policy else [],
            waypoint_index=terrain_policy.index if terrain_policy else 0,
            costmap=terrain_policy.plan.costmap if terrain_policy and terrain_policy.plan else None,
            cameras=obs.get("cameras"),
        ),
    )
    layers = env.terrain_layer_images()
    for layer, img in layers.items():
        _save_rgb(out_dir / f"{layer}_final.png", img)
    plan_payload = {
        "policy": name,
        "waypoints": [
            {"x": x, "y": y} for x, y in (terrain_policy.waypoints if terrain_policy else [])
        ],
        "index": terrain_policy.index if terrain_policy else 0,
        "n_segments": terrain_policy.plan.n_segments if terrain_policy and terrain_policy.plan else 0,
        "replans": terrain_policy.replans if terrain_policy else 0,
    }
    (out_dir / "plan.json").write_text(json.dumps(plan_payload, indent=2), encoding="utf-8")
    summary = {
        "steps_run": len(records),
        "cameras": names,
        "policy": name,
        "final_coverage_fraction": records[-1]["coverage_fraction"] if records else 0.0,
        "final_detections": info["detections"],
        "hand_signals_enabled": hand_signals,
        "n_drains": info.get("n_drains"),
        "n_banks": info.get("n_banks"),
        "final_pose": info.get("pose"),
        "final_imu": info.get("imu"),
        "final_gps": info.get("gps"),
        "n_waypoints": len(plan_payload["waypoints"]),
        "terrain_source": info.get("terrain_source"),
        "terrain_mode": env.cfg.perception.terrain_mode,
        "terminated_drain_drop": bool(info.get("drain_drop")),
        "terminated_tipover": bool(info.get("tipover")),
        "mission_loaded": bool(info.get("mission_loaded")),
        "geofence_advice": info.get("geofence_advice"),
        "living_advice": info.get("living_advice"),
        "log": records,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    env.close()
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dump Jim's Mower camera frames + detections")
    p.add_argument("--out", type=Path, default=Path("demo_out"))
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cameras", type=int, default=None, help="4, 5, or 6")
    p.add_argument("--hand-signals", action="store_true")
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Env YAML, scenario path, or bundled scenario name (suburban, …)",
    )
    p.add_argument(
        "--policy",
        choices=POLICIES,
        default="terrain",
        help="terrain (default coverage planner), scripted creep, or random wheels",
    )
    p.add_argument(
        "--terrain-observer",
        choices=("heuristic", "oracle", "blind", "learned"),
        default=None,
        help="override perception.terrain_mode (default: YAML, heuristic)",
    )
    p.add_argument(
        "--save-mission",
        type=Path,
        default=None,
        help="write map + uncut + pose on close (WAVE 2A resume)",
    )
    p.add_argument(
        "--load-mission",
        type=Path,
        default=None,
        help="restore map + uncut + pose before the first step",
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    summary = run_demo(
        args.out,
        steps=args.steps,
        seed=args.seed,
        cameras=args.cameras,
        hand_signals=args.hand_signals,
        config=args.config,
        policy=args.policy,
        terrain_observer=args.terrain_observer,
        load_mission=str(args.load_mission) if args.load_mission else None,
        save_mission=str(args.save_mission) if args.save_mission else None,
    )
    print(
        f"Wrote {summary['steps_run']} steps, policy={summary['policy']}, "
        f"observer={summary.get('terrain_source')}, "
        f"cameras={summary['cameras']} → {args.out}"
    )
    print(f"Final coverage: {summary['final_coverage_fraction']:.4f}")
    print(f"Detections on last step: {len(summary['final_detections'])}")
    print(f"Plan waypoints: {summary['n_waypoints']}")


if __name__ == "__main__":
    main()

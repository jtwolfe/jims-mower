"""Headless demo: dump multi-camera frames, detections, and a coverage map."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from jims_mower.bc import BcPolicy, maybe_load_bc
from jims_mower.bev import render_bev
from jims_mower.constants import BC_FEATURE_DIM, DEFAULT_BC_WEIGHTS, TERRAIN_MODE_CLI
from jims_mower.env import MowerEnv
from jims_mower.kinematics import sit_on_terrain, trimmer_xy
from jims_mower.planning import TerrainPolicy
from jims_mower.renderer import render_camera, render_topdown
from jims_mower.scenarios import load_source
from jims_mower.teach import TeachPolicy, keepouts_from_env
from jims_mower.types import Pose
from jims_mower.viewer import write_viewer_bundle

POLICIES = ("terrain", "scripted", "random", "bc", "teach")
DEFAULT_DEMO_STEPS = 160


DEFAULT_CAM_STRIDE = 10


def camera_dump_stride(steps: int, requested: Optional[int] = None) -> int:
    """How often to write camera PiP folders so long runs stay scrubbable.

    Default is every 10 steps. Always include step 0 and the last step via
    :func:`camera_dump_indices`. ``poses.json`` is still written every step.
    """
    _ = steps
    if requested is None:
        return DEFAULT_CAM_STRIDE
    return max(1, int(requested))


def camera_dump_indices(steps: int, stride: Optional[int] = None) -> set[int]:
    n = max(0, int(steps))
    if n <= 0:
        return {0}
    every = camera_dump_stride(n, stride)
    picks = set(range(0, n, every))
    picks.add(0)
    picks.add(n - 1)
    return picks


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


def _plan_overlay(env: MowerEnv, policy: Optional[Any]) -> np.ndarray:
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
    steps: int = DEFAULT_DEMO_STEPS,
    cam_stride: Optional[int] = None,
    dump_stride: Optional[int] = None,
    seed: int = 7,
    cameras: Optional[int] = None,
    hand_signals: bool = False,
    config: Optional[Any] = None,
    policy: str = "terrain",
    terrain_observer: Optional[str] = None,
    load_mission: Optional[str] = None,
    save_mission: Optional[str] = None,
    bc_weights: Optional[str] = None,
    log_bc: Optional[str] = None,
    yard_profile: Optional[str] = None,
) -> dict:
    name = (policy or "terrain").strip().lower()
    if name not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}; got {policy!r}")
    cfg, scenario = load_source(config)
    if cameras is not None:
        cfg.sensors.camera_count = cameras
        cfg.sensors.cameras = []
    if terrain_observer:
        from jims_mower.perception.terrain import normalize_terrain_mode

        cfg.perception.terrain_mode = normalize_terrain_mode(terrain_observer)
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
    if yard_profile:
        reset_opts["yard_profile"] = yard_profile
    obs, info = env.reset(seed=seed, options=reset_opts or None)
    out_dir.mkdir(parents=True, exist_ok=True)

    terrain_policy: Optional[TerrainPolicy] = None
    teach_policy: Optional[TeachPolicy] = None
    bc_policy: Optional[BcPolicy] = None
    bc_loaded = False
    rng = np.random.default_rng(seed)
    if name == "bc":
        bc_policy = maybe_load_bc(bc_weights, env.cfg.world.resolution_m)
        if bc_policy is not None:
            bc_policy.reset(obs, info)
            bc_loaded = True
        else:
            name = "terrain"
    if name == "teach":
        teach_policy = TeachPolicy(env.cfg, spec=env.geofence_spec())
        teach_policy.reset(obs, info)
    elif name == "terrain":
        terrain_policy = TerrainPolicy(env.cfg)
        terrain_policy.reset(obs, info)
    bc_log_feats: list[np.ndarray] = []
    bc_log_acts: list[np.ndarray] = []

    names = list(obs["cameras"].keys())
    records: list[dict] = []
    poses: list[dict] = [info.get("pose") or {}]
    dump_steps = camera_dump_indices(steps, cam_stride if cam_stride is not None else dump_stride)
    overlay_policy = teach_policy or terrain_policy

    def _dump_step(step_dir: Path, obs: dict, info: dict) -> None:
        step_dir.mkdir(parents=True, exist_ok=True)
        for cam_name, frame in obs["cameras"].items():
            _save_rgb(step_dir / f"cam_{cam_name}.png", frame)
        _save_rgb(step_dir / "montage.png", _montage(obs["cameras"], names))
        topdown = env.render()
        if topdown is not None:
            _save_rgb(step_dir / "topdown.png", topdown)
        _save_rgb(step_dir / "plan_overlay.png", _plan_overlay(env, overlay_policy))
        _save_rgb(
            step_dir / "bev.png",
            render_bev(
                env,
                obs,
                waypoints=overlay_policy.waypoints if overlay_policy else [],
                waypoint_index=overlay_policy.index if overlay_policy else 0,
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
                    "bc_loaded": bc_loaded,
                    "safe_mode": getattr(terrain_policy, "last_safe_mode", None)
                    if terrain_policy
                    else None,
                    "waypoint_index": overlay_policy.index if overlay_policy else None,
                    "n_waypoints": len(overlay_policy.waypoints) if overlay_policy else 0,
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
        elif name == "bc":
            assert bc_policy is not None
            action = bc_policy.act(obs, info)
        elif name == "teach":
            assert teach_policy is not None
            action = teach_policy.act(obs, info)
        else:
            assert terrain_policy is not None
            action = terrain_policy.act(obs, info)
        if log_bc:
            from jims_mower.features import extract_features

            bc_log_feats.append(
                extract_features(obs, info, resolution_m=env.cfg.world.resolution_m)
            )
            bc_log_acts.append(np.asarray(action, dtype=np.float32).reshape(-1)[:3])
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
                "pose": info.get("pose"),
            }
        )
        poses.append(info.get("pose") or {})
        if terminated or truncated or (teach_policy is not None and teach_policy.done):
            break

    topdown = env.render()
    if topdown is not None:
        _save_rgb(out_dir / "coverage_final.png", topdown)
    _save_rgb(out_dir / "plan_overlay_final.png", _plan_overlay(env, overlay_policy))
    _save_rgb(
        out_dir / "bev_final.png",
        render_bev(
            env,
            obs,
            waypoints=overlay_policy.waypoints if overlay_policy else [],
            waypoint_index=overlay_policy.index if overlay_policy else 0,
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
            {
                "x": x,
                "y": y,
                "z": float(env._terrain.sample(x, y)),
            }
            for x, y in (overlay_policy.waypoints if overlay_policy else [])
        ],
        "index": overlay_policy.index if overlay_policy else 0,
        "n_segments": terrain_policy.plan.n_segments if terrain_policy and terrain_policy.plan else 0,
        "replans": terrain_policy.replans if terrain_policy else 0,
    }
    (out_dir / "plan.json").write_text(json.dumps(plan_payload, indent=2), encoding="utf-8")
    summary = {
        "steps_run": len(records),
        "cameras": names,
        "policy": name,
        "bc_loaded": bc_loaded,
        "bc_weights": bc_weights or DEFAULT_BC_WEIGHTS,
        "safe_mode": getattr(terrain_policy, "last_safe_mode", None) if terrain_policy else None,
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
        "world": {
            "width_m": env.cfg.world.width_m,
            "height_m": env.cfg.world.height_m,
            "resolution_m": env.cfg.world.resolution_m,
        },
        "log": records,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    profile_payload = None
    if teach_policy is not None:
        taught = teach_policy.to_profile(keep_out=keepouts_from_env(env))
        profile_payload = taught.as_dict()
        (out_dir / "profile.json").write_text(json.dumps(profile_payload, indent=2), encoding="utf-8")
        summary["profile"] = str(out_dir / "profile.json")
        summary["keep_in_vertices"] = len(taught.keep_in)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_viewer_bundle(
        out_dir,
        env=env,
        poses=poses,
        plan=plan_payload,
        profile=profile_payload,
        cameras=names,
        policy=name,
    )
    if log_bc:
        dest = Path(log_bc)
        dest.mkdir(parents=True, exist_ok=True)
        x = (
            np.stack(bc_log_feats, axis=0)
            if bc_log_feats
            else np.zeros((0, BC_FEATURE_DIM), dtype=np.float32)
        )
        y = (
            np.stack(bc_log_acts, axis=0)
            if bc_log_acts
            else np.zeros((0, 3), dtype=np.float32)
        )
        np.save(dest / "features.npy", x)
        np.save(dest / "actions.npy", y)
        np.savez_compressed(dest / "dataset.npz", features=x, actions=y)
        (dest / "meta.json").write_text(
            json.dumps(
                {
                    "n_samples": int(x.shape[0]),
                    "policy": name,
                    "not_a_benchmark": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        summary["bc_log"] = str(dest)
        summary["bc_log_samples"] = int(x.shape[0])
    env.close()
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dump Jim's Mower camera frames + detections")
    p.add_argument("--out", type=Path, default=Path("demo_out"))
    p.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_DEMO_STEPS,
        help="episode length (default 160 — long enough to inspect in the viewer)",
    )
    p.add_argument(
        "--cam-stride",
        type=int,
        default=DEFAULT_CAM_STRIDE,
        help="write camera folders every N steps (default 10; always includes 0 and last)",
    )
    p.add_argument(
        "--dump-stride",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
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
        help="terrain (default), scripted, random, bc, or teach (perimeter → YardProfile)",
    )
    p.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="load a YardProfile JSON (geofence + home) into the env",
    )
    p.add_argument(
        "--bc-weights",
        type=Path,
        default=None,
        help="numpy BC weights (default bc_weights.npz); ignored unless --policy bc",
    )
    p.add_argument(
        "--log-bc",
        type=Path,
        default=None,
        help="write (obs→action) feature/action arrays while the demo runs",
    )
    p.add_argument(
        "--terrain-observer",
        choices=TERRAIN_MODE_CLI,
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
        cam_stride=args.dump_stride if args.dump_stride is not None else args.cam_stride,
        seed=args.seed,
        cameras=args.cameras,
        hand_signals=args.hand_signals,
        config=args.config,
        policy=args.policy,
        terrain_observer=args.terrain_observer,
        load_mission=str(args.load_mission) if args.load_mission else None,
        save_mission=str(args.save_mission) if args.save_mission else None,
        bc_weights=str(args.bc_weights) if args.bc_weights else None,
        log_bc=str(args.log_bc) if args.log_bc else None,
        yard_profile=str(args.profile) if args.profile else None,
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

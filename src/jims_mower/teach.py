"""Teach Boundary: follow a perimeter in sim, record a trail, write a YardProfile."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from jims_mower.env import MowerEnv
from jims_mower.geofence import GeofenceSpec
from jims_mower.planning.controller import tracking_action
from jims_mower.profile import (
    YardProfile,
    circle_polygon,
    perimeter_waypoints,
    trail_to_polygon,
    write_yard_profile,
)
from jims_mower.scenarios import load_source
from jims_mower.types import Pose


def _pose_from(obs: dict[str, Any], info: dict[str, Any]) -> Pose:
    raw = info.get("pose")
    if isinstance(raw, dict) and "x" in raw:
        return Pose(
            float(raw["x"]),
            float(raw["y"]),
            float(raw.get("theta", 0.0)),
            float(raw.get("z", 0.0)),
            float(raw.get("pitch", 0.0)),
            float(raw.get("roll", 0.0)),
        )
    arr = np.asarray(obs.get("pose", [0, 0, 0]), dtype=np.float32).reshape(-1)
    return Pose(
        float(arr[0]) if arr.size > 0 else 0.0,
        float(arr[1]) if arr.size > 1 else 0.0,
        float(arr[2]) if arr.size > 2 else 0.0,
        float(arr[3]) if arr.size > 3 else 0.0,
        float(arr[4]) if arr.size > 4 else 0.0,
        float(arr[5]) if arr.size > 5 else 0.0,
    )


class TeachPolicy:
    """Drive the authored (or inset-rectangle) perimeter. Trimmer stays off."""

    def __init__(
        self,
        cfg: Any,
        spec: Optional[GeofenceSpec] = None,
        *,
        margin_m: float = 0.80,
        arrive_m: float = 0.45,
        cruise: float = 0.85,
    ) -> None:
        self.cfg = cfg
        self.margin_m = float(margin_m)
        self.arrive_m = float(arrive_m)
        self.cruise = float(cruise)
        self.spec = spec or GeofenceSpec()
        self.waypoints: list[tuple[float, float]] = []
        self.index = 0
        self.trail: list[tuple[float, float]] = []
        self.last_advice = "ok"
        self.replans = 0
        self.plan = None
        self.last_safe_mode = "run"
        self.done = False

    def reset(self, obs: dict[str, Any], info: Optional[dict[str, Any]] = None) -> None:
        info = info or {}
        raw = (info.get("geofence_spec") or {}).get("keep_in") if isinstance(info, dict) else None
        keep_in = None
        if raw and len(raw) >= 3:
            keep_in = [(float(p[0]), float(p[1])) for p in raw]
        elif self.spec.keep_in and len(self.spec.keep_in) >= 3:
            keep_in = list(self.spec.keep_in)
        self.waypoints = perimeter_waypoints(
            self.cfg.world.width_m,
            self.cfg.world.height_m,
            margin_m=self.margin_m,
            keep_in=keep_in,
        )
        self.index = 0
        self.trail = []
        self.done = False
        self.last_advice = "ok"
        pose = _pose_from(obs, info)
        self.trail.append((pose.x, pose.y))

    def act(self, obs: dict[str, Any], info: dict[str, Any]) -> np.ndarray:
        pose = _pose_from(obs, info)
        self.trail.append((pose.x, pose.y))
        if not self.waypoints or self.done:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        while self.index < len(self.waypoints) - 1:
            tx, ty = self.waypoints[self.index]
            if math.hypot(pose.x - tx, pose.y - ty) <= self.arrive_m:
                self.index += 1
                continue
            break
        target = self.waypoints[min(self.index, len(self.waypoints) - 1)]
        wheels, dist, _err = tracking_action(
            pose,
            target,
            cruise=self.cruise,
            wheelbase_m=self.cfg.robot.wheelbase_m,
            turn_in_place_rad=0.65,
        )
        if self.index >= len(self.waypoints) - 1 and dist <= self.arrive_m:
            self.done = True
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return np.array([float(wheels[0]), float(wheels[1]), 0.0], dtype=np.float32)

    def planned_ring(self) -> list[tuple[float, float]]:
        pts = list(self.waypoints)
        if len(pts) >= 2 and math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < 1e-6:
            pts = pts[:-1]
        return pts

    def to_profile(
        self,
        *,
        name: str = "taught",
        keep_out: Optional[list[list[tuple[float, float]]]] = None,
        mesh: str = "yard.glb",
        home: Optional[dict[str, float]] = None,
    ) -> YardProfile:
        ring = trail_to_polygon(self.trail, fallback=self.planned_ring())
        if len(ring) < 3:
            ring = self.planned_ring()
        if home is None:
            if self.trail:
                home = {"x": float(self.trail[0][0]), "y": float(self.trail[0][1]), "theta": 0.0}
            elif ring:
                home = {"x": float(ring[0][0]), "y": float(ring[0][1]), "theta": 0.0}
            else:
                home = {"x": 1.0, "y": 1.0, "theta": 0.0}
        return YardProfile(
            name=name,
            width_m=float(self.cfg.world.width_m),
            height_m=float(self.cfg.world.height_m),
            resolution_m=float(self.cfg.world.resolution_m),
            keep_in=ring,
            keep_out=list(keep_out or []),
            home=home,
            mesh=mesh,
            trail=list(self.trail),
        )


def keepouts_from_env(env: MowerEnv, *, kinds: Optional[set[str]] = None) -> list[list[tuple[float, float]]]:
    kinds = kinds or {"tree", "furniture"}
    holes: list[list[tuple[float, float]]] = []
    for obst in env._yard.static():
        if obst.kind in kinds:
            holes.append(circle_polygon(obst.x, obst.y, obst.radius + 0.25, n=8))
    return holes


def run_teach(
    out_dir: Path,
    *,
    steps: int = 80,
    seed: int = 7,
    cameras: Optional[int] = 4,
    config: Optional[Any] = None,
    profile_name: str = "taught",
    auto_keepouts: bool = True,
    dump_cameras: bool = True,
) -> dict[str, Any]:
    """Perimeter run → YardProfile + viewer bundle (no fake mAP/FPS)."""
    from jims_mower.viewer import dump_step_frames, write_viewer_bundle

    cfg, scenario = load_source(config)
    if cameras is not None:
        cfg.sensors.camera_count = cameras
        cfg.sensors.cameras = []
    env = MowerEnv(config=cfg, scenario=scenario, render_mode="rgb_array")
    obs, info = env.reset(seed=seed)
    policy = TeachPolicy(env.cfg, spec=env.geofence_spec())
    policy.reset(obs, info)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(obs["cameras"].keys())
    records: list[dict[str, Any]] = []
    dump_steps = {0, max(0, steps // 2), max(0, steps - 1)}
    poses: list[dict[str, float]] = [info.get("pose") or {}]

    for t in range(int(steps)):
        if dump_cameras and t in dump_steps:
            dump_step_frames(out_dir / f"step_{t:03d}", obs, names=names)
        action = policy.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        poses.append(info.get("pose") or {})
        records.append(
            {
                "step": t,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "coverage_fraction": info.get("coverage_fraction"),
                "pose": info.get("pose"),
                "policy": "teach",
                "waypoint_index": policy.index,
            }
        )
        if terminated or truncated or policy.done:
            break

    keep_out = keepouts_from_env(env) if auto_keepouts else []
    profile = policy.to_profile(name=profile_name, keep_out=keep_out)
    profile_path = write_yard_profile(out_dir / "profile.json", profile)
    (out_dir / "trail.json").write_text(
        json.dumps({"points": [list(p) for p in policy.trail], "waypoints": [list(p) for p in policy.waypoints]}, indent=2),
        encoding="utf-8",
    )
    plan_payload = {
        "policy": "teach",
        "waypoints": [{"x": x, "y": y} for x, y in policy.waypoints],
        "index": policy.index,
        "n_segments": max(0, len(policy.waypoints) - 1),
        "replans": 0,
    }
    (out_dir / "plan.json").write_text(json.dumps(plan_payload, indent=2), encoding="utf-8")
    summary = {
        "steps_run": len(records),
        "cameras": names,
        "policy": "teach",
        "final_coverage_fraction": records[-1]["coverage_fraction"] if records else 0.0,
        "final_pose": info.get("pose"),
        "n_waypoints": len(policy.waypoints),
        "trail_points": len(policy.trail),
        "keep_in_vertices": len(profile.keep_in),
        "keep_out_count": len(profile.keep_out),
        "profile": str(profile_path),
        "world": {
            "width_m": env.cfg.world.width_m,
            "height_m": env.cfg.world.height_m,
            "resolution_m": env.cfg.world.resolution_m,
        },
        "not_a_benchmark": True,
        "log": records,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_viewer_bundle(
        out_dir,
        env=env,
        poses=poses,
        plan=plan_payload,
        profile=profile.as_dict(),
        cameras=names,
        policy="teach",
    )
    env.close()
    summary["out"] = str(out_dir)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Teach a yard boundary (perimeter run → YardProfile)")
    p.add_argument("--out", type=Path, default=Path("teach_out"))
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cameras", type=int, default=4)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--name", type=str, default="taught")
    p.add_argument("--no-keepouts", action="store_true", help="do not seed tree/furniture holes")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    summary = run_teach(
        args.out,
        steps=args.steps,
        seed=args.seed,
        cameras=args.cameras,
        config=args.config,
        profile_name=args.name,
        auto_keepouts=not args.no_keepouts,
    )
    print(
        f"Taught {summary['keep_in_vertices']} keep-in verts, "
        f"trail={summary['trail_points']} → {summary['profile']}"
    )


if __name__ == "__main__":
    main()

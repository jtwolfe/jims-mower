"""Multi-session mission state: coverage map + uncut grass + pose.

Grass persist already writes the yard mask. This adds pose and a CLI so a
later process can resume the same cut without replaying the episode.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from jims_mower.constants import MISSION_SCHEMA
from jims_mower.maps import GrassCoverageMap
from jims_mower.types import Pose


@dataclass
class MissionState:
    schema: str
    pose: Pose
    coverage: GrassCoverageMap
    scenario: str = ""
    seed: Optional[int] = None
    steps: int = 0

    def as_meta(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "pose": {
                "x": self.pose.x,
                "y": self.pose.y,
                "theta": self.pose.theta,
                "z": self.pose.z,
                "pitch": self.pose.pitch,
                "roll": self.pose.roll,
            },
            "scenario": self.scenario,
            "seed": self.seed,
            "steps": self.steps,
            "width_m": self.coverage.width_m,
            "height_m": self.coverage.height_m,
            "resolution_m": self.coverage.resolution_m,
            "cut_cells": self.coverage.cut_cell_count(),
            "uncut_cells": self.coverage.grass_cell_count() - self.coverage.cut_cell_count(),
        }


def save_mission(
    path: Union[str, Path],
    coverage: GrassCoverageMap,
    pose: Pose,
    *,
    scenario: str = "",
    seed: Optional[int] = None,
    steps: int = 0,
) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest,
        schema=np.asarray(MISSION_SCHEMA),
        cut=coverage.cut.astype(np.bool_),
        grass=coverage.grass.astype(np.bool_),
        width_m=np.float32(coverage.width_m),
        height_m=np.float32(coverage.height_m),
        resolution_m=np.float32(coverage.resolution_m),
        pose=np.array(
            [pose.x, pose.y, pose.theta, pose.z, pose.pitch, pose.roll],
            dtype=np.float32,
        ),
        scenario=np.asarray(scenario),
        seed=np.int64(-1 if seed is None else seed),
        steps=np.int64(steps),
    )
    sidecar = dest.with_suffix(".json")
    if dest.suffix == ".npz":
        sidecar = dest.with_name(dest.stem + ".json")
    meta = MissionState(MISSION_SCHEMA, pose, coverage, scenario, seed, steps).as_meta()
    sidecar.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return dest


def load_mission(path: Union[str, Path]) -> MissionState:
    src = Path(path)
    data = np.load(src, allow_pickle=False)
    schema = str(data["schema"]) if "schema" in data.files else MISSION_SCHEMA
    width = float(data["width_m"])
    height = float(data["height_m"])
    res = float(data["resolution_m"])
    coverage = GrassCoverageMap(width, height, res)
    coverage.cut = np.asarray(data["cut"], dtype=bool)
    coverage.grass = np.asarray(data["grass"], dtype=bool)
    if coverage.cut.shape != (coverage.rows, coverage.cols):
        raise ValueError(
            f"mission map shape {coverage.cut.shape} does not match "
            f"{coverage.rows}x{coverage.cols}"
        )
    arr = np.asarray(data["pose"], dtype=np.float32).reshape(-1)
    pose = Pose(
        float(arr[0]) if arr.size > 0 else 0.0,
        float(arr[1]) if arr.size > 1 else 0.0,
        float(arr[2]) if arr.size > 2 else 0.0,
        float(arr[3]) if arr.size > 3 else 0.0,
        float(arr[4]) if arr.size > 4 else 0.0,
        float(arr[5]) if arr.size > 5 else 0.0,
    )
    seed_raw = int(data["seed"]) if "seed" in data.files else -1
    return MissionState(
        schema=schema,
        pose=pose,
        coverage=coverage,
        scenario=str(data["scenario"]) if "scenario" in data.files else "",
        seed=None if seed_raw < 0 else seed_raw,
        steps=int(data["steps"]) if "steps" in data.files else 0,
    )


def apply_mission(coverage: GrassCoverageMap, state: MissionState) -> Pose:
    """Copy cut/uncut onto an existing map (same grid) and return the stored pose."""
    if state.coverage.cut.shape != coverage.cut.shape:
        raise ValueError(
            f"mission shape {state.coverage.cut.shape} does not match "
            f"yard {coverage.cut.shape}"
        )
    coverage.cut = state.coverage.cut.copy()
    coverage.grass = state.coverage.grass.copy()
    return state.pose


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Save or resume a Jim's Mower mission (map + pose)")
    sub = p.add_subparsers(dest="cmd", required=True)
    save = sub.add_parser("save", help="Run a short episode and write mission state")
    save.add_argument("--out", type=Path, required=True)
    save.add_argument("--config", type=str, default=None)
    save.add_argument("--steps", type=int, default=20)
    save.add_argument("--seed", type=int, default=7)
    save.add_argument("--cameras", type=int, default=4)
    save.add_argument("--policy", choices=("terrain", "scripted"), default="terrain")
    resume = sub.add_parser("resume", help="Load mission state and continue")
    resume.add_argument("--in", dest="src", type=Path, required=True)
    resume.add_argument("--out", type=Path, default=Path("mission_resume"))
    resume.add_argument("--config", type=str, default=None)
    resume.add_argument("--steps", type=int, default=20)
    resume.add_argument("--cameras", type=int, default=4)
    resume.add_argument("--policy", choices=("terrain", "scripted"), default="terrain")
    inspect = sub.add_parser("inspect", help="Print mission metadata")
    inspect.add_argument("--in", dest="src", type=Path, required=True)
    return p


def _run_demo_with_mission(
    *,
    out_dir: Path,
    steps: int,
    seed: int,
    cameras: int,
    config: Optional[str],
    policy: str,
    load_path: Optional[Path],
    save_path: Optional[Path],
) -> dict:
    from jims_mower.demo import run_demo

    return run_demo(
        out_dir,
        steps=steps,
        seed=seed,
        cameras=cameras,
        config=config,
        policy=policy,
        load_mission=str(load_path) if load_path else None,
        save_mission=str(save_path) if save_path else None,
    )


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "inspect":
        state = load_mission(args.src)
        print(json.dumps(state.as_meta(), indent=2))
        return
    if args.cmd == "save":
        summary = _run_demo_with_mission(
            out_dir=args.out.parent / (args.out.stem + "_demo"),
            steps=args.steps,
            seed=args.seed,
            cameras=args.cameras,
            config=args.config,
            policy=args.policy,
            load_path=None,
            save_path=args.out,
        )
        print(f"Wrote mission {args.out} coverage={summary['final_coverage_fraction']:.4f}")
        return
    state = load_mission(args.src)
    cfg = args.config or (state.scenario or None)
    summary = _run_demo_with_mission(
        out_dir=args.out,
        steps=args.steps,
        seed=state.seed if state.seed is not None else 7,
        cameras=args.cameras,
        config=cfg,
        policy=args.policy,
        load_path=args.src,
        save_path=args.out / "mission.npz" if args.out.suffix != ".npz" else args.out,
    )
    print(
        f"Resumed {args.src} → {args.out} "
        f"coverage={summary['final_coverage_fraction']:.4f}"
    )


if __name__ == "__main__":
    main()

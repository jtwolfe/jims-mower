"""CLI: jims-mower-record — log an episode + latency scorecard."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from jims_mower.episode import record_episode
from jims_mower.latency import LatencyDelays


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Record a Jim's Mower episode (obs, actions, maps)")
    p.add_argument("--out", type=Path, required=True, help="episode directory")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--cameras", type=int, default=None, help="4, 5, or 6")
    p.add_argument("--policy", choices=("terrain", "scripted", "random"), default="terrain")
    p.add_argument(
        "--terrain-observer",
        choices=("heuristic", "oracle", "blind"),
        default=None,
    )
    p.add_argument("--camera-delay-ms", type=float, default=0.0)
    p.add_argument("--plan-delay-ms", type=float, default=0.0)
    p.add_argument("--cmd-delay-ms", type=float, default=0.0)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    delays = LatencyDelays.from_ms(args.camera_delay_ms, args.plan_delay_ms, args.cmd_delay_ms)
    result = record_episode(
        args.out,
        steps=args.steps,
        seed=args.seed,
        config=args.config,
        policy=args.policy,
        cameras=args.cameras,
        delays=delays,
        terrain_observer=args.terrain_observer,
    )
    card = result["scorecard"]
    cam = card["timings_ms"]["camera_to_cmd"]
    print(
        f"Recorded {result['n_steps']} steps → {args.out} "
        f"(camera→cmd p50={cam['p50_ms']:.2f} ms; not a board FPS claim)"
    )


if __name__ == "__main__":
    main()

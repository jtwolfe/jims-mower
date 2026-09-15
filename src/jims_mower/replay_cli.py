"""CLI: jims-mower-replay — offline planner replay or env action playback."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from jims_mower.episode import replay_env, replay_offline
from jims_mower.latency import LatencyDelays


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Replay a recorded Jim's Mower episode")
    p.add_argument("episode", type=Path, help="directory written by jims-mower-record")
    p.add_argument(
        "--mode",
        choices=("offline", "env"),
        default="offline",
        help="offline: planner/controller only; env: apply recorded actions in the gym",
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--camera-delay-ms", type=float, default=0.0)
    p.add_argument("--plan-delay-ms", type=float, default=0.0)
    p.add_argument("--cmd-delay-ms", type=float, default=0.0)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.mode == "env":
        summary = replay_env(args.episode, out_dir=args.out)
        print(
            f"Env replay {summary['n_steps']} steps, "
            f"coverage={summary['final_coverage_fraction']:.4f}"
        )
        return
    delays = LatencyDelays.from_ms(args.camera_delay_ms, args.plan_delay_ms, args.cmd_delay_ms)
    summary = replay_offline(args.episode, delays=delays, out_dir=args.out)
    print(
        f"Offline replay {summary['n_steps']} steps, "
        f"max |Δaction|={summary['max_abs_action_delta']:.4g}"
    )


if __name__ == "__main__":
    main()

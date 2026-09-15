"""CLI: jims-mower-rl — 1-episode CPU smoke (REINFORCE / random-search / optional SB3)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from jims_mower.rl import smoke


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RL fine-tune scaffold (CPU smoke, no claimed scores)")
    p.add_argument(
        "--algo",
        choices=("reinforce", "random-search", "sb3"),
        default="reinforce",
        help="numpy REINFORCE (default), random-search, or optional SB3 PPO",
    )
    p.add_argument("--episodes", type=int, default=1, help="must be 1 for the smoke")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="optional JSON summary")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="alias kept for CI: 1-episode, no GPU",
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if int(args.episodes) != 1:
        raise SystemExit("jims-mower-rl smoke is 1 episode only (no fake training curves)")
    result = smoke(algo=args.algo, seed=args.seed, steps=args.steps)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"{result['algo']} smoke: steps={result.get('steps')} "
        f"gpu={result.get('gpu')} (not a benchmark)"
    )


if __name__ == "__main__":
    main()

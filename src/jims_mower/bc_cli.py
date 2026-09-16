"""CLI: jims-mower-bc — collect terrain demos and train the numpy BC stub."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from jims_mower.bc import collect_demos, default_weights_path, train_bc
from jims_mower.constants import DEFAULT_BC_WEIGHTS, TERRAIN_MODE_CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Behaviour-clone Jim's Mower terrain policy (numpy stub)")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="log (obs→action) from TerrainPolicy")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--steps", type=int, default=40)
    c.add_argument("--seed", type=int, default=7)
    c.add_argument("--config", type=str, default=None)
    c.add_argument("--cameras", type=int, default=None)
    c.add_argument(
        "--terrain-observer",
        choices=TERRAIN_MODE_CLI,
        default=None,
    )

    t = sub.add_parser("train", help="fit the tiny numpy MLP")
    t.add_argument("--in", dest="source", type=Path, required=True, help="collect dir or episode dir")
    t.add_argument("--out", type=Path, default=Path(DEFAULT_BC_WEIGHTS))
    t.add_argument("--epochs", type=int, default=80)
    t.add_argument("--lr", type=float, default=0.05)
    t.add_argument("--hidden", type=int, default=16)
    t.add_argument("--seed", type=int, default=0)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "collect":
        meta = collect_demos(
            args.out,
            steps=args.steps,
            seed=args.seed,
            config=args.config,
            cameras=args.cameras,
            terrain_observer=args.terrain_observer,
        )
        print(
            f"Collected {meta['n_samples']} (obs→action) pairs → {args.out} "
            "(teacher=terrain; not a benchmark)"
        )
        return
    result = train_bc(
        args.source,
        args.out,
        hidden=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )
    print(
        f"Wrote {args.out} from {result['n_samples']} samples "
        f"(train MSE={result['train_mse']:.4g}; not a published score). "
        f"Demo: jims-mower-demo --policy bc --bc-weights {args.out or default_weights_path()}"
    )


if __name__ == "__main__":
    main()

"""CLI: jims-mower-app — local owner API + phone shell."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from jims_mower.app.backend import make_backend
from jims_mower.app.server import serve_app


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Serve the WAVE UX-C owner app (JSON API + phone UI) against a sim or episode backend",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--backend",
        choices=("sim", "demo", "episode", "memory"),
        default="sim",
        help="sim/demo = live MowerEnv; episode = recorded directory; memory = kinematic stub",
    )
    p.add_argument("--config", default="geofence_movers", help="demo env / scenario name or YAML")
    p.add_argument("--episode", type=Path, default=None, help="jims-mower-record directory")
    p.add_argument("--yard", type=Path, default=None, help="YardProfile JSON to load (and persist on PUT)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cameras", type=int, default=4)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    kind = args.backend
    if args.episode is not None:
        kind = "episode"
    backend = make_backend(
        kind=kind,
        config=args.config,
        episode=args.episode,
        yard=args.yard,
        yard_path=args.yard,
        seed=args.seed,
        cameras=args.cameras,
    )
    print(f"jims-mower-app http://{args.host}:{args.port}/  backend={kind}")
    serve_app(backend, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

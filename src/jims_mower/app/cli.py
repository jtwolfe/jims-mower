"""CLI: jims-mower-app — local owner API + phone shell."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from jims_mower.app.backend import make_backend
from jims_mower.app.server import serve_app
from jims_mower.constants import APP_LIVE_PORT


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Serve the WAVE UX-C owner app (JSON API + phone UI) against a sim, episode, or live session",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument(
        "--port",
        type=int,
        default=None,
        help="listen port (default 8766 with --live, else 8765)",
    )
    p.add_argument(
        "--backend",
        choices=("sim", "demo", "episode", "memory", "live"),
        default="sim",
        help="sim/demo = live MowerEnv; episode = recorded directory; memory = kinematic stub; live = LiveSession",
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="own a live job (LiveSession + /api/live/control). Default yard acre_yard_demo, port 8766",
    )
    p.add_argument(
        "--config",
        default=None,
        help="demo env / scenario (live default: acre_yard_demo; sim default: geofence_movers)",
    )
    p.add_argument("--episode", type=Path, default=None, help="jims-mower-record directory")
    p.add_argument("--yard", type=Path, default=None, help="YardProfile JSON to load (and persist on PUT)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cameras", type=int, default=4)
    p.add_argument("--speed", default="5", help="live wall-clock multiplier: 1, 2, 5, or max")
    p.add_argument("--fast", action="store_true", help="live: mission_tiny + short budgets (CI)")
    p.add_argument("--steps", type=int, default=None, help="live episode budget")
    p.add_argument("--out", type=Path, default=Path("live_out"), help="live bundle directory")
    p.add_argument(
        "--first-run",
        action="store_true",
        help="first-run setup: pair → teach keep-in → save YardProfile → Start uses that fence",
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    kind = args.backend
    if args.episode is not None:
        kind = "episode"
    if args.live:
        kind = "live"
    live = kind == "live"
    port = int(args.port if args.port is not None else (APP_LIVE_PORT if live else 8765))
    if live:
        config = args.config or ("mission_tiny" if args.fast else "acre_yard_demo")
    else:
        config = args.config or "geofence_movers"
    backend = make_backend(
        kind=kind,
        config=config,
        episode=args.episode,
        yard=args.yard,
        yard_path=args.yard,
        seed=args.seed,
        cameras=args.cameras,
        fast=args.fast,
        speed=args.speed,
        steps=args.steps,
        out_dir=args.out,
        first_run=args.first_run,
    )
    print(f"jims-mower-app http://{args.host}:{port}/  backend={kind}  config={config}")
    if live:
        if args.first_run:
            print("First-run: Pair BT stub → Teach boundary → Save yard → Start job.")
        else:
            print("Phone owns the live job. Pair BT stub → Teach (optional) → Start / Pause / ESTOP.")
        print("Taught YardProfile is the job geofence. Demo confirm-fence is acre_yard_demo without teach.")
        print("Desktop viewer still: jims-mower-live --config acre_yard_demo --speed 5")
    serve_app(backend, host=args.host, port=port)


if __name__ == "__main__":
    main()

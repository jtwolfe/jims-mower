"""CLI: jims-mower-owner — phone overlay mock, or live phone app."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from jims_mower.constants import APP_LIVE_PORT
from jims_mower.owner import export_owner_overlay


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Owner phone: static overlay export, or --live to serve the app + live job"
    )
    p.add_argument("--out", type=Path, default=Path("owner_overlay.html"))
    p.add_argument("--config", type=str, default=None, help="scenario (live default acre_yard_demo)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cameras", type=int, default=4)
    p.add_argument(
        "--live",
        action="store_true",
        help="one command: phone app + LiveSession (acre_yard_demo on :8766)",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--speed", default="5")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--steps", type=int, default=None)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.live:
        from jims_mower.app.cli import main as app_main

        config = args.config or "acre_yard_demo"
        port = args.port if args.port is not None else APP_LIVE_PORT
        launch = [
            "--live",
            "--config",
            config,
            "--host",
            args.host,
            "--port",
            str(port),
            "--speed",
            str(args.speed),
            "--seed",
            str(args.seed),
            "--cameras",
            str(args.cameras),
        ]
        if args.fast:
            launch.append("--fast")
        if args.steps is not None:
            launch.extend(["--steps", str(args.steps)])
        app_main(launch)
        return
    payload = export_owner_overlay(
        args.out,
        config=args.config or "geofence_movers",
        seed=args.seed,
        cameras=args.cameras,
    )
    print(
        f"Owner overlay {payload.get('scenario')} "
        f"waypoints={len(payload.get('waypoints') or [])} → {args.out}"
    )


if __name__ == "__main__":
    main()

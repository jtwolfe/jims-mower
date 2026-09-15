"""CLI: jims-mower-owner — phone overlay mock (geofence + plan)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from jims_mower.owner import export_owner_overlay


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export a phone-sized yard overlay (HTML stub)")
    p.add_argument("--out", type=Path, default=Path("owner_overlay.html"))
    p.add_argument("--config", type=str, default="geofence_movers")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cameras", type=int, default=4)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    payload = export_owner_overlay(
        args.out,
        config=args.config,
        seed=args.seed,
        cameras=args.cameras,
    )
    print(
        f"Owner overlay {payload.get('scenario')} "
        f"waypoints={len(payload.get('waypoints') or [])} → {args.out}"
    )


if __name__ == "__main__":
    main()

"""CLI: jims-mower-telemetry — coverage / tip rate / drains / living near-misses JSON."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from jims_mower.constants import NEAR_MISS_LIVING_M
from jims_mower.telemetry import telemetry_from_demo_summary, telemetry_from_episode, write_telemetry


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Headless telemetry summary (not a benchmark)")
    p.add_argument("source", type=Path, help="episode directory or demo summary.json")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--near-miss-m", type=float, default=NEAR_MISS_LIVING_M)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    src = args.source
    if src.is_dir():
        payload = telemetry_from_episode(src, near_miss_m=args.near_miss_m)
    else:
        payload = telemetry_from_demo_summary(src)
    dest = args.out or (src / "telemetry.json" if src.is_dir() else src.with_name("telemetry.json"))
    write_telemetry(dest, payload)
    print(
        f"Telemetry steps={payload['steps']} coverage={payload['coverage_pct']:.2f}% "
        f"tips={payload['tip_count']} drains={payload['drain_entries']} "
        f"living_near_misses={payload['living_near_misses']} → {dest}"
    )


if __name__ == "__main__":
    main()

"""CLI: jims-mower-incident — scrub a recorded episode (cameras + hazard + advice)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from jims_mower.incident import write_incident_viewer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Incident replay viewer for a recorded episode")
    p.add_argument("episode", type=Path, help="directory written by jims-mower-record")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stride", type=int, default=1)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    index = write_incident_viewer(args.episode, args.out, stride=args.stride)
    print(f"Incident viewer {index['n_frames']} frames → {args.out}/index.html")


if __name__ == "__main__":
    main()

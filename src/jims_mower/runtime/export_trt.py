"""TensorRT export *placeholder* — no weights, no FPS, no mAP.

On an Orin you would export your own ONNX TerrainObserver / Detector
head and build an engine with ``trtexec``. This repo does not ship a
trained head. The script records the intended command and exits 0 on
``--dry-run`` so CI / docs can mention the path without inventing scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


PLACEHOLDER_NOTE = (
    "Software train→ONNX exists (jims-mower-train-terrain --onnx). "
    "No field-ready head is shipped. Point --onnx at a sim_only or "
    "your own export. Do not treat the printed trtexec line as a "
    "benchmark. fps_claim / map_claim / iou_claim stay null."
)


def planned_trtexec(onnx: Path, engine: Path, *, fp16: bool = True) -> list[str]:
    cmd = [
        "trtexec",
        f"--onnx={onnx}",
        f"--saveEngine={engine}",
        "--memPoolSize=workspace:256",
    ]
    if fp16:
        cmd.append("--fp16")
    return cmd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ONNX → TensorRT command (dry-run default; no field-ready head)"
    )
    p.add_argument("--onnx", type=Path, default=None, help="sim_only or your ONNX; not a field head")
    p.add_argument("--engine", type=Path, default=Path("terrain.engine"))
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print the intended command")
    p.add_argument("--out", type=Path, default=None, help="optional JSON sidecar")
    return p


def run_export(
    *,
    onnx: Optional[Path],
    engine: Path,
    fp16: bool = True,
    dry_run: bool = True,
    out: Optional[Path] = None,
) -> dict:
    cmd = planned_trtexec(onnx or Path("YOUR_HEAD.onnx"), engine, fp16=fp16)
    payload = {
        "dry_run": bool(dry_run or onnx is None or not Path(onnx).is_file()),
        "onnx": str(onnx) if onnx else None,
        "engine": str(engine),
        "command": cmd,
        "note": PLACEHOLDER_NOTE,
        "not_a_benchmark": True,
        "fps_claim": None,
        "map_claim": None,
        "iou_claim": None,
        "domain": "sim_only",
        "field_ready": False,
        "skipped_reason": None if (onnx is not None and Path(onnx).is_file() and not dry_run) else "no_onnx_or_dry_run",
    }
    if out is not None:
        dest = Path(out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    fp16 = not bool(args.no_fp16)
    result = run_export(
        onnx=args.onnx,
        engine=args.engine,
        fp16=fp16,
        dry_run=bool(args.dry_run or args.onnx is None),
        out=args.out,
    )
    print(" ".join(result["command"]))
    print(PLACEHOLDER_NOTE)
    if result["dry_run"]:
        sys.exit(0)


if __name__ == "__main__":
    main()

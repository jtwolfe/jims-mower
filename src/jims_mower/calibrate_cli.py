"""CLI: jims-mower-calibrate — stereo extrinsics checklist + YAML check.

Software bench only. A human on the rig still measures and commits
``extrinsics_stereo_measured.yaml``. No FPS / mAP.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from jims_mower.perception.calibration import (
    CHECKLIST,
    example_extrinsics_path,
    load_extrinsics,
    measured_template_path,
    run_gym_acceptance,
    validate_stereo_yaml,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Validate stereo extrinsics YAML (6–12 cm baseline) and print the "
            "hardware bench checklist. Does not invent field measurements."
        )
    )
    p.add_argument(
        "--yaml",
        type=Path,
        default=None,
        help="Extrinsics YAML (default: documented EXAMPLE extrinsics_stereo.yaml)",
    )
    p.add_argument(
        "--fixture",
        action="store_true",
        help="Run the gym lip / tape acceptance (synthetic stereo, not field mAP)",
    )
    p.add_argument(
        "--template",
        action="store_true",
        help="Validate the MEASURED template instead of the EXAMPLE file",
    )
    p.add_argument("--json", action="store_true", help="Print JSON instead of text")
    p.add_argument("--out", type=Path, default=None, help="Optional JSON report path")
    return p


def _text_report(payload: dict) -> str:
    lines = [
        "Jim's Mower stereo calibration bench",
        "See docs/CALIBRATION.md — software ready; human still tapes on hardware.",
        "",
        f"YAML: {payload['yaml']['path']}",
        f"Label: {payload['yaml']['label']}  (kind={payload['yaml']['kind']})",
    ]
    b = payload["yaml"].get("baseline_cm")
    pair = payload["yaml"].get("pair")
    if pair:
        lines.append(f"Pair: {pair[0]} / {pair[1]}")
    if b is not None:
        lines.append(f"Baseline: {b:.2f} cm  (band 6–12 cm)")
    lines.append("")
    lines.append("YAML checks:")
    for chk in payload["yaml"]["checks"]:
        mark = "x" if chk["ok"] else " "
        lines.append(f"  [{mark}] {chk['name']}: {chk['detail']}")
    lines.append("")
    lines.append("Hardware checklist (human on the rig):")
    for step in CHECKLIST:
        lines.append(f"  [ ] {step}")
    if payload.get("lip") is not None:
        lip = payload["lip"]
        lines.append("")
        lines.append("Gym fixture (ideal stereo, not a matcher / not COLMAP):")
        for t in payload.get("tape_identity") or []:
            mark = "x" if t["ok"] else " "
            lines.append(
                f"  [{mark}] tape {t['tape_m']:.2f} m → recon {t['recon_m']:.3f} m "
                f"(d={t['disparity_px']:.2f} px)"
            )
        mark = "x" if lip["ok"] else " "
        err = lip.get("tape_err_m")
        err_s = "n/a" if err is None else f"{err:.3f} m"
        lines.append(
            f"  [{mark}] lip @ {lip['tape_m']:.2f} m  cells_ok={lip['cells_ok']} "
            f"range_ok={lip['range_ok']} tape_err={err_s} "
            f"(tol {lip['tolerance_m']:.2f} m)"
        )
        lines.append("  fps_claim: null   map_claim: null")
    lines.append("")
    status = "PASS" if payload.get("ok") else "FAIL"
    lines.append(f"{status}  (not a field measurement)")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.yaml is not None:
        path = args.yaml
    elif args.template:
        path = measured_template_path()
    else:
        path = example_extrinsics_path()
    bundle = load_extrinsics(path)
    if args.fixture:
        payload = run_gym_acceptance(bundle)
    else:
        report = validate_stereo_yaml(bundle)
        payload = {
            "yaml": report.as_dict(),
            "tape_identity": None,
            "vertical_board": None,
            "lip": None,
            "ok": report.ok,
            "not_field_measurement": True,
            "fps_claim": None,
            "map_claim": None,
            "checklist": list(CHECKLIST),
        }
    if payload.get("checklist") is None:
        payload["checklist"] = list(CHECKLIST)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(_text_report(payload))
        if args.out is not None:
            print(f"wrote {args.out}")
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

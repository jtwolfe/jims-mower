"""First-boot / pre-fab bring-up walk (not a field scorecard).

``jims-mower-bringup`` runs existing software checks and prints
PASS / FAIL / SKIP. SKIP means no hardware or no measured bench —
it does **not** invent Wh, m/s, RPM, mAP, or RF range.

See ``docs/PACK_THERMAL.md``, ``docs/CALIBRATION.md``,
``docs/FAB_CHECKLIST.md``, ``docs/ESTOP.md``, ``docs/FIELD_TEST.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from jims_mower.config import EnvConfig, load_config
from jims_mower.hardware_estop import HardwareEstop
from jims_mower.onbox import OnboxLoop
from jims_mower.pack import meta_from_config
from jims_mower.perception.calibration import (
    example_extrinsics_path,
    load_extrinsics,
    validate_stereo_yaml,
)
from jims_mower.profile import ProfileError, load_yard_profile, parse_survey_origin
from jims_mower.runtime.gstreamer import gstreamer_available
from jims_mower.selftest import run_selftest

BRINGUP_SCHEMA = "jims_mower.bringup.v1"

DOCS = (
    ("PACK_THERMAL", "docs/PACK_THERMAL.md"),
    ("CALIBRATION", "docs/CALIBRATION.md"),
    ("FAB_CHECKLIST", "docs/FAB_CHECKLIST.md"),
    ("ESTOP", "docs/ESTOP.md"),
    ("FIELD_TEST", "docs/FIELD_TEST.md"),
    ("ONBOX", "docs/ONBOX.md"),
    ("SCALE", "docs/SCALE.md"),
)


def _example_yard_path() -> Path:
    packaged = Path(__file__).resolve().parent / "data" / "yards" / "example_profile.json"
    if packaged.is_file():
        return packaged
    return Path(__file__).resolve().parents[2] / "configs" / "yards" / "example_profile.json"


def _row(name: str, status: str, detail: str, **extra: Any) -> dict[str, Any]:
    blob = {"name": name, "status": str(status).upper(), "detail": detail}
    blob.update(extra)
    return blob


def check_selftest(config: Optional[Union[str, Path, dict, EnvConfig]] = None) -> dict[str, Any]:
    report = run_selftest(config)
    ok = bool(report.get("ok"))
    return _row(
        "selftest",
        "PASS" if ok else "FAIL",
        "gym streams (unloaded spin / IMU / entropy / HW ESTOP rails)"
        if ok
        else "self-test failed — see checks[]",
        checks=report.get("checks"),
        not_hardware_ate=True,
    )


def check_hw_estop_sim() -> dict[str, Any]:
    """Sim latch only. SKIP a physical paddle — we do not have one in CI."""
    hw = HardwareEstop()
    live = hw.filter_action(np.array([0.8, 0.8, 1.0], dtype=np.float32))
    hw.hit("bringup paddle")
    dead = hw.filter_action(np.array([0.8, 0.8, 1.0], dtype=np.float32))
    hw_clear_ignored = HardwareEstop()
    hw_clear_ignored.hit("bringup")
    # software "clear" is not this object — rails stay dead until reset()
    still = hw.filter_action(np.array([0.5, 0.5, 1.0], dtype=np.float32))
    hw.reset()
    restored = hw.filter_action(np.array([0.5, 0.5, 1.0], dtype=np.float32))
    ok = (
        float(np.max(np.abs(live))) > 0.0
        and float(np.max(np.abs(dead))) == 0.0
        and float(np.max(np.abs(still))) == 0.0
        and restored[0] > 0.4
    )
    return _row(
        "hw_estop_sim",
        "PASS" if ok else "FAIL",
        "sim paddle latch zeros rails; only hw_reset restores"
        if ok
        else "sim HardwareEstop latch failed",
        not_field_paddle=True,
    )


def check_watchdog_freeze(config: Optional[Union[str, Path, dict, EnvConfig]] = None) -> dict[str, Any]:
    loop = OnboxLoop.from_config(config or "configs/orin/bench.yaml")
    info = loop.step(np.array([0.7, 0.7, 0.0], dtype=np.float32))
    if info.get("watchdog_stalled"):
        return _row(
            "watchdog_freeze",
            "FAIL",
            "watchdog stalled on a fresh fake-CSI grab",
            info={k: info[k] for k in ("watchdog_reason", "cmd") if k in info},
        )
    loop.freeze_vision_stamp()
    stalled = False
    last = info
    for _ in range(8):
        last = loop.step(np.array([0.7, 0.7, 0.0], dtype=np.float32))
        if last.get("watchdog_stalled") and float(np.max(np.abs(last["cmd"][:2]))) == 0.0:
            stalled = True
            break
    return _row(
        "watchdog_freeze",
        "PASS" if stalled else "FAIL",
        "frozen vision stamps zeroed wheels (fake adapters, not a camera unplug)"
        if stalled
        else "watchdog did not zero wheels after a stamp freeze",
        watchdog_reason=last.get("watchdog_reason"),
        not_field_unplug=True,
    )


def check_calibrate_yaml(yaml_path: Optional[Path] = None) -> dict[str, Any]:
    path = yaml_path or example_extrinsics_path()
    try:
        bundle = load_extrinsics(path)
        report = validate_stereo_yaml(bundle)
    except Exception as exc:
        return _row("calibrate_yaml", "FAIL", str(exc), path=str(path))
    payload = report.as_dict()
    return _row(
        "calibrate_yaml",
        "PASS" if report.ok else "FAIL",
        f"{payload.get('label')}  measured={payload.get('measured')}  "
        f"(see docs/CALIBRATION.md — do not invent tape cm)",
        yaml=payload,
        path=str(path),
        not_field_measurement=True,
    )


def check_pack_measured(cfg: EnvConfig) -> dict[str, Any]:
    meta = meta_from_config(cfg)
    if meta.template:
        return _row(
            "pack_measured",
            "SKIP",
            "pack template on disk — not a bench claim. See docs/PACK_THERMAL.md",
            measured=False,
            template=True,
            acre_runtime_h=None,
        )
    if not meta.measured:
        return _row(
            "pack_measured",
            "SKIP",
            f"runtime.battery.measured is false (gym stub {meta.capacity_wh} Wh). "
            "No invented acre runtime. See docs/PACK_THERMAL.md",
            measured=False,
            capacity_wh=meta.capacity_wh,
            acre_runtime_h=None,
        )
    return _row(
        "pack_measured",
        "PASS",
        f"measured pack claim present ({meta.capacity_wh} Wh @ {meta.measured_at})",
        measured=True,
        capacity_wh=meta.capacity_wh,
        acre_runtime_h=None,
    )


def check_survey_origin(yard: Optional[Path] = None) -> dict[str, Any]:
    path = yard or _example_yard_path()
    if not path.is_file():
        return _row(
            "survey_origin",
            "SKIP",
            f"no yard profile at {path} — not inventing a peg",
            path=str(path),
        )
    try:
        profile = load_yard_profile(path)
        origin = profile.origin if hasattr(profile, "origin") else parse_survey_origin(None)
    except (ProfileError, OSError, ValueError) as exc:
        return _row("survey_origin", "FAIL", str(exc), path=str(path))
    blob = origin.as_dict() if hasattr(origin, "as_dict") else dict(origin)
    present = isinstance(blob, dict) and blob.get("frame")
    surveyed = bool(blob.get("surveyed"))
    if not present:
        return _row(
            "survey_origin",
            "FAIL",
            "yard profile missing origin.frame",
            path=str(path),
            origin=blob,
        )
    if not surveyed:
        return _row(
            "survey_origin",
            "PASS",
            "origin key present (surveyed: false — not a WGS84 field peg). "
            "See docs/SURVEY_ORIGIN.md",
            path=str(path),
            origin=blob,
            surveyed=False,
        )
    return _row(
        "survey_origin",
        "PASS",
        "surveyed origin present on the profile (still tape-stop on the lawn)",
        path=str(path),
        origin=blob,
        surveyed=True,
    )


def check_csi_hardware() -> dict[str, Any]:
    if gstreamer_available():
        return _row(
            "csi_hardware",
            "SKIP",
            "GStreamer bindings present, but live NVMM / physical CSI is not wired. "
            "Do not invent FPS.",
            gstreamer=True,
            fps_claim=None,
        )
    return _row(
        "csi_hardware",
        "SKIP",
        "no GStreamer / JetPack CSI in this environment",
        gstreamer=False,
        fps_claim=None,
    )


def run_bringup(
    config: Optional[Union[str, Path, dict, EnvConfig]] = None,
    *,
    yard: Optional[Path] = None,
    extrinsics: Optional[Path] = None,
) -> dict[str, Any]:
    cfg = load_config(config or "configs/orin/bench.yaml")
    checks = [
        check_selftest(cfg),
        check_hw_estop_sim(),
        check_watchdog_freeze(cfg),
        check_calibrate_yaml(extrinsics),
        check_pack_measured(cfg),
        check_survey_origin(yard),
        check_csi_hardware(),
    ]
    statuses = {c["status"] for c in checks}
    if "FAIL" in statuses:
        overall = "FAIL"
    elif any(c["status"] == "PASS" for c in checks):
        overall = "PASS"
    else:
        overall = "SKIP"
    return {
        "schema": BRINGUP_SCHEMA,
        "ok": overall != "FAIL",
        "status": overall,
        "checks": checks,
        "docs": [{"id": k, "path": p} for k, p in DOCS],
        "not_a_benchmark": True,
        "not_field_scorecard": True,
        "acre_runtime_h": None,
        "fps_claim": None,
        "map_claim": None,
    }


def format_bringup(report: dict[str, Any]) -> str:
    lines = [
        "Jim's Mower bring-up (pre-fab / first boot)",
        "PASS / FAIL / SKIP — SKIP means no hardware or unmeasured. "
        "No invented Wh / m/s / RPM / mAP / RF.",
        "",
    ]
    for chk in report.get("checks") or []:
        lines.append(f"  {chk['status']:<5} {chk['name']}: {chk['detail']}")
    lines.append("")
    lines.append("Docs:")
    for doc in report.get("docs") or []:
        lines.append(f"  {doc['id']}: {doc['path']}")
    lines.append("")
    lines.append(f"{report.get('status')}  (not a field scorecard)")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pre-fab / first-boot checks. Prints PASS/FAIL/SKIP. Never invents measurements."
    )
    p.add_argument("--config", default="configs/orin/bench.yaml")
    p.add_argument("--yard", type=Path, default=None, help="YardProfile JSON for origin check")
    p.add_argument("--extrinsics", type=Path, default=None, help="stereo YAML (default: EXAMPLE)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_bringup(args.config, yard=args.yard, extrinsics=args.extrinsics)
    text = json.dumps(report, indent=2) if args.json else format_bringup(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"bring-up {report['status']} → {args.out}")
    else:
        print(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

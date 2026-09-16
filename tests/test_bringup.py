"""Bring-up CLI: PASS/FAIL/SKIP, never invents measurements."""

from __future__ import annotations

import json
from pathlib import Path

from jims_mower.bringup import format_bringup, run_bringup
from jims_mower.config import load_config


def test_bringup_reports_pass_fail_skip(tmp_path: Path) -> None:
    report = run_bringup("configs/orin/bench.yaml")
    assert report["schema"] == "jims_mower.bringup.v1"
    assert report["not_a_benchmark"] is True
    assert report["acre_runtime_h"] is None
    assert report["fps_claim"] is None
    assert report["map_claim"] is None
    names = {c["name"]: c for c in report["checks"]}
    assert names["selftest"]["status"] in {"PASS", "FAIL"}
    assert names["hw_estop_sim"]["status"] == "PASS"
    assert names["watchdog_freeze"]["status"] == "PASS"
    assert names["calibrate_yaml"]["status"] == "PASS"
    assert names["pack_measured"]["status"] == "SKIP"
    assert names["pack_measured"]["acre_runtime_h"] is None
    assert names["survey_origin"]["status"] == "PASS"
    assert names["survey_origin"]["surveyed"] is False
    assert names["csi_hardware"]["status"] == "SKIP"
    assert names["csi_hardware"]["fps_claim"] is None
    text = format_bringup(report)
    assert "PASS" in text and "SKIP" in text
    assert "PACK_THERMAL" in text
    assert "CALIBRATION" in text
    assert "FAB_CHECKLIST" in text
    assert "ESTOP" in text
    assert "FIELD_TEST" in text
    dest = tmp_path / "bringup.json"
    dest.write_text(json.dumps(report), encoding="utf-8")
    assert dest.is_file()


def test_bringup_does_not_invent_pack_wh() -> None:
    cfg = load_config("configs/orin/bench.yaml")
    report = run_bringup(cfg)
    pack = next(c for c in report["checks"] if c["name"] == "pack_measured")
    assert pack["status"] == "SKIP"
    assert pack["measured"] is False
    assert "invent" in pack["detail"].lower() or "stub" in pack["detail"].lower()

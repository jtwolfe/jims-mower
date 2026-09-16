"""Stereo calibration bench: EXAMPLE YAML, gym lip / tape. Not field mAP."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from jims_mower.calibrate_cli import main as calibrate_main
from jims_mower.config import load_config
from jims_mower.perception.calibration import (
    LABEL_EXAMPLE,
    LABEL_TEMPLATE,
    LipFixture,
    STEREO_BASELINE_MAX_CM,
    STEREO_BASELINE_MIN_CM,
    dump_yaml_header_kind,
    example_extrinsics_path,
    load_extrinsics,
    measured_template_path,
    report_baseline_cm,
    run_gym_acceptance,
    run_lip_fixture,
    tape_vs_ideal_disparity,
    validate_stereo_yaml,
)
from jims_mower.perception.stereo import (
    StereoPairError,
    baseline_cm,
    find_stereo_pair,
    require_stereo_pair,
)
from jims_mower.types import CameraSpec

_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLE = _ROOT / "configs" / "orin" / "extrinsics_stereo.yaml"
_TEMPLATE = _ROOT / "configs" / "orin" / "extrinsics_stereo_measured.template.yaml"
_LOOKAROUND = _ROOT / "configs" / "orin" / "extrinsics_6cam.yaml"


def test_example_yaml_baseline_in_six_to_twelve_cm() -> None:
    bundle = load_extrinsics(_EXAMPLE)
    assert bundle.pair is not None
    b = report_baseline_cm(bundle.cameras)
    assert STEREO_BASELINE_MIN_CM <= b <= STEREO_BASELINE_MAX_CM
    assert b == pytest.approx(8.0, abs=1e-6)
    assert baseline_cm(bundle.pair) == pytest.approx(8.0, abs=1e-6)


def test_example_file_is_not_marked_measured() -> None:
    cfg = load_config(_EXAMPLE)
    assert cfg.calibration.measured is False
    assert cfg.calibration.template is False
    report = validate_stereo_yaml(_EXAMPLE)
    assert report.ok
    assert report.label == LABEL_EXAMPLE
    assert report.measured is False
    assert dump_yaml_header_kind(_EXAMPLE) == LABEL_EXAMPLE


def test_measured_template_flags() -> None:
    cfg = load_config(_TEMPLATE)
    assert cfg.calibration.measured is True
    assert cfg.calibration.template is True
    report = validate_stereo_yaml(_TEMPLATE)
    assert report.ok
    assert report.label == LABEL_TEMPLATE
    assert dump_yaml_header_kind(_TEMPLATE) == LABEL_TEMPLATE


def test_packaged_example_matches_repo_copy() -> None:
    packaged = example_extrinsics_path()
    assert packaged.is_file()
    repo = load_extrinsics(_EXAMPLE)
    pkg = load_extrinsics(packaged)
    assert repo.meta.measured is False
    assert pkg.meta.measured is False
    assert repo.baseline_cm == pytest.approx(pkg.baseline_cm or 0.0)


def test_lookaround_yaml_rejected_as_pair() -> None:
    bundle = load_extrinsics(_LOOKAROUND)
    assert bundle.pair is None
    with pytest.raises(StereoPairError):
        require_stereo_pair(bundle.cameras)
    report = validate_stereo_yaml(bundle)
    assert report.ok is False
    names = {c.name for c in report.checks}
    assert "stereo_pair" in names


def test_example_filename_cannot_claim_measured(tmp_path: Path) -> None:
    dest = tmp_path / "extrinsics_stereo.yaml"
    dest.write_text(_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    raw = yaml.safe_load(dest.read_text(encoding="utf-8"))
    raw["calibration"]["measured"] = True
    dest.write_text(yaml.safe_dump(raw), encoding="utf-8")
    report = validate_stereo_yaml(dest)
    assert report.ok is False
    assert any(c.name == "example_not_measured" and not c.ok for c in report.checks)


def test_measured_file_requires_tape_baseline(tmp_path: Path) -> None:
    dest = tmp_path / "extrinsics_stereo_measured.yaml"
    raw = yaml.safe_load(_TEMPLATE.read_text(encoding="utf-8"))
    raw["calibration"]["template"] = False
    raw["calibration"]["measured"] = True
    raw["calibration"]["tape_baseline_cm"] = None
    dest.write_text(yaml.safe_dump(raw), encoding="utf-8")
    report = validate_stereo_yaml(dest)
    assert report.ok is False
    assert any(c.name == "tape_baseline_recorded" and not c.ok for c in report.checks)

    raw["calibration"]["tape_baseline_cm"] = 8.2
    dest.write_text(yaml.safe_dump(raw), encoding="utf-8")
    report2 = validate_stereo_yaml(dest)
    assert report2.ok
    assert report2.measured is True
    assert report2.template is False


def test_tape_identity_in_near_field_band() -> None:
    pair = find_stereo_pair(load_config(_EXAMPLE).resolved_cameras())
    assert pair is not None
    for tape in (0.80, 1.50, 2.00, 2.50, 3.50, 4.00):
        chk = tape_vs_ideal_disparity(tape, pair, width=80, fov_deg=70.0)
        assert chk.ok
        assert chk.abs_err_m < 1e-9


def test_lip_fixture_stamps_correct_cells() -> None:
    pair = find_stereo_pair(load_config(_EXAMPLE).resolved_cameras())
    assert pair is not None
    result = run_lip_fixture(pair, LipFixture(tape_m=2.0))
    assert result.cells_ok
    assert result.range_ok
    assert result.ok
    assert result.fps_claim is None
    assert result.map_claim is None
    assert result.not_colmap is True
    assert result.tape_err_m is not None
    assert result.tape_err_m <= result.tolerance_m
    assert result.cells_stamped > 0
    assert result.hit_cells


def test_gym_acceptance_bundle() -> None:
    payload = run_gym_acceptance(_EXAMPLE)
    assert payload["ok"]
    assert payload["not_field_measurement"] is True
    assert payload["fps_claim"] is None
    assert payload["lip"]["ok"]
    assert all(t["ok"] for t in payload["tape_identity"])
    assert payload["yaml"]["label"] == LABEL_EXAMPLE


def test_calibrate_cli_example(tmp_path: Path) -> None:
    out = tmp_path / "cal.json"
    rc = calibrate_main(["--yaml", str(_EXAMPLE), "--fixture", "--out", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["yaml"]["measured"] is False
    assert payload["fps_claim"] is None


def test_calibrate_cli_template() -> None:
    rc = calibrate_main(["--template", "--json"])
    assert rc == 0


def test_measured_template_path_exists() -> None:
    path = measured_template_path()
    assert path.is_file()
    assert "template" in path.name


def test_require_pair_rejects_wide_lookaround() -> None:
    cams = [
        CameraSpec("front_left", 0.20, 0.20, 0.38, 0.0, -22.0),
        CameraSpec("front_right", 0.20, -0.20, 0.38, 0.0, -22.0),
    ]
    with pytest.raises(StereoPairError):
        require_stereo_pair(cams)

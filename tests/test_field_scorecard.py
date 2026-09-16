"""Field scorecard schema. Tips / drains / leftover — not mAP."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from jims_mower.field_scorecard import (
    FIELD_SCORECARD_SCHEMA,
    FieldScorecardError,
    empty_scorecard,
    gym_dryrun_scorecard,
    load_scorecard,
    scorecard_template_path,
    validate_scorecard,
    write_scorecard,
)

_ROOT = Path(__file__).resolve().parents[1]
_TEMPLATE = _ROOT / "configs" / "field" / "scorecard.template.yaml"


def test_empty_scorecard_has_null_claims() -> None:
    blank = empty_scorecard()
    assert blank["schema"] == FIELD_SCORECARD_SCHEMA
    assert blank["field_run"] is False
    assert blank["acre_runtime_h"] is None
    assert blank["map_claim"] is None
    assert blank["iou_claim"] is None
    assert blank["fps_claim"] is None
    assert set(blank["score"]) == {
        "tips",
        "drain_entries",
        "leftover_uncut_cells",
        "leftover_uncut_m2",
        "estop_pulls",
    }
    assert all(v is None for v in blank["score"].values())
    assert blank["domain"] == ""
    assert blank["field_ready"] is False


def test_template_loads_unrun() -> None:
    card = load_scorecard(_TEMPLATE)
    assert card.field_run is False
    assert card.pack_measured is False
    assert card.acre_runtime_h is None
    assert card.score["tips"] is None
    assert scorecard_template_path().is_file()


def test_refuses_map_claim() -> None:
    raw = empty_scorecard()
    raw["map_claim"] = 0.42
    with pytest.raises(FieldScorecardError, match="mAP"):
        validate_scorecard(raw)


def test_refuses_acre_runtime_without_field_and_pack() -> None:
    raw = empty_scorecard()
    raw["acre_runtime_h"] = 5.5
    with pytest.raises(FieldScorecardError, match="acre_runtime"):
        validate_scorecard(raw)
    raw["field_run"] = True
    raw["pack_measured"] = True
    card = validate_scorecard(raw)
    assert card.acre_runtime_h == pytest.approx(5.5)


def test_refuses_unknown_score_key() -> None:
    raw = empty_scorecard()
    raw["score"]["mAP"] = 0.9
    with pytest.raises(FieldScorecardError):
        validate_scorecard(raw)


def test_packaged_template_matches_repo() -> None:
    repo = yaml.safe_load(_TEMPLATE.read_text(encoding="utf-8"))
    packaged = yaml.safe_load(scorecard_template_path().read_text(encoding="utf-8"))
    assert repo["schema"] == packaged["schema"]
    assert repo["field_run"] is False
    assert packaged["field_run"] is False


def test_write_scorecard_roundtrip(tmp_path: Path) -> None:
    raw = gym_dryrun_scorecard()
    raw["score"] = {
        "tips": 1,
        "drain_entries": 0,
        "leftover_uncut_cells": 12,
        "leftover_uncut_m2": 0.48,
        "estop_pulls": 1,
    }
    dest = tmp_path / "gym.yaml"
    write_scorecard(dest, raw)
    card = load_scorecard(dest)
    assert card.domain == "gym_dryrun"
    assert card.field_ready is False
    assert card.score["tips"] == 1
    assert card.score["leftover_uncut_cells"] == 12

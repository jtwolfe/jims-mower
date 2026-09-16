"""Residential-acre field scorecard — tips / drains / leftover / ESTOP.

Not mAP, IoU, or FPS. The template is empty; a field run is a human
fill. See ``docs/FIELD_TEST.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

FIELD_SCORECARD_SCHEMA = "jims_mower.field_scorecard.v1"
FORBIDDEN_CLAIM_KEYS = frozenset(
    {
        "map",
        "mAP",
        "iou",
        "IoU",
        "fps",
        "FPS",
        "map_claim",
        "iou_claim",
        "fps_claim",
    }
)
SCORE_KEYS = (
    "tips",
    "drain_entries",
    "leftover_uncut_cells",
    "leftover_uncut_m2",
    "estop_pulls",
)
PREFLIGHT_KEYS = ("self_test", "hw_estop_paddle", "soc", "rain_flag")
MISSION_KEYS = (
    "teach_boundary",
    "surveyed_origin",
    "explore",
    "map_ready",
    "mow",
    "return_home",
)
CHECK_KEYS = (
    "living_interlock",
    "tip_ramp_recovery",
    "rain_skip",
    "soc_skip",
    "day2_resume",
)


class FieldScorecardError(ValueError):
    """Scorecard is missing, claims mAP/IoU/FPS, or invents acre runtime."""


@dataclass
class FieldScorecard:
    schema: str
    field_run: bool
    pack_measured: bool
    notes: str
    preflight: dict[str, Any]
    mission: dict[str, Any]
    checks: dict[str, Any]
    score: dict[str, Any]
    acre_runtime_h: Optional[float] = None
    path: Optional[Path] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "field_run": self.field_run,
            "pack_measured": self.pack_measured,
            "preflight": dict(self.preflight),
            "mission": dict(self.mission),
            "checks": dict(self.checks),
            "score": dict(self.score),
            "acre_runtime_h": self.acre_runtime_h,
            "map_claim": None,
            "iou_claim": None,
            "fps_claim": None,
            "notes": self.notes,
        }
        payload.update(self.extra)
        return payload


def scorecard_template_path() -> Path:
    packaged = Path(__file__).resolve().parent / "data" / "field" / "scorecard.template.yaml"
    if packaged.is_file():
        return packaged
    return Path(__file__).resolve().parents[2] / "configs" / "field" / "scorecard.template.yaml"


def empty_scorecard() -> dict[str, Any]:
    """Empty artifact a human fills after a residential-acre run."""
    return {
        "schema": FIELD_SCORECARD_SCHEMA,
        "field_run": False,
        "pack_measured": False,
        "preflight": {k: None for k in PREFLIGHT_KEYS},
        "mission": {k: None for k in MISSION_KEYS},
        "checks": {k: None for k in CHECK_KEYS},
        "score": {k: None for k in SCORE_KEYS},
        "acre_runtime_h": None,
        "map_claim": None,
        "iou_claim": None,
        "fps_claim": None,
        "notes": "",
    }


def _claim_forbidden(node: Any, *, where: str) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in FORBIDDEN_CLAIM_KEYS and value not in (None, False, ""):
                raise FieldScorecardError(
                    f"{where}.{key} must stay null — score tips / drain "
                    "entries / leftover uncut / ESTOP pulls, not mAP/IoU/FPS"
                )
            _claim_forbidden(value, where=f"{where}.{key}")
    elif isinstance(node, list):
        for i, item in enumerate(node):
            _claim_forbidden(item, where=f"{where}[{i}]")


def validate_scorecard(raw: dict[str, Any]) -> FieldScorecard:
    if not isinstance(raw, dict):
        raise FieldScorecardError("scorecard must be a mapping")
    _claim_forbidden(raw, where="scorecard")
    schema = str(raw.get("schema") or "")
    if schema != FIELD_SCORECARD_SCHEMA:
        raise FieldScorecardError(
            f"scorecard schema must be {FIELD_SCORECARD_SCHEMA}; got {schema!r}"
        )
    field_run = bool(raw.get("field_run", False))
    pack_measured = bool(raw.get("pack_measured", False))
    acre = raw.get("acre_runtime_h")
    if acre is not None and not (field_run and pack_measured):
        raise FieldScorecardError(
            "acre_runtime_h must stay null until a field run *and* a "
            "measured pack (docs/PACK_THERMAL.md)"
        )
    score = raw.get("score") or {}
    if not isinstance(score, dict):
        raise FieldScorecardError("score must be a mapping")
    extra_score = [k for k in score if k not in SCORE_KEYS]
    if extra_score:
        raise FieldScorecardError(
            f"score keys {extra_score} are not allowed — use {list(SCORE_KEYS)}"
        )
    preflight = raw.get("preflight") or {}
    mission = raw.get("mission") or {}
    checks = raw.get("checks") or {}
    if not isinstance(preflight, dict) or not isinstance(mission, dict) or not isinstance(checks, dict):
        raise FieldScorecardError("preflight / mission / checks must be mappings")
    known = {
        "schema",
        "field_run",
        "pack_measured",
        "preflight",
        "mission",
        "checks",
        "score",
        "acre_runtime_h",
        "map_claim",
        "iou_claim",
        "fps_claim",
        "notes",
        "yard",
        "date",
        "operator",
    }
    extra = {k: raw[k] for k in raw if k not in known}
    return FieldScorecard(
        schema=schema,
        field_run=field_run,
        pack_measured=pack_measured,
        notes=str(raw.get("notes") or ""),
        preflight=dict(preflight),
        mission=dict(mission),
        checks=dict(checks),
        score=dict(score),
        acre_runtime_h=None if acre is None else float(acre),
        extra=extra,
    )


def load_scorecard(path: Optional[Path] = None) -> FieldScorecard:
    target = Path(path) if path is not None else scorecard_template_path()
    if not target.is_file():
        raise FieldScorecardError(f"scorecard not found: {target}")
    data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    card = validate_scorecard(data)
    card.path = target
    return card

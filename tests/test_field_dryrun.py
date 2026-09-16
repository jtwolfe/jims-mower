"""Gym field-scorecard dry-run: writes a valid card, not a field test."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from jims_mower.constants import FIELD_DRYRUN_DOMAIN, FIELD_SCORECARD_SCHEMA, SIGNAL_TO_ID
from jims_mower.env import MowerEnv, _nearest_detection_signal
from jims_mower.field_dryrun import (
    check_hand_signal,
    check_living_interlock,
    check_schedule_skips,
    format_dryrun,
    run_field_dryrun,
)
from jims_mower.field_scorecard import FieldScorecardError, load_scorecard, validate_scorecard
from jims_mower.perception.detect import gym_red_bias_signal, paint_kind_blob
from jims_mower.types import Detection


def test_gym_red_bias_person_is_stop() -> None:
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    bbox = (6, 4, 10, 12)
    painted = paint_kind_blob(frame, "person", bbox)
    assert gym_red_bias_signal(painted, bbox) == "stop"
    assert gym_red_bias_signal(frame, bbox) is None


def test_nearest_detection_signal_without_world_xy() -> None:
    front = Detection("person", "front", (8, 6, 10, 12), 0.7, None, "stop", "person")
    rear = Detection("person", "rear", (4, 4, 8, 8), 0.7, None, "go", "person")
    assert _nearest_detection_signal([rear, front], (0.0, 0.0)) == "stop"
    located = Detection("person", "front", (1, 1, 4, 4), 0.7, (2.0, 1.0), "back", "person")
    assert _nearest_detection_signal([front, located], (2.0, 1.0)) == "back"


def test_appearance_red_bias_fills_icd_and_policy() -> None:
    row = check_hand_signal()
    assert row["status"] == "PASS"
    assert row["hand_signal"] == SIGNAL_TO_ID["stop"]
    assert row["hand_signal_source"] == "appearance_crop"
    assert row["field_confusion_matrix"] is None
    assert row["gym_only"] is True


def test_appearance_living_interlock_front_vs_behind() -> None:
    row = check_living_interlock()
    assert row["status"] == "PASS"
    assert row["front_trimmer_enabled"] is False
    assert row["rear_trimmer_enabled"] is True


def test_schedule_flags_skip() -> None:
    rain, soc = check_schedule_skips()
    assert rain["status"] == "PASS"
    assert soc["status"] == "PASS"


def test_field_dryrun_writes_valid_scorecard(tmp_path: Path) -> None:
    dest = tmp_path / "scorecard.yaml"
    report = run_field_dryrun(scenario="mission_tiny", steps=180, seed=3, out=dest, work_dir=tmp_path)
    assert report["ok"] is True
    assert report["status"] == "PASS"
    assert dest.is_file()
    card = load_scorecard(dest)
    assert card.schema == FIELD_SCORECARD_SCHEMA
    assert card.domain == FIELD_DRYRUN_DOMAIN
    assert card.field_ready is False
    assert card.field_run is False
    assert card.acre_runtime_h is None
    assert card.as_dict()["map_claim"] is None
    assert card.as_dict()["iou_claim"] is None
    assert card.as_dict()["fps_claim"] is None
    assert set(card.score) == {
        "tips",
        "drain_entries",
        "leftover_uncut_cells",
        "leftover_uncut_m2",
        "estop_pulls",
    }
    assert all(card.score[k] is not None for k in card.score)
    assert "mAP" not in card.score
    assert card.checks["living_interlock"] == "PASS"
    assert card.checks["hand_signal"] == "PASS"
    assert card.checks["rain_skip"] == "PASS"
    assert card.checks["soc_skip"] == "PASS"
    assert card.checks["day2_resume"] == "PASS"
    assert card.mission["teach_boundary"] == "PASS"
    assert card.mission["surveyed_origin"] == "PASS"
    raw = yaml.safe_load(dest.read_text(encoding="utf-8"))
    assert raw["domain"] == FIELD_DRYRUN_DOMAIN
    assert raw["field_ready"] is False
    text = format_dryrun(report)
    assert "hand_signal" in text
    assert "living_interlock" in text
    assert "not a field test" in text.lower()


def test_gym_dryrun_cannot_claim_field_ready() -> None:
    raw = {
        "schema": FIELD_SCORECARD_SCHEMA,
        "field_run": False,
        "pack_measured": False,
        "domain": FIELD_DRYRUN_DOMAIN,
        "field_ready": True,
        "preflight": {},
        "mission": {},
        "checks": {},
        "score": {"tips": 0, "drain_entries": 0, "leftover_uncut_cells": 0, "leftover_uncut_m2": 0.0, "estop_pulls": 0},
        "acre_runtime_h": None,
        "notes": "",
    }
    try:
        validate_scorecard(raw)
        raise AssertionError("expected FieldScorecardError")
    except FieldScorecardError as exc:
        assert "gym_dryrun" in str(exc)


def test_field_dryrun_cli_exits_zero(tmp_path: Path) -> None:
    dest = tmp_path / "cli_scorecard.yaml"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "jims_mower.field_dryrun",
            "--scenario",
            "mission_tiny",
            "--steps",
            "80",
            "--out",
            str(dest),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert dest.is_file()
    card = load_scorecard(dest)
    assert card.domain == FIELD_DRYRUN_DOMAIN
    assert card.field_ready is False
    assert "hand_signal" in proc.stdout
    assert "living_interlock" in proc.stdout


def test_appearance_hand_signal_in_env_obs() -> None:
    env = MowerEnv(
        config={
            "dt": 0.1,
            "max_steps": 8,
            "sensors": {"width": 32, "height": 24, "camera_count": 4},
            "world": {
                "width_m": 8.0,
                "height_m": 8.0,
                "resolution_m": 0.20,
                "n_people": 0,
                "n_dogs": 0,
                "n_cats": 0,
                "n_birds": 0,
                "n_trees": 0,
                "n_furniture": 0,
                "n_toys": 0,
                "terrain": {"enabled": False},
            },
            "perception": {"detector_backend": "appearance", "interlock_source": "detections"},
            "curriculum": {"hand_signals": True},
        }
    )
    env.reset(seed=1)
    images = {name: np.zeros((24, 32, 3), dtype=np.uint8) for name in env.camera_index}
    images["front"] = paint_kind_blob(images["front"], "person", (8, 6, 12, 14))
    env._camera_images = lambda: images  # type: ignore[method-assign]
    obs, _r, _t, _c, info = env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
    env.close()
    assert int(obs["hand_signal"]) == SIGNAL_TO_ID["stop"]
    assert info["hand_signal_name"] == "stop"
    assert info["hand_signal_source"] == "appearance_crop"
    dets = info.get("detections") or []
    assert any(d.get("hand_signal") == "stop" for d in dets if isinstance(d, dict))

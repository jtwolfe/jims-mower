"""Overnight farm dry-run and a tiny live matrix."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jims_mower.farm import FarmThresholdError, main, plan_matrix, run_farm


def test_plan_matrix() -> None:
    matrix = plan_matrix([0, 1], ["suburban", "paddock"])
    assert len(matrix) == 4
    assert matrix[0] == {"seed": 0, "scenario": "suburban"}


def test_farm_dry_run(tmp_path: Path) -> None:
    summary = run_farm(
        tmp_path,
        seeds=[0, 1],
        scenarios=["suburban", "orchard"],
        steps=20,
        dry_run=True,
    )
    assert summary["dry_run"] is True
    assert summary["n_planned"] == 4
    assert summary["breached"] is False
    assert "n_tips" not in summary or summary.get("n_episodes") is None
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "matrix.json").is_file()
    matrix = json.loads((tmp_path / "matrix.json").read_text(encoding="utf-8"))
    assert len(matrix) == 4


def test_farm_cli_dry_run(tmp_path: Path) -> None:
    main(
        [
            "--dry-run",
            "--seeds",
            "0",
            "--scenarios",
            "playground",
            "--out",
            str(tmp_path),
        ]
    )
    payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert payload["n_planned"] == 1


def test_farm_tiny_live(tmp_path: Path) -> None:
    summary = run_farm(
        tmp_path,
        seeds=[0],
        scenarios=["playground"],
        steps=2,
        cameras=4,
        policy="scripted",
        dry_run=False,
        max_tips=0,
        max_drain_entries=0,
        terrain_observer="oracle",
    )
    assert summary["n_episodes"] == 1
    assert summary["n_tips"] == 0
    assert summary["n_drain_entries"] == 0
    assert summary["breached"] is False
    assert (tmp_path / "playground_seed0.json").is_file()


def test_farm_threshold_breach(tmp_path: Path) -> None:
    with pytest.raises(FarmThresholdError):
        run_farm(
            tmp_path,
            seeds=[0],
            scenarios=["playground"],
            steps=1,
            cameras=4,
            policy="scripted",
            max_tips=-1,
            max_drain_entries=0,
            terrain_observer="oracle",
        )

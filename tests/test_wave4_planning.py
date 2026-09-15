"""Wet-slope cost, energy-aware strips, multi-yard sequence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jims_mower.constants import HAZARD_STEEP
from jims_mower.planning.costmap import STEEP_COST, build_costmap
from jims_mower.planning.coverage import _energy_reorder, plan_coverage
from jims_mower.sequence import load_sequence, run_sequence


def test_wet_slope_adds_cost() -> None:
    n = 16
    hazard = np.zeros((n, n), dtype=np.float32)
    slope = np.zeros((n, n), dtype=np.float32)
    hazard[8, 8] = HAZARD_STEEP
    slope[8, 8] = 0.15
    dry = build_costmap(
        hazard,
        slope,
        resolution_m=0.2,
        width_m=3.2,
        height_m=3.2,
        max_climb_slope_rad=0.32,
        drain_clearance_m=0.0,
        margin_m=0.0,
        wet=False,
    )
    wet = build_costmap(
        hazard,
        slope,
        resolution_m=0.2,
        width_m=3.2,
        height_m=3.2,
        max_climb_slope_rad=0.32,
        drain_clearance_m=0.0,
        margin_m=0.0,
        wet=True,
        wet_slope_extra=3.0,
    )
    assert dry.cost[8, 8] == pytest.approx(STEEP_COST)
    assert wet.cost[8, 8] == pytest.approx(STEEP_COST + 3.0)
    assert not wet.blocked[8, 8]


def test_energy_reorder_prefers_nearby_short() -> None:
    segs = [
        [(0, 10), (0, 11), (0, 12), (0, 13)],
        [(1, 1)],
    ]
    ordered = _energy_reorder(segs, (1, 1), battery_soc=0.10, limp_soc=0.15)
    assert ordered[0] == segs[1]


def test_energy_aware_plan_coverage() -> None:
    n = 12
    hazard = np.zeros((n, n), dtype=np.float32)
    slope = np.zeros((n, n), dtype=np.float32)
    cm = build_costmap(
        hazard,
        slope,
        resolution_m=0.25,
        width_m=3.0,
        height_m=3.0,
        max_climb_slope_rad=0.4,
        drain_clearance_m=0.0,
        margin_m=0.0,
    )
    plan = plan_coverage(
        cm,
        (0.4, 0.4),
        strip_spacing_m=0.5,
        waypoint_stride_m=0.5,
        battery_soc=0.08,
        limp_soc=0.15,
        energy_aware=True,
    )
    assert plan.waypoints


def test_sequence_dry_run(tmp_path: Path) -> None:
    summary = run_sequence(
        tmp_path,
        yards=["paddock", "suburban"],
        steps=4,
        dry_run=True,
    )
    assert summary["dry_run"] is True
    assert summary["yards"] == ["paddock", "suburban"]
    assert summary["n_planned"] == 2
    assert (tmp_path / "summary.json").is_file()


def test_load_bundled_sequence() -> None:
    data = load_sequence(Path("configs/sequences/paddock_then_suburban.yaml"))
    assert data["yards"] == ["paddock", "suburban"]
    assert "jims_mower.sequence" in data["schema"]

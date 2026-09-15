"""Costmap from slope + hazard labels."""

from __future__ import annotations

import math

import numpy as np
import pytest

from jims_mower.constants import HAZARD_DRAIN, HAZARD_DRAIN_EDGE, HAZARD_NONE, HAZARD_STEEP
from jims_mower.planning.costmap import BLOCKED_COST, STEEP_COST, build_costmap


def _empty(n: int = 20) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((n, n), dtype=np.float32), np.zeros((n, n), dtype=np.float32)


def test_free_cells_have_unit_cost() -> None:
    hazard, slope = _empty(16)
    cm = build_costmap(
        hazard,
        slope,
        resolution_m=0.2,
        width_m=3.2,
        height_m=3.2,
        max_climb_slope_rad=0.32,
        drain_clearance_m=0.0,
        margin_m=0.0,
    )
    assert not cm.blocked[8, 8]
    assert cm.cost[8, 8] == pytest.approx(1.0)


def test_channel_is_forbidden() -> None:
    hazard, slope = _empty(20)
    hazard[10, :] = HAZARD_DRAIN
    cm = build_costmap(
        hazard,
        slope,
        resolution_m=0.2,
        width_m=4.0,
        height_m=4.0,
        max_climb_slope_rad=0.32,
        drain_clearance_m=0.0,
        margin_m=0.0,
    )
    assert bool(cm.blocked[10, 5])
    assert not math.isfinite(cm.cost[10, 5])
    assert cm.cost[10, 5] == BLOCKED_COST or not math.isfinite(float(cm.cost[10, 5]))


def test_lip_is_blocked_for_reroute() -> None:
    hazard, slope = _empty(16)
    hazard[8, 8] = HAZARD_DRAIN_EDGE
    cm = build_costmap(
        hazard,
        slope,
        resolution_m=0.2,
        width_m=3.2,
        height_m=3.2,
        max_climb_slope_rad=0.32,
        drain_clearance_m=0.0,
        margin_m=0.0,
    )
    assert cm.blocked[8, 8]


def test_drain_clearance_inflates_channel() -> None:
    hazard, slope = _empty(20)
    hazard[10, 10] = HAZARD_DRAIN
    cm = build_costmap(
        hazard,
        slope,
        resolution_m=0.2,
        width_m=4.0,
        height_m=4.0,
        max_climb_slope_rad=0.32,
        drain_clearance_m=0.4,
        margin_m=0.0,
    )
    # 0.4 m / 0.2 m = 2 cells of disk inflation.
    assert cm.blocked[10, 12]
    assert cm.blocked[12, 10]
    assert not cm.blocked[10, 15]


def test_steep_below_climb_cap_is_slow_corridor() -> None:
    hazard, slope = _empty(16)
    hazard[6, 6] = HAZARD_STEEP
    slope[6, 6] = 0.25
    cm = build_costmap(
        hazard,
        slope,
        resolution_m=0.2,
        width_m=3.2,
        height_m=3.2,
        max_climb_slope_rad=0.32,
        drain_clearance_m=0.0,
        margin_m=0.0,
    )
    assert not cm.blocked[6, 6]
    assert cm.cost[6, 6] == pytest.approx(STEEP_COST)


def test_steep_above_climb_cap_is_blocked() -> None:
    hazard, slope = _empty(16)
    hazard[6, 6] = HAZARD_STEEP
    slope[6, 6] = 0.50
    cm = build_costmap(
        hazard,
        slope,
        resolution_m=0.2,
        width_m=3.2,
        height_m=3.2,
        max_climb_slope_rad=0.32,
        drain_clearance_m=0.0,
        margin_m=0.0,
    )
    assert cm.blocked[6, 6]


def test_occupancy_blocks_cells() -> None:
    hazard, slope = _empty(12)
    occ = np.zeros((12, 12), dtype=np.float32)
    occ[4, 4] = 1.0
    cm = build_costmap(
        hazard,
        slope,
        resolution_m=0.2,
        width_m=2.4,
        height_m=2.4,
        max_climb_slope_rad=0.32,
        drain_clearance_m=0.0,
        occupancy=occ,
        occupancy_inflate_m=0.0,
        margin_m=0.0,
    )
    assert cm.blocked[4, 4]
    assert not cm.blocked[4, 8]


def test_yard_margin_blocks_edges() -> None:
    hazard, slope = _empty(20)
    cm = build_costmap(
        hazard,
        slope,
        resolution_m=0.2,
        width_m=4.0,
        height_m=4.0,
        max_climb_slope_rad=0.32,
        drain_clearance_m=0.0,
        margin_m=0.4,
    )
    assert cm.blocked[0, 10]
    assert cm.blocked[10, 0]
    assert not cm.blocked[10, 10]


def test_world_cell_roundtrip_and_oob_blocked() -> None:
    hazard, slope = _empty(10)
    cm = build_costmap(
        hazard,
        slope,
        resolution_m=0.5,
        width_m=5.0,
        height_m=5.0,
        max_climb_slope_rad=0.32,
        drain_clearance_m=0.0,
        margin_m=0.0,
    )
    x, y = cm.cell_to_world(2, 3)
    assert cm.world_to_cell(x, y) == (2, 3)
    assert cm.is_blocked_world(-1.0, 1.0)
    assert cm.nearest_free(1.25, 1.25) == (2, 2)


def test_rejects_shape_mismatch() -> None:
    hazard, _ = _empty(8)
    slope = np.zeros((7, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        build_costmap(
            hazard,
            slope,
            resolution_m=0.2,
            width_m=1.6,
            height_m=1.6,
            max_climb_slope_rad=0.3,
        )


def test_free_label_constant() -> None:
    assert HAZARD_NONE == 0

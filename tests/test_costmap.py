"""Costmap from slope + hazard labels."""

from __future__ import annotations

import math

import numpy as np
import pytest

from jims_mower.constants import HAZARD_DRAIN, HAZARD_DRAIN_EDGE, HAZARD_NONE, HAZARD_STEEP
from jims_mower.planning.costmap import BLOCKED_COST, CONTOUR_COST, STEEP_COST, build_costmap


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


def test_steep_between_climb_and_tip_is_contour_not_blocked() -> None:
    hazard, slope = _empty(16)
    hazard[6, 6] = HAZARD_STEEP
    slope[6, 6] = 0.35
    cm = build_costmap(
        hazard,
        slope,
        resolution_m=0.2,
        width_m=3.2,
        height_m=3.2,
        max_climb_slope_rad=0.32,
        tip_lethal_slope_rad=0.38,
        contour_cost=CONTOUR_COST,
        drain_clearance_m=0.0,
        margin_m=0.0,
    )
    assert not cm.blocked[6, 6]
    assert cm.cost[6, 6] == pytest.approx(CONTOUR_COST)


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


def test_low_confidence_inflates_free_cost() -> None:
    hazard, slope = _empty(16)
    conf = np.ones((16, 16), dtype=np.float32)
    conf[8, 8] = 0.10
    cm = build_costmap(
        hazard,
        slope,
        resolution_m=0.2,
        width_m=3.2,
        height_m=3.2,
        max_climb_slope_rad=0.32,
        drain_clearance_m=0.0,
        margin_m=0.0,
        confidence=conf,
        uncertainty_inflate=2.0,
        uncertain_hazard_boost=0.0,
    )
    assert not cm.blocked[8, 8]
    assert cm.cost[8, 8] == pytest.approx(1.0 * (1.0 + 2.0 * 0.90))
    assert cm.cost[8, 9] == pytest.approx(1.0)
    assert cm.confidence is not None
    assert cm.confidence[8, 8] == pytest.approx(0.10)


def test_uncertain_steep_hint_gets_boost() -> None:
    hazard, slope = _empty(16)
    hazard[6, 6] = HAZARD_STEEP
    slope[6, 6] = 0.20
    conf = np.ones((16, 16), dtype=np.float32)
    conf[6, 6] = 0.10
    cm = build_costmap(
        hazard,
        slope,
        resolution_m=0.2,
        width_m=3.2,
        height_m=3.2,
        max_climb_slope_rad=0.32,
        drain_clearance_m=0.0,
        margin_m=0.0,
        confidence=conf,
        uncertainty_inflate=0.0,
        uncertain_hazard_boost=4.0,
        uncertain_confidence_floor=0.25,
    )
    assert not cm.blocked[6, 6]
    assert cm.cost[6, 6] > STEEP_COST


def test_elevation_prior_floors_flat_observer_slope() -> None:
    hazard, slope = _empty(20)
    # Observer left slope at 0; prior is a 0.14 rad grade along +x.
    yy = (np.arange(20) + 0.5) * 0.2
    xx = (np.arange(20) + 0.5) * 0.2
    gx, gy = np.meshgrid(xx, yy)
    prior = (0.14 * (gx - 2.0)).astype(np.float32)
    cm = build_costmap(
        hazard,
        slope,
        resolution_m=0.2,
        width_m=4.0,
        height_m=4.0,
        max_climb_slope_rad=0.40,
        drain_clearance_m=0.0,
        margin_m=0.0,
        elevation_prior=prior,
        prior_blend=1.0,
    )
    assert float(np.mean(cm.cost)) >= 1.0
    # Grade is below the climb cap, so the yard stays free (not the old
    # "almost-flat observer vs sloped physics" fight).
    assert not bool(cm.blocked[10, 10])


def test_confidence_shape_mismatch() -> None:
    hazard, slope = _empty(8)
    with pytest.raises(ValueError):
        build_costmap(
            hazard,
            slope,
            resolution_m=0.2,
            width_m=1.6,
            height_m=1.6,
            max_climb_slope_rad=0.3,
            confidence=np.ones((7, 8), dtype=np.float32),
        )

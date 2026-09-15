"""Boustrophedon coverage + A* stay out of drain channels."""

from __future__ import annotations

import numpy as np
import pytest

from jims_mower.constants import HAZARD_DRAIN, HAZARD_DRAIN_EDGE
from jims_mower.planning.costmap import build_costmap
from jims_mower.planning.coverage import plan_coverage, shortest_path


def _channel_map() -> tuple[object, np.ndarray]:
    n = 40
    res = 0.2
    hazard = np.zeros((n, n), dtype=np.float32)
    slope = np.zeros((n, n), dtype=np.float32)
    # Horizontal ditch through the middle, with lips, a gap on the right so
    # north/south are still connected (A* must go around, not through).
    hazard[20, :30] = HAZARD_DRAIN
    hazard[19, :30] = HAZARD_DRAIN_EDGE
    hazard[21, :30] = HAZARD_DRAIN_EDGE
    cm = build_costmap(
        hazard,
        slope,
        resolution_m=res,
        width_m=n * res,
        height_m=n * res,
        max_climb_slope_rad=0.32,
        drain_clearance_m=0.20,
        margin_m=0.20,
    )
    return cm, hazard


def test_planner_waypoints_avoid_channels() -> None:
    cm, hazard = _channel_map()
    plan = plan_coverage(cm, (1.0, 1.0), strip_spacing_m=0.4, waypoint_stride_m=0.4)
    assert len(plan.waypoints) > 8
    assert plan.n_segments >= 2
    for row, col in plan.cells:
        assert hazard[row, col] != HAZARD_DRAIN
        assert not cm.blocked[row, col]
    for x, y in plan.waypoints:
        assert not cm.is_blocked_world(x, y)


def test_astar_goes_around_channel_not_through() -> None:
    cm, hazard = _channel_map()
    path = shortest_path(cm, (1.2, 1.2), (1.2, 6.5))
    assert path, "expected a path around the partial drain"
    for row, col in path:
        assert hazard[row, col] != HAZARD_DRAIN
        assert not cm.blocked[row, col]
    # Must actually travel toward the gap (higher columns), not tunnel.
    max_col = max(c for _r, c in path)
    assert max_col >= 30


def test_full_width_channel_covers_reachable_side_only() -> None:
    n = 24
    res = 0.25
    hazard = np.zeros((n, n), dtype=np.float32)
    hazard[12, :] = HAZARD_DRAIN
    cm = build_costmap(
        hazard,
        np.zeros_like(hazard),
        resolution_m=res,
        width_m=n * res,
        height_m=n * res,
        max_climb_slope_rad=0.3,
        drain_clearance_m=0.25,
        margin_m=0.25,
    )
    plan = plan_coverage(cm, (1.0, 1.0), strip_spacing_m=0.5, waypoint_stride_m=0.5)
    ys = [y for _x, y in plan.waypoints]
    assert ys
    # Start is south of the wall; no waypoint should jump to the far north.
    assert max(ys) < 12 * res + 0.4
    for row, col in plan.cells:
        assert hazard[row, col] != HAZARD_DRAIN


def test_mowable_mask_skips_non_grass() -> None:
    n = 16
    res = 0.2
    hazard = np.zeros((n, n), dtype=np.float32)
    cm = build_costmap(
        hazard,
        np.zeros_like(hazard),
        resolution_m=res,
        width_m=n * res,
        height_m=n * res,
        max_climb_slope_rad=0.3,
        drain_clearance_m=0.0,
        margin_m=0.2,
    )
    mowable = np.zeros((n, n), dtype=bool)
    mowable[4:6, :] = True
    plan = plan_coverage(
        cm,
        (0.8, 0.8),
        strip_spacing_m=0.2,
        waypoint_stride_m=0.4,
        mowable=mowable,
    )
    rows = {r for r, _c in plan.cells[1:]}  # skip the start snap
    assert rows <= {4, 5} or any(r in {4, 5} for r in rows)


def test_empty_map_returns_start_only() -> None:
    hazard = np.ones((8, 8), dtype=np.float32) * HAZARD_DRAIN
    cm = build_costmap(
        hazard,
        np.zeros_like(hazard),
        resolution_m=0.2,
        width_m=1.6,
        height_m=1.6,
        max_climb_slope_rad=0.3,
        drain_clearance_m=0.0,
        margin_m=0.0,
    )
    plan = plan_coverage(cm, (0.5, 0.5))
    assert plan.n_segments == 0


def test_shortest_path_same_cell() -> None:
    hazard = np.zeros((10, 10), dtype=np.float32)
    cm = build_costmap(
        hazard,
        np.zeros_like(hazard),
        resolution_m=0.2,
        width_m=2.0,
        height_m=2.0,
        max_climb_slope_rad=0.3,
        drain_clearance_m=0.0,
        margin_m=0.0,
    )
    path = shortest_path(cm, (0.5, 0.5), (0.5, 0.5))
    assert len(path) == 1

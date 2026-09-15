"""Frontier exploration through known-safe cells only."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from jims_mower.planning.costmap import BLOCKED_COST, Costmap
from jims_mower.planning.coverage import shortest_path
from jims_mower.planning.observed import ObservedMap, downsample_frontiers, frontiers, nearest_frontier


@dataclass
class ExplorePlan:
    """Path to a frontier (or empty when the map is complete / stuck)."""

    waypoints: list[tuple[float, float]]
    cells: list[tuple[int, int]] = field(default_factory=list)
    target: Optional[tuple[int, int]] = None
    frontier_cells: list[tuple[int, int]] = field(default_factory=list)
    no_frontier: bool = False
    unreachable_frontiers: int = 0

    @property
    def done(self) -> bool:
        return self.no_frontier or not self.waypoints


def explore_costmap(omap: ObservedMap, keep_in: Optional[np.ndarray] = None) -> Costmap:
    """Transit only through known-safe cells. Unknown is blocked."""
    blocked = omap.unknown_blocked(keep_in)
    cost = np.full(blocked.shape, 1.0, dtype=np.float32)
    cost[blocked] = BLOCKED_COST
    return Costmap(
        cost=cost,
        blocked=blocked,
        width_m=omap.width_m,
        height_m=omap.height_m,
        resolution_m=omap.resolution_m,
        confidence=omap.confidence.copy(),
    )


def plan_explore(
    omap: ObservedMap,
    start_xy: tuple[float, float],
    *,
    keep_in: Optional[np.ndarray] = None,
    waypoint_stride_m: float = 0.40,
) -> ExplorePlan:
    """A* from the robot to the nearest reachable frontier.

    The goal is a known-free cell that touches unknown. Local camera/ToF
    stamps then grow known space; the planner never assumes unknown is safe.
    """
    cm = explore_costmap(omap, keep_in)
    raw = frontiers(omap.observed, omap.free, keep_in=keep_in)
    thin = downsample_frontiers(raw, min_sep=2, limit=80)
    start = cm.nearest_free(*start_xy)
    if start is None:
        return ExplorePlan(waypoints=[], frontier_cells=thin, no_frontier=not thin)

    stride = max(1, int(round(float(waypoint_stride_m) / max(omap.resolution_m, 1e-6))))
    if not thin:
        return ExplorePlan(waypoints=[], frontier_cells=[], no_frontier=True)

    # Try nearest first, then a few farther frontiers if A* cannot connect.
    ordered = sorted(thin, key=lambda rc: (rc[0] - start[0]) ** 2 + (rc[1] - start[1]) ** 2)
    unreachable = 0
    for target in ordered[:16]:
        if cm.blocked[target]:
            snapped = cm.nearest_free(*cm.cell_to_world(*target))
            if snapped is None:
                unreachable += 1
                continue
            target = snapped
        path = shortest_path(cm, start, target, world=False)
        if not path:
            unreachable += 1
            continue
        cells = path[::stride]
        if cells[-1] != path[-1]:
            cells.append(path[-1])
        waypoints = [cm.cell_to_world(r, c) for r, c in cells]
        return ExplorePlan(
            waypoints=waypoints,
            cells=cells,
            target=target,
            frontier_cells=thin,
            no_frontier=False,
            unreachable_frontiers=unreachable,
        )
    return ExplorePlan(
        waypoints=[],
        frontier_cells=thin,
        no_frontier=False,
        unreachable_frontiers=unreachable + max(0, len(ordered) - 16),
    )


def pick_frontier_world(
    omap: ObservedMap,
    start_xy: tuple[float, float],
    *,
    keep_in: Optional[np.ndarray] = None,
) -> Optional[tuple[float, float]]:
    raw = frontiers(omap.observed, omap.free, keep_in=keep_in)
    start = omap.world_to_cell(*start_xy)
    if start is None:
        start = (int(omap.rows / 2), int(omap.cols / 2))
    cell = nearest_frontier(raw, start)
    if cell is None:
        return None
    return omap.cell_to_world(*cell)

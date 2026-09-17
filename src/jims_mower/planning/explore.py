"""Frontier exploration through known-safe cells only."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from jims_mower.planning.costmap import BLOCKED_COST, CONTOUR_COST, Costmap, slope_from_elevation
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
    skipped_frontiers: int = 0

    @property
    def done(self) -> bool:
        return self.no_frontier or not self.waypoints


def explore_costmap(
    omap: ObservedMap,
    keep_in: Optional[np.ndarray] = None,
    *,
    max_climb_slope_rad: Optional[float] = None,
    tip_lethal_slope_rad: Optional[float] = None,
    contour_cost: float = CONTOUR_COST,
) -> Costmap:
    """Transit only through known-safe cells. Unknown is blocked.

    Tip-dangerous grades (above the tip-safe margin) are lethal. Between
    ``max_climb`` and that margin, A* prefers a contour (high cost).
    Climbable hills stay free so explore can climb or walk around.
    """
    blocked = omap.unknown_blocked(keep_in)
    cost = np.full(blocked.shape, 1.0, dtype=np.float32)
    if max_climb_slope_rad is not None or tip_lethal_slope_rad is not None:
        slope = slope_from_elevation(omap.elevation, omap.resolution_m)
        climb = float(max_climb_slope_rad) if max_climb_slope_rad is not None else 0.32
        lethal = float(tip_lethal_slope_rad) if tip_lethal_slope_rad is not None else climb
        if lethal < climb:
            lethal = climb
        contour = (slope >= climb) & (slope < lethal) & ~blocked
        cost[contour] = np.maximum(cost[contour], float(contour_cost))
        lethal_mask = (slope >= lethal) & ~blocked
        blocked = blocked | lethal_mask
    cost[blocked] = BLOCKED_COST
    return Costmap(
        cost=cost,
        blocked=blocked,
        width_m=omap.width_m,
        height_m=omap.height_m,
        resolution_m=omap.resolution_m,
        confidence=omap.confidence.copy(),
    )


def _cell_near_skip(
    cell: tuple[int, int],
    skip: Iterable[tuple[int, int]],
    *,
    radius: int = 3,
) -> bool:
    r2 = max(1, int(radius)) ** 2
    cr, cc = cell
    for sr, sc in skip:
        if (cr - sr) ** 2 + (cc - sc) ** 2 <= r2:
            return True
    return False


def plan_explore(
    omap: ObservedMap,
    start_xy: tuple[float, float],
    *,
    keep_in: Optional[np.ndarray] = None,
    waypoint_stride_m: float = 0.40,
    skip_cells: Optional[Iterable[tuple[int, int]]] = None,
    avoid_xy: Optional[tuple[float, float]] = None,
    cluster_cells: int = 3,
    max_climb_slope_rad: Optional[float] = None,
    tip_lethal_slope_rad: Optional[float] = None,
    contour_cost: float = CONTOUR_COST,
) -> ExplorePlan:
    """A* from the robot to the nearest reachable frontier.

    The goal is a known-free cell that touches unknown. Local camera/ToF
    stamps then grow known space; the planner never assumes unknown is safe.
    Learned blockage cells and ``skip_cells`` (failed no-progress frontiers)
    are not retried — they count as unreachable so the owner line is honest.
    """
    cm = explore_costmap(
        omap,
        keep_in,
        max_climb_slope_rad=max_climb_slope_rad,
        tip_lethal_slope_rad=tip_lethal_slope_rad,
        contour_cost=contour_cost,
    )
    ignore = np.asarray(omap.blockage, dtype=bool)
    raw = frontiers(omap.observed, omap.free, keep_in=keep_in, ignore_unknown=ignore)
    thin = downsample_frontiers(raw, min_sep=2, limit=80)
    start = cm.nearest_free(*start_xy)
    if start is None:
        return ExplorePlan(waypoints=[], frontier_cells=thin, no_frontier=not thin)

    stride = max(1, int(round(float(waypoint_stride_m) / max(omap.resolution_m, 1e-6))))
    if not thin:
        return ExplorePlan(waypoints=[], frontier_cells=[], no_frontier=True)

    skip = list(skip_cells or [])
    skipped = 0
    candidates: list[tuple[int, int]] = []
    for cell in thin:
        if bool(ignore[cell]) or (skip and _cell_near_skip(cell, skip, radius=cluster_cells)):
            skipped += 1
            continue
        candidates.append(cell)

    # Try nearest first, then farther frontiers if A* cannot connect.
    # After a nearby failure, prefer a different heading so we do not
    # oscillate on the same free/unknown lip.
    def _score(rc: tuple[int, int]) -> tuple[float, float]:
        dist2 = float((rc[0] - start[0]) ** 2 + (rc[1] - start[1]) ** 2)
        penalty = 0.0
        if avoid_xy is not None:
            ax, ay = omap.cell_to_world(*rc)
            penalty = -((ax - avoid_xy[0]) ** 2 + (ay - avoid_xy[1]) ** 2)
        return (dist2 + 0.35 * penalty, dist2)

    ordered = sorted(candidates, key=_score)
    unreachable = 0
    try_n = min(len(ordered), 28)
    for target in ordered[:try_n]:
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
        leftover = max(0, len(ordered) - try_n)
        return ExplorePlan(
            waypoints=waypoints,
            cells=cells,
            target=target,
            frontier_cells=thin,
            no_frontier=False,
            unreachable_frontiers=unreachable + skipped + leftover,
            skipped_frontiers=skipped,
        )
    leftover = max(0, len(ordered) - try_n)
    return ExplorePlan(
        waypoints=[],
        frontier_cells=thin,
        no_frontier=not candidates,
        unreachable_frontiers=unreachable + skipped + leftover,
        skipped_frontiers=skipped,
    )


def pick_frontier_world(
    omap: ObservedMap,
    start_xy: tuple[float, float],
    *,
    keep_in: Optional[np.ndarray] = None,
) -> Optional[tuple[float, float]]:
    raw = frontiers(
        omap.observed,
        omap.free,
        keep_in=keep_in,
        ignore_unknown=np.asarray(omap.blockage, dtype=bool),
    )
    start = omap.world_to_cell(*start_xy)
    if start is None:
        start = (int(omap.rows / 2), int(omap.cols / 2))
    cell = nearest_frontier(raw, start)
    if cell is None:
        return None
    return omap.cell_to_world(*cell)

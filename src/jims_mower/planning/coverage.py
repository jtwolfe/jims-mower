"""Boustrophedon coverage on a costmap, with A* between strip endpoints."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from jims_mower.planning.costmap import BLOCKED_COST, Costmap

_NEIGHBORS = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


@dataclass
class CoveragePlan:
    """World-frame waypoints plus the costmap they were planned on."""

    waypoints: list[tuple[float, float]]
    cells: list[tuple[int, int]] = field(default_factory=list)
    costmap: Optional[Costmap] = None
    n_segments: int = 0

    def remaining(self, index: int) -> list[tuple[float, float]]:
        if index < 0:
            return list(self.waypoints)
        return list(self.waypoints[index:])


def shortest_path(
    costmap: Costmap,
    start: tuple[float, float] | tuple[int, int],
    goal: tuple[float, float] | tuple[int, int],
    *,
    world: bool = True,
) -> list[tuple[int, int]]:
    """A* on the costmap. ``start`` / ``goal`` are world XY when ``world``."""
    if world:
        sc = costmap.world_to_cell(*start) or costmap.nearest_free(*start)
        gc = costmap.world_to_cell(*goal) or costmap.nearest_free(*goal)
    else:
        sc = (int(start[0]), int(start[1]))
        gc = (int(goal[0]), int(goal[1]))
        if not (0 <= sc[0] < costmap.rows and 0 <= sc[1] < costmap.cols):
            sc = None
        if not (0 <= gc[0] < costmap.rows and 0 <= gc[1] < costmap.cols):
            gc = None
    if sc is None or gc is None:
        return []
    if costmap.blocked[sc]:
        snapped = costmap.nearest_free(*costmap.cell_to_world(*sc))
        if snapped is None:
            return []
        sc = snapped
    if costmap.blocked[gc]:
        snapped = costmap.nearest_free(*costmap.cell_to_world(*gc))
        if snapped is None:
            return []
        gc = snapped
    return _astar(costmap, sc, gc)


def _astar(
    costmap: Costmap,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    if start == goal:
        return [start]
    rows, cols = costmap.rows, costmap.cols
    gx, gy = goal

    def h(r: int, c: int) -> float:
        return math.hypot(r - gx, c - gy)

    open_heap: list[tuple[float, int, int, int]] = []
    seq = 0
    heapq.heappush(open_heap, (h(*start), seq, start[0], start[1]))
    came: dict[tuple[int, int], tuple[int, int]] = {}
    gscore = {start: 0.0}
    closed: set[tuple[int, int]] = set()

    while open_heap:
        _, _, r, c = heapq.heappop(open_heap)
        node = (r, c)
        if node in closed:
            continue
        if node == goal:
            return _reconstruct(came, node)
        closed.add(node)
        g0 = gscore[node]
        for dr, dc, step in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if costmap.blocked[nr, nc]:
                continue
            edge = 0.5 * (float(costmap.cost[r, c]) + float(costmap.cost[nr, nc])) * step
            if not math.isfinite(edge) or edge >= BLOCKED_COST:
                continue
            cand = g0 + edge
            nxt = (nr, nc)
            if cand + 1e-9 < gscore.get(nxt, math.inf):
                came[nxt] = node
                gscore[nxt] = cand
                seq += 1
                heapq.heappush(open_heap, (cand + h(nr, nc), seq, nr, nc))
    return []


def _reconstruct(
    came: dict[tuple[int, int], tuple[int, int]],
    node: tuple[int, int],
) -> list[tuple[int, int]]:
    path = [node]
    while node in came:
        node = came[node]
        path.append(node)
    path.reverse()
    return path


def _downsample(cells: list[tuple[int, int]], stride: int) -> list[tuple[int, int]]:
    if not cells:
        return []
    stride = max(1, int(stride))
    out = list(cells[::stride])
    if out[-1] != cells[-1]:
        out.append(cells[-1])
    return out


def _dedup_adjacent(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for cell in cells:
        if not out or out[-1] != cell:
            out.append(cell)
    return out


def _energy_reorder(
    segments: list[list[tuple[int, int]]],
    start_cell: Optional[tuple[int, int]],
    battery_soc: float,
    limp_soc: float,
) -> list[list[tuple[int, int]]]:
    """When SOC is low, prefer short nearby strips over a long boustrophedon."""
    if not segments or start_cell is None:
        return segments
    soc = float(battery_soc)
    if soc > float(limp_soc) + 0.15:
        return segments

    def _len(seg: list[tuple[int, int]]) -> int:
        return len(seg)

    def _dist(seg: list[tuple[int, int]]) -> int:
        r, c = seg[0]
        return (r - start_cell[0]) ** 2 + (c - start_cell[1]) ** 2

    # Closest-then-shortest: dump remaining energy on nearby grass.
    return sorted(segments, key=lambda seg: (_dist(seg), _len(seg)))


def plan_coverage(
    costmap: Costmap,
    start_xy: tuple[float, float],
    *,
    strip_spacing_m: float = 0.28,
    waypoint_stride_m: float = 0.32,
    mowable: Optional[np.ndarray] = None,
    battery_soc: Optional[float] = None,
    limp_soc: float = 0.15,
    energy_aware: bool = False,
) -> CoveragePlan:
    """Lawnmower (boustrophedon) strips on free cells, A* across gaps.

    ``mowable`` (same shape as the costmap) restricts which free cells are
    sweep targets — typically uncut grass. Transit A* may still cross any
    unblocked cell so the robot can go around a drain.
    """
    res = max(costmap.resolution_m, 1e-6)
    strip_step = max(1, int(round(float(strip_spacing_m) / res)))
    stride = max(1, int(round(float(waypoint_stride_m) / res)))

    if mowable is None:
        sweep = ~costmap.blocked
    else:
        if mowable.shape != costmap.blocked.shape:
            raise ValueError("mowable mask shape must match the costmap")
        sweep = np.asarray(mowable, dtype=bool) & ~costmap.blocked

    segments: list[list[tuple[int, int]]] = []
    row_ids = list(range(strip_step // 2, costmap.rows, strip_step))
    for i, row in enumerate(row_ids):
        cols = range(costmap.cols) if i % 2 == 0 else range(costmap.cols - 1, -1, -1)
        run: list[tuple[int, int]] = []
        for col in cols:
            if sweep[row, col]:
                run.append((row, col))
            elif run:
                segments.append(run)
                run = []
        if run:
            segments.append(run)

    start_cell = costmap.nearest_free(*start_xy)
    if segments and start_cell is not None:
        if energy_aware and battery_soc is not None:
            segments = _energy_reorder(segments, start_cell, float(battery_soc), limp_soc)
        else:
            def _seg_key(seg: list[tuple[int, int]]) -> int:
                r, c = seg[0]
                return (r - start_cell[0]) ** 2 + (c - start_cell[1]) ** 2

            best_i = min(range(len(segments)), key=lambda i: _seg_key(segments[i]))
            segments = segments[best_i:] + segments[:best_i]

    path_cells: list[tuple[int, int]] = []
    if start_cell is not None:
        path_cells.append(start_cell)

    cursor = start_cell
    n_seg = 0
    for seg in segments:
        if n_seg == 0 and start_cell is not None and len(seg) > 1:
            # Join the first lane at the nearest cell instead of driving
            # to the far left/right end first.
            j = min(
                range(len(seg)),
                key=lambda k: (seg[k][0] - start_cell[0]) ** 2 + (seg[k][1] - start_cell[1]) ** 2,
            )
            seg = seg[j:]
        sampled = _downsample(seg, stride)
        if not sampled:
            continue
        target = sampled[0]
        if cursor is not None and cursor != target:
            connector = _astar(costmap, cursor, target)
            if not connector:
                continue
            path_cells.extend(_downsample(connector[1:], stride))
        elif cursor is None:
            if costmap.blocked[target]:
                continue
            path_cells.append(target)
        path_cells.extend(sampled[1:])
        cursor = sampled[-1]
        n_seg += 1

    path_cells = _dedup_adjacent(path_cells)
    waypoints = [costmap.cell_to_world(r, c) for r, c in path_cells]
    return CoveragePlan(
        waypoints=waypoints,
        cells=path_cells,
        costmap=costmap,
        n_segments=n_seg,
    )

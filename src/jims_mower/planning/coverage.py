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
    skipped_segments: int = 0
    planned_mowable_cells: int = 0
    reachable_mowable_cells: int = 0
    unreachable_mowable_cells: int = 0
    unmapped_mowable_cells: int = 0
    planned_coverage_fraction: float = 0.0
    unreachable_cells: list[tuple[int, int]] = field(default_factory=list)
    orientation_rad: float = 0.0

    def remaining(self, index: int) -> list[tuple[float, float]]:
        if index < 0:
            return list(self.waypoints)
        return list(self.waypoints[index:])

    def as_metrics(self) -> dict[str, float | int]:
        return {
            "n_segments": int(self.n_segments),
            "skipped_segments": int(self.skipped_segments),
            "planned_mowable_cells": int(self.planned_mowable_cells),
            "reachable_mowable_cells": int(self.reachable_mowable_cells),
            "unreachable_mowable_cells": int(self.unreachable_mowable_cells),
            "unmapped_mowable_cells": int(self.unmapped_mowable_cells),
            "planned_coverage_fraction": float(self.planned_coverage_fraction),
            "n_waypoints": len(self.waypoints),
            "orientation_rad": float(self.orientation_rad),
        }


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


def choose_strip_orientation(
    elevation: Optional[np.ndarray],
    mowable: np.ndarray,
    *,
    resolution_m: float,
    slope_bias: float = 0.55,
) -> float:
    """Strip travel heading from the yard principal axis, biased off the slope.

    Strips follow the long axis of the mowable region. When a mean grade is
    present, they rotate toward the contour (perpendicular to ascent) so the
    robot is not forced into repeated uphill–downhill runs.
    """
    mask = np.asarray(mowable, dtype=bool)
    axis = _principal_axis_rad(mask)
    if elevation is None or not np.any(mask):
        return axis
    elev = np.asarray(elevation, dtype=np.float32)
    if elev.shape != mask.shape:
        return axis
    res = max(float(resolution_m), 1e-6)
    gy, gx = np.gradient(elev, res)
    mx = float(gx[mask].mean())
    my = float(gy[mask].mean())
    mag = math.hypot(mx, my)
    if mag < 0.02:
        return axis
    # Contour heading is perpendicular to the ascent vector.
    contour = math.atan2(mx, -my)
    # Align contour with the principal axis (flip 180° if needed).
    err = _wrap_pi(contour - axis)
    if abs(err) > math.pi / 2:
        contour = _wrap_pi(contour + math.pi)
        err = _wrap_pi(contour - axis)
    mix = float(np.clip(slope_bias * min(1.0, mag / 0.12), 0.0, 1.0))
    return _wrap_pi(axis + mix * err)


def _principal_axis_rad(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if xs.size < 8:
        return 0.0
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    pts -= pts.mean(axis=0)
    cov = pts.T @ pts
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, int(np.argmax(evals))]
    return math.atan2(float(axis[1]), float(axis[0]))


def _wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    rows, cols = mask.shape
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                i, j = stack.pop()
                cells.append((i, j))
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols and mask[ni, nj] and not seen[ni, nj]:
                        seen[ni, nj] = True
                        stack.append((ni, nj))
            comps.append(cells)
    return comps


def reachable_mask(
    costmap: Costmap,
    start: tuple[int, int],
    *,
    blocked: Optional[np.ndarray] = None,
) -> np.ndarray:
    """8-connected flood of unblocked cells from ``start``."""
    block = costmap.blocked if blocked is None else np.asarray(blocked, dtype=bool)
    reach = np.zeros((costmap.rows, costmap.cols), dtype=bool)
    if block[start]:
        return reach
    stack = [start]
    reach[start] = True
    rows, cols = costmap.rows, costmap.cols
    while stack:
        r, c = stack.pop()
        for dr, dc, _step in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if reach[nr, nc] or block[nr, nc]:
                continue
            reach[nr, nc] = True
            stack.append((nr, nc))
    return reach


def drop_unmapped_islands(
    costmap: Costmap,
    sweep: np.ndarray,
    start: tuple[int, int],
    soft_transit: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop fog islands that are only cut off by unknown, not a real obstacle.

    Camera stamps on a taught rectangle leave disconnected observed patches.
    Those are unmapped leftover, not ``unreachable`` grass the job failed to
    reach. A drain / shed / pond that still blocks after treating unknown as
    transit stays unreachable.
    """
    mask = np.asarray(sweep, dtype=bool)
    reach = reachable_mask(costmap, start)
    isolated = mask & ~reach
    if soft_transit is None or not np.any(isolated):
        return mask, np.zeros_like(mask, dtype=bool)
    soft = np.asarray(soft_transit, dtype=bool)
    if soft.shape != mask.shape:
        return mask, np.zeros_like(mask, dtype=bool)
    opened = costmap.blocked & ~soft
    reach_if = reachable_mask(costmap, start, blocked=opened)
    fog = isolated & reach_if
    cleaned = mask.copy()
    cleaned[fog] = False
    return cleaned, fog


def _strips_on_mask(
    sweep: np.ndarray,
    strip_step: int,
    *,
    axis: str,
) -> list[list[tuple[int, int]]]:
    segments: list[list[tuple[int, int]]] = []
    rows, cols = sweep.shape
    if axis == "col":
        col_ids = list(range(strip_step // 2, cols, strip_step))
        for i, col in enumerate(col_ids):
            row_range = range(rows) if i % 2 == 0 else range(rows - 1, -1, -1)
            run: list[tuple[int, int]] = []
            for row in row_range:
                if sweep[row, col]:
                    run.append((row, col))
                elif run:
                    segments.append(run)
                    run = []
            if run:
                segments.append(run)
        return segments
    row_ids = list(range(strip_step // 2, rows, strip_step))
    for i, row in enumerate(row_ids):
        col_range = range(cols) if i % 2 == 0 else range(cols - 1, -1, -1)
        run = []
        for col in col_range:
            if sweep[row, col]:
                run.append((row, col))
            elif run:
                segments.append(run)
                run = []
        if run:
            segments.append(run)
    return segments


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
    orientation_rad: Optional[float] = None,
    elevation: Optional[np.ndarray] = None,
    soft_transit: Optional[np.ndarray] = None,
) -> CoveragePlan:
    """Lawnmower (boustrophedon) strips on free cells, A* across gaps.

    ``mowable`` (same shape as the costmap) restricts which free cells are
    sweep targets — typically uncut grass. Transit A* may still cross any
    unblocked cell so the robot can go around a drain.

    Disconnected components that A* cannot reach because of a real
    obstacle are counted as ``unreachable_mowable_cells``. Fog islands
    that are only cut off by unknown (``soft_transit``) are dropped from
    the planned set instead of inflating unreachable.
    Strip axis follows ``orientation_rad`` (0 = east–west travel) or a
    principal-axis / slope choice when ``elevation`` is given.
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

    heading = (
        float(orientation_rad)
        if orientation_rad is not None
        else choose_strip_orientation(elevation, sweep, resolution_m=res)
    )
    axis = "col" if abs(math.sin(heading)) > abs(math.cos(heading)) else "row"

    start_cell = costmap.nearest_free(*start_xy)
    unmapped_n = 0
    unreachable: list[tuple[int, int]] = []
    if start_cell is not None and int(sweep.sum()):
        cleaned, fog = drop_unmapped_islands(costmap, sweep, start_cell, soft_transit)
        unmapped_n = int(fog.sum())
        sweep = cleaned
        reach = reachable_mask(costmap, start_cell)
        isolated = sweep & ~reach
        if np.any(isolated):
            ys, xs = np.where(isolated)
            unreachable.extend((int(r), int(c)) for r, c in zip(ys.tolist(), xs.tolist()))
            sweep[isolated] = False
    planned_n = int(sweep.sum()) + len(unreachable)

    segments: list[list[tuple[int, int]]] = _strips_on_mask(sweep, strip_step, axis=axis)

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
    skipped = 0
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
                skipped += 1
                unreachable.extend(seg)
                continue
            path_cells.extend(_downsample(connector[1:], stride))
        elif cursor is None:
            if costmap.blocked[target]:
                skipped += 1
                unreachable.extend(seg)
                continue
            path_cells.append(target)
        path_cells.extend(sampled[1:])
        cursor = sampled[-1]
        n_seg += 1

    path_cells = _dedup_adjacent(path_cells)
    waypoints = [costmap.cell_to_world(r, c) for r, c in path_cells]
    unreachable = _dedup_adjacent(sorted(set(unreachable)))
    unreach_n = len(unreachable)
    reachable_n = max(0, planned_n - unreach_n)
    frac = (reachable_n / planned_n) if planned_n else 0.0
    return CoveragePlan(
        waypoints=waypoints,
        cells=path_cells,
        costmap=costmap,
        n_segments=n_seg,
        skipped_segments=skipped,
        planned_mowable_cells=planned_n,
        reachable_mowable_cells=reachable_n,
        unreachable_mowable_cells=unreach_n,
        unmapped_mowable_cells=unmapped_n,
        planned_coverage_fraction=frac,
        unreachable_cells=unreachable,
        orientation_rad=heading,
    )

"""Yard state: spawn obstacles and step living agents."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from jims_mower.constants import (
    ALL_KINDS,
    CUTTER_RISK_KINDS,
    DEFAULT_HEIGHTS,
    DEFAULT_RADII,
    DEFAULT_SPEEDS,
    DENSITY_COUNTS,
    SOFT_KINDS,
    STATIC_KINDS,
    TRAJECTORY_MODES,
)
from jims_mower.types import Obstacle, Pose, Trajectory


@dataclass
class Yard:
    width_m: float
    height_m: float
    obstacles: list[Obstacle] = field(default_factory=list)

    def static(self) -> list[Obstacle]:
        return [o for o in self.obstacles if o.kind in STATIC_KINDS]

    def living(self) -> list[Obstacle]:
        return [o for o in self.obstacles if o.kind not in STATIC_KINDS]


def _random_heading(rng: np.random.Generator) -> float:
    return float(rng.uniform(-math.pi, math.pi))


def _overlaps(
    x: float,
    y: float,
    radius: float,
    others: list[Obstacle],
    keepout: list[tuple[float, float, float]],
) -> bool:
    for ox, oy, r in keepout:
        if math.hypot(x - ox, y - oy) < radius + r:
            return True
    for obst in others:
        extra = 0.25 * max(obst.length_m, 0.0)
        if math.hypot(x - obst.x, y - obst.y) < radius + obst.radius + extra + 0.15:
            return True
    return False


def _decorate_kind(kind: str, obst: Obstacle, rng: np.random.Generator) -> Obstacle:
    if kind in SOFT_KINDS or kind in CUTTER_RISK_KINDS:
        obst.soft = True
        obst.cutter_risk = True
        if obst.length_m <= 0.0:
            obst.length_m = 2.6 if kind == "hose" else 3.2
            obst.heading = _random_heading(rng)
    return obst


def spawn_obstacle(
    rng: np.random.Generator,
    kind: str,
    yard: Yard,
    keepout: list[tuple[float, float, float]],
    *,
    name: str = "",
    tries: int = 80,
    x: Optional[float] = None,
    y: Optional[float] = None,
    heading: Optional[float] = None,
    length_m: float = 0.0,
    trajectory: Optional[Trajectory] = None,
) -> Optional[Obstacle]:
    radius = DEFAULT_RADII[kind]
    margin = radius + 0.6
    if x is not None and y is not None:
        if _overlaps(x, y, radius, yard.obstacles, keepout):
            return None
        obst = Obstacle(
            kind=kind,
            x=x,
            y=y,
            radius=radius,
            z=DEFAULT_HEIGHTS[kind],
            heading=_random_heading(rng) if heading is None else float(heading),
            name=name or kind,
            length_m=length_m,
            trajectory=trajectory,
        )
        return _decorate_kind(kind, obst, rng)
    for _ in range(tries):
        sx = float(rng.uniform(margin, yard.width_m - margin))
        sy = float(rng.uniform(margin, yard.height_m - margin))
        if _overlaps(sx, sy, radius, yard.obstacles, keepout):
            continue
        obst_heading = _random_heading(rng) if heading is None else float(heading)
        speed = DEFAULT_SPEEDS.get(kind, 0.0)
        obst = Obstacle(
            kind=kind,
            x=sx,
            y=sy,
            radius=radius,
            z=DEFAULT_HEIGHTS[kind],
            vx=speed * math.cos(obst_heading),
            vy=speed * math.sin(obst_heading),
            heading=obst_heading,
            name=name or kind,
            length_m=length_m,
            trajectory=trajectory,
        )
        return _decorate_kind(kind, obst, rng)
    return None


def _counts_from_mapping(counts: dict[str, int]) -> dict[str, int]:
    return {
        "person": int(counts.get("person", 0)),
        "dog": int(counts.get("dog", 0)),
        "cat": int(counts.get("cat", 0)),
        "bird": int(counts.get("bird", 0)),
        "tree": int(counts.get("tree", 0)),
        "furniture": int(counts.get("furniture", 0)),
        "toy": int(counts.get("toy", 0)),
        "hose": int(counts.get("hose", 0)),
        "cord": int(counts.get("cord", 0)),
    }


def place_obstacle(
    kind: str,
    x: float,
    y: float,
    *,
    radius: Optional[float] = None,
    heading: float = 0.0,
    name: str = "",
    z: Optional[float] = None,
    vx: float = 0.0,
    vy: float = 0.0,
    trajectory: Optional[Trajectory] = None,
) -> Obstacle:
    """Place one authored obstacle (scenario DSL)."""
    if kind not in ALL_KINDS:
        raise ValueError(f"Unknown obstacle kind {kind!r}")
    obst = Obstacle(
        kind=kind,
        x=float(x),
        y=float(y),
        radius=float(radius if radius is not None else DEFAULT_RADII[kind]),
        z=float(z if z is not None else DEFAULT_HEIGHTS[kind]),
        vx=float(vx),
        vy=float(vy),
        heading=float(heading),
        name=name or kind,
        trajectory=trajectory,
    )
    if kind in CUTTER_RISK_KINDS:
        obst.soft = True
        obst.cutter_risk = True
    return obst


def spawn_yard(
    rng: np.random.Generator,
    width_m: float,
    height_m: float,
    counts: dict[str, int],
    robot_keepout: tuple[float, float, float],
    extra_keepout: Optional[list[tuple[float, float, float]]] = None,
    *,
    layout: str = "random",
    orchard_rows: int = 3,
    orchard_cols: int = 4,
    explicit: Optional[list[Obstacle]] = None,
    mover_mode: str = "wander",
    density: str = "default",
) -> Yard:
    yard = Yard(width_m=width_m, height_m=height_m)
    keepout = [robot_keepout]
    if extra_keepout:
        keepout.extend(extra_keepout)
    for obst in explicit or []:
        yard.obstacles.append(obst)
        keepout.append((obst.x, obst.y, obst.radius))
    resolved = _counts_from_mapping(counts)
    overlay = DENSITY_COUNTS.get(str(density), None)
    if overlay:
        resolved.update(overlay)
    if layout == "orchard":
        _place_orchard_trees(yard, keepout, orchard_rows, orchard_cols)
        resolved["tree"] = 0
    elif layout == "playground":
        _place_playground_toys(rng, yard, keepout, resolved["toy"])
        resolved["toy"] = 0
    order = ("tree", "furniture", "toy", "hose", "cord", "person", "dog", "cat", "bird")
    for kind in order:
        n = int(resolved.get(kind, 0))
        for i in range(n):
            obst = spawn_obstacle(rng, kind, yard, keepout, name=f"{kind}_{i}")
            if obst is not None:
                yard.obstacles.append(obst)
    _assign_default_paths(yard, rng, mover_mode)
    return yard


def _place_orchard_trees(
    yard: Yard,
    keepout: list[tuple[float, float, float]],
    rows: int,
    cols: int,
) -> None:
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    margin = 1.4
    xs = np.linspace(margin, yard.width_m - margin, cols)
    ys = np.linspace(margin, yard.height_m - margin, rows)
    idx = 0
    for y in ys:
        for x in xs:
            fx, fy = float(x), float(y)
            if _overlaps(fx, fy, DEFAULT_RADII["tree"], yard.obstacles, keepout):
                continue
            yard.obstacles.append(
                Obstacle(
                    kind="tree",
                    x=fx,
                    y=fy,
                    radius=DEFAULT_RADII["tree"],
                    z=DEFAULT_HEIGHTS["tree"],
                    name=f"tree_{idx}",
                )
            )
            idx += 1


def _place_playground_toys(
    rng: np.random.Generator,
    yard: Yard,
    keepout: list[tuple[float, float, float]],
    n_toys: int,
) -> None:
    """Cluster toys in one play area instead of scattering them."""
    cx = 0.32 * yard.width_m
    cy = 0.38 * yard.height_m
    spread = min(2.4, 0.28 * min(yard.width_m, yard.height_m))
    placed = 0
    for i in range(max(0, n_toys) * 4):
        if placed >= n_toys:
            break
        x = float(np.clip(rng.normal(cx, spread * 0.35), 0.6, yard.width_m - 0.6))
        y = float(np.clip(rng.normal(cy, spread * 0.35), 0.6, yard.height_m - 0.6))
        obst = spawn_obstacle(rng, "toy", yard, keepout, name=f"toy_{placed}", x=x, y=y)
        if obst is not None:
            yard.obstacles.append(obst)
            placed += 1
    for i in range(placed, n_toys):
        obst = spawn_obstacle(rng, "toy", yard, keepout, name=f"toy_{i}")
        if obst is not None:
            yard.obstacles.append(obst)


def _assign_default_paths(yard: Yard, rng: np.random.Generator, mode: str) -> None:
    """Give unauthored living agents a short patrol when the yard asks for it."""
    key = str(mode or "wander").strip().lower()
    if key not in TRAJECTORY_MODES or key == "wander":
        return
    for obst in yard.living():
        if obst.trajectory is not None and obst.trajectory.waypoints:
            continue
        span = 2.4 if obst.kind == "person" else 3.2
        axis = 0 if float(rng.random()) < 0.5 else 1
        a = (obst.x, obst.y)
        if axis == 0:
            b = (
                float(np.clip(obst.x + span, obst.radius + 0.3, yard.width_m - obst.radius - 0.3)),
                obst.y,
            )
        else:
            b = (
                obst.x,
                float(np.clip(obst.y + span, obst.radius + 0.3, yard.height_m - obst.radius - 0.3)),
            )
        obst.trajectory = Trajectory(mode=key if key != "line" else "patrol", waypoints=[a, b])


def _clip_to_yard(x: float, y: float, obst: Obstacle, yard: Yard) -> tuple[float, float, bool]:
    lo = obst.radius + 0.15
    hi_x = yard.width_m - obst.radius - 0.15
    hi_y = yard.height_m - obst.radius - 0.15
    bounced = x < lo or x > hi_x or y < lo or y > hi_y
    return float(np.clip(x, lo, hi_x)), float(np.clip(y, lo, hi_y)), bounced


def _follow_waypoints(obst: Obstacle, dt: float, yard: Yard) -> None:
    traj = obst.trajectory
    assert traj is not None
    pts = traj.waypoints
    if not pts:
        return
    speed = traj.speed_mps if traj.speed_mps > 0.0 else DEFAULT_SPEEDS.get(obst.kind, 0.0)
    idx = int(np.clip(traj.index, 0, len(pts) - 1))
    tx, ty = pts[idx]
    dx, dy = tx - obst.x, ty - obst.y
    dist = math.hypot(dx, dy)
    if dist < 0.12:
        mode = traj.mode
        if mode == "loop":
            traj.index = (idx + 1) % len(pts)
        elif mode == "line":
            if idx < len(pts) - 1:
                traj.index = idx + 1
            else:
                obst.vx = 0.0
                obst.vy = 0.0
                return
        else:  # patrol
            nxt = idx + traj.direction
            if nxt < 0 or nxt >= len(pts):
                traj.direction = -traj.direction
                nxt = idx + traj.direction
            traj.index = int(np.clip(nxt, 0, len(pts) - 1))
        tx, ty = pts[traj.index]
        dx, dy = tx - obst.x, ty - obst.y
        dist = math.hypot(dx, dy)
    if dist < 1e-6:
        obst.vx = 0.0
        obst.vy = 0.0
        return
    obst.heading = math.atan2(dy, dx)
    obst.vx = speed * math.cos(obst.heading)
    obst.vy = speed * math.sin(obst.heading)
    nx, ny, bounced = _clip_to_yard(obst.x + obst.vx * dt, obst.y + obst.vy * dt, obst, yard)
    if bounced:
        traj.direction = -traj.direction
        obst.heading = math.atan2(ty - ny, tx - nx)
    obst.x = nx
    obst.y = ny


def step_movers(
    yard: Yard,
    dt: float,
    rng: np.random.Generator,
    heading_jitter: float = 0.35,
) -> None:
    """Advance people/animals along authored paths or a random walk; bounce off the fence."""
    for obst in yard.living():
        traj = obst.trajectory
        if traj is not None and traj.mode in TRAJECTORY_MODES and traj.mode != "wander" and traj.waypoints:
            _follow_waypoints(obst, dt, yard)
            continue
        obst.heading = obst.heading + float(rng.uniform(-heading_jitter, heading_jitter))
        speed = DEFAULT_SPEEDS.get(obst.kind, 0.0)
        if traj is not None and traj.speed_mps > 0.0:
            speed = traj.speed_mps
        obst.vx = speed * math.cos(obst.heading)
        obst.vy = speed * math.sin(obst.heading)
        nx, ny, bounced = _clip_to_yard(obst.x + obst.vx * dt, obst.y + obst.vy * dt, obst, yard)
        if bounced:
            if nx != obst.x + obst.vx * dt:
                obst.heading = math.atan2(obst.vy, -obst.vx)
            if ny != obst.y + obst.vy * dt:
                obst.heading = math.atan2(-obst.vy, obst.vx)
        obst.x = nx
        obst.y = ny


def robot_start_pose(width_m: float, height_m: float) -> Pose:
    return Pose(0.5 * width_m, 0.5 * height_m, 0.0)


def point_to_segment_distance(
    px: float,
    py: float,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> float:
    dx = x1 - x0
    dy = y1 - y0
    length2 = dx * dx + dy * dy
    if length2 < 1e-12:
        return math.hypot(px - x0, py - y0)
    t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / length2))
    return math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))

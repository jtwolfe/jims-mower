"""Yard state: spawn obstacles and step living agents."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from jims_mower.constants import (
    DEFAULT_HEIGHTS,
    DEFAULT_RADII,
    DEFAULT_SPEEDS,
    STATIC_KINDS,
)
from jims_mower.types import Obstacle, Pose


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
        if math.hypot(x - obst.x, y - obst.y) < radius + obst.radius + 0.15:
            return True
    return False


def spawn_obstacle(
    rng: np.random.Generator,
    kind: str,
    yard: Yard,
    keepout: list[tuple[float, float, float]],
    *,
    name: str = "",
    tries: int = 80,
) -> Optional[Obstacle]:
    radius = DEFAULT_RADII[kind]
    margin = radius + 0.6
    for _ in range(tries):
        x = float(rng.uniform(margin, yard.width_m - margin))
        y = float(rng.uniform(margin, yard.height_m - margin))
        if _overlaps(x, y, radius, yard.obstacles, keepout):
            continue
        heading = _random_heading(rng)
        speed = DEFAULT_SPEEDS.get(kind, 0.0)
        return Obstacle(
            kind=kind,
            x=x,
            y=y,
            radius=radius,
            z=DEFAULT_HEIGHTS[kind],
            vx=speed * math.cos(heading),
            vy=speed * math.sin(heading),
            heading=heading,
            name=name or kind,
        )
    return None


def spawn_yard(
    rng: np.random.Generator,
    width_m: float,
    height_m: float,
    counts: dict[str, int],
    robot_keepout: tuple[float, float, float],
) -> Yard:
    yard = Yard(width_m=width_m, height_m=height_m)
    keepout = [robot_keepout]
    order = ("tree", "furniture", "toy", "person", "dog", "cat", "bird")
    for kind in order:
        n = int(counts.get(kind, 0))
        for i in range(n):
            obst = spawn_obstacle(rng, kind, yard, keepout, name=f"{kind}_{i}")
            if obst is not None:
                yard.obstacles.append(obst)
    return yard


def step_movers(
    yard: Yard,
    dt: float,
    rng: np.random.Generator,
    heading_jitter: float = 0.35,
) -> None:
    """Random-walk people/animals; bounce off the fence."""
    for obst in yard.living():
        obst.heading = obst.heading + float(rng.uniform(-heading_jitter, heading_jitter))
        speed = DEFAULT_SPEEDS.get(obst.kind, 0.0)
        obst.vx = speed * math.cos(obst.heading)
        obst.vy = speed * math.sin(obst.heading)
        nx = obst.x + obst.vx * dt
        ny = obst.y + obst.vy * dt
        lo = obst.radius + 0.15
        hi_x = yard.width_m - obst.radius - 0.15
        hi_y = yard.height_m - obst.radius - 0.15
        if nx < lo or nx > hi_x:
            obst.heading = math.atan2(obst.vy, -obst.vx)
            nx = float(np.clip(nx, lo, hi_x))
        if ny < lo or ny > hi_y:
            obst.heading = math.atan2(-obst.vy, obst.vx)
            ny = float(np.clip(ny, lo, hi_y))
        obst.x = nx
        obst.y = ny


def robot_start_pose(width_m: float, height_m: float) -> Pose:
    return Pose(0.5 * width_m, 0.5 * height_m, 0.0)

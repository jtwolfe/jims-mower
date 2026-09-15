"""Coverage-without-collision reward."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from jims_mower.config import RewardConfig
from jims_mower.constants import LIVING_KINDS


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    coverage: float
    time_penalty: float
    collision: float
    completion: float
    newly_cut: int
    done_success: bool


def compute_reward(
    cfg: RewardConfig,
    *,
    newly_cut: int,
    grass_cells: int,
    coverage_fraction: float,
    collision_kind: Optional[str],
    out_of_bounds: bool,
) -> RewardBreakdown:
    coverage = 0.0
    if grass_cells > 0 and newly_cut:
        coverage = cfg.coverage_scale * (newly_cut / grass_cells)
    time_pen = -abs(cfg.time_penalty)
    collision = 0.0
    if out_of_bounds:
        collision = -abs(cfg.out_of_bounds)
    elif collision_kind in LIVING_KINDS:
        collision = -abs(cfg.collision_living)
    elif collision_kind is not None:
        collision = -abs(cfg.collision_static)

    done_success = coverage_fraction >= cfg.completion_threshold and collision == 0.0
    completion = cfg.completion_bonus if done_success else 0.0
    total = coverage + time_pen + collision + completion
    return RewardBreakdown(
        total=float(total),
        coverage=float(coverage),
        time_penalty=float(time_pen),
        collision=float(collision),
        completion=float(completion),
        newly_cut=int(newly_cut),
        done_success=done_success,
    )

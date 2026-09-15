"""Optional semantic raster: grass / non-grass / drain / bank / static.

Built from coverage + hazard + occupancy. Not a claimed mIoU head.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from jims_mower.constants import (
    HAZARD_DRAIN,
    HAZARD_DRAIN_EDGE,
    HAZARD_STEEP,
    SEMANTIC_BANK,
    SEMANTIC_BUILDING,
    SEMANTIC_BUNKER,
    SEMANTIC_DRAIN,
    SEMANTIC_FREE,
    SEMANTIC_GARDEN,
    SEMANTIC_GRASS,
    SEMANTIC_NON_GRASS,
    SEMANTIC_PATH,
    SEMANTIC_STATIC,
    STRUCTURE_BUILDING,
    STRUCTURE_BUNKER,
    STRUCTURE_GARDEN,
    STRUCTURE_GREEN,
    STRUCTURE_PATH,
    STRUCTURE_POND,
)


def semantic_raster(
    coverage: np.ndarray,
    hazard: np.ndarray,
    occupancy: Optional[np.ndarray] = None,
    structure: Optional[np.ndarray] = None,
) -> np.ndarray:
    """uint8 layer aligned with the grass grid.

    Priority (high → low): static occupancy, drain, bank, non-grass, grass.
    Free (0) is unused grass-grid cells that somehow have no coverage value.
    """
    cov = np.asarray(coverage, dtype=np.float32)
    haz = np.asarray(hazard, dtype=np.float32)
    if cov.shape != haz.shape:
        raise ValueError("coverage and hazard must share shape")
    out = np.full(cov.shape, SEMANTIC_FREE, dtype=np.uint8)
    grass = cov >= 0.0
    out[grass] = SEMANTIC_GRASS
    out[~grass] = SEMANTIC_NON_GRASS
    bank = haz >= HAZARD_STEEP
    out[bank] = SEMANTIC_BANK
    drain = haz >= HAZARD_DRAIN_EDGE
    # Channel / lip win over bank so a swale on a slope stays drain.
    out[drain] = SEMANTIC_DRAIN
    if structure is not None:
        struct = np.asarray(structure)
        if struct.shape != cov.shape:
            raise ValueError("structure shape must match coverage")
        out[struct == STRUCTURE_PATH] = SEMANTIC_PATH
        out[struct == STRUCTURE_BUNKER] = SEMANTIC_BUNKER
        out[np.isin(struct, (STRUCTURE_GARDEN, STRUCTURE_GREEN))] = SEMANTIC_GARDEN
        out[struct == STRUCTURE_BUILDING] = SEMANTIC_BUILDING
        out[struct == STRUCTURE_POND] = SEMANTIC_NON_GRASS
    if occupancy is not None:
        occ = np.asarray(occupancy, dtype=np.float32)
        if occ.shape != cov.shape:
            raise ValueError("occupancy shape must match coverage")
        # Static detections sit on top of terrain so the planner can see
        # a person/tree even when the cell is also a bank.
        out[occ > 0.5] = SEMANTIC_STATIC
    return out

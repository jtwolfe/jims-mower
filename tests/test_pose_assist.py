"""Sparse landmark revisit. Still not_slam — not a shipped SLAM stack."""

from __future__ import annotations

import numpy as np
import pytest

from jims_mower.mapping import LoopClosureStub
from jims_mower.types import Pose

# Stated gym band: taught vertices stay inside this after a revisit pull.
FENCE_REVISIT_TOL_M = 0.75


def test_landmark_revisit_reduces_fence_drift() -> None:
    loop = LoopClosureStub(cell_m=1.0, min_step_gap=2, match=0.5, max_correct_m=1.50)
    square = [(1.0, 1.0), (5.0, 1.0), (5.0, 5.0), (1.0, 5.0)]
    loop.teach_vertices(square)
    occ = np.ones((24, 24), dtype=np.float32)
    # First pass: visit each vertex (stores fingerprints).
    for x, y in square:
        loop.update(occ, Pose(x, y, 0.0), 0.25)
    # Revisit first vertex with 1.1 m drift.
    drifted = Pose(2.05, 1.15, 0.0)
    loop.update(occ, drifted, 0.25)
    info = loop.as_info()
    assert info["not_slam"] is True
    assert info["pose_assist"] is True
    assert info["max_correct_m"] == pytest.approx(1.50)
    corrected = loop.apply(drifted)
    err = float(np.hypot(corrected.x - square[0][0], corrected.y - square[0][1]))
    drifted_err = float(np.hypot(drifted.x - square[0][0], drifted.y - square[0][1]))
    assert err < drifted_err
    assert err <= FENCE_REVISIT_TOL_M
    verts = [loop.apply(Pose(x + 1.1, y + 0.15, 0.0)) for x, y in square]
    # Same translation applied to the drifted square.
    mean = loop.fence_error_m([(p.x, p.y) for p in verts], taught=square)
    assert mean <= loop.max_correct_m


def test_loop_stub_still_not_slam_without_landmarks() -> None:
    loop = LoopClosureStub()
    info = loop.as_info()
    assert info["not_slam"] is True
    assert info["pose_assist"] is False
    assert info["loop_closure"] is False

"""Hazard-map action mask zeros forward commands into a drain cone."""

from __future__ import annotations

import numpy as np
import pytest

from jims_mower.action_mask import forward_hazard, mask_action
from jims_mower.constants import HAZARD_DRAIN, HAZARD_DRAIN_EDGE, HAZARD_STEEP


def _obs(hazard: np.ndarray) -> dict:
    return {
        "pose": np.array([0.6, 0.6, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "hazard": hazard,
    }


def test_mask_zeros_forward_on_drain_ahead() -> None:
    grid = np.zeros((16, 16), dtype=np.float32)
    # Pose (0.6, 0.6) heading +x, resolution 0.2 → cells around col 3–6.
    grid[:, 5:8] = HAZARD_DRAIN
    action = np.array([0.7, 0.7, 1.0], dtype=np.float32)
    out = mask_action(action, _obs(grid), resolution_m=0.2)
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(0.0)
    assert out[2] == pytest.approx(0.0)
    assert forward_hazard(_obs(grid), resolution_m=0.2) >= HAZARD_DRAIN_EDGE


def test_mask_allows_reverse_on_drain() -> None:
    grid = np.zeros((16, 16), dtype=np.float32)
    grid[:, 5:8] = HAZARD_DRAIN_EDGE
    action = np.array([-0.5, -0.5, 1.0], dtype=np.float32)
    out = mask_action(action, _obs(grid), resolution_m=0.2)
    assert out[0] < 0.0
    assert out[1] < 0.0
    assert out[2] == pytest.approx(0.0)


def test_mask_scales_steep_forward() -> None:
    grid = np.zeros((16, 16), dtype=np.float32)
    grid[:, 5:8] = HAZARD_STEEP
    action = np.array([0.8, 0.8, 1.0], dtype=np.float32)
    out = mask_action(action, _obs(grid), resolution_m=0.2)
    assert 0.0 < float(out[0]) < 0.8
    assert float(out[2]) == pytest.approx(1.0)


def test_mask_passthrough_on_free() -> None:
    grid = np.zeros((16, 16), dtype=np.float32)
    action = np.array([0.4, 0.35, 1.0], dtype=np.float32)
    out = mask_action(action, _obs(grid), resolution_m=0.2)
    assert out[0] == pytest.approx(0.4)
    assert out[1] == pytest.approx(0.35)

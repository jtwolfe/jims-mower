"""Grass coverage and occupancy rasters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jims_mower.maps import (
    GrassCoverageMap,
    occupancy_from_detections,
    occupancy_from_obstacles,
)
from jims_mower.types import Detection, Obstacle


def test_new_map_is_uncut() -> None:
    m = GrassCoverageMap(4.0, 4.0, 0.5)
    assert m.coverage_fraction() == pytest.approx(0.0)
    assert m.grass_cell_count() == 64
    assert m.cut_cell_count() == 0


def test_mark_circle_counts_new_cells_once() -> None:
    m = GrassCoverageMap(4.0, 4.0, 0.5)
    n1 = m.mark_circle(2.0, 2.0, 0.6)
    n2 = m.mark_circle(2.0, 2.0, 0.6)
    assert n1 > 0
    assert n2 == 0
    assert m.cut_cell_count() == n1


def test_out_of_bounds_world_to_cell() -> None:
    m = GrassCoverageMap(2.0, 2.0, 0.5)
    assert m.world_to_cell(-0.1, 0.5) is None
    assert m.world_to_cell(0.2, 0.2) == (0, 0)


def test_exclude_tree_removes_grass() -> None:
    m = GrassCoverageMap(4.0, 4.0, 0.5)
    before = m.grass_cell_count()
    m.exclude_circle(1.0, 1.0, 0.6)
    assert m.grass_cell_count() < before
    assert m.as_float().min() == pytest.approx(-1.0)


def test_reset_clears_cut_and_restores_grass() -> None:
    m = GrassCoverageMap(3.0, 3.0, 0.5)
    m.exclude_circle(0.5, 0.5, 0.4)
    m.mark_circle(2.0, 2.0, 0.5)
    m.reset()
    assert m.coverage_fraction() == pytest.approx(0.0)
    assert bool(m.grass.all())


def test_sample_world_values() -> None:
    m = GrassCoverageMap(2.0, 2.0, 0.5)
    assert m.sample_world(0.25, 0.25) == pytest.approx(0.0)
    m.mark_circle(0.25, 0.25, 0.4)
    assert m.sample_world(0.25, 0.25) == pytest.approx(1.0)
    assert m.sample_world(-1.0, 0.0) == pytest.approx(-1.0)


def test_cell_to_world_roundtrip() -> None:
    m = GrassCoverageMap(4.0, 2.0, 0.5)
    x, y = m.cell_to_world(0, 0)
    assert m.world_to_cell(x, y) == (0, 0)


def test_reject_bad_resolution() -> None:
    with pytest.raises(ValueError):
        GrassCoverageMap(2.0, 2.0, 0.0)


def test_occupancy_from_obstacles() -> None:
    grid = occupancy_from_obstacles(
        (10, 10),
        [Obstacle("tree", 1.0, 1.0, 0.4)],
        0.2,
    )
    assert grid.max() == pytest.approx(1.0)
    assert grid.min() == pytest.approx(0.0)


def test_occupancy_from_detections_skips_missing_xy() -> None:
    dets = [
        Detection("person", "front", (1, 1, 4, 4), 0.9, world_xy=None),
        Detection("dog", "front", (1, 1, 4, 4), 0.9, world_xy=(1.0, 1.0)),
    ]
    grid = occupancy_from_detections((10, 10), dets, 0.2)
    assert float(grid.sum()) > 0.0


def test_exclude_mask_drops_drain_cells() -> None:
    m = GrassCoverageMap(2.0, 2.0, 0.5)
    keep = np.ones((4, 4), dtype=bool)
    keep[0, 0] = False
    m.exclude_mask(keep)
    assert m.grass_cell_count() == 15
    assert m.as_float()[0, 0] == pytest.approx(-1.0)


def test_regenerate_grows_cut_cells_back() -> None:
    m = GrassCoverageMap(4.0, 4.0, 0.5)
    m.mark_circle(2.0, 2.0, 2.0)
    before = m.cut_cell_count()
    assert before > 4
    grown = m.regenerate(np.random.default_rng(0), 0.25)
    assert grown > 0
    assert m.cut_cell_count() == before - grown


def test_grass_state_roundtrip(tmp_path: Path) -> None:
    m = GrassCoverageMap(3.0, 3.0, 0.5)
    m.exclude_circle(0.4, 0.4, 0.4)
    m.mark_circle(2.0, 2.0, 0.8)
    path = tmp_path / "yard_grass.npz"
    m.save_state(path)
    other = GrassCoverageMap(3.0, 3.0, 0.5)
    other.load_state(path)
    assert np.array_equal(other.cut, m.cut)
    assert np.array_equal(other.grass, m.grass)


def test_as_float_dtype() -> None:
    m = GrassCoverageMap(2.0, 2.0, 0.5)
    arr = m.as_float()
    assert arr.dtype == np.float32
    assert arr.shape == (4, 4)

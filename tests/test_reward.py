"""Coverage reward shaping."""

from __future__ import annotations

import pytest

from jims_mower.config import RewardConfig
from jims_mower.reward import compute_reward


def _cfg() -> RewardConfig:
    return RewardConfig(
        coverage_scale=1.0,
        time_penalty=0.01,
        collision_living=50.0,
        collision_static=20.0,
        out_of_bounds=10.0,
        completion_bonus=15.0,
        completion_threshold=0.95,
    )


def test_time_penalty_always_applies() -> None:
    b = compute_reward(
        _cfg(),
        newly_cut=0,
        grass_cells=100,
        coverage_fraction=0.0,
        collision_kind=None,
        out_of_bounds=False,
    )
    assert b.total == pytest.approx(-0.01)
    assert b.newly_cut == 0


def test_coverage_scaled_by_map_size() -> None:
    b = compute_reward(
        _cfg(),
        newly_cut=10,
        grass_cells=100,
        coverage_fraction=0.1,
        collision_kind=None,
        out_of_bounds=False,
    )
    assert b.coverage == pytest.approx(0.1)
    assert b.total == pytest.approx(0.09)


def test_recut_is_zero_new_cells() -> None:
    b = compute_reward(
        _cfg(),
        newly_cut=0,
        grass_cells=50,
        coverage_fraction=0.4,
        collision_kind=None,
        out_of_bounds=False,
    )
    assert b.coverage == pytest.approx(0.0)


def test_living_collision_penalty() -> None:
    b = compute_reward(
        _cfg(),
        newly_cut=0,
        grass_cells=100,
        coverage_fraction=0.2,
        collision_kind="person",
        out_of_bounds=False,
    )
    assert b.collision == pytest.approx(-50.0)
    assert b.done_success is False


def test_static_collision_penalty() -> None:
    b = compute_reward(
        _cfg(),
        newly_cut=0,
        grass_cells=100,
        coverage_fraction=0.2,
        collision_kind="tree",
        out_of_bounds=False,
    )
    assert b.collision == pytest.approx(-20.0)


def test_out_of_bounds_penalty() -> None:
    b = compute_reward(
        _cfg(),
        newly_cut=5,
        grass_cells=100,
        coverage_fraction=0.2,
        collision_kind=None,
        out_of_bounds=True,
    )
    assert b.collision == pytest.approx(-10.0)


def test_completion_bonus() -> None:
    b = compute_reward(
        _cfg(),
        newly_cut=1,
        grass_cells=100,
        coverage_fraction=0.96,
        collision_kind=None,
        out_of_bounds=False,
    )
    assert b.done_success is True
    assert b.completion == pytest.approx(15.0)


def test_no_completion_on_collision() -> None:
    b = compute_reward(
        _cfg(),
        newly_cut=1,
        grass_cells=100,
        coverage_fraction=0.99,
        collision_kind="dog",
        out_of_bounds=False,
    )
    assert b.done_success is False
    assert b.completion == pytest.approx(0.0)

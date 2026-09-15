"""Yard spawn and mover integration."""

from __future__ import annotations

import numpy as np
import pytest

from jims_mower.world import robot_start_pose, spawn_yard, step_movers


def test_robot_starts_in_center() -> None:
    pose = robot_start_pose(12.0, 8.0)
    assert pose.x == pytest.approx(6.0)
    assert pose.y == pytest.approx(4.0)
    assert pose.theta == pytest.approx(0.0)


def test_spawn_respects_counts_and_keepout() -> None:
    rng = np.random.default_rng(0)
    yard = spawn_yard(
        rng,
        10.0,
        10.0,
        {"person": 1, "dog": 1, "tree": 2, "toy": 1},
        (5.0, 5.0, 1.5),
    )
    kinds = [o.kind for o in yard.obstacles]
    assert kinds.count("person") == 1
    assert kinds.count("dog") == 1
    assert kinds.count("tree") == 2
    for obst in yard.obstacles:
        assert (obst.x - 5.0) ** 2 + (obst.y - 5.0) ** 2 >= 1.5**2


def test_static_vs_living_split() -> None:
    rng = np.random.default_rng(1)
    yard = spawn_yard(
        rng, 10.0, 10.0, {"person": 1, "tree": 1, "furniture": 1}, (5.0, 5.0, 1.0)
    )
    assert {o.kind for o in yard.static()} <= {"tree", "furniture", "toy"}
    assert {o.kind for o in yard.living()} <= {"person", "dog", "cat", "bird"}


def test_movers_stay_in_yard() -> None:
    rng = np.random.default_rng(2)
    yard = spawn_yard(rng, 8.0, 8.0, {"person": 1, "dog": 1, "cat": 1}, (1.0, 1.0, 0.8))
    for _ in range(40):
        step_movers(yard, 0.2, rng)
    for obst in yard.living():
        assert 0.1 < obst.x < 7.9
        assert 0.1 < obst.y < 7.9


def test_hose_and_cord_are_soft_clutter() -> None:
    rng = np.random.default_rng(3)
    yard = spawn_yard(
        rng,
        10.0,
        10.0,
        {"hose": 1, "cord": 1, "tree": 1},
        (5.0, 5.0, 1.2),
    )
    clutter = [o for o in yard.obstacles if o.kind in {"hose", "cord"}]
    assert len(clutter) == 2
    for obst in clutter:
        assert obst.is_soft
        assert obst.is_cutter_risk
        assert obst.length_m > 1.0


def test_orchard_layout_places_tree_rows() -> None:
    rng = np.random.default_rng(4)
    yard = spawn_yard(
        rng,
        12.0,
        10.0,
        {"tree": 12},
        (6.0, 5.0, 1.2),
        layout="orchard",
        orchard_rows=3,
        orchard_cols=4,
    )
    trees = [o for o in yard.obstacles if o.kind == "tree"]
    ys = {round(o.y, 1) for o in trees}
    assert len(trees) >= 6
    assert len(ys) >= 2


def test_empty_counts_spawn_nothing() -> None:
    yard = spawn_yard(np.random.default_rng(0), 6.0, 6.0, {}, (3.0, 3.0, 1.0))
    assert yard.obstacles == []

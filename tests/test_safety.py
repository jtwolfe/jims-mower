"""Trimmer interlock and collisions."""

from __future__ import annotations

import math

import pytest

from jims_mower.safety import (
    cutter_risk_hit,
    first_collision,
    in_yard,
    is_body_collision,
    is_living,
    nearest_living,
    terrain_hazards,
    trimmer_interlock,
)
from jims_mower.types import Obstacle, Pose


def _person(x: float, y: float) -> Obstacle:
    return Obstacle("person", x, y, 0.25, z=0.9)


def _tree(x: float, y: float) -> Obstacle:
    return Obstacle("tree", x, y, 0.35, z=0.4)


def test_living_kinds() -> None:
    assert is_living("person")
    assert is_living("dog")
    assert is_living("cat")
    assert is_living("bird")
    assert not is_living("tree")
    assert not is_living("furniture")
    assert not is_living("toy")
    assert not is_living("hose")
    assert not is_living("cord")


def test_nearest_living_empty() -> None:
    dist, obst = nearest_living((0.0, 0.0), [])
    assert math.isinf(dist)
    assert obst is None


def test_nearest_living_ignores_trees() -> None:
    dist, obst = nearest_living((0.0, 0.0), [_tree(0.2, 0.0), _person(2.0, 0.0)])
    assert obst is not None and obst.kind == "person"
    assert dist == pytest.approx(2.0)


def test_nearest_picks_closer_animal() -> None:
    dog = Obstacle("dog", 1.0, 0.0, 0.2)
    cat = Obstacle("cat", 3.0, 0.0, 0.12)
    dist, obst = nearest_living((0.0, 0.0), [cat, dog])
    assert obst is dog
    assert dist == pytest.approx(1.0)


def test_interlock_off_when_not_requested() -> None:
    pose = Pose(0.0, 0.0, 0.0)
    d = trimmer_interlock(False, pose, [_person(5.0, 0.0)], offset_m=0.32, safety_radius_m=1.5)
    assert d.trimmer_enabled is False
    assert d.requested is False
    assert d.blocked_reason is None


def test_interlock_allows_when_far() -> None:
    pose = Pose(0.0, 0.0, 0.0)
    d = trimmer_interlock(True, pose, [_person(6.0, 0.0)], offset_m=0.32, safety_radius_m=1.5)
    assert d.trimmer_enabled is True


def test_interlock_blocks_person() -> None:
    pose = Pose(0.0, 0.0, 0.0)
    d = trimmer_interlock(True, pose, [_person(1.0, 0.0)], offset_m=0.32, safety_radius_m=1.5)
    assert d.trimmer_enabled is False
    assert d.blocked_reason is not None
    assert "person" in d.blocked_reason


@pytest.mark.parametrize("kind", ["dog", "cat", "bird"])
def test_interlock_blocks_animals(kind: str) -> None:
    pose = Pose(0.0, 0.0, 0.0)
    obst = Obstacle(kind, 0.8, 0.0, 0.15, z=0.3)
    d = trimmer_interlock(True, pose, [obst], offset_m=0.32, safety_radius_m=1.5)
    assert d.trimmer_enabled is False
    assert d.nearest_kind == kind


@pytest.mark.parametrize("kind", ["tree", "furniture", "toy"])
def test_interlock_ignores_static(kind: str) -> None:
    pose = Pose(0.0, 0.0, 0.0)
    obst = Obstacle(kind, 0.6, 0.0, 0.2)
    d = trimmer_interlock(True, pose, [obst], offset_m=0.32, safety_radius_m=1.5)
    assert d.trimmer_enabled is True


def test_interlock_uses_trimmer_hub_not_body() -> None:
    # Person sits 1.2 m in front of the body. Body-center distance is 1.2;
    # trimmer (0.32 m forward) is 0.88 m away — inside a 1.0 m bubble.
    pose = Pose(0.0, 0.0, 0.0)
    person = _person(1.2, 0.0)
    d = trimmer_interlock(True, pose, [person], offset_m=0.32, safety_radius_m=1.0)
    assert d.trimmer_enabled is False
    assert d.nearest_living_m == pytest.approx(0.88, abs=1e-6)


def test_interlock_rejects_bad_radius() -> None:
    with pytest.raises(ValueError):
        trimmer_interlock(True, Pose(0, 0, 0), [], offset_m=0.3, safety_radius_m=0.0)


def test_body_collision_person() -> None:
    pose = Pose(0.0, 0.0, 0.0)
    assert is_body_collision(pose, _person(0.3, 0.0), 0.28)


def test_no_collision_when_far() -> None:
    pose = Pose(0.0, 0.0, 0.0)
    assert not is_body_collision(pose, _person(3.0, 0.0), 0.28)


def test_high_bird_does_not_collide() -> None:
    pose = Pose(0.0, 0.0, 0.0)
    bird = Obstacle("bird", 0.1, 0.0, 0.08, z=1.4)
    assert not is_body_collision(pose, bird, 0.28)


def test_low_bird_does_collide() -> None:
    pose = Pose(0.0, 0.0, 0.0)
    bird = Obstacle("bird", 0.1, 0.0, 0.08, z=0.2)
    assert is_body_collision(pose, bird, 0.28)


def test_first_collision_returns_first_hit() -> None:
    pose = Pose(0.0, 0.0, 0.0)
    hit = first_collision(pose, [_tree(4.0, 0.0), _person(0.2, 0.0)], 0.28)
    assert hit is not None and hit.kind == "person"


def test_terrain_hazards_without_field() -> None:
    ev = terrain_hazards(
        Pose(1.0, 1.0, 0.0),
        None,
        length_m=0.5,
        track_m=0.4,
        tip_roll_rad=0.4,
        tip_pitch_rad=0.45,
        wheel_drop_m=0.08,
        steep_slope_rad=0.3,
    )
    assert ev.advice == "ok"
    assert ev.tipover is False
    assert ev.drain_drop is False


def test_soft_hose_is_not_a_body_collision() -> None:
    pose = Pose(0.0, 0.0, 0.0)
    hose = Obstacle("hose", 0.1, 0.0, 0.06, z=0.03, length_m=2.0, soft=True, cutter_risk=True)
    assert hose.is_soft
    assert not is_body_collision(pose, hose, 0.28)
    assert first_collision(pose, [hose], 0.28) is None


def test_cutter_risk_flag_when_trimmer_hits_cord() -> None:
    cord = Obstacle("cord", 0.32, 0.0, 0.04, z=0.02, length_m=1.5, heading=1.2)
    hit = cutter_risk_hit((0.32, 0.0), [cord], 0.16)
    assert hit is cord
    miss = cutter_risk_hit((3.0, 3.0), [cord], 0.16)
    assert miss is None


def test_in_yard_and_out() -> None:
    pose = Pose(4.0, 4.0, 0.0)
    assert in_yard(pose, 8.0, 8.0, 0.28)
    assert not in_yard(Pose(-0.1, 4.0, 0.0), 8.0, 8.0, 0.28)
    assert not in_yard(Pose(7.9, 4.0, 0.0), 8.0, 8.0, 0.28)

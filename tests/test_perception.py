"""Pluggable detector, grass hook, and hand-signal curriculum."""

from __future__ import annotations

import numpy as np
import pytest

from jims_mower.constants import HAND_SIGNALS, UNCUT_GRASS_RGB
from jims_mower.perception import (
    BlindDetector,
    ColorGrassObserver,
    HandSignalCurriculum,
    MockDetector,
    grass_fraction,
    uncut_grass_mask,
)
from jims_mower.perception.mock import category_for, detection_from_obstacle
from jims_mower.types import CameraSpec, Obstacle, PerceptionContext, Pose


def _front() -> CameraSpec:
    return CameraSpec("front", 0.25, 0.0, 0.38, 0.0, -8.0)


def test_category_for() -> None:
    assert category_for("person") == "person"
    assert category_for("dog") == "animal"
    assert category_for("tree") == "static"


def test_mock_projects_person_in_front() -> None:
    pose = Pose(1.0, 4.0, 0.0)
    person = Obstacle("person", 3.0, 4.0, 0.25, z=0.9)
    det = detection_from_obstacle(person, _front(), pose, 80, 60, include_signal=False)
    assert det is not None
    assert det.label == "person"
    assert det.camera == "front"
    assert det.world_xy == (3.0, 4.0)
    assert det.hand_signal is None
    x, y, w, h = det.bbox
    assert w > 0 and h > 0
    assert 0 <= x < 80 and 0 <= y < 60


def test_mock_skips_object_behind() -> None:
    pose = Pose(4.0, 4.0, 0.0)
    person = Obstacle("person", 1.0, 4.0, 0.25, z=0.9)
    det = detection_from_obstacle(person, _front(), pose, 80, 60, include_signal=False)
    assert det is None


def test_mock_includes_hand_signal_when_asked() -> None:
    pose = Pose(1.0, 4.0, 0.0)
    person = Obstacle("person", 3.0, 4.0, 0.25, z=0.9, hand_signal="stop")
    det = detection_from_obstacle(person, _front(), pose, 80, 60, include_signal=True)
    assert det is not None
    assert det.hand_signal == "stop"
    hidden = detection_from_obstacle(person, _front(), pose, 80, 60, include_signal=False)
    assert hidden is not None and hidden.hand_signal is None


def test_blind_detector_is_empty() -> None:
    ctx = PerceptionContext(Pose(0, 0, 0), [_front()], [])
    assert BlindDetector().detect({"front": np.zeros((24, 32, 3), np.uint8)}, ctx) == []


def test_mock_detector_protocol() -> None:
    pose = Pose(1.0, 4.0, 0.0)
    obstacles = [
        Obstacle("person", 3.0, 4.0, 0.25, z=0.9),
        Obstacle("dog", 2.8, 3.5, 0.2, z=0.35),
        Obstacle("tree", 3.2, 4.4, 0.35, z=0.4),
    ]
    images = {"front": np.zeros((24, 32, 3), np.uint8)}
    ctx = PerceptionContext(pose, [_front()], obstacles, (32, 24), False)
    dets = MockDetector().detect(images, ctx)
    labels = {d.label for d in dets}
    assert "person" in labels
    assert "dog" in labels
    assert all(d.category for d in dets)


def test_grass_fraction_on_uncut_color() -> None:
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:] = UNCUT_GRASS_RGB
    assert grass_fraction(img) == pytest.approx(1.0)
    mask = uncut_grass_mask(img)
    assert bool(mask.all())


def test_grass_fraction_on_sky() -> None:
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:] = (135, 186, 230)
    assert grass_fraction(img) == pytest.approx(0.0)


def test_color_observer_per_camera() -> None:
    images = {
        "front": np.broadcast_to(
            np.array(UNCUT_GRASS_RGB, dtype=np.uint8), (8, 8, 3)
        ).copy(),
        "rear": np.zeros((8, 8, 3), dtype=np.uint8),
    }
    out = ColorGrassObserver().estimate(images)
    assert out["front"] > 0.9
    assert out["rear"] == pytest.approx(0.0)


def test_grass_hook_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        uncut_grass_mask(np.zeros((8, 8), dtype=np.uint8))


def test_curriculum_disabled_clears_signals() -> None:
    people = [Obstacle("person", 1.0, 1.0, 0.25, hand_signal="go")]
    cur = HandSignalCurriculum(False)
    cur.assign(people, np.random.default_rng(0))
    assert people[0].hand_signal is None


def test_curriculum_assigns_known_signals() -> None:
    people = [Obstacle("person", 1.0, 1.0, 0.25), Obstacle("dog", 2.0, 2.0, 0.2)]
    cur = HandSignalCurriculum(True, hold_steps=2)
    rng = np.random.default_rng(1)
    cur.assign(people, rng)
    assert people[0].hand_signal in HAND_SIGNALS
    assert people[1].hand_signal is None


def test_curriculum_rotates_after_hold() -> None:
    people = [Obstacle("person", 1.0, 1.0, 0.25)]
    cur = HandSignalCurriculum(True, hold_steps=2)
    rng = np.random.default_rng(2)
    cur.assign(people, rng)
    first = people[0].hand_signal
    cur.maybe_rotate(people, rng)
    assert people[0].hand_signal == first
    cur.maybe_rotate(people, rng)
    # After hold expires it reassigns; may or may not change, but stays valid.
    assert people[0].hand_signal in HAND_SIGNALS


def test_nearest_person_signal() -> None:
    obstacles = [
        Obstacle("person", 5.0, 0.0, 0.25, hand_signal="follow"),
        Obstacle("person", 1.0, 0.0, 0.25, hand_signal="stop"),
    ]
    assert HandSignalCurriculum.nearest_person_signal(obstacles, (0.0, 0.0)) == "stop"

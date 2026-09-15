"""WAVE 4 perception hooks: appearance, hand-signal stub, grass net, semantic."""

from __future__ import annotations

import numpy as np
import pytest

from jims_mower.constants import KIND_RGB, LONG_GRASS_RGB, SEMANTIC_DRAIN, SEMANTIC_GRASS, UNCUT_GRASS_RGB
from jims_mower.perception import (
    FeatureGrassObserver,
    HandSignalClassifier,
    MockDetector,
    grass_observer_from_mode,
    refine_category,
    semantic_raster,
)
from jims_mower.perception.classify import appearance_features, classify_hand_signal_features, crop_bbox
from jims_mower.perception.mock import category_for as cat_for
from jims_mower.types import CameraSpec, Obstacle, PerceptionContext, Pose


def test_category_includes_toy() -> None:
    assert cat_for("toy") == "toy"
    assert cat_for("person") == "person"
    assert cat_for("dog") == "animal"


def test_refine_category_matches_palette() -> None:
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[:] = KIND_RGB["person"]
    label, category, score = refine_category(img, (2, 2, 12, 12), "person")
    assert category == "person"
    assert score > 0.4
    assert label == "person"


def test_hand_signal_classifier_on_dark_crop() -> None:
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    assert HandSignalClassifier().classify(img, (1, 1, 10, 10)) == "stop"
    bright = np.full((16, 16, 3), 200, dtype=np.uint8)
    bright[:, :, 1] = 220
    name = classify_hand_signal_features(appearance_features(bright))
    assert name in {"go", "follow", "back", "stop"}


def test_crop_bbox_empty_off_frame() -> None:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    assert crop_bbox(img, (20, 20, 4, 4)).size == 0


def test_feature_grass_observer() -> None:
    green = np.broadcast_to(np.array(UNCUT_GRASS_RGB, dtype=np.uint8), (10, 10, 3)).copy()
    sky = np.zeros((10, 10, 3), dtype=np.uint8)
    sky[:] = (135, 186, 230)
    out = FeatureGrassObserver().estimate({"front": green, "rear": sky})
    assert out["front"] > 0.5
    assert out["rear"] < 0.35
    assert isinstance(grass_observer_from_mode("feature"), FeatureGrassObserver)


def test_semantic_raster_priority() -> None:
    cov = np.ones((6, 6), dtype=np.float32)
    cov[0, 0] = -1.0
    haz = np.zeros((6, 6), dtype=np.float32)
    haz[2, 2] = 3.0
    occ = np.zeros((6, 6), dtype=np.float32)
    occ[4, 4] = 1.0
    sem = semantic_raster(cov, haz, occ)
    assert sem[1, 1] == SEMANTIC_GRASS
    assert sem[2, 2] == SEMANTIC_DRAIN
    assert int(sem[4, 4]) == 5  # static


def test_mock_detector_appearance_sets_category() -> None:
    pose = Pose(1.0, 4.0, 0.0)
    obstacles = [
        Obstacle("person", 3.0, 4.0, 0.25, z=0.9),
        Obstacle("toy", 2.6, 4.1, 0.10, z=0.08),
    ]
    img = np.zeros((24, 32, 3), dtype=np.uint8)
    img[:] = KIND_RGB["person"]
    ctx = PerceptionContext(pose, [CameraSpec("front", 0.25, 0.0, 0.38, 0.0, -8.0)], obstacles, (32, 24))
    dets = MockDetector().detect({"front": img}, ctx)
    cats = {d.category for d in dets}
    assert "person" in cats or "toy" in cats
    assert all(d.category for d in dets)


def test_long_grass_constant_differs() -> None:
    assert LONG_GRASS_RGB != UNCUT_GRASS_RGB


def test_semantic_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        semantic_raster(np.zeros((2, 2)), np.zeros((3, 3)))

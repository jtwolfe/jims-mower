"""CV-2 gym grass coverage observer. Not field mAP / IoU."""

from __future__ import annotations

import numpy as np
import pytest

from jims_mower.constants import COV_CUT, COV_NON_GRASS, COV_UNCUT, HAZARD_DRAIN, UNCUT_GRASS_RGB
from jims_mower.env import MowerEnv
from jims_mower.perception.grass import (
    ClassAwareGrassObserver,
    ColorGrassObserver,
    classify_coverage_rgb,
    coverage_error,
    cut_grass_mask,
    grass_observer_from_mode,
    paint_lawn_strip_fixture,
    uncut_fraction_from_labels,
)


def test_lawn_strip_fixture_error_is_small() -> None:
    image, truth = paint_lawn_strip_fixture(40, 56, sky_rows=8, uncut_frac=0.40, cut_frac=0.35)
    pred = classify_coverage_rgb(image)
    err = coverage_error(pred, truth)
    assert err["map_claim"] is None
    assert err["iou_claim"] is None
    assert err["field_strip"] is False
    assert err["uncut_fraction_abs_error"] < 0.08
    assert err["cut_fraction_abs_error"] < 0.08
    assert err["grass_vs_nongrass_accuracy"] > 0.90
    assert err["pixel_accuracy"] > 0.85


def test_class_aware_uses_terrain_labels() -> None:
    image, _truth = paint_lawn_strip_fixture(24, 32, sky_rows=4)
    # Force the uncut strip to non-grass via a drain class map.
    terrain = np.zeros(image.shape[:2], dtype=np.uint8)
    terrain[4:, :10] = HAZARD_DRAIN
    labels = ClassAwareGrassObserver().classify(image, terrain)
    assert int((labels[4:, :10] == COV_NON_GRASS).sum()) == int(labels[4:, :10].size)
    detail = ClassAwareGrassObserver().estimate_detail(
        {"front": image},
        {"front": terrain},
    )
    assert detail.used_terrain_classes is True
    assert detail.map_claim is None
    assert detail.per_camera["front"] < uncut_fraction_from_labels(classify_coverage_rgb(image))


def test_class_aware_falls_back_to_colour() -> None:
    img = np.zeros((12, 12, 3), dtype=np.uint8)
    img[:] = UNCUT_GRASS_RGB
    out = ClassAwareGrassObserver().estimate({"front": img})
    assert out["front"] > 0.9
    assert isinstance(grass_observer_from_mode("class"), ClassAwareGrassObserver)
    assert isinstance(grass_observer_from_mode("color"), ColorGrassObserver)


def test_cut_mask_on_tan() -> None:
    from jims_mower.constants import CUT_GRASS_RGB

    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[:] = CUT_GRASS_RGB
    assert bool(cut_grass_mask(img).all())
    labels = classify_coverage_rgb(img)
    assert int((labels == COV_CUT).sum()) == 64


def test_env_default_coverage_source_is_gym_grid() -> None:
    env = MowerEnv(
        config={
            "sensors": {"width": 16, "height": 12, "camera_count": 4},
            "world": {
                "width_m": 6.0,
                "height_m": 6.0,
                "resolution_m": 0.30,
                "n_people": 0,
                "n_dogs": 0,
                "n_cats": 0,
                "n_birds": 0,
                "n_trees": 0,
                "n_furniture": 0,
                "n_toys": 0,
                "terrain": {"enabled": False},
            },
        }
    )
    _obs, info = env.reset(seed=2)
    assert info["coverage_source"] == "gym_grid"
    assert info["gym_coverage_fraction"] == pytest.approx(info["coverage_fraction"])
    env.close()


def test_env_observer_coverage_source_flag() -> None:
    env = MowerEnv(
        config={
            "sensors": {"width": 16, "height": 12, "camera_count": 4},
            "perception": {"grass_mode": "class", "coverage_source": "observer"},
            "world": {
                "width_m": 6.0,
                "height_m": 6.0,
                "resolution_m": 0.30,
                "n_people": 0,
                "n_dogs": 0,
                "n_cats": 0,
                "n_birds": 0,
                "n_trees": 0,
                "n_furniture": 0,
                "n_toys": 0,
                "terrain": {"enabled": False},
            },
        }
    )
    _obs, info = env.reset(seed=3)
    assert info["coverage_source"] == "observer"
    assert info["observer_coverage_fraction"] is not None
    assert 0.0 <= float(info["observer_coverage_fraction"]) <= 1.0
    assert info["gym_coverage_fraction"] == pytest.approx(env._coverage.coverage_fraction())
    env.close()

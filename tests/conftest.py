"""Shared fixtures."""

from __future__ import annotations

import pytest

from jims_mower.config import EnvConfig, load_config


def tiny_config_dict() -> dict:
    return {
        "dt": 0.1,
        "max_steps": 40,
        "sensors": {"width": 32, "height": 24, "camera_count": 4, "fov_deg": 70.0},
        "world": {
            "width_m": 8.0,
            "height_m": 8.0,
            "resolution_m": 0.20,
            "n_people": 1,
            "n_dogs": 1,
            "n_cats": 0,
            "n_birds": 0,
            "n_trees": 1,
            "n_furniture": 0,
            "n_toys": 1,
        },
        "robot": {
            "trimmer": {"safety_radius_m": 1.2, "offset_m": 0.32, "radius_m": 0.16}
        },
    }


@pytest.fixture
def tiny_cfg() -> EnvConfig:
    return load_config(tiny_config_dict())

"""S2R-4: drive.scale halves distance. No invented Kv."""

from __future__ import annotations

import math

import numpy as np
import pytest

from jims_mower.config import ConfigError, load_config
from jims_mower.env import MowerEnv


def _flat(scale: float) -> dict:
    return {
        "dt": 0.1,
        "max_steps": 20,
        "sensors": {
            "width": 16,
            "height": 12,
            "camera_count": 4,
            "gps": {"dropout_prob": 0.0},
        },
        "world": {
            "width_m": 10.0,
            "height_m": 10.0,
            "resolution_m": 0.25,
            "n_people": 0,
            "n_dogs": 0,
            "n_cats": 0,
            "n_birds": 0,
            "n_trees": 0,
            "n_furniture": 0,
            "n_toys": 0,
            "terrain": {"enabled": False},
        },
        "robot": {
            "max_wheel_speed_mps": 1.2,
            "drive": {"scale": float(scale), "measured": False},
            "trimmer": {"scale": 1.0, "measured": False},
        },
        "perception": {"terrain_mode": "oracle", "detector_backend": "blind"},
    }


def _drive_distance(scale: float, steps: int = 8) -> float:
    env = MowerEnv(config=_flat(scale))
    _obs, info = env.reset(seed=1)
    x0, y0 = float(info["pose"]["x"]), float(info["pose"]["y"])
    last = info
    for _ in range(steps):
        _obs, _r, _t, _c, last = env.step(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    env.close()
    return math.hypot(float(last["pose"]["x"]) - x0, float(last["pose"]["y"]) - y0)


def test_scale_half_halves_distance() -> None:
    full = _drive_distance(1.0)
    half = _drive_distance(0.5)
    assert full > 0.4
    assert half == pytest.approx(0.5 * full, rel=0.08, abs=0.04)
    vmax = 1.2
    dt = 0.1
    expected_full = vmax * 1.0 * dt * 8
    assert full == pytest.approx(expected_full, rel=0.12)


def test_drive_scale_defaults_unmeasured() -> None:
    cfg = load_config()
    assert cfg.robot.drive.scale == pytest.approx(1.0)
    assert cfg.robot.drive.measured is False
    assert cfg.robot.trimmer.scale == pytest.approx(1.0)
    assert cfg.robot.trimmer.measured is False
    assert cfg.robot.wheel_speed_mps() == pytest.approx(cfg.robot.max_wheel_speed_mps)


def test_rejects_nonpositive_scale() -> None:
    with pytest.raises(ConfigError):
        load_config({"robot": {"drive": {"scale": 0.0}}})
    with pytest.raises(ConfigError):
        load_config({"robot": {"trimmer": {"scale": -1.0}}})

"""Height field, slope attitude, and drain / tip hazards."""

from __future__ import annotations

import math

import numpy as np
import pytest

from jims_mower.constants import TERRAIN_DRAIN, TERRAIN_DRAIN_EDGE
from jims_mower.kinematics import sit_on_terrain, wheel_positions
from jims_mower.safety import terrain_hazards
from jims_mower.terrain import HeightField, generate_terrain
from jims_mower.types import Pose


def test_empty_field_is_flat() -> None:
    hf = HeightField.empty(4.0, 4.0, 0.2)
    assert hf.sample(1.0, 1.0) == pytest.approx(0.0)
    assert hf.sample_label(1.0, 1.0) == 0
    assert hf.sample_slope(1.0, 1.0) == pytest.approx(0.0, abs=1e-5)


def test_plane_pitch_and_roll() -> None:
    # z = 0.2 x  → facing +x should pitch nose-up.
    hf = HeightField.from_function(6.0, 6.0, 0.1, lambda x, y: 0.2 * x)
    posed = sit_on_terrain(Pose(3.0, 3.0, 0.0), hf, 0.50, 0.40)
    assert posed.pitch == pytest.approx(math.atan(0.2), abs=0.03)
    assert posed.roll == pytest.approx(0.0, abs=0.03)
    assert posed.z == pytest.approx(0.2 * 3.0, abs=0.05)

    # z = 0.2 y  → left side higher, positive roll (body +y is left).
    hf = HeightField.from_function(6.0, 6.0, 0.1, lambda x, y: 0.2 * y)
    posed = sit_on_terrain(Pose(3.0, 3.0, 0.0), hf, 0.50, 0.40)
    assert posed.roll == pytest.approx(math.atan(0.2), abs=0.03)
    assert posed.pitch == pytest.approx(0.0, abs=0.03)


def test_generated_yard_has_drains_and_banks() -> None:
    rng = np.random.default_rng(4)
    hf = generate_terrain(
        rng,
        12.0,
        12.0,
        0.10,
        n_drains=2,
        n_banks=2,
        keepout=[(6.0, 6.0, 1.8)],
    )
    assert len(hf.drains) >= 1
    assert int((hf.labels == TERRAIN_DRAIN).sum()) > 0
    assert hf.elevation.min() < -0.05
    # Start keepout stays near grade.
    assert abs(hf.sample(6.0, 6.0)) < 0.08


def test_disabled_terrain_is_flat() -> None:
    hf = generate_terrain(np.random.default_rng(0), 8.0, 8.0, 0.2, enabled=False, n_drains=4)
    assert hf.drains == []
    assert float(np.abs(hf.elevation).max()) == pytest.approx(0.0)


def test_drain_drop_when_wheel_in_channel() -> None:
    hf = HeightField.empty(6.0, 6.0, 0.10)
    # Deep ditch along y through x=3.4, wide enough to catch the right wheels.
    yy = (np.arange(hf.rows) + 0.5) * hf.resolution_m
    xx = (np.arange(hf.cols) + 0.5) * hf.resolution_m
    gx, gy = np.meshgrid(xx, yy)
    ditch = np.abs(gx - 3.35) < 0.18
    hf.elevation[ditch] = -0.22
    hf.labels[ditch] = TERRAIN_DRAIN
    hf.recompute_slope()
    pose = sit_on_terrain(Pose(3.0, 3.0, 0.0), hf, 0.50, 0.40)
    ev = terrain_hazards(
        pose,
        hf,
        length_m=0.50,
        track_m=0.40,
        tip_roll_rad=0.40,
        tip_pitch_rad=0.45,
        wheel_drop_m=0.08,
        steep_slope_rad=0.30,
    )
    assert ev.drain_drop is True
    assert ev.advice == "stop"
    assert any(ev.wheels_in_drain)


def test_tipover_on_extreme_roll() -> None:
    hf = HeightField.from_function(6.0, 6.0, 0.1, lambda x, y: 1.2 * y)
    pose = sit_on_terrain(Pose(3.0, 3.0, 0.0), hf, 0.50, 0.40)
    ev = terrain_hazards(
        pose,
        hf,
        length_m=0.50,
        track_m=0.40,
        tip_roll_rad=0.40,
        tip_pitch_rad=0.45,
        wheel_drop_m=0.08,
        steep_slope_rad=0.30,
    )
    assert ev.tipover is True
    assert ev.advice == "stop"


def test_reroute_when_drain_ahead() -> None:
    hf = HeightField.empty(6.0, 6.0, 0.10)
    # Channel across the path 0.55 m in front of a robot at (2, 3) heading +x.
    for row in range(hf.rows):
        for col in range(hf.cols):
            x = (col + 0.5) * 0.10
            if abs(x - 2.55) < 0.12:
                hf.labels[row, col] = TERRAIN_DRAIN_EDGE
    hf.recompute_slope()
    pose = Pose(2.0, 3.0, 0.0)
    ev = terrain_hazards(
        pose,
        hf,
        length_m=0.50,
        track_m=0.40,
        tip_roll_rad=0.40,
        tip_pitch_rad=0.45,
        wheel_drop_m=0.08,
        steep_slope_rad=0.30,
        look_ahead_m=0.55,
    )
    assert ev.advice == "reroute"
    assert ev.drain_drop is False


def test_wheel_positions_track() -> None:
    fl, fr, rl, rr = wheel_positions(Pose(0.0, 0.0, 0.0), 0.50, 0.40)
    assert fl[1] == pytest.approx(0.20)
    assert fr[1] == pytest.approx(-0.20)
    assert rl[0] == pytest.approx(-0.25)
    assert rr[0] == pytest.approx(-0.25)


def test_outside_sample_is_zero() -> None:
    hf = HeightField.empty(2.0, 2.0, 0.2)
    assert hf.sample(-1.0, 0.5) == pytest.approx(0.0)
    assert hf.sample_label(-1.0, 0.5) == 0

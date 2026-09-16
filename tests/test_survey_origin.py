"""MAP-5: surveyed-origin model + GNSS/pose stop short of the tape."""

from __future__ import annotations

import numpy as np
import pytest

from jims_mower.env import MowerEnv
from jims_mower.geofence import (
    GeofenceSpec,
    geofence_advice,
    geofence_advice_from_gnss,
    gps_enu_from_world,
    gps_world_from_enu,
)
from jims_mower.profile import (
    SurveyOrigin,
    YardProfile,
    parse_survey_origin,
    parse_yard_profile,
    world_xy_from_enu,
)
from jims_mower.types import Pose


KEEP_IN_LOCAL = [(1.0, 1.0), (7.0, 1.0), (7.0, 7.0), (1.0, 7.0)]


def test_origin_default_is_identity() -> None:
    origin = SurveyOrigin()
    assert origin.e_m == 0.0
    assert origin.surveyed is False
    assert origin.has_wgs84() is False
    x, y = world_xy_from_enu(3.0, 4.0, origin)
    assert (x, y) == pytest.approx((3.0, 4.0))


def test_origin_surveyed_requires_lat_lon() -> None:
    with pytest.raises(Exception, match="surveyed"):
        parse_survey_origin({"surveyed": True, "e_m": 0, "n_m": 0})


def test_keep_in_is_metres_from_origin() -> None:
    origin = SurveyOrigin(e_m=2.0, n_m=1.0, lat_deg=None, lon_deg=None)
    profile = YardProfile(
        name="peg",
        keep_in=list(KEEP_IN_LOCAL),
        home={"x": 1.5, "y": 1.5, "theta": 0.0},
        origin=origin,
        width_m=12.0,
        height_m=10.0,
    )
    world = profile.keep_in_world()
    assert world[0] == pytest.approx((3.0, 2.0))
    assert world[1] == pytest.approx((9.0, 2.0))
    home = profile.home_pose()
    assert home.x == pytest.approx(3.5)
    assert home.y == pytest.approx(2.5)
    spec = profile.geofence_spec()
    assert spec.origin is not None
    assert spec.origin["e_m"] == pytest.approx(2.0)


def test_profile_roundtrip_keeps_origin(tmp_path) -> None:
    profile = YardProfile(
        name="anchor",
        keep_in=list(KEEP_IN_LOCAL),
        origin=SurveyOrigin(e_m=0.5, n_m=-0.25, lat_deg=-33.8, lon_deg=151.2, surveyed=True),
    )
    dest = tmp_path / "yard.json"
    dest.write_text(__import__("json").dumps(profile.as_dict(), indent=2), encoding="utf-8")
    loaded = parse_yard_profile(__import__("json").loads(dest.read_text(encoding="utf-8")))
    assert loaded.origin.surveyed is True
    assert loaded.origin.lat_deg == pytest.approx(-33.8)
    assert loaded.origin.e_m == pytest.approx(0.5)
    assert loaded.keep_in[0] == (1.0, 1.0)


def test_gnss_enu_roundtrip() -> None:
    origin = {"e_m": 4.0, "n_m": 2.0, "u_m": 0.5}
    world = np.array([6.0, 5.0, 1.5, 1.0], dtype=np.float32)
    enu = gps_enu_from_world(world, origin)
    assert enu[0] == pytest.approx(2.0)
    assert enu[1] == pytest.approx(3.0)
    assert enu[2] == pytest.approx(1.0)
    assert enu[3] == pytest.approx(1.0)
    back = gps_world_from_enu(enu, origin)
    np.testing.assert_allclose(back, world, atol=1e-5)


def test_pose_and_valid_gnss_stop_short_of_tape() -> None:
    spec = GeofenceSpec(keep_in=list(KEEP_IN_LOCAL))
    # Heading +x toward the east tape at x=7 from just inside.
    pose = Pose(6.80, 4.0, 0.0)
    assert geofence_advice(pose, spec, slow_m=0.8, stop_m=0.28, look_ahead_m=0.55) == "stop"
    gnss_ok = geofence_advice_from_gnss(
        pose,
        spec,
        gnss_xy=(6.78, 4.0),
        gnss_valid=True,
        slow_m=0.8,
        stop_m=0.28,
        look_ahead_m=0.55,
    )
    assert gnss_ok == "stop"
    mid = Pose(4.0, 4.0, 0.0)
    assert geofence_advice_from_gnss(mid, spec, gnss_xy=(4.0, 4.0), gnss_valid=True) == "ok"


def test_invalid_gnss_falls_back_to_pose() -> None:
    spec = GeofenceSpec(keep_in=list(KEEP_IN_LOCAL))
    pose = Pose(6.80, 4.0, 0.0)
    # Stale GNSS that still says "middle of the yard" must not override pose stop.
    advice = geofence_advice_from_gnss(
        pose,
        spec,
        gnss_xy=(4.0, 4.0),
        gnss_valid=False,
        slow_m=0.8,
        stop_m=0.28,
        look_ahead_m=0.55,
    )
    assert advice == "stop"


def test_env_stops_before_keep_in_tape() -> None:
    keep_in = [(1.0, 1.0), (6.0, 1.0), (6.0, 6.0), (1.0, 6.0)]
    origin = SurveyOrigin(e_m=0.0, n_m=0.0)
    profile = YardProfile(
        name="tape",
        width_m=8.0,
        height_m=8.0,
        resolution_m=0.25,
        keep_in=keep_in,
        home={"x": 3.0, "y": 3.5, "theta": 0.0},
        origin=origin,
    )
    cfg = {
        "sensors": {"width": 16, "height": 12, "camera_count": 4, "gps": {"dropout_prob": 0.0, "horiz_noise_std_m": 0.0}},
        "world": {
            "width_m": 8.0,
            "height_m": 8.0,
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
        "planner": {"geofence_stop_m": 0.28, "geofence_slow_m": 0.80},
        "max_steps": 80,
    }
    env = MowerEnv(config=cfg)
    obs, info = env.reset(seed=3, options={"yard_profile": profile, "resize_world": False})
    assert info["gnss_valid"] in {True, False}
    assert info["survey_origin"]["frame"] == "local_enu"
    tape_x = 6.0
    max_x = env._pose.x
    crossed = False
    for _ in range(40):
        advice = str(info.get("geofence_advice") or "ok")
        if advice == "stop":
            action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            action = np.array([0.85, 0.85, 0.0], dtype=np.float32)
        obs, _r, terminated, truncated, info = env.step(action)
        max_x = max(max_x, env._pose.x)
        if env._pose.x >= tape_x:
            crossed = True
            break
        if terminated or truncated:
            break
    env.close()
    assert not crossed
    assert max_x < tape_x
    assert info["geofence_advice"] in {"slow", "stop", "ok"}


def test_offset_origin_gnss_matches_world_tape() -> None:
    origin = SurveyOrigin(e_m=2.0, n_m=1.0)
    profile = YardProfile(name="off", keep_in=list(KEEP_IN_LOCAL), origin=origin)
    spec = profile.geofence_spec()
    # Local east tape at 7 → world x = 9.
    pose = Pose(8.80, 5.0, 0.0)
    enu = gps_enu_from_world(np.array([8.80, 5.0, 0.0, 1.0], dtype=np.float32), origin.as_dict())
    world = gps_world_from_enu(enu, origin.as_dict())
    advice = geofence_advice_from_gnss(
        pose,
        spec,
        gnss_xy=(float(world[0]), float(world[1])),
        gnss_valid=True,
        stop_m=0.28,
        look_ahead_m=0.55,
    )
    assert advice == "stop"
    assert float(enu[0]) == pytest.approx(6.80)

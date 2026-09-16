"""YardProfile schema: load, save, validate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jims_mower.constants import SURVEY_SCHEMA, YARD_PROFILE_SCHEMA
from jims_mower.geofence import GeofenceSpec
from jims_mower.yard_profile import (
    YardProfileError,
    default_yard_profile,
    load_yard_profile,
    save_yard_profile,
    validate_yard_profile,
    yard_profile_from_geofence,
    yard_profile_from_survey,
)


def test_default_roundtrip(tmp_path: Path) -> None:
    profile = default_yard_profile()
    dest = tmp_path / "yard.json"
    save_yard_profile(profile, dest)
    loaded = load_yard_profile(dest)
    assert loaded.schema == YARD_PROFILE_SCHEMA
    assert loaded.name == "example_yard"
    assert loaded.home["x"] == pytest.approx(2.0)
    assert len(loaded.keep_in) == 4
    assert len(loaded.keep_out[0]) == 4
    assert loaded.radio["primary"] == "lora"
    assert loaded.radio["bluetooth"] is True
    assert loaded.radio["wifi"]["enabled"] is False
    assert loaded.radio["lora"]["enabled"] is True
    assert loaded.schedule["days"] == ["mon", "wed", "fri"]
    assert loaded.schedule["timezone"] == "local"
    assert loaded.schedule["min_soc"] == pytest.approx(0.25)
    assert loaded.schedule["skip_rain"] is True
    assert "stub" not in loaded.schedule["note"]
    assert loaded.geofence_spec().has_polygons()


def test_validate_rejects_bad_schema() -> None:
    with pytest.raises(YardProfileError, match="schema"):
        validate_yard_profile({"name": "x"})
    with pytest.raises(YardProfileError, match="unsupported"):
        validate_yard_profile({"schema": "nope.v0"})


def test_validate_polygons_and_paths() -> None:
    base = default_yard_profile().as_dict()
    bad = dict(base, keep_in=[[0, 0], [1, 1]])
    with pytest.raises(YardProfileError, match="3 vertices"):
        validate_yard_profile(bad)
    with pytest.raises(YardProfileError, match="relative"):
        validate_yard_profile(dict(base, mesh="/abs/mesh.json"))
    with pytest.raises(YardProfileError, match="parent"):
        validate_yard_profile(dict(base, mesh="../secret.json"))


def test_validate_radio_and_schedule() -> None:
    base = default_yard_profile().as_dict()
    with pytest.raises(YardProfileError, match="primary"):
        validate_yard_profile(dict(base, radio={**base["radio"], "primary": "zigbee"}))
    with pytest.raises(YardProfileError, match="HH:MM"):
        validate_yard_profile(dict(base, schedule={**base["schedule"], "start_local": "9am"}))
    with pytest.raises(YardProfileError, match="days"):
        validate_yard_profile(dict(base, schedule={**base["schedule"], "days": ["funday"]}))
    with pytest.raises(YardProfileError, match="IANA"):
        validate_yard_profile(dict(base, schedule={**base["schedule"], "timezone": "Nope/Zone"}))
    with pytest.raises(YardProfileError, match="min_soc"):
        validate_yard_profile(dict(base, schedule={**base["schedule"], "min_soc": 4}))


def test_from_geofence_and_survey() -> None:
    spec = GeofenceSpec(keep_in=[(0, 0), (4, 0), (4, 3), (0, 3)], keep_out=[[(1, 1), (2, 1), (2, 2), (1, 2)]])
    profile = yard_profile_from_geofence(spec, name="gf", width_m=5, height_m=4)
    assert profile.keep_in[0] == (0.0, 0.0)
    assert profile.home["x"] == 0.0
    survey = {
        "schema": SURVEY_SCHEMA,
        "name": "surveyed",
        "geofence": [[1, 1], [8, 1], [8, 6], [1, 6]],
        "width_m": 9,
        "height_m": 7,
    }
    lifted = yard_profile_from_survey(survey)
    assert lifted.name == "surveyed"
    assert len(lifted.keep_in) == 4


def test_example_file_loads() -> None:
    repo = Path(__file__).resolve().parents[1] / "configs" / "yards" / "example_profile.json"
    profile = load_yard_profile(repo)
    assert profile.mesh in {"yard.glb", "maps/example_yard.mesh.json"} or profile.mesh.endswith(".glb")
    raw = json.loads(repo.read_text(encoding="utf-8"))
    assert validate_yard_profile(raw)["schema"] == YARD_PROFILE_SCHEMA

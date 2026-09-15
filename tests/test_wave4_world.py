"""Seasonal overlays, survey import, new yards, curriculum."""

from __future__ import annotations

from pathlib import Path

from jims_mower.config import load_config
from jims_mower.curriculum_sched import load_curriculum, next_stage, stage_names
from jims_mower.env import MowerEnv
from jims_mower.overlays import apply_overlay, apply_season
from jims_mower.scenarios import list_scenarios, load_source
from jims_mower.survey import SurveyError, load_survey


def test_new_scenarios_registered() -> None:
    names = list_scenarios()
    for name in ("flat", "narrow_gate", "fence_line", "property_scale"):
        assert name in names


def test_property_scale_is_coarse_and_large() -> None:
    cfg, _ = load_source("property_scale")
    assert cfg.world.width_m >= 40.0
    assert cfg.world.resolution_m >= 0.25


def test_narrow_gate_has_keepouts() -> None:
    _cfg, scen = load_source("narrow_gate")
    assert scen is not None
    assert len(scen.keepout) >= 2


def test_seasonal_long_grass() -> None:
    cfg = load_config({"world": {"season": "long_grass"}})
    assert cfg.world.grass.enabled is True
    assert cfg.world.grass.regenerate_frac >= 0.22
    apply_season(cfg, "leaf_clutter")
    assert cfg.world.n_toys >= 8


def test_apply_overlay_file() -> None:
    cfg = load_config()
    apply_overlay(cfg, "long_grass")
    assert cfg.world.season == "long_grass"


def test_survey_import() -> None:
    scen = load_survey(Path("configs/surveys/example_yard.json"))
    assert scen.name == "example_yard"
    assert len(scen.geofence) >= 3
    assert len(scen.drains) >= 2


def test_survey_rejects_missing_schema() -> None:
    try:
        load_survey({"name": "x", "geofence": [[0, 0], [1, 0], [1, 1]]})
        raise AssertionError("expected SurveyError")
    except SurveyError:
        pass


def test_curriculum_order() -> None:
    payload = load_curriculum()
    assert stage_names(payload) == ["flat", "suburban", "wet", "night"]
    nxt = next_stage("flat", payload)
    assert nxt is not None and nxt["scenario"] == "suburban"
    assert next_stage("night", payload) is None


def test_flat_and_gate_env_reset() -> None:
    for name in ("flat", "narrow_gate"):
        cfg, scen = load_source(name)
        cfg.sensors.camera_count = 4
        cfg.sensors.cameras = []
        cfg.sensors.width = 16
        cfg.sensors.height = 12
        env = MowerEnv(config=cfg, scenario=scen, render_mode=None)
        obs, info = env.reset(seed=0)
        assert "occupancy" in obs
        env.close()


def test_extrinsics_yaml_loads() -> None:
    cfg = load_config("configs/orin/extrinsics_6cam.yaml")
    cams = cfg.resolved_cameras()
    assert len(cams) == 6
    assert {c.name for c in cams} >= {"front", "rear", "left", "right"}

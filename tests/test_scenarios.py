"""WAVE 1C scenario loaders plus WAVE 1A DSL (geofence, weather, extras)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jims_mower.config import ConfigError, load_config
from jims_mower.constants import WORLD_LAYOUTS
from jims_mower.env import MowerEnv
from jims_mower.safety import in_yard, point_in_polygon
from jims_mower.scenarios import (
    ScenarioError,
    list_scenarios,
    load_dsl_scenario,
    load_scenario,
    load_source,
    looks_like_scenario,
    parse_scenario,
    scenario_path,
)
from jims_mower.types import Pose

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = ROOT / "configs" / "scenarios"

REQUIRED = (
    "suburban",
    "rural_paddock",
    "playground",
    "orchard",
    "terrace",
    "kerb_gutter",
    "wet_swale",
    "night_porch",
)

DSL_EXTRAS = ("paddock", "night_dawn", "wet_slope", "gradient_yard")


def _shrink(cfg):
    cfg.sensors.width = 16
    cfg.sensors.height = 12
    cfg.sensors.camera_count = 4
    cfg.sensors.cameras = []
    return cfg


def test_at_least_six_named_scenarios() -> None:
    names = list_scenarios()
    assert len(names) >= 6
    for name in REQUIRED:
        assert name in names
        assert scenario_path(name).is_file()
    for name in DSL_EXTRAS:
        assert name in names
        assert scenario_path(name).is_file()


def test_all_scenario_yamls_load() -> None:
    for name in list_scenarios():
        cfg = load_scenario(name)
        assert cfg.world.layout in WORLD_LAYOUTS
        assert cfg.weather.pack in {"clear", "dawn", "dusk", "night", "rain"}


def test_load_scenario_missing() -> None:
    with pytest.raises(ConfigError):
        load_scenario("not-a-real-yard")


def test_demo_config_path_is_load_config() -> None:
    path = Path("configs/scenarios/suburban.yaml")
    cfg = load_config(path)
    assert cfg.world.layout == "suburban"
    assert cfg.world.n_hoses >= 1


def test_suburban_and_playground_reset() -> None:
    sub = MowerEnv(config=_shrink(load_scenario("suburban")))
    obs, info = sub.reset(seed=4)
    kinds = [o.kind for o in sub._yard.obstacles]
    assert kinds.count("hose") >= 1
    assert info["layout"] == "suburban"
    assert "front" in obs["cameras"]
    sub.close()

    play = MowerEnv(config=_shrink(load_scenario("playground")))
    _, info = play.reset(seed=5)
    toys = [o for o in play._yard.obstacles if o.kind == "toy"]
    cords = [o for o in play._yard.obstacles if o.kind == "cord"]
    assert len(toys) >= 6
    assert len(cords) >= 1
    assert info["layout"] == "playground"
    play.close()


def test_orchard_and_wet_swale_reset() -> None:
    orchard = MowerEnv(config=_shrink(load_scenario("orchard")))
    _, info = orchard.reset(seed=6)
    trees = [o for o in orchard._yard.obstacles if o.kind == "tree"]
    ys = {round(o.y, 1) for o in trees}
    assert len(trees) >= 6
    assert len(ys) >= 2
    assert info["weather_pack"] == "dawn"
    orchard.close()

    swale = MowerEnv(config=_shrink(load_scenario("wet_swale")))
    _, info = swale.reset(seed=7)
    assert info["n_puddles"] >= 1
    assert info["n_drains"] >= 1
    assert info["layout"] == "swale"
    assert swale._appearance.wet_specular is True
    swale.close()


def test_terrace_and_kerb_structured_terrain() -> None:
    terrace = MowerEnv(config=_shrink(load_scenario("terrace")))
    _, info = terrace.reset(seed=8)
    assert info["n_banks"] >= 2
    terrace.close()

    kerb = MowerEnv(config=_shrink(load_scenario("kerb_gutter")))
    _, info = kerb.reset(seed=9)
    assert info["n_drains"] >= 1
    assert any(d.kind == "gutter" for d in kerb._terrain.drains)
    kerb.close()


def test_bundled_dsl_scenarios_registered() -> None:
    names = set(list_scenarios())
    assert set(DSL_EXTRAS) <= names
    assert "schema" not in names


def test_each_dsl_scenario_loads() -> None:
    for name in DSL_EXTRAS:
        scn = load_dsl_scenario(name)
        assert scn.name == name
        assert scn.config.world.width_m > 0
        env = MowerEnv(config=scn, render_mode=None)
        obs, info = env.reset(seed=0)
        assert "cameras" in obs
        assert info["scenario"] == name
        assert "weather" in info
        env.close()


def test_looks_like_scenario_vs_env_yaml() -> None:
    assert looks_like_scenario({"kind": "scenario", "name": "x"})
    assert looks_like_scenario({"name": "x", "weather": {"wet": True}})
    assert not looks_like_scenario({"dt": 0.1, "sensors": {"camera_count": 4}})


def test_load_source_env_yaml_unchanged() -> None:
    cfg, scn = load_source(ROOT / "configs" / "steep_yard.yaml")
    assert scn is None
    assert cfg.world.terrain.n_drains >= 1


def test_load_source_1c_suburban_is_envconfig() -> None:
    cfg, scn = load_source("suburban")
    assert scn is None
    assert cfg.world.layout == "suburban"


def test_gradient_yard_tilts_and_crosses_drain() -> None:
    scn = load_dsl_scenario("gradient_yard")
    assert scn.name == "gradient_yard"
    assert scn.config.world.terrain.base_gradient.effective_slope_rad() > 0.10
    assert len(scn.drains) == 1
    env = MowerEnv(config=_shrink(scn.config), scenario=scn, render_mode=None)
    _, info = env.reset(seed=3)
    assert info["n_drains"] >= 1
    low = env._terrain.sample(1.5, 6.0)
    high = env._terrain.sample(10.5, 6.0)
    assert high > low + 0.8
    env.close()


def test_explicit_drain_is_carved() -> None:
    scn = load_dsl_scenario("paddock")
    env = MowerEnv(config=scn, render_mode=None)
    env.reset(seed=1)
    assert len(env._terrain.drains) >= 1
    labels = env.oracle_labels()
    assert int((labels["hazard"] >= 2).sum()) > 0
    env.close()


def test_paddock_places_authored_trees() -> None:
    scn = load_dsl_scenario("paddock")
    trees = [o for o in scn.obstacles if o.kind == "tree"]
    assert len(trees) >= 2
    env = MowerEnv(config=scn, render_mode=None)
    env.reset(seed=2)
    kinds = [o.kind for o in env._yard.obstacles]
    assert kinds.count("tree") >= 2
    env.close()


def test_rejects_bad_weather() -> None:
    with pytest.raises(ScenarioError):
        parse_scenario({"name": "x", "weather": {"lighting": "noon"}})
    with pytest.raises(ScenarioError):
        parse_scenario({"name": "x", "weather": {"night": True, "dawn": True}})


def test_rejects_short_geofence() -> None:
    with pytest.raises(ScenarioError):
        parse_scenario({"name": "x", "geofence": [[0, 0], [1, 0]]})


def test_rejects_unknown_obstacle() -> None:
    with pytest.raises(ScenarioError):
        parse_scenario({"name": "x", "obstacles": [{"kind": "dragon", "x": 1, "y": 1}]})


def test_missing_name() -> None:
    with pytest.raises(ScenarioError):
        parse_scenario({"kind": "scenario", "weather": {"wet": True}})


def test_geofence_point_in_polygon() -> None:
    square = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
    assert point_in_polygon(2.0, 2.0, square)
    assert not point_in_polygon(5.0, 2.0, square)
    pose = Pose(2.0, 2.0, 0.0)
    assert in_yard(pose, 8.0, 8.0, 0.2, geofence=square)
    assert not in_yard(Pose(6.0, 6.0, 0.0), 8.0, 8.0, 0.2, geofence=square)


def test_dawn_cameras_darker_than_day() -> None:
    base = {
        "kind": "scenario",
        "name": "lit",
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
        "sensors": {"width": 32, "height": 24, "camera_count": 4},
    }
    day = parse_scenario({**base, "weather": {"lighting": "day"}})
    dawn = parse_scenario({**base, "weather": {"lighting": "dawn"}})
    env_day = MowerEnv(config=day, render_mode=None)
    env_dawn = MowerEnv(config=dawn, render_mode=None)
    obs_day, _ = env_day.reset(seed=4)
    obs_dawn, info_dawn = env_dawn.reset(seed=4)
    assert info_dawn["weather"]["dawn"] is True
    mean_day = float(np.mean(obs_day["cameras"]["front"]))
    mean_dawn = float(np.mean(obs_dawn["cameras"]["front"]))
    assert mean_dawn < mean_day * 0.85
    env_day.close()
    env_dawn.close()


def test_wet_slope_flag() -> None:
    scn = load_dsl_scenario("wet_slope")
    assert scn.weather.wet is True
    assert len(scn.banks) >= 1
    _, info = MowerEnv(config=scn).reset(seed=3)
    assert info["weather"]["wet"] is True


def test_scenario_yaml_files_exist() -> None:
    for name in list(REQUIRED) + list(DSL_EXTRAS):
        assert (SCENARIO_DIR / f"{name}.yaml").is_file()
    assert (SCENARIO_DIR / "schema.yaml").is_file()

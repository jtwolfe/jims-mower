"""WAVE 1C scenario loaders and a couple of env resets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jims_mower.config import ConfigError, load_config
from jims_mower.constants import WORLD_LAYOUTS
from jims_mower.env import MowerEnv
from jims_mower.scenarios import list_scenarios, load_scenario, scenario_path


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

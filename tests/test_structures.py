"""Path / building / bunker layers and golf scenarios."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jims_mower.constants import (
    BUNKER_RGB,
    PATH_RGB,
    STRUCTURE_BUILDING,
    STRUCTURE_BUNKER,
    STRUCTURE_PATH,
    TERRAIN_BUNKER,
    TERRAIN_PATH,
    WORLD_LAYOUTS,
)
from jims_mower.env import MowerEnv
from jims_mower.perception.cv_terrain import classify_structure_rgb
from jims_mower.planning.costmap import BLOCKED_COST, build_costmap
from jims_mower.scenarios import list_scenarios, load_dsl_scenario, parse_scenario
from jims_mower.structures import (
    BunkerFeature,
    PathFeature,
    PolygonFeature,
    apply_bunkers,
    apply_paths,
    apply_polygons,
    rasterize_polyline,
)
from jims_mower.terrain import HeightField, apply_dem_npy, bundled_dem_path


def test_golf_scenarios_registered() -> None:
    names = list_scenarios()
    assert "golf_rough" in names
    assert "golf_fairway_snip" in names
    assert "golf_rough" in WORLD_LAYOUTS
    assert "golf_fairway" in WORLD_LAYOUTS


def test_golf_rough_has_uneven_mesh_and_path() -> None:
    scn = load_dsl_scenario("golf_rough")
    assert scn.paths
    assert scn.bunkers
    assert scn.buildings
    assert scn.greens
    assert scn.garden_beds
    scn.config.sensors.width = 16
    scn.config.sensors.height = 12
    scn.config.sensors.camera_count = 4
    scn.config.sensors.cameras = []
    env = MowerEnv(config=scn, render_mode=None)
    obs, info = env.reset(seed=3)
    elev = env._terrain.elevation
    assert float(elev.max() - elev.min()) > 0.20
    assert int((env._terrain.labels == TERRAIN_PATH).sum()) > 20
    assert int((env._terrain.labels == TERRAIN_BUNKER).sum()) > 10
    assert int((obs["structure"] == STRUCTURE_PATH).sum()) > 10
    grass = env._coverage.grass
    assert not bool(np.all(grass[env._terrain.labels == TERRAIN_PATH]))
    env.close()
    del info


def test_golf_fairway_snip_blocks_green_and_path() -> None:
    scn = load_dsl_scenario("golf_fairway_snip")
    assert scn.greens
    scn.config.sensors.width = 16
    scn.config.sensors.height = 12
    scn.config.sensors.camera_count = 4
    scn.config.sensors.cameras = []
    env = MowerEnv(config=scn, render_mode=None)
    obs, _info = env.reset(seed=2)
    struct = obs["structure"]
    assert int((struct == STRUCTURE_PATH).sum()) > 8
    assert int((struct == STRUCTURE_BUNKER).sum()) > 8
    env.close()


def test_path_polyline_yaml() -> None:
    scn = parse_scenario(
        {
            "name": "path_yard",
            "kind": "scenario",
            "paths": [{"vertices": [[1, 1], [4, 1], [4, 3]], "width_m": 0.8}],
            "buildings": [{"vertices": [[0.5, 3.5], [1.5, 3.5], [1.5, 4.5], [0.5, 4.5]]}],
            "bunkers": [{"x": 2.5, "y": 2.5, "radius_m": 0.6}],
            "world": {"width_m": 6.0, "height_m": 6.0, "layout": "random"},
        }
    )
    assert len(scn.paths) == 1
    assert scn.paths[0].width_m == pytest.approx(0.8)
    assert len(scn.buildings) == 1
    assert scn.bunkers[0].radius_m == pytest.approx(0.6)


def test_costmap_blocks_building_and_costs_path() -> None:
    n = 16
    hazard = np.zeros((n, n), dtype=np.float32)
    slope = np.zeros((n, n), dtype=np.float32)
    struct = np.zeros((n, n), dtype=np.uint8)
    struct[4, :] = STRUCTURE_PATH
    struct[10:13, 10:13] = STRUCTURE_BUILDING
    struct[2, 2] = STRUCTURE_BUNKER
    cm = build_costmap(
        hazard,
        slope,
        resolution_m=0.2,
        width_m=3.2,
        height_m=3.2,
        max_climb_slope_rad=0.4,
        drain_clearance_m=0.0,
        margin_m=0.0,
        structure=struct,
        path_cost=8.0,
        bunker_cost=12.0,
    )
    assert cm.cost[4, 8] == pytest.approx(8.0)
    assert cm.blocked[11, 11]
    assert not np.isfinite(cm.cost[11, 11]) or cm.cost[11, 11] == BLOCKED_COST
    assert cm.blocked[2, 2]


def test_apply_structures_on_heightfield() -> None:
    hf = HeightField.empty(6.0, 6.0, 0.2)
    apply_paths(
        hf.labels,
        hf.elevation,
        hf.resolution_m,
        [PathFeature(vertices=((0.5, 1.0), (5.0, 1.0)), width_m=0.6)],
    )
    apply_polygons(
        hf.labels,
        hf.elevation,
        hf.resolution_m,
        [PolygonFeature(vertices=((4.0, 4.0), (5.4, 4.0), (5.4, 5.4), (4.0, 5.4)), kind="building")],
    )
    apply_bunkers(
        hf.labels,
        hf.elevation,
        hf.resolution_m,
        [BunkerFeature(x=2.0, y=3.5, radius_m=0.8, depth_m=0.2)],
    )
    assert int((hf.labels == TERRAIN_PATH).sum()) > 5
    assert int((hf.labels == TERRAIN_BUNKER).sum()) > 5


def test_classify_path_and_bunker_swatches() -> None:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[:4] = PATH_RGB
    img[4:] = BUNKER_RGB
    labels = classify_structure_rgb(img)
    assert int((labels[:4] == STRUCTURE_PATH).mean()) > 0.7
    assert int((labels[4:] == STRUCTURE_BUNKER).mean()) > 0.7


def test_dem_npy_hook(tmp_path: Path) -> None:
    hf = HeightField.empty(4.0, 4.0, 0.5)
    patch = np.linspace(-0.2, 0.2, 8 * 8, dtype=np.float32).reshape(8, 8)
    dest = tmp_path / "tiny_dem.npy"
    np.save(dest, patch)
    before = hf.elevation.copy()
    apply_dem_npy(hf, dest)
    assert hf.elevation.shape == before.shape
    assert float(np.abs(hf.elevation - before).max()) > 0.05


def test_bundled_dem_fixture() -> None:
    dest = bundled_dem_path()
    assert dest.is_file()
    hf = HeightField.empty(6.0, 6.0, 0.5)
    before = hf.elevation.copy()
    apply_dem_npy(hf, dest)
    assert hf.elevation.shape == before.shape
    assert float(np.abs(hf.elevation - before).max()) > 0.02
    assert abs(float(hf.elevation.mean() - before.mean())) < 0.02


def test_rasterize_polyline_width() -> None:
    mask = rasterize_polyline((20, 20), 0.2, [(0.4, 2.0), (3.6, 2.0)], 0.6)
    assert int(mask.sum()) > 10
    assert bool(mask[10, 10])

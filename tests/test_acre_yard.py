"""One-acre explore/mow yard: load, area, layers, cheap smoke."""

from __future__ import annotations

from jims_mower.constants import (
    STRUCTURE_BUILDING,
    STRUCTURE_BUNKER,
    STRUCTURE_GARDEN,
    STRUCTURE_PATH,
    STRUCTURE_POND,
    TERRAIN_POND,
    WORLD_LAYOUTS,
)
from jims_mower.env import MowerEnv
from jims_mower.mesh import mesh_from_env
from jims_mower.scenarios import list_scenarios, load_dsl_scenario, load_source

ACRE_M2 = 4046.8564224
ACRE_TOL = 0.05


def _shrink(scn):
    scn.config.sensors.width = 16
    scn.config.sensors.height = 12
    scn.config.sensors.camera_count = 4
    scn.config.sensors.cameras = []
    return scn


def test_acre_yard_registered() -> None:
    names = list_scenarios()
    assert "acre_yard" in names
    assert "acre_yard_demo" in names
    assert "acre_yard" in WORLD_LAYOUTS


def test_acre_yard_is_about_one_acre() -> None:
    cfg, scn = load_source("acre_yard")
    assert scn is not None
    area_m2 = cfg.world.width_m * cfg.world.height_m
    acres = area_m2 / ACRE_M2
    assert abs(acres - 1.0) <= ACRE_TOL
    n_cells = int(round(cfg.world.width_m / cfg.world.resolution_m)) * int(
        round(cfg.world.height_m / cfg.world.resolution_m)
    )
    assert n_cells <= 20_000
    assert cfg.world.resolution_m >= 0.40


def test_acre_yard_authors_property_mix() -> None:
    scn = load_dsl_scenario("acre_yard")
    trees = [o for o in scn.obstacles if o.kind == "tree"]
    assert len(trees) >= 8
    assert scn.config.world.n_trees == 0
    assert scn.garden_beds
    assert scn.bunkers and len(scn.bunkers) >= 2
    assert scn.paths
    assert scn.buildings and len(scn.buildings) >= 1
    assert scn.ponds
    assert scn.drains
    assert scn.banks
    assert scn.geofence and len(scn.geofence) >= 4
    assert scn.keepout
    pond_verts = list(scn.ponds[0].vertices)
    assert pond_verts in [list(p) for p in scn.keepout]
    assert scn.config.world.terrain.multi_scale_amp_m > 0.0
    assert scn.config.planner.strip_spacing_m >= 1.0


def test_acre_yard_env_smoke() -> None:
    scn = _shrink(load_dsl_scenario("acre_yard"))
    env = MowerEnv(config=scn, render_mode=None)
    obs, info = env.reset(seed=3)
    assert info["scenario"] == "acre_yard"
    assert info["layout"] == "acre_yard"
    labels = env._terrain.labels
    struct = obs["structure"]
    assert int((labels == TERRAIN_POND).sum()) > 8
    assert int((struct == STRUCTURE_POND).sum()) > 8
    assert int((struct == STRUCTURE_PATH).sum()) > 8
    assert int((struct == STRUCTURE_BUILDING).sum()) > 4
    assert int((struct == STRUCTURE_BUNKER).sum()) > 4
    assert int((struct == STRUCTURE_GARDEN).sum()) > 4
    assert env._terrain.puddles or info["n_puddles"] >= 1
    assert info["n_drains"] >= 1
    assert info["n_banks"] >= 1
    grass = env._coverage.grass
    assert not bool(grass[labels == TERRAIN_POND].any())
    elev = env._terrain.elevation
    assert float(elev.max() - elev.min()) > 0.25
    for _ in range(3):
        obs, _reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert "cameras" in obs
        if terminated or truncated:
            break
    mesh = mesh_from_env(env, stride=4)
    assert not mesh.is_empty()
    assert mesh.vertex_count >= 16
    env.close()


def test_acre_yard_demo_is_same_acre_with_faster_calibrate() -> None:
    cfg, scn = load_source("acre_yard_demo")
    full, _full_scn = load_source("acre_yard")
    assert scn is not None
    assert cfg.world.width_m == full.world.width_m
    assert cfg.world.height_m == full.world.height_m
    assert cfg.world.layout == "acre_yard"
    assert scn.ponds and scn.buildings
    assert cfg.mission.calibrate_confirm_m >= 20.0
    assert cfg.mission.max_calibrate_steps < full.mission.max_calibrate_steps
    assert cfg.mission.explore_complete < full.mission.explore_complete
    assert cfg.mission.explore_complete <= 0.45
    assert cfg.mission.max_explore_steps <= 1400
    area = (cfg.mission.calibrate_confirm_m or 0.0) < 80.0
    assert area
    # Tighter keep-in than the full fence, still covers pond + sheds.
    xs = [p[0] for p in scn.geofence]
    ys = [p[1] for p in scn.geofence]
    assert min(xs) > 1.6 and max(xs) < 68.4
    assert min(ys) > 1.6 and max(ys) < 56.4

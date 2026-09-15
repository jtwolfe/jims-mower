"""Terrain policy on the steep-yard config stays out of drains / tips."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from jims_mower.config import load_config
from jims_mower.constants import HAZARD_DRAIN
from jims_mower.env import MowerEnv
from jims_mower.planning import TerrainPolicy, build_costmap, plan_coverage


def _steep_cfg() -> dict:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "steep_yard.yaml")
    # Keep the drain/bank field; drop walkers so a person collision is not
    # this test's concern.
    cfg.world.n_people = 0
    cfg.world.n_dogs = 0
    cfg.world.n_cats = 0
    cfg.world.n_birds = 0
    cfg.max_steps = 80
    return cfg


def test_plan_from_steep_yard_obs_avoids_channels() -> None:
    env = MowerEnv(config=_steep_cfg(), render_mode=None)
    obs, info = env.reset(seed=5)
    policy = TerrainPolicy(env.cfg)
    plan = policy.reset(obs, info)
    assert plan.waypoints
    hazard = obs["hazard"]
    res = env.cfg.world.resolution_m
    for x, y in plan.waypoints:
        cell = (int(y / res), int(x / res))
        if 0 <= cell[0] < hazard.shape[0] and 0 <= cell[1] < hazard.shape[1]:
            assert float(hazard[cell]) != float(HAZARD_DRAIN)
    env.close()


def test_terrain_policy_no_drain_or_tip_on_steep_yard() -> None:
    env = MowerEnv(config=_steep_cfg(), render_mode=None)
    obs, info = env.reset(seed=5)
    policy = TerrainPolicy(env.cfg)
    policy.reset(obs, info)
    coverage0 = float(info["coverage_fraction"])
    for _ in range(80):
        action = policy.act(obs, info)
        obs, _reward, terminated, truncated, info = env.step(action)
        assert not info["drain_drop"]
        assert not info["tipover"]
        if terminated or truncated:
            break
    assert float(info["coverage_fraction"]) >= coverage0
    assert policy.last_advice in {"ok", "slow", "reroute", "stop"}
    env.close()


def test_build_costmap_from_env_hazard() -> None:
    env = MowerEnv(config=_steep_cfg(), render_mode=None)
    obs, _info = env.reset(seed=2)
    cm = build_costmap(
        obs["hazard"],
        obs["slope"],
        resolution_m=env.cfg.world.resolution_m,
        width_m=env.cfg.world.width_m,
        height_m=env.cfg.world.height_m,
        max_climb_slope_rad=env.cfg.planner.max_climb_slope_rad,
        drain_clearance_m=env.cfg.planner.drain_clearance_m,
        occupancy=obs["occupancy"],
        occupancy_inflate_m=env.cfg.planner.occupancy_inflate_m,
        margin_m=env.cfg.robot.collision_radius_m,
    )
    assert np.any(cm.blocked)
    pose = obs["pose"]
    plan = plan_coverage(cm, (float(pose[0]), float(pose[1])))
    for row, col in plan.cells:
        assert not cm.blocked[row, col]
    env.close()

"""Heuristic CV observer + planner stays out of channels on fixed seeds.

These are episode metrics (drain_drop / coverage), not detector mAP.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from jims_mower.config import load_config
from jims_mower.constants import HAZARD_DRAIN, HAZARD_DRAIN_EDGE, TERRAIN_DRAIN
from jims_mower.env import MowerEnv
from jims_mower.perception import HeuristicTerrainObserver
from jims_mower.planning import TerrainPolicy


def _steep_cfg(*, mode: str, max_steps: int = 80):
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "steep_yard.yaml")
    cfg.world.n_people = 0
    cfg.world.n_dogs = 0
    cfg.world.n_cats = 0
    cfg.world.n_birds = 0
    cfg.max_steps = max_steps
    cfg.perception.terrain_mode = mode
    return cfg


def _true_channel_hits(env: MowerEnv, waypoints: list[tuple[float, float]]) -> int:
    res = env.cfg.world.resolution_m
    hits = 0
    for x, y in waypoints:
        cell = env._terrain.world_to_cell(x, y)
        if cell is None:
            continue
        if int(env._terrain.labels[cell]) == TERRAIN_DRAIN:
            hits += 1
    return hits


def _run_policy(mode: str, seed: int, steps: int = 80) -> dict:
    env = MowerEnv(config=_steep_cfg(mode=mode, max_steps=steps), render_mode=None)
    if mode == "heuristic":
        env.terrain_observer = HeuristicTerrainObserver()
    obs, info = env.reset(seed=seed)
    policy = TerrainPolicy(env.cfg)
    plan = policy.reset(obs, info)
    coverage0 = float(info["coverage_fraction"])
    drain_drop = False
    tipover = False
    for _ in range(steps):
        action = policy.act(obs, info)
        obs, _reward, terminated, truncated, info = env.step(action)
        drain_drop = drain_drop or bool(info["drain_drop"])
        tipover = tipover or bool(info["tipover"])
        if terminated or truncated:
            break
    out = {
        "source": info.get("terrain_source"),
        "coverage0": coverage0,
        "coverage": float(info["coverage_fraction"]),
        "drain_drop": drain_drop,
        "tipover": tipover,
        "plan_channel_hits": _true_channel_hits(env, plan.waypoints),
        "n_waypoints": len(plan.waypoints),
        "hazard_drain_cells": int((obs["hazard"] >= HAZARD_DRAIN).sum()),
    }
    env.close()
    return out


def test_heuristic_sees_drains_on_steep_yard_reset() -> None:
    env = MowerEnv(config=_steep_cfg(mode="heuristic"), render_mode=None)
    obs, info = env.reset(seed=5)
    assert info["terrain_source"] == "heuristic"
    assert int((obs["hazard"] >= HAZARD_DRAIN_EDGE).sum()) > 0
    env.close()


def test_heuristic_plan_skips_observed_channels() -> None:
    """First plan avoids cells the cameras marked. Unseen ditch ends are OK.

    Completeness vs the god-view field is not required at spawn; the episode
    tests check that the robot still does not drive into a channel.
    """
    env = MowerEnv(config=_steep_cfg(mode="heuristic"), render_mode=None)
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
    true_ch = env._terrain.labels == TERRAIN_DRAIN
    recall = float(((hazard >= HAZARD_DRAIN_EDGE) & true_ch).sum()) / max(1, int(true_ch.sum()))
    assert recall >= 0.5
    env.close()


def test_heuristic_policy_no_channel_entry_seed5() -> None:
    result = _run_policy("heuristic", seed=5, steps=80)
    assert result["source"] == "heuristic"
    assert result["drain_drop"] is False
    assert result["tipover"] is False
    assert result["coverage"] >= result["coverage0"]


def test_heuristic_policy_no_channel_entry_seed7() -> None:
    result = _run_policy("heuristic", seed=7, steps=80)
    assert result["drain_drop"] is False
    assert result["tipover"] is False
    assert result["coverage"] >= result["coverage0"]


def test_heuristic_coverage_vs_oracle_no_mAP_claim() -> None:
    """Both stay out of channels. Coverage is reported, not scored as mAP."""
    heur = _run_policy("heuristic", seed=5, steps=60)
    ora = _run_policy("oracle", seed=5, steps=60)
    assert heur["drain_drop"] is False and ora["drain_drop"] is False
    assert heur["source"] == "heuristic"
    assert ora["source"] == "oracle"
    # Heuristic may cut less (partial map). Both must still cut some grass
    # or at least not lose coverage. No detector-accuracy number here.
    assert heur["coverage"] >= heur["coverage0"]
    assert ora["coverage"] >= ora["coverage0"]
    assert heur["hazard_drain_cells"] > 0
    assert ora["hazard_drain_cells"] > 0

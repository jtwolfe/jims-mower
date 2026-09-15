"""Episode scorecards and frozen-seed gates."""

from __future__ import annotations

import json
from pathlib import Path

from jims_mower.config import load_config
from jims_mower.env import MowerEnv
from jims_mower.metrics import evaluate_episode, write_scorecard


def _flat_safe_cfg() -> dict:
    return {
        "dt": 0.1,
        "max_steps": 20,
        "sensors": {"width": 32, "height": 24, "camera_count": 4, "fov_deg": 70.0},
        "world": {
            "width_m": 8.0,
            "height_m": 8.0,
            "resolution_m": 0.20,
            "n_people": 0,
            "n_dogs": 0,
            "n_cats": 0,
            "n_birds": 0,
            "n_trees": 0,
            "n_furniture": 0,
            "n_toys": 0,
            "terrain": {"enabled": False},
        },
        "robot": {
            "trimmer": {"safety_radius_m": 1.2, "offset_m": 0.32, "radius_m": 0.16}
        },
    }


def _steep_no_people():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "steep_yard.yaml")
    cfg.world.n_people = 0
    cfg.world.n_dogs = 0
    cfg.world.n_cats = 0
    cfg.world.n_birds = 0
    cfg.max_steps = 40
    cfg.perception.terrain_mode = "oracle"
    return cfg


def test_scorecard_fields_and_json(tmp_path: Path) -> None:
    env = MowerEnv(config=_flat_safe_cfg(), render_mode=None)
    card = evaluate_episode(env, seed=11, steps=8, policy="scripted", close=True)
    assert card.seed == 11
    assert card.steps == 8
    assert card.tip_count == 0
    assert card.drain_entries == 0
    assert card.coverage_pct >= 0.0
    assert set(card.terrain_advice) == {"ok", "slow", "reroute", "stop"}
    assert sum(card.terrain_advice.values()) == 8
    path = tmp_path / "card.json"
    write_scorecard(path, card)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["tip_count"] == 0
    assert payload["policy"] == "scripted"


def test_frozen_seed_11_flat_no_tips() -> None:
    env = MowerEnv(config=_flat_safe_cfg(), render_mode=None)
    card = evaluate_episode(env, seed=11, steps=10, policy="scripted", close=True)
    assert card.tip_count == 0
    assert card.drain_entries == 0
    assert card.coverage_pct > 0.0


def test_frozen_seed_5_steep_oracle_no_drain() -> None:
    env = MowerEnv(config=_steep_no_people(), render_mode=None)
    card = evaluate_episode(env, seed=5, steps=12, policy="terrain", close=True)
    assert card.tip_count == 0
    assert card.drain_entries == 0
    assert card.coverage_pct >= 0.0
    assert card.terrain_advice["stop"] + card.terrain_advice["ok"] + card.terrain_advice[
        "slow"
    ] + card.terrain_advice["reroute"] == card.steps


def test_frozen_seed_7_steep_oracle_no_drain() -> None:
    env = MowerEnv(config=_steep_no_people(), render_mode=None)
    card = evaluate_episode(env, seed=7, steps=12, policy="terrain", close=True)
    assert card.tip_count == 0
    assert card.drain_entries == 0

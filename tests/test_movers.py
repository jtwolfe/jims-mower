"""Moving people/animals: trajectories, occupancy updates, living advice."""

from __future__ import annotations

import numpy as np
import pytest

from jims_mower.env import MowerEnv
from jims_mower.planning.controller import TerrainPolicy
from jims_mower.safety import living_advice
from jims_mower.types import Obstacle, Trajectory
from jims_mower.world import Yard, spawn_yard, step_movers


def test_patrol_reverses_at_waypoint() -> None:
    yard = Yard(width_m=10.0, height_m=10.0)
    person = Obstacle(
        "person",
        2.0,
        5.0,
        0.25,
        z=0.9,
        trajectory=Trajectory(mode="patrol", waypoints=[(2.0, 5.0), (6.0, 5.0)], speed_mps=1.0),
    )
    yard.obstacles.append(person)
    rng = np.random.default_rng(0)
    xs = []
    for _ in range(80):
        step_movers(yard, 0.1, rng)
        xs.append(person.x)
    assert max(xs) > 5.0
    # After reaching the far end the agent walks back.
    assert xs[-1] < max(xs) - 0.3


def test_loop_visits_all_corners() -> None:
    yard = Yard(width_m=10.0, height_m=10.0)
    dog = Obstacle(
        "dog",
        2.0,
        2.0,
        0.2,
        trajectory=Trajectory(
            mode="loop",
            waypoints=[(2.0, 2.0), (5.0, 2.0), (5.0, 5.0), (2.0, 5.0)],
            speed_mps=1.2,
        ),
    )
    yard.obstacles.append(dog)
    rng = np.random.default_rng(1)
    hit = [False, False, False, False]
    for _ in range(200):
        step_movers(yard, 0.1, rng)
        if dog.x > 4.5 and dog.y < 2.6:
            hit[1] = True
        if dog.x > 4.5 and dog.y > 4.5:
            hit[2] = True
        if dog.x < 2.6 and dog.y > 4.5:
            hit[3] = True
    assert hit[1] and hit[2] and hit[3]


def test_living_advice_ladder() -> None:
    assert living_advice(4.5, "person").advice == "ok"
    assert living_advice(2.5, "person").advice == "slow"
    assert living_advice(1.4, "dog").advice == "reroute"
    assert living_advice(0.4, "cat").advice == "stop"


def test_occupancy_moves_with_person() -> None:
    env = MowerEnv(
        config={
            "sensors": {"width": 16, "height": 12, "camera_count": 4},
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
        }
    )
    obs, _ = env.reset(seed=4)
    person = Obstacle(
        "person",
        2.0,
        4.0,
        0.25,
        z=0.9,
        trajectory=Trajectory(mode="line", waypoints=[(2.0, 4.0), (6.0, 4.0)], speed_mps=1.0),
    )
    env._yard.obstacles.append(person)
    obs, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
    first = obs["occupancy"].copy()
    for _ in range(12):
        obs, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
    later = obs["occupancy"]
    assert float(first.sum()) > 0.0
    assert float(later.sum()) > 0.0
    # The occupied blob shifted with the walker.
    assert not np.array_equal(first, later)
    assert info["living_advice"] in {"ok", "slow", "reroute", "stop"}
    env.close()


def test_controller_stops_for_nearby_person() -> None:
    from jims_mower.config import EnvConfig
    from jims_mower.planning.coverage import CoveragePlan

    cfg = EnvConfig()
    policy = TerrainPolicy(cfg)
    policy.fusion.reset(0.0, 0.0, 0.0)
    policy.plan = CoveragePlan(waypoints=[(3.0, 0.0)])
    policy.index = 0
    obs = {
        "pose": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "gps": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "imu": np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0], dtype=np.float32),
        "hazard": np.zeros((12, 12), dtype=np.float32),
        "slope": np.zeros((12, 12), dtype=np.float32),
        "occupancy": np.zeros((12, 12), dtype=np.float32),
        "coverage": np.zeros((12, 12), dtype=np.float32),
        "hand_signal": 0,
    }
    info = {
        "terrain_advice": "ok",
        "living_advice": "stop",
        "geofence_advice": "ok",
        "pose": {"x": 0.0, "y": 0.0, "theta": 0.0, "z": 0.0, "pitch": 0.0, "roll": 0.0},
    }
    action = policy.act(obs, info)
    assert action[0] == pytest.approx(0.0)
    assert action[1] == pytest.approx(0.0)
    assert policy.last_advice == "stop"


def test_density_sparse_skips_people() -> None:
    rng = np.random.default_rng(0)
    yard = spawn_yard(
        rng,
        10.0,
        10.0,
        {"person": 3, "dog": 0, "tree": 0},
        (5.0, 5.0, 1.0),
        density="sparse",
    )
    assert all(o.kind != "person" for o in yard.obstacles)
    assert any(o.kind == "dog" for o in yard.obstacles)

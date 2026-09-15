"""Multi-session mission save / load (map + uncut + pose)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jims_mower.env import MowerEnv
from jims_mower.maps import GrassCoverageMap
from jims_mower.mission import apply_mission, load_mission, save_mission
from jims_mower.types import Pose


def test_mission_roundtrip(tmp_path: Path) -> None:
    coverage = GrassCoverageMap(4.0, 4.0, 0.5)
    coverage.mark_circle(2.0, 2.0, 0.8)
    pose = Pose(1.2, 2.3, 0.4, 0.01, 0.0, 0.0)
    path = tmp_path / "mission.npz"
    save_mission(path, coverage, pose, scenario="suburban", seed=7, steps=12)
    assert path.is_file()
    assert path.with_name("mission.json").is_file()
    state = load_mission(path)
    assert state.schema.startswith("jims_mower.mission")
    assert state.pose.x == pytest.approx(1.2)
    assert state.pose.y == pytest.approx(2.3)
    assert state.scenario == "suburban"
    assert state.seed == 7
    assert np.array_equal(state.coverage.cut, coverage.cut)


def test_env_save_load_mission(tmp_path: Path) -> None:
    path = tmp_path / "yard_mission.npz"
    cfg = {
        "sensors": {"width": 16, "height": 12, "camera_count": 4},
        "world": {
            "width_m": 8.0,
            "height_m": 8.0,
            "resolution_m": 0.2,
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
    env = MowerEnv(config=cfg)
    env.reset(seed=8, options={"save_mission": str(path)})
    for _ in range(8):
        env.step(np.array([0.7, 0.7, 1.0], dtype=np.float32))
    cut = env._coverage.cut.copy()
    pose = (env._pose.x, env._pose.y, env._pose.theta)
    env.close()
    assert path.is_file()

    env2 = MowerEnv(config=cfg)
    _, info = env2.reset(seed=99, options={"load_mission": str(path)})
    assert info["mission_loaded"] is True
    assert np.array_equal(env2._coverage.cut, cut)
    assert env2._pose.x == pytest.approx(pose[0], abs=0.15)
    assert env2._pose.y == pytest.approx(pose[1], abs=0.15)
    env2.close()


def test_apply_mission_rejects_shape_mismatch() -> None:
    src = GrassCoverageMap(3.0, 3.0, 0.5)
    dest = GrassCoverageMap(4.0, 4.0, 0.5)
    from jims_mower.mission import MissionState

    state = MissionState("jims_mower.mission.v1", Pose(0, 0, 0), src)
    with pytest.raises(ValueError):
        apply_mission(dest, state)


def test_demo_mission_flags(tmp_path: Path) -> None:
    from jims_mower.demo import run_demo

    mission = tmp_path / "demo_mission.npz"
    run_demo(
        tmp_path / "a",
        steps=4,
        seed=5,
        cameras=4,
        policy="scripted",
        save_mission=str(mission),
    )
    assert mission.is_file()
    summary = run_demo(
        tmp_path / "b",
        steps=2,
        seed=5,
        cameras=4,
        policy="scripted",
        load_mission=str(mission),
    )
    assert summary["mission_loaded"] is True

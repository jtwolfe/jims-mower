"""Persistent occupancy, height fusion stub, loop-closure stub."""

from __future__ import annotations

import numpy as np

from jims_mower.env import MowerEnv
from jims_mower.mapping import LoopClosureStub, PersistentOccupancy, fuse_height_rgb_tof
from jims_mower.types import CameraSpec, Detection, Pose


def test_persistent_occupancy_decays() -> None:
    persist = PersistentOccupancy(10, 10, decay=0.5)
    det = Detection("person", "front", (1, 1, 4, 4), 0.9, world_xy=(1.0, 1.0))
    first = persist.update([det], resolution_m=0.2).copy()
    persist.update([], resolution_m=0.2)
    later = persist.grid
    assert float(first.max()) > 0.5
    assert float(later.max()) < float(first.max())
    persist.reset()
    assert float(persist.grid.max()) == 0.0


def test_tof_paints_local_occupancy() -> None:
    persist = PersistentOccupancy(16, 16, decay=0.2, tof_hit_m=0.20)
    pose = Pose(1.6, 1.6, 0.0)
    tof = np.array([0.08, 1.0, 1.0, 1.0], dtype=np.float32)
    grid = persist.update([], resolution_m=0.2, tof=tof, pose=pose)
    assert float(grid.max()) > 0.0


def test_height_fusion_returns_shape() -> None:
    pose = Pose(2.0, 2.0, 0.0, z=0.1)
    cam = CameraSpec("front", 0.25, 0.0, 0.38, 0.0, -22.0)
    images = {"front": np.zeros((12, 16, 3), dtype=np.uint8)}
    tof = np.array([0.12, 0.12, 0.12, 0.12], dtype=np.float32)
    elev = fuse_height_rgb_tof(
        images,
        [cam],
        pose,
        tof,
        shape=(10, 10),
        resolution_m=0.4,
        world_size=(4.0, 4.0),
    )
    assert elev.shape == (10, 10)
    assert elev.dtype == np.float32


def test_loop_closure_stub_not_slam() -> None:
    loop = LoopClosureStub(cell_m=1.0, min_step_gap=2, match=0.5)
    occ = np.zeros((12, 12), dtype=np.float32)
    occ[3:6, 3:6] = 1.0
    pose_a = Pose(0.8, 0.8, 0.0)
    pose_b = Pose(3.2, 3.2, 0.0)
    loop.update(occ, pose_a, 0.2)
    loop.update(occ, pose_b, 0.2)
    hit = loop.update(occ, pose_a, 0.2)
    info = loop.as_info()
    assert info["not_slam"] is True
    if hit is not None:
        assert hit.score >= 0.5


def test_env_emits_semantic_and_height() -> None:
    env = MowerEnv(
        config={
            "sensors": {"width": 16, "height": 12, "camera_count": 4},
            "world": {
                "width_m": 6.0,
                "height_m": 6.0,
                "resolution_m": 0.30,
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
    obs, info = env.reset(seed=1)
    assert "semantic" in info
    assert info["semantic"].shape == obs["occupancy"].shape
    assert "height_fused" in info
    assert info.get("not_slam") is True
    env.close()

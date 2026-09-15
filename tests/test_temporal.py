"""Hysteresis / decay changes hazard maps; person/dog tracklets associate."""

from __future__ import annotations

import numpy as np

from jims_mower.constants import HAZARD_DRAIN
from jims_mower.env import MowerEnv
from jims_mower.perception import DetectionTracklets, HazardHysteresis, HeuristicTerrainObserver
from jims_mower.types import CameraSpec, Detection, PerceptionContext, Pose


def test_hysteresis_needs_confirm_then_decays() -> None:
    filt = HazardHysteresis(confirm=1.6, decay=0.45, forget=0.20, hit_boost=1.0)
    incoming = np.zeros((6, 6), dtype=np.float32)
    incoming[2, 3] = HAZARD_DRAIN
    conf = np.zeros_like(incoming)
    conf[2, 3] = 1.0
    first = filt.update(incoming, conf)
    assert float(first[2, 3]) == 0.0  # not enough evidence yet
    second = filt.update(incoming, conf)
    assert float(second[2, 3]) == 0.0  # decay-then-boost still below confirm
    third = filt.update(incoming, conf)
    assert float(third[2, 3]) == float(HAZARD_DRAIN)
    blank = np.zeros_like(incoming)
    after = third
    for _ in range(12):
        after = filt.update(blank, blank)
    assert float(after[2, 3]) == 0.0
    assert not np.array_equal(third, after)


def test_heuristic_temporal_changes_maps() -> None:
    pose = Pose(2.0, 4.0, 0.0)
    cam = CameraSpec("front", 0.25, 0.0, 0.38, 0.0, -22.0)
    ctx = PerceptionContext(
        pose,
        [cam],
        [],
        (16, 12),
        map_shape=(20, 20),
        resolution_m=0.4,
        world_size=(8.0, 8.0),
    )
    drain = np.zeros((12, 16, 3), dtype=np.uint8)
    drain[:] = (58, 42, 28)
    grass = np.zeros((12, 16, 3), dtype=np.uint8)
    grass[:] = (46, 140, 58)
    imu = np.zeros(6, dtype=np.float32)
    imu[2] = 9.81
    gps = np.zeros(4, dtype=np.float32)
    obs = HeuristicTerrainObserver(temporal=True)
    obs._filter = HazardHysteresis(confirm=0.5, decay=0.35, forget=0.12)
    stamped = obs.estimate({"front": drain}, imu, gps, ctx)
    assert int((stamped.hazard >= 2).sum()) > 0
    later = stamped
    for _ in range(10):
        later = obs.estimate({"front": grass}, imu, gps, ctx)
    assert int((later.hazard >= 2).sum()) < int((stamped.hazard >= 2).sum())


def test_tracklets_keep_id_when_person_moves() -> None:
    tracker = DetectionTracklets(max_dist_m=1.5, max_missed=3)
    a = Detection("person", "front", (1, 1, 4, 8), 0.9, world_xy=(2.0, 3.0), category="person")
    t1 = tracker.update([a])
    assert len(t1) == 1
    tid = t1[0].track_id
    b = Detection("person", "front", (2, 2, 4, 8), 0.88, world_xy=(2.3, 3.1), category="person")
    t2 = tracker.update([b])
    assert len(t2) == 1
    assert t2[0].track_id == tid
    assert t2[0].hits == 2


def test_tracklets_drop_after_misses() -> None:
    tracker = DetectionTracklets(max_dist_m=1.0, max_missed=1)
    dog = Detection("dog", "left", (0, 0, 3, 3), 0.8, world_xy=(1.0, 1.0), category="animal")
    tracker.update([dog])
    tracker.update([])
    gone = tracker.update([])
    assert gone == []


def test_env_info_has_tracklets() -> None:
    env = MowerEnv(
        config={
            "sensors": {"width": 16, "height": 12, "camera_count": 4},
            "world": {
                "width_m": 8.0,
                "height_m": 8.0,
                "resolution_m": 0.25,
                "n_people": 1,
                "n_dogs": 1,
                "n_cats": 0,
                "n_birds": 0,
                "n_trees": 0,
                "n_furniture": 0,
                "n_toys": 0,
                "terrain": {"enabled": False},
            },
        }
    )
    _, info = env.reset(seed=4)
    assert "tracklets" in info
    assert isinstance(info["tracklets"], list)
    env.close()

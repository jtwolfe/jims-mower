"""CV-3 / PLN-4: appearance dets from RGB, not god-view obstacles."""

from __future__ import annotations

import numpy as np

from jims_mower.env import MowerEnv
from jims_mower.perception.detect import AppearanceDetector, paint_kind_blob
from jims_mower.perception.temporal import DetectionTracklets
from jims_mower.safety import trimmer_interlock, trimmer_interlock_from_detections
from jims_mower.types import CameraSpec, Detection, Obstacle, PerceptionContext, Pose


def _ctx(obstacles: list[Obstacle]) -> PerceptionContext:
    cam = CameraSpec("front", 0.25, 0.0, 0.38, 0.0, -8.0)
    return PerceptionContext(Pose(2.0, 4.0, 0.0), [cam], obstacles, (32, 24), False)


def _tiny_appearance(**extra) -> dict:
    cfg = {
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
        "perception": {
            "detector_backend": "appearance",
            "interlock_source": "detections",
            "terrain_mode": "heuristic",
        },
        "robot": {"trimmer": {"safety_radius_m": 1.5, "offset_m": 0.32}},
    }
    cfg.update(extra)
    return cfg


def test_appearance_detects_painted_person_not_oracle_list() -> None:
    obstacles = [Obstacle("person", 7.0, 4.0, 0.25, z=0.9)]
    ctx = _ctx(obstacles)
    blank = {"front": np.zeros((24, 32, 3), dtype=np.uint8)}
    det = AppearanceDetector()
    assert det.detect(blank, ctx) == []
    assert det.map_claim is None
    painted = paint_kind_blob(blank["front"], "person", (10, 6, 10, 12))
    found = det.detect({"front": painted}, ctx)
    assert any(d.label == "person" for d in found)
    assert all(d.world_xy is None for d in found)


def test_appearance_detects_animal_and_obstacle_blobs() -> None:
    det = AppearanceDetector()
    ctx = _ctx([])
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    frame = paint_kind_blob(frame, "dog", (2, 2, 8, 8))
    frame = paint_kind_blob(frame, "furniture", (18, 10, 8, 8))
    found = det.detect({"front": frame}, ctx)
    labels = {d.label for d in found}
    assert "dog" in labels
    assert "furniture" in labels


def test_dets_interlock_trips_on_front_blob_not_oracle_behind() -> None:
    person_behind = Detection(
        label="person",
        camera="rear",
        bbox=(4, 4, 8, 8),
        confidence=0.8,
        world_xy=None,
        category="person",
    )
    empty = trimmer_interlock_from_detections(
        True, [person_behind], safety_radius_m=1.5
    )
    assert empty.trimmer_enabled is True
    front = Detection(
        label="person",
        camera="front",
        bbox=(8, 8, 10, 12),
        confidence=0.8,
        world_xy=None,
        category="person",
    )
    trip = trimmer_interlock_from_detections(True, [front], safety_radius_m=1.5)
    assert trip.trimmer_enabled is False
    assert trip.requested is True


def test_gym_appearance_painted_front_blob_kills_trimmer() -> None:
    env = MowerEnv(config=_tiny_appearance())
    obs, info = env.reset(seed=3)
    env._yard.obstacles.append(Obstacle("person", env._pose.x - 2.5, env._pose.y, 0.25, z=0.9))

    def _painted_front() -> dict[str, np.ndarray]:
        images = {name: np.zeros((24, 32, 3), dtype=np.uint8) for name in env.camera_index}
        images["front"] = paint_kind_blob(images["front"], "person", (8, 6, 12, 14))
        return images

    env._camera_images = _painted_front  # type: ignore[method-assign]
    _obs, _r, _t, _c, info = env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
    assert info["trimmer_requested"] is True
    assert info["trimmer_enabled"] is False
    labels = [d.get("label") for d in info.get("detections") or [] if isinstance(d, dict)]
    assert "person" in labels
    env.close()


def test_gym_appearance_oracle_person_behind_does_not_trip() -> None:
    """God-view list has a person behind; front RGB does not. Trimmer stays on."""
    env = MowerEnv(config=_tiny_appearance())
    obs, info = env.reset(seed=4)
    behind = Obstacle("person", env._pose.x - 2.0, env._pose.y, 0.25, z=0.9)
    env._yard.obstacles.append(behind)
    god = trimmer_interlock(
        True,
        env._pose,
        env._yard.obstacles,
        offset_m=0.32,
        safety_radius_m=1.5,
    )
    assert god.trimmer_enabled is False

    def _no_person_rgb() -> dict[str, np.ndarray]:
        return {name: np.zeros((24, 32, 3), dtype=np.uint8) for name in env.camera_index}

    env._camera_images = _no_person_rgb  # type: ignore[method-assign]
    _obs, _r, _t, _c, info = env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
    assert info["trimmer_requested"] is True
    assert info["trimmer_enabled"] is True
    env.close()


def test_appearance_tracklets_associate_by_iou() -> None:
    tracker = DetectionTracklets()
    a = Detection("person", "front", (10, 8, 8, 10), 0.7, None, None, "person")
    b = Detection("person", "front", (11, 9, 8, 10), 0.7, None, None, "person")
    t0 = tracker.update([a])
    t1 = tracker.update([b])
    assert len(t0) == 1 and len(t1) == 1
    assert t0[0].track_id == t1[0].track_id
    assert t1[0].hits == 2


def test_gym_rendered_person_in_front_trips_trimmer() -> None:
    """Rendered-camera path: KIND_RGB disk in the front view, not oracle projection."""
    import math

    env = MowerEnv(config=_tiny_appearance())
    _obs, info = env.reset(seed=5)
    pose = env._pose
    env._yard.obstacles = [
        Obstacle(
            "person",
            pose.x + 1.1 * math.cos(pose.theta),
            pose.y + 1.1 * math.sin(pose.theta),
            0.28,
            z=0.9,
        )
    ]
    _obs, _r, _t, _c, info = env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
    labels = [d.get("label") for d in info.get("detections") or [] if isinstance(d, dict)]
    env.close()
    assert "person" in labels
    assert info["trimmer_requested"] is True
    assert info["trimmer_enabled"] is False


def test_config_accepts_appearance_backend() -> None:
    from jims_mower.config import load_config

    cfg = load_config({"perception": {"detector_backend": "appearance"}})
    assert cfg.perception.detector_backend == "appearance"
    assert cfg.perception.interlock_source == "auto"

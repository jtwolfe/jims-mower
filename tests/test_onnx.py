"""Train → ONNX → observer. No published IoU / mAP / FPS."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jims_mower.config import ConfigError, load_config
from jims_mower.env import MowerEnv
from jims_mower.export import export_dataset
from jims_mower.perception import (
    AppearanceDetector,
    MockDetector,
    OnnxTerrainObserver,
    detector_from_mode,
    terrain_observer_from_mode,
)
from jims_mower.perception.learn import load_weights
from jims_mower.perception.onnx_io import (
    OnnxExportError,
    export_terrain_onnx,
    onnx_available,
    onnxruntime_available,
)
from jims_mower.perception.train import train_from_export
from jims_mower.runtime.export_trt import run_export
from jims_mower.types import CameraSpec, Obstacle, PerceptionContext, Pose


def _tiny_cfg() -> dict:
    return {
        "dt": 0.1,
        "max_steps": 8,
        "sensors": {"width": 32, "height": 24, "camera_count": 4, "fov_deg": 70.0},
        "world": {
            "width_m": 8.0,
            "height_m": 8.0,
            "resolution_m": 0.20,
            "n_people": 1,
            "n_dogs": 0,
            "n_cats": 0,
            "n_birds": 0,
            "n_trees": 0,
            "n_furniture": 0,
            "n_toys": 0,
            "terrain": {"enabled": True, "n_drains": 1, "n_banks": 1, "noise_amp_m": 0.0},
        },
        "robot": {
            "trimmer": {"safety_radius_m": 1.2, "offset_m": 0.32, "radius_m": 0.16}
        },
    }


def test_onnx_mode_falls_back_without_file() -> None:
    observer = terrain_observer_from_mode("onnx")
    assert isinstance(observer, OnnxTerrainObserver)
    assert observer.backend == "heuristic"
    assert observer.iou_claim is None
    assert observer.map_claim is None
    assert observer.fps_claim is None
    env = MowerEnv(config=_tiny_cfg(), terrain_observer=observer)
    obs, info = env.reset(seed=3)
    assert info["terrain_source"] == "heuristic"
    assert obs["hazard"].shape == obs["coverage"].shape
    assert obs["elevation"].shape == obs["coverage"].shape
    env.close()


def test_trt_mode_falls_back_without_engine() -> None:
    observer = terrain_observer_from_mode("trt")
    env = MowerEnv(config=_tiny_cfg(), terrain_observer=observer)
    obs, info = env.reset(seed=4)
    assert info["terrain_source"] == "heuristic"
    assert obs["hazard"].shape == obs["coverage"].shape
    env.close()


def test_config_accepts_onnx_and_trt_modes() -> None:
    onnx_cfg = load_config({"perception": {"terrain_mode": "onnx", "onnx_path": ""}})
    assert onnx_cfg.perception.terrain_mode == "onnx"
    trt_cfg = load_config({"perception": {"terrain_mode": "trt"}})
    assert trt_cfg.perception.terrain_mode == "trt"
    with pytest.raises(ConfigError):
        load_config({"perception": {"terrain_mode": "magic"}})


def test_env_honors_onnx_mode_from_config() -> None:
    env = MowerEnv(config={**_tiny_cfg(), "perception": {"terrain_mode": "onnx"}})
    assert isinstance(env.terrain_observer, OnnxTerrainObserver)
    _obs, info = env.reset(seed=1)
    assert info["terrain_source"] == "heuristic"
    env.close()


def test_default_gym_stays_heuristic() -> None:
    cfg = load_config()
    assert cfg.perception.terrain_mode == "heuristic"
    assert cfg.perception.detector_backend == "mock"
    env = MowerEnv(config=_tiny_cfg())
    _obs, info = env.reset(seed=5)
    assert info["terrain_source"] == "heuristic"
    env.close()


def test_train_stub_keeps_claims_null(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    export_dataset(dataset, steps=2, seed=7, config=_tiny_cfg(), cameras=4, policy="scripted")
    stats = train_from_export(
        dataset, tmp_path / "terrain_mlp.npz", hidden=4, epochs=3, seed=7, backend="numpy"
    )
    assert Path(stats["weights"]).is_file()
    assert stats["iou_claim"] is None
    assert stats["map_claim"] is None
    assert stats["fps_claim"] is None
    assert stats["field_ready"] is False
    assert stats["domain"] == "sim_only"


@pytest.mark.skipif(not onnx_available(), reason="onnx extra not installed")
def test_train_writes_onnx_file(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    export_dataset(
        dataset,
        steps=2,
        seed=7,
        config=_tiny_cfg(),
        cameras=4,
        policy="scripted",
        adapter="fake_csi",
    )
    onnx_path = tmp_path / "terrain_seg.onnx"
    stats = train_from_export(
        dataset,
        tmp_path / "terrain_mlp.npz",
        hidden=4,
        epochs=4,
        seed=7,
        backend="numpy",
        onnx_path=onnx_path,
    )
    assert onnx_path.is_file()
    assert Path(stats["onnx"]) == onnx_path
    sidecar = json.loads(onnx_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["schema"] == "jims_mower.terrain_onnx.v1"
    assert sidecar["domain"] == "sim_only"
    assert sidecar["field_ready"] is False
    assert sidecar["iou_claim"] is None
    assert sidecar["map_claim"] is None
    assert sidecar["fps_claim"] is None
    assert sidecar["classes"] == ["grass", "bank", "lip", "drain"]
    import onnx

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)


@pytest.mark.skipif(
    not (onnx_available() and onnxruntime_available()),
    reason="onnx + onnxruntime extras not installed",
)
def test_onnx_observer_loads_and_matches_icd_shape(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    export_dataset(dataset, steps=2, seed=2, config=_tiny_cfg(), cameras=4, policy="scripted")
    weights = tmp_path / "terrain_mlp.npz"
    onnx_path = tmp_path / "terrain_seg.onnx"
    train_from_export(
        dataset, weights, hidden=4, epochs=4, seed=2, backend="numpy", onnx_path=onnx_path
    )
    observer = OnnxTerrainObserver(onnx_path, temporal=False, stride=2)
    assert observer.backend == "onnx"
    assert observer.iou_claim is None
    env = MowerEnv(config=_tiny_cfg(), terrain_observer=observer)
    observation, info = env.reset(seed=2)
    assert info["terrain_source"] == "onnx"
    assert observation["hazard"].shape == observation["coverage"].shape
    assert observation["elevation"].shape == observation["coverage"].shape
    assert observation["slope"].shape == observation["coverage"].shape
    assert observation["confidence"].shape == observation["coverage"].shape
    observation, _, _, _, info = env.step(np.array([0.3, 0.3, 0.0], dtype=np.float32))
    assert info["terrain_source"] == "onnx"
    env.close()
    # Numpy weights still load beside the ONNX.
    loaded = load_weights(weights)
    assert loaded.n_classes == 4


def test_export_onnx_without_extra_is_clear(tmp_path: Path) -> None:
    if onnx_available():
        pytest.skip("onnx extra is installed")
    from jims_mower.perception.learn import init_weights

    model = init_weights(hidden=0)
    with pytest.raises(OnnxExportError, match="onnx is not installed"):
        export_terrain_onnx(model, tmp_path / "x.onnx")


def test_trtexec_dry_run_still_honest(tmp_path: Path) -> None:
    payload = run_export(onnx=tmp_path / "missing.onnx", engine=tmp_path / "x.engine", dry_run=True)
    assert payload["fps_claim"] is None
    assert payload["map_claim"] is None
    assert payload["iou_claim"] is None
    assert payload["field_ready"] is False
    assert payload["dry_run"] is True
    assert payload["command"][0] == "trtexec"


def test_appearance_detector_ignores_obstacles() -> None:
    pose = Pose(1.0, 4.0, 0.0)
    obstacles = [Obstacle("person", 3.0, 4.0, 0.25, z=0.9)]
    cam = CameraSpec("front", 0.25, 0.0, 0.38, 0.0, -8.0)
    images = {"front": np.zeros((24, 32, 3), dtype=np.uint8)}
    ctx = PerceptionContext(pose, [cam], obstacles, (32, 24), False)
    det = AppearanceDetector()
    assert det.detect(images, ctx) == []
    assert det.map_claim is None
    assert det.fps_claim is None
    mocked = MockDetector(appearance=False).detect(images, ctx)
    assert any(item.label == "person" for item in mocked)


def test_detector_from_mode_default_is_mock() -> None:
    assert isinstance(detector_from_mode("mock"), MockDetector)
    assert detector_from_mode("appearance").detect({}, PerceptionContext(Pose(0, 0, 0), [], [])) == []


def test_appearance_backend_leaves_detections_empty_and_interlock_consumes_key() -> None:
    cfg = {**_tiny_cfg(), "perception": {"detector_backend": "appearance", "terrain_mode": "heuristic"}}
    env = MowerEnv(config=cfg)
    obs, info = env.reset(seed=8)
    assert "detections" in obs
    assert "detections" in info
    assert info["detections"] == []
    assert int((obs["detections"][:, 0] > 0).sum()) == 0
    env.close()
    default = MowerEnv(config=_tiny_cfg())
    _obs, info = default.reset(seed=8)
    assert isinstance(default.detector, MockDetector)
    default.close()

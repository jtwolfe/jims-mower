"""Exporter → numpy train → LearnedTerrainObserver. No claimed mAP."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from jims_mower.env import MowerEnv
from jims_mower.export import export_dataset
from jims_mower.perception import LearnedTerrainObserver, terrain_observer_from_mode
from jims_mower.perception.learn import fit_numpy, load_weights, save_weights
from jims_mower.perception.train import samples_from_export, train_from_export


def _tiny_cfg() -> dict:
    return {
        "dt": 0.1,
        "max_steps": 8,
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
            "terrain": {"enabled": True, "n_drains": 1, "n_banks": 1, "noise_amp_m": 0.0},
        },
        "robot": {
            "trimmer": {"safety_radius_m": 1.2, "offset_m": 0.32, "radius_m": 0.16}
        },
    }


def test_numpy_fit_writes_and_loads(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x = rng.random((80, 7), dtype=np.float32)
    y = rng.integers(0, 4, size=80)
    model, stats = fit_numpy(x, y, hidden=4, epochs=3, lr=0.2, seed=0)
    assert stats["backend"] == "numpy"
    assert "not mAP" in stats["note"]
    path = save_weights(model, tmp_path / "w.npz")
    loaded = load_weights(path)
    assert loaded.n_features == 7
    proba = loaded.predict_proba(x[:4])
    assert proba.shape == (4, 4)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_train_from_export_and_learned_observer(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    export_dataset(dataset, steps=2, seed=7, config=_tiny_cfg(), cameras=4, policy="scripted")
    meta = __import__("json").loads((dataset / "meta.json").read_text(encoding="utf-8"))
    assert meta.get("camera_specs")
    weights = tmp_path / "terrain_mlp.npz"
    stats = train_from_export(dataset, weights, hidden=6, epochs=8, seed=7, backend="numpy")
    assert Path(stats["weights"]).is_file()
    assert stats["backend"] == "numpy"
    assert stats["n_samples"] >= 8 or stats.get("n_pixels", 0) >= 8
    x, y, info = samples_from_export(dataset, stride=3, max_per_class=80, seed=1)
    assert x.shape[1] == 7
    assert y.size == x.shape[0]
    assert sum(info["class_counts"]) == y.size

    obs = LearnedTerrainObserver(weights, temporal=False, stride=2)
    env = MowerEnv(config=_tiny_cfg(), terrain_observer=obs)
    observation, info = env.reset(seed=7)
    assert info["terrain_source"] == "learned"
    assert observation["hazard"].shape == observation["coverage"].shape
    observation, _, _, _, info = env.step(np.array([0.3, 0.3, 0.0], dtype=np.float32))
    assert info["terrain_source"] == "learned"
    env.close()


def test_learned_mode_runs_without_weights() -> None:
    observer = terrain_observer_from_mode("learned")
    env = MowerEnv(config=_tiny_cfg(), terrain_observer=observer)
    obs, info = env.reset(seed=3)
    assert info["terrain_source"] == "learned"
    assert obs["hazard"].shape == obs["coverage"].shape
    env.close()


def test_heuristic_path_still_runs() -> None:
    env = MowerEnv(config={**_tiny_cfg(), "perception": {"terrain_mode": "heuristic"}})
    obs, info = env.reset(seed=5)
    assert info["terrain_source"] == "heuristic"
    assert obs["hazard"].dtype == np.float32
    env.close()

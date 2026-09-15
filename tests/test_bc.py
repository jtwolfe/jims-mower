"""Behaviour cloning collect / train / demo --policy bc."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jims_mower.bc import BcPolicy, collect_demos, maybe_load_bc, train_bc
from jims_mower.bc_cli import main as bc_main
from jims_mower.constants import BC_FEATURE_DIM
from jims_mower.demo import run_demo
from jims_mower.features import extract_features


def _tiny() -> dict:
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
        "perception": {"terrain_mode": "oracle"},
        "robot": {
            "trimmer": {"safety_radius_m": 1.2, "offset_m": 0.32, "radius_m": 0.16}
        },
    }


def test_extract_features_fixed_dim() -> None:
    obs = {
        "pose": np.zeros(6, dtype=np.float32),
        "imu": np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0], dtype=np.float32),
        "gps": np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32),
        "tof": np.ones(4, dtype=np.float32),
        "hazard": np.zeros((8, 8), dtype=np.float32),
        "occupancy": np.zeros((8, 8), dtype=np.float32),
        "coverage": np.zeros((8, 8), dtype=np.float32),
        "trimmer_enabled": np.array([1.0], dtype=np.float32),
        "hand_signal": 0,
    }
    feats = extract_features(obs, {"terrain_advice": "ok"}, resolution_m=0.2)
    assert feats.shape == (BC_FEATURE_DIM,)
    assert feats.dtype == np.float32


def test_collect_train_and_policy_act(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    meta = collect_demos(logs, steps=6, seed=2, config=_tiny(), cameras=4)
    assert meta["n_samples"] >= 1
    assert (logs / "features.npy").is_file()
    assert meta["not_a_benchmark"] is True
    weights = tmp_path / "bc_weights.npz"
    result = train_bc(logs, weights, epochs=12, seed=0)
    assert weights.is_file()
    assert result["not_a_benchmark"] is True
    assert "train_mse" in result
    policy = BcPolicy.load(weights, resolution_m=0.2)
    obs = {
        "pose": np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "imu": np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0], dtype=np.float32),
        "gps": np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32),
        "tof": np.ones(4, dtype=np.float32) * 0.3,
        "hazard": np.zeros((10, 10), dtype=np.float32),
        "occupancy": np.zeros((10, 10), dtype=np.float32),
        "coverage": np.zeros((10, 10), dtype=np.float32),
        "trimmer_enabled": np.array([1.0], dtype=np.float32),
        "hand_signal": 0,
    }
    action = policy.act(obs, {"terrain_advice": "ok"})
    assert action.shape == (3,)
    assert -1.0 <= float(action[0]) <= 1.0
    assert 0.0 <= float(action[2]) <= 1.0


def test_demo_policy_bc_loads_weights(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    collect_demos(logs, steps=5, seed=1, config=_tiny(), cameras=4)
    weights = tmp_path / "w.npz"
    train_bc(logs, weights, epochs=8, seed=1)
    out = tmp_path / "demo"
    summary = run_demo(
        out,
        steps=3,
        seed=1,
        cameras=4,
        policy="bc",
        config=_tiny(),  # type: ignore[arg-type]
        bc_weights=str(weights),
    )
    assert summary["bc_loaded"] is True
    assert summary["policy"] == "bc"
    assert summary["steps_run"] >= 1


def test_demo_policy_bc_falls_back_without_weights(tmp_path: Path) -> None:
    summary = run_demo(
        tmp_path / "demo",
        steps=2,
        seed=3,
        cameras=4,
        policy="bc",
        config=_tiny(),  # type: ignore[arg-type]
        bc_weights=str(tmp_path / "missing.npz"),
    )
    assert summary["bc_loaded"] is False
    assert summary["policy"] == "terrain"
    assert summary["steps_run"] >= 1


def test_maybe_load_missing() -> None:
    assert maybe_load_bc("/no/such/bc.npz", 0.1) is None


def test_bc_cli(tmp_path: Path) -> None:
    logs = tmp_path / "cli-logs"
    # CLI collect uses load_source; pass a real yaml-less default via tiny file.
    cfg = tmp_path / "tiny.yaml"
    import yaml

    yaml.safe_dump(_tiny(), cfg.open("w"))
    bc_main(["collect", "--out", str(logs), "--steps", "3", "--seed", "1", "--cameras", "4", "--config", str(cfg)])
    weights = tmp_path / "cli.npz"
    bc_main(["train", "--in", str(logs), "--out", str(weights), "--epochs", "5"])
    assert weights.is_file()

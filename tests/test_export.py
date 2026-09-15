"""Dataset exporter folder layout and determinism."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jims_mower.export import export_dataset
from tests.conftest import tiny_config_dict


def _tiny_export_cfg() -> dict:
    cfg = tiny_config_dict()
    cfg["max_steps"] = 8
    cfg["world"]["n_people"] = 0
    cfg["world"]["n_dogs"] = 0
    cfg["world"]["n_toys"] = 0
    cfg["world"]["n_trees"] = 0
    cfg["world"]["terrain"] = {"enabled": True, "n_drains": 1, "n_banks": 0, "noise_amp_m": 0.0}
    return cfg


def test_export_layout(tmp_path: Path) -> None:
    meta = export_dataset(
        tmp_path,
        steps=2,
        seed=7,
        config=_tiny_export_cfg(),
        cameras=4,
        policy="scripted",
    )
    assert meta["schema"] == "jims_mower.dataset.v1"
    assert meta["seed"] == 7
    assert meta["frames"] >= 2
    assert (tmp_path / "LAYOUT.md").is_file()
    assert (tmp_path / "meta.json").is_file()
    assert (tmp_path / "coco.json").is_file()
    assert (tmp_path / "images" / "000000_front.png").is_file()
    assert (tmp_path / "labels" / "000000_hazard.png").is_file()
    assert (tmp_path / "labels" / "000000_grass.png").is_file()
    assert (tmp_path / "labels" / "000000_elevation.npy").is_file()
    assert (tmp_path / "labels" / "000000_slope.npy").is_file()
    assert (tmp_path / "frames" / "000000.json").is_file()
    sidecar = json.loads((tmp_path / "frames" / "000000.json").read_text(encoding="utf-8"))
    assert len(sidecar["imu"]) == 6
    assert len(sidecar["gps"]) == 4
    assert sidecar["tof"] is not None
    assert len(sidecar["tof"]) == 4
    assert sidecar["labels"]["hazard"].endswith("hazard.png")
    coco = json.loads((tmp_path / "coco.json").read_text(encoding="utf-8"))
    assert coco["images"]
    assert coco["semantic"]["hazard"]["3"] == "channel"
    elev = np.load(tmp_path / "labels" / "000000_elevation.npy")
    assert elev.ndim == 2
    assert elev.dtype == np.float32


def test_export_omits_tof_when_requested(tmp_path: Path) -> None:
    export_dataset(
        tmp_path,
        steps=1,
        seed=1,
        config=_tiny_export_cfg(),
        cameras=4,
        policy="scripted",
        include_tof=False,
    )
    sidecar = json.loads((tmp_path / "frames" / "000000.json").read_text(encoding="utf-8"))
    assert sidecar["tof"] is None


def test_export_deterministic(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    cfg = _tiny_export_cfg()
    export_dataset(a, steps=1, seed=11, config=cfg, cameras=4, policy="scripted")
    export_dataset(b, steps=1, seed=11, config=cfg, cameras=4, policy="scripted")
    haz_a = (a / "labels" / "000000_hazard.png").read_bytes()
    haz_b = (b / "labels" / "000000_hazard.png").read_bytes()
    assert haz_a == haz_b
    rgb_a = (a / "images" / "000000_front.png").read_bytes()
    rgb_b = (b / "images" / "000000_front.png").read_bytes()
    assert rgb_a == rgb_b

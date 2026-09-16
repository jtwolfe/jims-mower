"""Dataset exporter folder layout and determinism."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jims_mower.export import export_dataset


def _tiny_export_cfg() -> dict:
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
            "terrain": {"enabled": True, "n_drains": 1, "n_banks": 0, "noise_amp_m": 0.0},
        },
        "robot": {
            "trimmer": {"safety_radius_m": 1.2, "offset_m": 0.32, "radius_m": 0.16}
        },
    }


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
    assert meta["camera_specs"]
    assert meta["world_size"] == [8.0, 8.0]
    assert "domain_randomization" in meta
    assert meta["domain_randomization"]["enabled"] is False
    assert meta["source"] == "renderer"
    assert meta["split"]["n_train"] + meta["split"]["n_val"] == meta["frames"]


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


def test_export_domain_rand_flag(tmp_path: Path) -> None:
    meta = export_dataset(
        tmp_path,
        steps=1,
        seed=2,
        config=_tiny_export_cfg(),
        cameras=4,
        policy="scripted",
        domain_rand=True,
    )
    assert meta["domain_randomization"]["enabled"] is True
    assert meta["domain_randomization"]["lighting"] is True


def test_export_fake_csi_dataset_v1_and_split(tmp_path: Path) -> None:
    """CV-8 harness: FakeCsi frames land in jims_mower.dataset.v1 with a split."""
    meta = export_dataset(
        tmp_path,
        steps=4,
        seed=3,
        config=_tiny_export_cfg(),
        cameras=4,
        policy="scripted",
        adapter="fake_csi",
        val_frac=0.25,
    )
    assert meta["schema"] == "jims_mower.dataset.v1"
    assert meta["source"] == "fake_csi"
    assert meta["adapter"] == "fake_csi"
    assert meta["map_claim"] is None
    assert meta["fps_claim"] is None
    split = meta["split"]
    assert split["rule"] == "last_frac_val"
    assert split["n_train"] + split["n_val"] == meta["frames"]
    assert split["val_frames"]
    assert (tmp_path / "split.json").is_file()
    disk = json.loads((tmp_path / "split.json").read_text(encoding="utf-8"))
    assert disk["train_frames"] == split["train_frames"]
    # Fake CSI still writes named RGB at the contract size.
    assert (tmp_path / "images" / "000000_front.png").is_file()


def test_assign_frame_split_last_frac() -> None:
    from jims_mower.dataset import assign_frame_split

    split = assign_frame_split(10, val_frac=0.20)
    assert split["train_frames"] == list(range(8))
    assert split["val_frames"] == [8, 9]


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

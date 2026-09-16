"""Farm quarantine, dataset schema, studies extra kinds, watchdog, Gst, TRT."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jims_mower.constants import DATASET_SCHEMA
from jims_mower.dataset import DatasetSchemaError, validate_dataset_meta
from jims_mower.export import export_dataset
from jims_mower.farm import run_farm
from jims_mower.perception.trt import TrtDetector, TrtTerrainObserver, tensorrt_available
from jims_mower.runtime.gstreamer import FakeGstAdapter, GstNotAvailable, GstNvmmAdapter, gstreamer_available
from jims_mower.runtime.watchdog import SensorWatchdog
from jims_mower.studies import plan_axis_matrix, run_study
from jims_mower.types import PerceptionContext, Pose


def test_dataset_schema_required() -> None:
    with pytest.raises(DatasetSchemaError):
        validate_dataset_meta({"seed": 1})
    with pytest.raises(DatasetSchemaError):
        validate_dataset_meta({"schema": "nope"})
    assert validate_dataset_meta({"schema": DATASET_SCHEMA}) == DATASET_SCHEMA


def test_export_enforces_schema(tmp_path: Path) -> None:
    meta = export_dataset(
        tmp_path,
        steps=1,
        seed=3,
        config={
            "sensors": {"width": 16, "height": 12, "camera_count": 4},
            "world": {
                "width_m": 6.0,
                "height_m": 6.0,
                "resolution_m": 0.3,
                "n_people": 0,
                "n_dogs": 0,
                "n_cats": 0,
                "n_birds": 0,
                "n_trees": 0,
                "n_furniture": 0,
                "n_toys": 0,
                "terrain": {"enabled": False},
            },
        },
        cameras=4,
        policy="scripted",
    )
    assert meta["schema"] == DATASET_SCHEMA
    dumped = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    validate_dataset_meta(dumped)


def test_farm_quarantine_recorded(tmp_path: Path) -> None:
    summary = run_farm(
        tmp_path,
        seeds=[0],
        scenarios=["suburban", "playground"],
        steps=2,
        dry_run=True,
        quarantine=["playground"],
    )
    assert "playground" in summary["quarantined"]
    assert summary["dry_run"] is True


def test_farm_flake_quarantines(tmp_path: Path) -> None:
    summary = run_farm(
        tmp_path,
        seeds=[0],
        scenarios=["not-a-real-yard"],
        steps=1,
        dry_run=False,
        flake_budget=1,
    )
    assert summary["flakes"]
    assert "not-a-real-yard" in summary["quarantined"]
    assert (tmp_path / "quarantine.json").is_file()


def test_study_pitch_dry_run(tmp_path: Path) -> None:
    matrix = plan_axis_matrix(kind="pitch", seeds=[0], scenario="steep_yard")
    assert any(abs(item["front_pitch_deg"] + 22.0) < 1e-6 for item in matrix)
    summary = run_study(
        tmp_path,
        seeds=[0],
        cameras=[4],
        tof_counts=[4],
        imu_scales=[1.0],
        kind="pitch",
        dry_run=True,
    )
    assert summary["kind"] == "pitch"
    assert summary["fps_claim"] is None
    assert summary["n_planned"] >= 2


def test_watchdog_stops_frozen_imu() -> None:
    wd = SensorWatchdog(enabled=True, imu_stall_s=0.15, vision_stall_s=9.0, dt=0.10)
    imu = np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0], dtype=np.float32)
    cams = {"front": np.zeros((4, 4, 3), dtype=np.uint8)}
    cams["front"][0, 0] = 1
    wd.observe(imu, cams)
    wd.observe(imu, cams)
    wd.observe(imu, cams)
    assert wd.stalled
    assert wd.reason == "imu_stall"
    held = wd.filter_action(np.array([0.4, 0.4, 1.0], dtype=np.float32))
    assert float(np.max(np.abs(held))) == 0.0


def test_watchdog_stops_frozen_vision_stamp() -> None:
    wd = SensorWatchdog(enabled=True, imu_stall_s=9.0, vision_stall_s=0.15, dt=0.10)
    imu = np.array([0.0, 0.0, 9.81, 0.0, 0.1, 0.0], dtype=np.float32)
    cams = {"front": np.ones((4, 4, 3), dtype=np.uint8)}
    wd.observe(imu, cams, imu_stamp_s=0.0, vision_stamp_s=1.0)
    wd.observe(imu * 2, cams, imu_stamp_s=0.1, vision_stamp_s=1.0)
    wd.observe(imu * 3, cams, imu_stamp_s=0.2, vision_stamp_s=1.0)
    assert wd.stalled
    assert wd.reason == "vision_stall"


def test_gstreamer_stub_not_required() -> None:
    assert gstreamer_available() is False
    real = GstNvmmAdapter()
    with pytest.raises(GstNotAvailable):
        real.capture()
    fake = FakeGstAdapter(["front"])
    frames = fake.capture({"front": np.ones((24, 32, 3), dtype=np.uint8)})
    assert frames["front"].shape == (60, 80, 3)
    assert fake.fps_claim is None


def test_trt_placeholder_falls_back() -> None:
    assert tensorrt_available() is False
    det = TrtDetector("/no/such.engine")
    assert det.backend == "mock"
    assert det.fps_claim is None
    ctx = PerceptionContext(Pose(0, 0, 0), [], [])
    assert det.detect({}, ctx) == []
    obs = TrtTerrainObserver("/no/such.engine")
    assert obs.backend == "numpy"
    assert obs.fps_claim is None

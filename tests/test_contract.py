"""Versioned runtime contract dataclasses + required-field checks."""

from __future__ import annotations

import pytest

from jims_mower.contract import (
    CONTRACT_VERSION,
    JSON_SCHEMAS,
    MESSAGE_KINDS,
    CameraFrame,
    DetectionSet,
    GpsFix,
    ImuSample,
    Plan,
    SafeState,
    TerrainMaps,
    TofArray,
    WheelCommand,
    schema_for,
    validate_payload,
)


def test_all_kinds_have_schemas() -> None:
    assert set(MESSAGE_KINDS) == set(JSON_SCHEMAS)
    for kind in MESSAGE_KINDS:
        schema = schema_for(kind)
        assert "version" in schema["required"]


def test_camera_frame_roundtrip() -> None:
    msg = CameraFrame(name="front", width=80, height=60)
    assert msg.version == CONTRACT_VERSION
    again = CameraFrame.from_dict(msg.to_dict())
    assert again.name == "front"
    assert again.width == 80


def test_imu_gps_tof_plan_command() -> None:
    imu = ImuSample(accel_mps2=(0.0, 0.0, 9.81), gyro_radps=(0.0, 0.0, 0.1))
    gps = GpsFix(x=1.0, y=2.0, z=0.1, valid=1.0)
    tof = TofArray(ranges_m=(0.2, 0.2, 0.2, 0.2))
    plan = Plan(waypoints=[(1.0, 2.0), (1.4, 2.0)], index=1, advice="slow")
    cmd = WheelCommand(left=0.4, right=0.35, trimmer=1.0)
    maps = TerrainMaps(resolution_m=0.1, source="heuristic")
    dets = DetectionSet.from_dict(
        {
            "version": CONTRACT_VERSION,
            "detections": [
                {
                    "label": "person",
                    "camera": "front",
                    "bbox": [1, 2, 8, 10],
                    "confidence": 0.9,
                }
            ],
        }
    )
    assert imu.to_dict()["version"] == CONTRACT_VERSION
    assert gps.to_dict()["valid"] == 1.0
    assert tof.ranges_m[0] == pytest.approx(0.2)
    assert plan.waypoints[0] == (1.0, 2.0)
    assert cmd.trimmer == pytest.approx(1.0)
    assert maps.confidence == "confidence"
    assert dets.detections[0].label == "person"
    validate_payload("WheelCommand", cmd.to_dict())
    validate_payload("GpsFix", gps.to_dict())
    safe = SafeState(mode="estop", hold=True, trimmer_allowed=False)
    assert safe.version == CONTRACT_VERSION
    validate_payload("SafeState", safe.to_dict())


def test_missing_version_rejected() -> None:
    with pytest.raises(ValueError):
        validate_payload("GpsFix", {"x": 1.0, "y": 2.0})


def test_missing_required_field() -> None:
    with pytest.raises(ValueError):
        validate_payload("CameraFrame", {"version": "1", "name": "front"})

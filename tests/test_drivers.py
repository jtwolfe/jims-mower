"""Fake I2C / UART / CSI drivers publish runtime-contract payloads."""

from __future__ import annotations

import numpy as np
import pytest

from jims_mower.contract import validate_payload
from jims_mower.env import MowerEnv
from jims_mower.runtime.bus import InProcessBus
from jims_mower.constants import GRAVITY_MPS2
from jims_mower.runtime.drivers import (
    FakeCsiDriver,
    FakeGnssDriver,
    FakeImuDriver,
    FakeTofDriver,
    SensorRig,
    level_rest_imu,
    publish_obs,
)
from jims_mower.runtime.ros2_stubs import ROS2_TOPICS, documented_topics, ros2_available


def _tiny() -> dict:
    return {
        "dt": 0.1,
        "max_steps": 20,
        "sensors": {"width": 16, "height": 12, "camera_count": 4, "tof": {"count": 2}},
        "world": {
            "width_m": 8.0,
            "height_m": 8.0,
            "resolution_m": 0.25,
            "n_people": 0,
            "n_dogs": 0,
            "n_cats": 0,
            "n_birds": 0,
            "n_trees": 0,
            "n_furniture": 0,
            "n_toys": 0,
            "terrain": {"enabled": False},
        },
        "perception": {"terrain_mode": "blind"},
    }


def test_fake_drivers_match_contract() -> None:
    env = MowerEnv(config=_tiny(), render_mode=None)
    obs, _info = env.reset(seed=3)
    bus, msgs = publish_obs(obs, stamp_s=0.1)
    kinds = {m.kind for m in msgs}
    assert {"ImuSample", "GpsFix", "TofArray", "CameraFrame"} <= kinds
    assert bus.qsize("ImuSample") == 1
    assert bus.qsize("CameraFrame") == 4
    imu = bus.get("ImuSample")
    assert imu is not None
    assert imu.bus == "i2c"
    validate_payload("ImuSample", imu.payload)
    assert imu.payload["accel_mps2"][2] == pytest.approx(float(obs["imu"][2]), abs=1e-5)
    gps = bus.get("GpsFix")
    assert gps is not None and gps.bus == "uart"
    tof = bus.get("TofArray")
    assert tof is not None and tof.bus == "i2c"
    assert len(tof.payload["ranges_m"]) == 4
    cams = bus.drain("CameraFrame")
    assert all(c.bus == "csi" for c in cams)
    assert {c.payload["name"] for c in cams} == set(obs["cameras"])
    env.close()


def test_driver_classes_use_documented_buses() -> None:
    bus = InProcessBus()
    assert FakeImuDriver.bus_name == "i2c"
    assert FakeGnssDriver.bus_name == "uart"
    assert FakeCsiDriver.bus_name == "csi"
    assert FakeTofDriver.bus_name == "i2c"
    rig = SensorRig(bus)
    obs = {
        "imu": np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0], dtype=np.float32),
        "gps": np.array([1.0, 2.0, 0.1, 1.0], dtype=np.float32),
        "tof": np.array([0.2, 0.2, 0.0, 0.0], dtype=np.float32),
        "cameras": {"front": np.zeros((12, 16, 3), dtype=np.uint8)},
    }
    rig.publish(obs, stamp_s=1.5)
    assert bus.get("ImuSample").payload["stamp_s"] == pytest.approx(1.5)


def test_fake_imu_level_rest() -> None:
    bus = InProcessBus()
    imu = FakeImuDriver(bus)
    rest = imu.publish_level_rest(stamp_s=0.2)
    assert rest.payload["accel_mps2"][0] == pytest.approx(0.0)
    assert rest.payload["accel_mps2"][1] == pytest.approx(0.0)
    assert rest.payload["accel_mps2"][2] == pytest.approx(GRAVITY_MPS2, abs=1e-5)
    assert rest.payload["gyro_radps"] == pytest.approx((0.0, 0.0, 0.0))
    missing = imu.publish({}, stamp_s=0.3)
    assert missing.payload["accel_mps2"][2] == pytest.approx(GRAVITY_MPS2, abs=1e-5)
    arr = level_rest_imu()
    assert arr.shape == (6,)
    assert arr[2] == pytest.approx(GRAVITY_MPS2, abs=1e-5)


def test_fake_gnss_valid_toggles() -> None:
    bus = InProcessBus()
    gnss = FakeGnssDriver(bus)
    obs = {"gps": np.array([1.0, 2.0, 0.1, 1.0], dtype=np.float32)}
    gnss.set_valid(True)
    first = gnss.publish(obs, stamp_s=1.0)
    assert first.payload["valid"] == pytest.approx(1.0)
    bit = gnss.toggle_valid()
    assert bit == pytest.approx(0.0)
    second = gnss.publish(obs, stamp_s=1.1)
    assert second.payload["valid"] == pytest.approx(0.0)
    gnss.toggle_valid()
    third = gnss.publish(obs, stamp_s=1.2)
    assert third.payload["valid"] == pytest.approx(1.0)


def test_fake_tof_board_shortens_one_corner() -> None:
    bus = InProcessBus()
    tof = FakeTofDriver(bus)
    obs = {"tof": np.array([0.20, 0.20, 0.20, 0.20], dtype=np.float32)}
    tof.publish(obs, stamp_s=0.0)
    before = bus.get("TofArray")
    assert before is not None
    tof.slide_board("FL", 0.05)
    tof.publish(obs, stamp_s=0.1)
    after = bus.get("TofArray")
    assert after is not None
    assert after.payload["ranges_m"][0] == pytest.approx(0.15, abs=1e-5)
    assert after.payload["ranges_m"][1] == pytest.approx(0.20, abs=1e-5)


def test_ros2_stubs_are_import_safe() -> None:
    assert ros2_available() is False
    topics = documented_topics()
    assert topics["ImuSample"] == "/jims_mower/imu"
    assert set(ROS2_TOPICS) >= {"ImuSample", "GpsFix", "CameraFrame", "WheelCommand"}

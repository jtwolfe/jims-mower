"""Fake I2C IMU, UART GNSS, CSI camera, and I2C ToF publishers.

These modules take a gym observation (or a recorded episode frame) and
publish ``runtime_contract`` payloads onto an :class:`InProcessBus`.
They are **not** Linux i2c-dev / GStreamer drivers. On the Orin, replace
the publish path with the real bus; keep the same message kinds.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from jims_mower.contract import (
    CONTRACT_VERSION,
    CameraFrame,
    GpsFix,
    ImuSample,
    TofArray,
)
from jims_mower.runtime.bus import BusMessage, InProcessBus

# Class addresses / device nodes — documentation only, not probed.
I2C_IMU_ADDR = 0x68  # BMI088 / ICM-42688-P class
I2C_TOF_ADDR = 0x29  # VL53L1X class
UART_GNSS_DEV = "/dev/ttyUSB0"
CSI_DEVICE = "nvarguscamerasrc"


def _stamp(obs: dict[str, Any], fallback: float = 0.0) -> float:
    raw = obs.get("stamp_s")
    if raw is None:
        return float(fallback)
    return float(np.asarray(raw).reshape(-1)[0])


def imu_from_obs(obs: dict[str, Any], *, stamp_s: float = 0.0) -> ImuSample:
    arr = np.asarray(obs.get("imu", [0, 0, 9.81, 0, 0, 0]), dtype=np.float32).reshape(-1)
    if arr.size < 6:
        arr = np.pad(arr, (0, 6 - arr.size))
    return ImuSample(
        accel_mps2=(float(arr[0]), float(arr[1]), float(arr[2])),
        gyro_radps=(float(arr[3]), float(arr[4]), float(arr[5])),
        stamp_s=_stamp(obs, stamp_s),
        frame_id="imu",
        version=CONTRACT_VERSION,
    )


def gps_from_obs(obs: dict[str, Any], *, stamp_s: float = 0.0) -> GpsFix:
    arr = np.asarray(obs.get("gps", [0, 0, 0, 0]), dtype=np.float32).reshape(-1)
    if arr.size < 4:
        arr = np.pad(arr, (0, 4 - arr.size))
    return GpsFix(
        x=float(arr[0]),
        y=float(arr[1]),
        z=float(arr[2]),
        valid=float(arr[3]),
        stamp_s=_stamp(obs, stamp_s),
        frame_id="gps",
        version=CONTRACT_VERSION,
    )


def tof_from_obs(obs: dict[str, Any], *, stamp_s: float = 0.0) -> TofArray:
    arr = np.asarray(obs.get("tof", [0, 0, 0, 0]), dtype=np.float32).reshape(-1)
    if arr.size < 4:
        arr = np.pad(arr, (0, 4 - int(arr.size)))
    return TofArray(
        ranges_m=(float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])),
        stamp_s=_stamp(obs, stamp_s),
        frame_id="tof",
        version=CONTRACT_VERSION,
    )


def cameras_from_obs(obs: dict[str, Any], *, stamp_s: float = 0.0) -> list[CameraFrame]:
    frames: list[CameraFrame] = []
    cameras = obs.get("cameras") or {}
    stamp = _stamp(obs, stamp_s)
    for name, image in cameras.items():
        arr = np.asarray(image)
        h = int(arr.shape[0]) if arr.ndim >= 2 else 0
        w = int(arr.shape[1]) if arr.ndim >= 2 else 0
        frames.append(
            CameraFrame(
                name=str(name),
                width=max(w, 1),
                height=max(h, 1),
                encoding="rgb8",
                stamp_s=stamp,
                frame_id=f"camera/{name}",
                version=CONTRACT_VERSION,
            )
        )
    return frames


class FakeImuDriver:
    """Publish gym ``obs['imu']`` as if a 6-axis chip sat on I2C."""

    bus_name = "i2c"
    address = I2C_IMU_ADDR
    kind = "ImuSample"

    def __init__(self, bus: InProcessBus) -> None:
        self.bus = bus

    def publish(self, obs: dict[str, Any], *, stamp_s: float = 0.0) -> BusMessage:
        sample = imu_from_obs(obs, stamp_s=stamp_s)
        return self.bus.publish(self.kind, sample.to_dict(), bus=self.bus_name)


class FakeGnssDriver:
    """Publish gym ``obs['gps']`` as if a UART GNSS module produced a fix."""

    bus_name = "uart"
    device = UART_GNSS_DEV
    kind = "GpsFix"

    def __init__(self, bus: InProcessBus) -> None:
        self.bus = bus

    def publish(self, obs: dict[str, Any], *, stamp_s: float = 0.0) -> BusMessage:
        fix = gps_from_obs(obs, stamp_s=stamp_s)
        return self.bus.publish(self.kind, fix.to_dict(), bus=self.bus_name)


class FakeCsiDriver:
    """Publish gym camera metadata as if CSI / NVMM capture filled the rig."""

    bus_name = "csi"
    device = CSI_DEVICE
    kind = "CameraFrame"

    def __init__(self, bus: InProcessBus) -> None:
        self.bus = bus

    def publish(self, obs: dict[str, Any], *, stamp_s: float = 0.0) -> list[BusMessage]:
        out: list[BusMessage] = []
        for frame in cameras_from_obs(obs, stamp_s=stamp_s):
            out.append(self.bus.publish(self.kind, frame.to_dict(), bus=self.bus_name))
        return out


class FakeTofDriver:
    """Publish gym ``obs['tof']`` as if four downward ToF sensors sat on I2C."""

    bus_name = "i2c"
    address = I2C_TOF_ADDR
    kind = "TofArray"

    def __init__(self, bus: InProcessBus) -> None:
        self.bus = bus

    def publish(self, obs: dict[str, Any], *, stamp_s: float = 0.0) -> BusMessage:
        sample = tof_from_obs(obs, stamp_s=stamp_s)
        return self.bus.publish(self.kind, sample.to_dict(), bus=self.bus_name)


class SensorRig:
    """I2C IMU + UART GNSS + CSI cameras + I2C ToF on one in-process bus."""

    def __init__(self, bus: Optional[InProcessBus] = None) -> None:
        self.bus = bus or InProcessBus()
        self.imu = FakeImuDriver(self.bus)
        self.gnss = FakeGnssDriver(self.bus)
        self.csi = FakeCsiDriver(self.bus)
        self.tof = FakeTofDriver(self.bus)

    def publish(self, obs: dict[str, Any], *, stamp_s: float = 0.0) -> list[BusMessage]:
        msgs = [
            self.imu.publish(obs, stamp_s=stamp_s),
            self.gnss.publish(obs, stamp_s=stamp_s),
            self.tof.publish(obs, stamp_s=stamp_s),
        ]
        msgs.extend(self.csi.publish(obs, stamp_s=stamp_s))
        return msgs


def publish_obs(
    obs: dict[str, Any],
    *,
    bus: Optional[InProcessBus] = None,
    stamp_s: float = 0.0,
) -> tuple[InProcessBus, list[BusMessage]]:
    """Convenience: stand up a rig, publish one gym / recorded observation."""
    rig = SensorRig(bus)
    return rig.bus, rig.publish(obs, stamp_s=stamp_s)

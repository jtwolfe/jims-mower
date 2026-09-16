"""Fake I2C IMU, UART GNSS, CSI camera, and I2C ToF publishers.

These modules take a gym observation (or a recorded episode frame) and
publish ``runtime_contract`` payloads onto an :class:`InProcessBus`.
They are **not** Linux i2c-dev / GStreamer drivers. Addresses below are
**documentation** (not probed). On the Orin, replace the publish path
with the real bus; keep the same message kinds.

:class:`FakeCsiDriver` also fills ``obs["cameras"]`` at the ICD contract
size via :class:`~jims_mower.runtime.gstreamer.FakeGstAdapter` so the
bench watchdog sees named live frames + fresh stamps.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import numpy as np

from jims_mower.constants import GRAVITY_MPS2
from jims_mower.contract import (
    CONTRACT_VERSION,
    CameraFrame,
    GpsFix,
    ImuSample,
    TofArray,
)
from jims_mower.runtime.bus import BusMessage, InProcessBus
from jims_mower.runtime.capture import (
    DEFAULT_CONTRACT_HEIGHT,
    DEFAULT_CONTRACT_WIDTH,
    CaptureResult,
    names_from_specs,
)
from jims_mower.runtime.gstreamer import FakeGstAdapter
from jims_mower.types import CameraSpec

# Class addresses / device nodes — documentation only, not probed.
I2C_IMU_ADDR = 0x68  # BMI088 / ICM-42688-P class
I2C_TOF_ADDR = 0x29  # VL53L1X class
UART_GNSS_DEV = "/dev/ttyUSB0"
CSI_DEVICE = "nvarguscamerasrc"

# ICD rest / ToF corner names (gym fixture + FakeTofDriver).
TOF_CORNER_INDEX = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}
LEVEL_REST_IMU = (0.0, 0.0, float(GRAVITY_MPS2), 0.0, 0.0, 0.0)


def level_rest_imu() -> np.ndarray:
    """Body specific force + gyro at level rest: ``(0, 0, g, 0, 0, 0)``."""
    return np.array(LEVEL_REST_IMU, dtype=np.float32)


def _stamp(obs: dict[str, Any], fallback: float = 0.0) -> float:
    raw = obs.get("stamp_s")
    if raw is None:
        return float(fallback)
    return float(np.asarray(raw).reshape(-1)[0])


def imu_from_obs(obs: dict[str, Any], *, stamp_s: float = 0.0) -> ImuSample:
    raw = obs.get("imu")
    if raw is None:
        arr = level_rest_imu()
    else:
        arr = np.asarray(raw, dtype=np.float32).reshape(-1)
        if arr.size < 6:
            arr = np.pad(arr, (0, 6 - arr.size))
    return ImuSample(
        accel_mps2=(float(arr[0]), float(arr[1]), float(arr[2])),
        gyro_radps=(float(arr[3]), float(arr[4]), float(arr[5])),
        stamp_s=_stamp(obs, stamp_s),
        frame_id="imu",
        version=CONTRACT_VERSION,
    )


def gps_from_obs(
    obs: dict[str, Any],
    *,
    stamp_s: float = 0.0,
    valid: Optional[float] = None,
) -> GpsFix:
    arr = np.asarray(obs.get("gps", [0, 0, 0, 0]), dtype=np.float32).reshape(-1)
    if arr.size < 4:
        arr = np.pad(arr, (0, 4 - arr.size))
    bit = float(arr[3] if valid is None else valid)
    return GpsFix(
        x=float(arr[0]),
        y=float(arr[1]),
        z=float(arr[2]),
        valid=bit,
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
    """Publish gym ``obs['imu']`` as if a 6-axis chip sat on I2C.

    Missing ``imu`` key → level rest ``(0, 0, g, 0, 0, 0)``. Address
    ``0x68`` is documentation only.
    """

    bus_name = "i2c"
    address = I2C_IMU_ADDR
    kind = "ImuSample"

    def __init__(self, bus: InProcessBus) -> None:
        self.bus = bus

    def publish(self, obs: dict[str, Any], *, stamp_s: float = 0.0) -> BusMessage:
        sample = imu_from_obs(obs, stamp_s=stamp_s)
        return self.bus.publish(self.kind, sample.to_dict(), bus=self.bus_name)

    def publish_level_rest(self, *, stamp_s: float = 0.0) -> BusMessage:
        return self.publish({"imu": level_rest_imu()}, stamp_s=stamp_s)


class FakeGnssDriver:
    """Publish gym ``obs['gps']`` as if a UART GNSS module produced a fix.

    ``set_valid`` / ``toggle_valid`` force the ICD ``valid`` bit so gym
    dropout tests are writable. Device node is documentation only.
    """

    bus_name = "uart"
    device = UART_GNSS_DEV
    kind = "GpsFix"

    def __init__(self, bus: InProcessBus) -> None:
        self.bus = bus
        self._valid_override: Optional[float] = None

    def set_valid(self, valid: bool) -> None:
        self._valid_override = 1.0 if valid else 0.0

    def toggle_valid(self) -> float:
        current = 1.0 if self._valid_override is None else float(self._valid_override)
        self._valid_override = 0.0 if current >= 0.5 else 1.0
        return float(self._valid_override)

    def publish(self, obs: dict[str, Any], *, stamp_s: float = 0.0) -> BusMessage:
        fix = gps_from_obs(obs, stamp_s=stamp_s, valid=self._valid_override)
        return self.bus.publish(self.kind, fix.to_dict(), bus=self.bus_name)


class FakeCsiDriver:
    """Fake CSI: fill ``obs["cameras"]`` and publish ``CameraFrame`` metadata.

    Uses :class:`FakeGstAdapter` so CI gets named live frames at the ICD
    contract size with fresh stamps. Swap this publish path for
    :class:`~jims_mower.runtime.gstreamer.GstNvmmAdapter` on the Orin.
    """

    bus_name = "csi"
    device = CSI_DEVICE
    kind = "CameraFrame"

    def __init__(
        self,
        bus: Optional[InProcessBus] = None,
        names: Optional[Sequence[str]] = None,
        *,
        width: int = DEFAULT_CONTRACT_WIDTH,
        height: int = DEFAULT_CONTRACT_HEIGHT,
        cameras: Optional[Iterable[CameraSpec]] = None,
        dt: float = 0.10,
    ) -> None:
        self.bus = bus or InProcessBus()
        self.adapter = FakeGstAdapter(
            names if names is not None else names_from_specs(cameras),
            width=width,
            height=height,
            cameras=cameras,
            dt=dt,
        )

    @property
    def names(self) -> list[str]:
        return list(self.adapter.names)

    def grab(
        self,
        frames: Optional[dict[str, np.ndarray]] = None,
        *,
        stamp_s: Optional[float] = None,
    ) -> CaptureResult:
        return self.adapter.grab(frames, stamp_s=stamp_s)

    def fill_obs(
        self,
        obs: Optional[dict[str, Any]] = None,
        frames: Optional[dict[str, np.ndarray]] = None,
        *,
        stamp_s: Optional[float] = None,
    ) -> dict[str, Any]:
        return self.adapter.fill_obs(obs, frames=frames, stamp_s=stamp_s)

    def publish(self, obs: dict[str, Any], *, stamp_s: float = 0.0) -> list[BusMessage]:
        if not (obs.get("cameras")):
            self.fill_obs(obs, stamp_s=stamp_s)
        out: list[BusMessage] = []
        for frame in cameras_from_obs(obs, stamp_s=stamp_s):
            out.append(self.bus.publish(self.kind, frame.to_dict(), bus=self.bus_name))
        return out


class FakeTofDriver:
    """Publish gym ``obs['tof']`` as if four downward ToF sensors sat on I2C.

    ``slide_board(corner)`` shortens that FL/FR/RL/RR range — gym stand-in
    for a board under a wheel. Address ``0x29`` is documentation only.
    """

    bus_name = "i2c"
    address = I2C_TOF_ADDR
    kind = "TofArray"

    def __init__(self, bus: InProcessBus) -> None:
        self.bus = bus
        self._board: Optional[tuple[int, float]] = None

    def slide_board(self, corner: str, thickness_m: float = 0.04) -> None:
        key = str(corner).upper()
        if key not in TOF_CORNER_INDEX:
            raise ValueError(f"ToF corner must be one of {tuple(TOF_CORNER_INDEX)}; got {corner!r}")
        self._board = (TOF_CORNER_INDEX[key], float(thickness_m))

    def clear_board(self) -> None:
        self._board = None

    def publish(self, obs: dict[str, Any], *, stamp_s: float = 0.0) -> BusMessage:
        sample = tof_from_obs(obs, stamp_s=stamp_s)
        ranges = list(sample.ranges_m)
        if self._board is not None:
            idx, thick = self._board
            ranges[idx] = max(0.0, float(ranges[idx]) - float(thick))
        sample = TofArray(
            ranges_m=(float(ranges[0]), float(ranges[1]), float(ranges[2]), float(ranges[3])),
            stamp_s=sample.stamp_s,
            frame_id=sample.frame_id,
            version=sample.version,
        )
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

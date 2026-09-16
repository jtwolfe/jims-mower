"""GStreamer / NVMM capture adapter.

On an Orin this would pull CSI frames through ``nvarguscamerasrc`` into
NVMM, downsample them in :mod:`jims_mower.runtime.capture`, and write
``obs["cameras"]``. CI and the default laptop install do **not** require
GStreamer, PyGObject, or Jetson NVMM.

:class:`GstNvmmAdapter` raises :class:`GstNotAvailable` without Gst
(construction is cheap). :class:`FakeGstAdapter` fills the same dict from
numpy buffers — named frames at the ICD contract size with fresh stamps
so :class:`~jims_mower.runtime.watchdog.SensorWatchdog` stays happy.

Default names match :class:`~jims_mower.types.CameraSpec` and prefer the
field stereo pair (``stereo_left`` / ``stereo_right`` + mono). No FPS
claim. Physical CSI still needs JetPack + real cameras.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import numpy as np

from jims_mower.runtime.capture import (
    DEFAULT_CONTRACT_HEIGHT,
    DEFAULT_CONTRACT_WIDTH,
    DOWNSAMPLE_NOTE,
    FIELD_RIG_NAMES,
    CaptureResult,
    StampClock,
    frames_to_obs,
    names_from_specs,
)
from jims_mower.types import CameraSpec

GST_NOTE = (
    "GStreamer / NVMM is an on-box capture path. CI uses FakeCsiDriver / "
    "FakeGstAdapter. Do not install pygobject in the default extra. "
    "Physical CSI on Orin still needs JetPack + real cameras."
)


class GstNotAvailable(RuntimeError):
    """GStreamer bindings are not present (expected in CI)."""


def gstreamer_available() -> bool:
    try:
        import gi  # noqa: F401
    except Exception:
        return False
    try:
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst  # noqa: F401
    except Exception:
        return False
    return True


class GstNvmmAdapter:
    """Real NVMM path. Construction is cheap; capture fails without Gst.

    When JetPack/Gst exists, ``grab`` / ``fill_obs`` should capture CSI,
    call :func:`jims_mower.runtime.capture.downsample_rgb`, and write
    named frames at ``width`` × ``height``. That last hop is **not**
    wired in this repo — swap the publish path on the board.
    """

    pipeline = (
        "nvarguscamerasrc ! nvvidconv ! video/x-raw,format=RGB ! appsink"
    )

    def __init__(
        self,
        names: Optional[Sequence[str]] = None,
        *,
        width: int = DEFAULT_CONTRACT_WIDTH,
        height: int = DEFAULT_CONTRACT_HEIGHT,
        cameras: Optional[Iterable[CameraSpec]] = None,
        dt: float = 0.10,
    ) -> None:
        self.names = list(names) if names is not None else names_from_specs(cameras)
        if not self.names:
            self.names = list(FIELD_RIG_NAMES)
        self.width = int(width)
        self.height = int(height)
        self.available = gstreamer_available()
        self.note = GST_NOTE
        self.downsample_note = DOWNSAMPLE_NOTE
        self.fps_claim = None
        self.clock = StampClock(dt=dt)

    @property
    def last_stamp_s(self) -> float:
        return self.clock.stamp_s

    def grab(
        self,
        frames: Optional[dict[str, np.ndarray]] = None,
        *,
        stamp_s: Optional[float] = None,
    ) -> CaptureResult:
        del frames, stamp_s
        if not self.available:
            raise GstNotAvailable(GST_NOTE)
        raise GstNotAvailable("live NVMM capture is not wired in this repo")

    def capture(
        self,
        frames: Optional[dict[str, np.ndarray]] = None,
        *,
        stamp_s: Optional[float] = None,
    ) -> dict[str, np.ndarray]:
        return self.grab(frames, stamp_s=stamp_s).cameras

    def fill_obs(
        self,
        obs: Optional[dict[str, Any]] = None,
        frames: Optional[dict[str, np.ndarray]] = None,
        *,
        stamp_s: Optional[float] = None,
    ) -> dict[str, Any]:
        return self.grab(frames, stamp_s=stamp_s).fill_obs(obs)


class FakeGstAdapter:
    """Laptop stand-in: named contract-size frames + fresh stamps.

    Pass gym / recorded rasters in ``frames`` and they are downsampled
    to ``width`` × ``height``. With no buffers, each camera gets a
    synthetic live frame so CI / bench can exercise the watchdog
    without JetPack.
    """

    def __init__(
        self,
        names: Optional[Sequence[str]] = None,
        *,
        width: int = DEFAULT_CONTRACT_WIDTH,
        height: int = DEFAULT_CONTRACT_HEIGHT,
        cameras: Optional[Iterable[CameraSpec]] = None,
        dt: float = 0.10,
    ) -> None:
        self.names = list(names) if names is not None else names_from_specs(cameras)
        if not self.names:
            self.names = list(FIELD_RIG_NAMES)
        self.width = int(width)
        self.height = int(height)
        self.available = True
        self.note = GST_NOTE
        self.downsample_note = DOWNSAMPLE_NOTE
        self.fps_claim = None
        self.clock = StampClock(dt=dt)

    @property
    def last_stamp_s(self) -> float:
        return self.clock.stamp_s

    def grab(
        self,
        frames: Optional[dict[str, np.ndarray]] = None,
        *,
        stamp_s: Optional[float] = None,
    ) -> CaptureResult:
        stamp = self.clock.next(stamp_s)
        return frames_to_obs(
            frames,
            self.names,
            width=self.width,
            height=self.height,
            stamp_s=stamp,
            tick=self.clock.tick,
        )

    def capture(
        self,
        frames: Optional[dict[str, np.ndarray]] = None,
        *,
        stamp_s: Optional[float] = None,
    ) -> dict[str, np.ndarray]:
        return self.grab(frames, stamp_s=stamp_s).cameras

    def fill_obs(
        self,
        obs: Optional[dict[str, Any]] = None,
        frames: Optional[dict[str, np.ndarray]] = None,
        *,
        stamp_s: Optional[float] = None,
    ) -> dict[str, Any]:
        return self.grab(frames, stamp_s=stamp_s).fill_obs(obs)

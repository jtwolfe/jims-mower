"""GStreamer / NVMM capture adapter *stub*.

On an Orin this would pull CSI frames through ``nvarguscamerasrc`` into
NVMM and copy (or map) them into ``obs["cameras"]``. CI and the default
laptop install do **not** require GStreamer, PyGObject, or Jetson NVMM.

Use :class:`FakeGstAdapter` in tests — it fills the same dict from numpy
buffers the gym already produced. A missing GStreamer install raises
``GstNotAvailable`` from :class:`GstNvmmAdapter.capture`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

GST_NOTE = (
    "GStreamer / NVMM is an on-box capture path. CI uses FakeCsiDriver / "
    "FakeGstAdapter. Do not install pygobject in the default extra."
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
    """Real NVMM path. Construction is cheap; capture fails without Gst."""

    pipeline = "nvarguscamerasrc ! nvvidconv ! video/x-raw,format=RGB ! appsink"

    def __init__(self, names: Optional[list[str]] = None) -> None:
        self.names = list(names or ("front", "rear", "left", "right"))
        self.available = gstreamer_available()
        self.note = GST_NOTE
        self.fps_claim = None

    def capture(self) -> dict[str, np.ndarray]:
        if not self.available:
            raise GstNotAvailable(GST_NOTE)
        raise GstNotAvailable("live NVMM capture is not wired in this repo")


class FakeGstAdapter:
    """Laptop stand-in: copies gym / recorded rasters into obs['cameras']."""

    def __init__(self, names: Optional[list[str]] = None) -> None:
        self.names = list(names or ("front", "rear", "left", "right"))
        self.available = True
        self.note = GST_NOTE
        self.fps_claim = None

    def capture(self, frames: Optional[dict[str, np.ndarray]] = None) -> dict[str, np.ndarray]:
        if frames is None:
            return {name: np.zeros((8, 8, 3), dtype=np.uint8) for name in self.names}
        return {name: np.asarray(frames[name]) for name in frames}

"""CSI-like frames → ``obs["cameras"]`` at the ICD contract size.

This is the software path that will swallow real CSI once JetPack /
GStreamer exist on the Orin. Downsample happens **here**
(:func:`downsample_rgb`) *before* frames are written into
``obs["cameras"]``. CI uses :class:`~jims_mower.runtime.gstreamer.FakeGstAdapter`
/ :class:`~jims_mower.runtime.drivers.FakeCsiDriver`. :class:`~jims_mower.runtime.gstreamer.GstNvmmAdapter`
raises cleanly without Gst.

Names match :class:`~jims_mower.types.CameraSpec`. The bench / Orin path
prefers the field stereo pair (``stereo_left`` / ``stereo_right``) plus
side/rear mono from ``configs/orin/extrinsics_stereo.yaml``. Gym
look-around names (``front_left`` / ``front_right``) stay valid for
legacy scenarios.

No FPS claim. ``fps_claim`` stays ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np

from jims_mower.types import CameraSpec

# ICD / sim default. Env may pass ``sensors.width`` × ``sensors.height``.
DEFAULT_CONTRACT_WIDTH = 80
DEFAULT_CONTRACT_HEIGHT = 60

# Preferred field rig (HARDWARE_DESIGN §7 / extrinsics_stereo.yaml).
FIELD_STEREO_NAMES = ("stereo_left", "stereo_right")
FIELD_MONO_NAMES = ("front", "rear", "left", "right")
FIELD_RIG_NAMES = FIELD_STEREO_NAMES + FIELD_MONO_NAMES

# Default gym look-around. front_left / front_right are NOT a stereo pair.
GYM_LOOKAROUND_NAMES = ("front", "front_left", "front_right", "rear", "left", "right")

# Downsample lives in this module — document this path in SIM_TO_REAL / JETSON.
DOWNSAMPLE_NOTE = (
    "Downsample to the ICD contract size happens in "
    "jims_mower.runtime.capture.downsample_rgb, before obs['cameras'] is filled. "
    "Default contract is 80×60 (sensors.width × sensors.height). No FPS claim."
)


def names_from_specs(cameras: Optional[Iterable[CameraSpec]]) -> list[str]:
    """Same names as ``CameraSpec``. Empty → preferred field stereo + mono."""
    if cameras:
        names = [str(cam.name) for cam in cameras]
        if names:
            return names
    return list(FIELD_RIG_NAMES)


def is_field_stereo_name(name: str) -> bool:
    return str(name) in FIELD_STEREO_NAMES


def field_rig_yaml_path() -> Path:
    """Example stereo extrinsics (not a measured calibration)."""
    packaged = Path(__file__).resolve().parents[1] / "data" / "orin" / "extrinsics_stereo.yaml"
    if packaged.is_file():
        return packaged
    return Path(__file__).resolve().parents[3] / "configs" / "orin" / "extrinsics_stereo.yaml"


def load_field_camera_specs() -> list[CameraSpec]:
    """Load example field-rig names / poses from the stereo YAML."""
    from jims_mower.config import load_config

    return list(load_config(field_rig_yaml_path()).resolved_cameras())


def downsample_rgb(
    frame: Any,
    width: int = DEFAULT_CONTRACT_WIDTH,
    height: int = DEFAULT_CONTRACT_HEIGHT,
) -> np.ndarray:
    """Nearest-neighbour resize to ``(height, width, 3)`` uint8 RGB.

    This is the **only** downsample step on the CSI → obs path. Call it
    before writing ``obs["cameras"][name]``. Already-correct frames are
    returned as a contiguous uint8 view (no scale).
    """
    w = max(1, int(width))
    h = max(1, int(height))
    arr = np.asarray(frame)
    if arr.size == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3:
        return np.zeros((h, w, 3), dtype=np.uint8)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    elif arr.shape[-1] < 3:
        arr = np.pad(arr, ((0, 0), (0, 0), (0, 3 - arr.shape[-1])))
    src_h, src_w = int(arr.shape[0]), int(arr.shape[1])
    if src_h == h and src_w == w:
        return np.ascontiguousarray(arr, dtype=np.uint8)
    if src_h <= 0 or src_w <= 0:
        return np.zeros((h, w, 3), dtype=np.uint8)
    rows = np.clip((np.arange(h) * src_h / h).astype(np.intp), 0, src_h - 1)
    cols = np.clip((np.arange(w) * src_w / w).astype(np.intp), 0, src_w - 1)
    return np.ascontiguousarray(arr[np.ix_(rows, cols)], dtype=np.uint8)


def ensure_contract_rgb(
    frame: Any,
    width: int = DEFAULT_CONTRACT_WIDTH,
    height: int = DEFAULT_CONTRACT_HEIGHT,
) -> np.ndarray:
    """Downsample only when the frame is not already the contract size."""
    arr = np.asarray(frame)
    if (
        arr.ndim == 3
        and arr.shape[0] == int(height)
        and arr.shape[1] == int(width)
        and arr.shape[2] == 3
        and arr.dtype == np.uint8
    ):
        return arr
    return downsample_rgb(arr, width, height)


def _name_color(name: str) -> tuple[int, int, int]:
    """Stable synthetic tint per camera so named frames are distinguishable."""
    table = {
        "stereo_left": (40, 80, 200),
        "stereo_right": (60, 140, 220),
        "front": (46, 140, 58),
        "front_left": (80, 160, 70),
        "front_right": (90, 150, 50),
        "rear": (160, 90, 40),
        "left": (120, 80, 160),
        "right": (180, 140, 50),
    }
    if name in table:
        return table[name]
    digest = sum(ord(ch) for ch in name) % 180
    return (40 + digest, 80, 200 - digest // 2)


def synthetic_live_frame(
    name: str,
    width: int,
    height: int,
    *,
    stamp_s: float = 0.0,
    tick: int = 0,
) -> np.ndarray:
    """Named uint8 RGB that changes with stamp/tick (fake CSI, not a scene)."""
    w = max(1, int(width))
    h = max(1, int(height))
    r, g, b = _name_color(str(name))
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[..., 0] = r
    frame[..., 1] = g
    frame[..., 2] = b
    # Live pixel so SensorWatchdog fingerprints move when stamps do.
    phase = int(abs(float(stamp_s)) * 1000.0 + int(tick)) & 0xFF
    frame[0, 0] = (phase, (phase + 40) & 0xFF, (phase + 80) & 0xFF)
    if w > 1:
        frame[0, 1, 0] = (ord(str(name)[0]) + phase) & 0xFF
    return frame


@dataclass
class CaptureResult:
    """Named contract-size frames plus the vision stamp for the watchdog."""

    cameras: dict[str, np.ndarray]
    stamp_s: float
    names: list[str] = field(default_factory=list)

    def fill_obs(self, obs: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Write ``obs["cameras"]`` (and ``stamp_s``) at the ICD contract size."""
        out: dict[str, Any] = {} if obs is None else obs
        out["cameras"] = self.cameras
        out["stamp_s"] = float(self.stamp_s)
        return out


def frames_to_obs(
    frames: Optional[dict[str, Any]],
    names: Sequence[str],
    *,
    width: int = DEFAULT_CONTRACT_WIDTH,
    height: int = DEFAULT_CONTRACT_HEIGHT,
    stamp_s: float = 0.0,
    tick: int = 0,
) -> CaptureResult:
    """Build a :class:`CaptureResult` — downsample happens here."""
    wanted = [str(n) for n in names] or list(FIELD_RIG_NAMES)
    src = frames or {}
    cameras: dict[str, np.ndarray] = {}
    for name in wanted:
        if name in src:
            cameras[name] = downsample_rgb(src[name], width, height)
        else:
            cameras[name] = synthetic_live_frame(
                name, width, height, stamp_s=stamp_s, tick=tick
            )
    return CaptureResult(cameras=cameras, stamp_s=float(stamp_s), names=wanted)


class StampClock:
    """Monotonic capture stamps. Default step is the gym ``dt`` (0.10 s)."""

    def __init__(self, *, dt: float = 0.10, start_s: float = 0.0) -> None:
        self.dt = float(dt)
        self._stamp = float(start_s)
        self._tick = 0
        self.started = False

    def reset(self, start_s: float = 0.0) -> None:
        self._stamp = float(start_s)
        self._tick = 0
        self.started = False

    @property
    def stamp_s(self) -> float:
        return float(self._stamp)

    @property
    def tick(self) -> int:
        return int(self._tick)

    def next(self, stamp_s: Optional[float] = None) -> float:
        if stamp_s is not None:
            self._stamp = float(stamp_s)
            self._tick += 1
            self.started = True
            return self._stamp
        if not self.started:
            self.started = True
            self._tick = 1
            return self._stamp
        self._stamp += self.dt
        self._tick += 1
        return self._stamp


def adapter_kind(raw: Union[str, None]) -> str:
    """Normalize ``runtime.cameras.adapter``."""
    text = (raw or "renderer").strip().lower()
    if text in {"", "none", "gym", "renderer"}:
        return "renderer"
    if text in {"fake_gst", "fake-gst", "fake"}:
        return "fake_gst"
    if text in {"fake_csi", "fake-csi", "csi"}:
        return "fake_csi"
    if text in {"gst", "gst_nvmm", "gstreamer", "nvmm"}:
        return "gst"
    return text

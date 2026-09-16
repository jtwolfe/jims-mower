"""Appearance detector (CV-3): RGB blobs, not god-view obstacles.

``MockDetector`` stays the gym default (projects ``context.obstacles``).
This path **must ignore** that list. It finds synthetic person / animal /
obstacle colours in the camera image (palette + connected blobs). That is
not a published mAP. ``map_claim`` / ``fps_claim`` stay null.

A future ONNX box head can sit behind the same type (``sim_only`` until
you measure). Without a real box graph we do **not** invent boxes from a
terrain MLP.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from jims_mower.constants import KIND_RGB
from jims_mower.perception.base import BlindDetector, Detector
from jims_mower.perception.classify import HandSignalClassifier, crop_bbox
from jims_mower.perception.mock import MockDetector, category_for
from jims_mower.types import Detection, PerceptionContext

# Palette match in RGB. Obstacle disks are unshaded KIND_RGB. Assign
# each pixel to the nearest kind so dog ≠ person (those centroids are
# ~70 apart). Cord is near-black — use a tight ball so empty frames
# and grass do not become a yard-sized "cord".
_PALETTE_MAX_DIST = 55.0
_DARK_MAX_DIST = 16.0
_DARK_KINDS = frozenset({"cord"})
_MIN_BLOB_PIXELS = 8
_DETECT_KINDS = (
    "person",
    "dog",
    "cat",
    "bird",
    "toy",
    "furniture",
    "tree",
    "hose",
    "cord",
)


def paint_kind_blob(
    image: np.ndarray,
    kind: str,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Paint a solid KIND_RGB rectangle. Gym / unit fixture, not a label protocol."""
    out = np.asarray(image).copy()
    if out.ndim != 3 or out.shape[2] < 3:
        raise ValueError("image must be HxWx3")
    x, y, w, h = (int(v) for v in bbox)
    rgb = KIND_RGB.get(kind, (128, 128, 128))
    y0 = max(0, y)
    x0 = max(0, x)
    y1 = min(out.shape[0], y + max(h, 0))
    x1 = min(out.shape[1], x + max(w, 0))
    if y1 > y0 and x1 > x0:
        out[y0:y1, x0:x1, :3] = np.asarray(rgb, dtype=np.uint8)
    return out


def _connected_bboxes(mask: np.ndarray, min_pixels: int) -> list[tuple[int, int, int, int]]:
    """4-connected components → bounding boxes. No scipy."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        min_x = max_x = x0
        min_y = max_y = y0
        count = 0
        while stack:
            y, x = stack.pop()
            count += 1
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if count >= min_pixels:
            boxes.append((int(min_x), int(min_y), int(max_x - min_x + 1), int(max_y - min_y + 1)))
    return boxes


def detect_palette_blobs(
    image: np.ndarray,
    camera: str,
    *,
    max_dist: float = _PALETTE_MAX_DIST,
    min_pixels: int = _MIN_BLOB_PIXELS,
) -> list[Detection]:
    """Find KIND_RGB blobs in one RGB frame. Not mAP. No obstacle list."""
    if image is None or np.asarray(image).size == 0:
        return []
    frame = np.asarray(image)
    if frame.ndim != 3 or frame.shape[2] < 3:
        return []
    kinds = [(k, KIND_RGB[k]) for k in _DETECT_KINDS if k in KIND_RGB]
    if not kinds:
        return []
    arr = np.asarray(frame[..., :3], dtype=np.float32)
    targets = np.asarray([rgb for _, rgb in kinds], dtype=np.float32)
    dist = np.linalg.norm(arr[:, :, None, :] - targets.reshape(1, 1, -1, 3), axis=3)
    nearest = dist.argmin(axis=2)
    min_d = dist.min(axis=2)
    out: list[Detection] = []
    for i, (kind, _rgb) in enumerate(kinds):
        limit = _DARK_MAX_DIST if kind in _DARK_KINDS else float(max_dist)
        mask = (nearest == i) & (min_d <= limit)
        for bbox in _connected_bboxes(mask, min_pixels):
            x, y, w, h = bbox
            area = float(w * h)
            conf = float(np.clip(0.35 + 0.01 * area, 0.20, 0.85))
            out.append(
                Detection(
                    label=kind,
                    camera=str(camera),
                    bbox=bbox,
                    confidence=conf,
                    world_xy=None,
                    hand_signal=None,
                    category=category_for(kind),
                    depth_m=0.0,
                )
            )
    return out


class AppearanceDetector:
    """RGB appearance detector. Does not read ``context.obstacles``.

    ``map_claim`` / ``fps_claim`` / ``iou_claim`` stay null. Gym blobs
    are ``sim_only`` — not a field head.
    """

    def __init__(
        self,
        onnx_path: Optional[Union[str, Path]] = None,
        *,
        session=None,
        gym_hand_signals: bool = True,
    ) -> None:
        self.onnx_path = str(onnx_path) if onnx_path else ""
        self.session = session
        self.map_claim = None
        self.fps_claim = None
        self.iou_claim = None
        self.sim_only = True
        self.gym_hand_signals = bool(gym_hand_signals)
        self._signals = HandSignalClassifier()
        if self.session is None and self.onnx_path:
            dest = Path(self.onnx_path)
            if dest.is_file():
                from jims_mower.perception.onnx_io import load_onnx_session, onnxruntime_available

                if onnxruntime_available():
                    self.session = load_onnx_session(dest)
        self.backend = "onnx" if self.session is not None else "appearance"

    def detect(
        self,
        images: dict[str, np.ndarray],
        context: PerceptionContext,
    ) -> list[Detection]:
        # Honest: never project context.obstacles.
        _ = context.obstacles
        if self.session is not None:
            # No shipped detector graph — do not invent boxes from a terrain MLP.
            pass
        out: list[Detection] = []
        for name, frame in (images or {}).items():
            out.extend(detect_palette_blobs(frame, name))
        if self.gym_hand_signals and context.hand_signals_enabled:
            use_cls = bool(context.hand_signal_classifier)
            signed: list[Detection] = []
            for det in out:
                if det.label != "person":
                    signed.append(det)
                    continue
                frame = images.get(det.camera)
                signal = None
                if frame is not None:
                    if use_cls:
                        signal = self._signals.classify(frame, det.bbox)
                    else:
                        signal = gym_red_bias_signal(frame, det.bbox)
                signed.append(
                    Detection(
                        label=det.label,
                        camera=det.camera,
                        bbox=det.bbox,
                        confidence=det.confidence,
                        world_xy=det.world_xy,
                        hand_signal=signal,
                        category=det.category,
                        depth_m=det.depth_m,
                    )
                )
            out = signed
        return out


def gym_red_bias_signal(image: np.ndarray, bbox: tuple[int, int, int, int]) -> Optional[str]:
    """Gym-only stop/go from red vs green bias on a person crop.

    KIND_RGB person is already red-biased, so a painted person blob reads
    ``stop``. Green-biased crops may read ``go``. This is **not** a gesture
    model. No confusion matrix. Drop it on real clips if it is unreliable.
    """
    crop = crop_bbox(image, bbox)
    if crop.size == 0:
        return None
    mean = crop.reshape(-1, 3).astype(np.float32).mean(axis=0)
    r, g, b = float(mean[0]), float(mean[1]), float(mean[2])
    if r > g + 12.0 and r > b:
        return "stop"
    if g > r + 8.0 and g > b:
        return "go"
    return None


def detector_from_mode(
    mode: str,
    *,
    onnx_path: Optional[Union[str, Path]] = None,
    engine_path: Optional[Union[str, Path]] = None,
    inner: Optional[Detector] = None,
) -> Detector:
    key = (mode or "mock").strip().lower()
    if key in {"blind"}:
        return BlindDetector()
    if key in {"onnx", "appearance"}:
        return AppearanceDetector(onnx_path)
    if key in {"trt", "tensorrt"}:
        from jims_mower.perception.trt import TrtDetector

        return TrtDetector(engine_path, inner or MockDetector())
    return inner or MockDetector()

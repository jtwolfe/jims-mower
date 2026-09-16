"""Appearance / ONNX detector scaffolding (CV-3). Not a published mAP.

``MockDetector`` stays the gym default (projects ``context.obstacles``).
This path **must ignore** that god-view list. Without an ONNX file it
returns ``[]``. Living interlock still consumes the ``detections`` ICD
key — empty means the interlock does not fire, which is honest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from jims_mower.perception.base import BlindDetector, Detector
from jims_mower.perception.mock import MockDetector
from jims_mower.types import Detection, PerceptionContext


class AppearanceDetector:
    """Appearance-based detector that can load ONNX later.

    Does not read ``context.obstacles``. ``map_claim`` / ``fps_claim``
    stay null. A future box head can sit behind the same type.
    """

    def __init__(
        self,
        onnx_path: Optional[Union[str, Path]] = None,
        *,
        session=None,
    ) -> None:
        self.onnx_path = str(onnx_path) if onnx_path else ""
        self.session = session
        self.map_claim = None
        self.fps_claim = None
        self.iou_claim = None
        if self.session is None and self.onnx_path:
            dest = Path(self.onnx_path)
            if dest.is_file():
                from jims_mower.perception.onnx_io import load_onnx_session, onnxruntime_available

                if onnxruntime_available():
                    self.session = load_onnx_session(dest)
        self.backend = "onnx" if self.session is not None else "empty"

    def detect(
        self,
        images: dict[str, np.ndarray],
        context: PerceptionContext,
    ) -> list[Detection]:
        del images
        # Honest: no god-view projector. A later ONNX box head would run here.
        _ = context.obstacles
        if self.session is None:
            return []
        # No shipped detector graph — do not invent boxes from a terrain MLP.
        return []


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

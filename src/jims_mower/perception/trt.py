"""TensorRT *load-weights* placeholders behind Detector / TerrainObserver.

No engine is shipped. CI never imports TensorRT. When ``engine_path`` is
missing the wrappers delegate to the numpy / mock heads already in-tree.
``fps_claim`` is always ``None``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from jims_mower.perception.mock import MockDetector
from jims_mower.perception.terrain import (
    HeuristicTerrainObserver,
    LearnedTerrainObserver,
    TerrainEstimate,
    TerrainObserver,
)
from jims_mower.types import Detection, PerceptionContext


def tensorrt_available() -> bool:
    try:
        import tensorrt  # noqa: F401
    except Exception:
        return False
    return True


def _engine_ready(path: Optional[Union[str, Path]]) -> bool:
    if not path:
        return False
    dest = Path(path)
    return dest.is_file() and dest.suffix.lower() in {".engine", ".plan", ".trt"}


class TrtDetector:
    """Detector backend hook. Falls back to ``MockDetector`` without an engine."""

    def __init__(
        self,
        engine_path: Optional[Union[str, Path]] = None,
        inner: Optional[object] = None,
    ) -> None:
        self.engine_path = str(engine_path) if engine_path else ""
        self.inner = inner or MockDetector()
        self.backend = "trt" if (_engine_ready(engine_path) and tensorrt_available()) else "mock"
        self.fps_claim = None
        self.map_claim = None

    def detect(
        self,
        images: dict[str, np.ndarray],
        context: PerceptionContext,
    ) -> list[Detection]:
        # A real Orin build would deserialize the engine here. This repo
        # does not ship weights; keep the mock / inner path honest.
        return self.inner.detect(images, context)


class TrtTerrainObserver:
    """TerrainObserver backend hook. Falls back to heuristic / learned numpy."""

    def __init__(
        self,
        engine_path: Optional[Union[str, Path]] = None,
        inner: Optional[TerrainObserver] = None,
        *,
        weights_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.engine_path = str(engine_path) if engine_path else ""
        if inner is not None:
            self.inner = inner
        elif weights_path and Path(weights_path).is_file():
            self.inner = LearnedTerrainObserver(weights_path)
        else:
            self.inner = HeuristicTerrainObserver()
        self.backend = "trt" if (_engine_ready(engine_path) and tensorrt_available()) else "numpy"
        self.fps_claim = None
        self.map_claim = None

    def reset(self) -> None:
        reset = getattr(self.inner, "reset", None)
        if callable(reset):
            reset()

    def estimate(
        self,
        images: dict[str, np.ndarray],
        imu: np.ndarray,
        gps: np.ndarray,
        context: PerceptionContext,
    ) -> TerrainEstimate:
        return self.inner.estimate(images, imu, gps, context)

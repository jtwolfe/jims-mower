"""Pluggable perception protocols. Swap these on the real robot."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from jims_mower.types import Detection, PerceptionContext


@runtime_checkable
class Detector(Protocol):
    """Object detector. Sim ships a mock; hardware should inject a real one."""

    def detect(
        self,
        images: dict[str, np.ndarray],
        context: PerceptionContext,
    ) -> list[Detection]:
        ...


@runtime_checkable
class GrassObserver(Protocol):
    """Per-image grass-coverage estimate (color heuristic or a learned head)."""

    def estimate(self, images: dict[str, np.ndarray]) -> dict[str, float]:
        ...


class BlindDetector:
    """Working stub: always returns no detections."""

    def detect(
        self,
        images: dict[str, np.ndarray],
        context: PerceptionContext,
    ) -> list[Detection]:
        del images, context
        return []

"""Perception plugins: detectors, grass observers, hand-signal labels."""

from jims_mower.perception.base import BlindDetector, Detector, GrassObserver
from jims_mower.perception.grass import ColorGrassObserver, grass_fraction, uncut_grass_mask
from jims_mower.perception.hand_signals import HandSignalCurriculum
from jims_mower.perception.mock import MockDetector, detection_from_obstacle

__all__ = [
    "BlindDetector",
    "ColorGrassObserver",
    "Detector",
    "GrassObserver",
    "HandSignalCurriculum",
    "MockDetector",
    "detection_from_obstacle",
    "grass_fraction",
    "uncut_grass_mask",
]

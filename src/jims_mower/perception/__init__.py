"""Perception plugins: detectors, grass observers, hand-signal labels."""

from jims_mower.perception.base import BlindDetector, Detector, GrassObserver
from jims_mower.perception.grass import ColorGrassObserver, grass_fraction, uncut_grass_mask
from jims_mower.perception.hand_signals import HandSignalCurriculum
from jims_mower.perception.mock import MockDetector, detection_from_obstacle
from jims_mower.perception.cv_terrain import classify_terrain_rgb, drain_pixel_fraction
from jims_mower.perception.terrain import (
    BlindTerrainObserver,
    HeuristicTerrainObserver,
    OracleTerrainObserver,
    TerrainEstimate,
    TerrainObserver,
    terrain_observer_from_mode,
)

__all__ = [
    "BlindDetector",
    "BlindTerrainObserver",
    "ColorGrassObserver",
    "Detector",
    "GrassObserver",
    "HandSignalCurriculum",
    "HeuristicTerrainObserver",
    "MockDetector",
    "OracleTerrainObserver",
    "TerrainEstimate",
    "TerrainObserver",
    "classify_terrain_rgb",
    "detection_from_obstacle",
    "drain_pixel_fraction",
    "grass_fraction",
    "terrain_observer_from_mode",
    "uncut_grass_mask",
]

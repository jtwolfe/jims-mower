"""Perception plugins: detectors, grass observers, hand-signal labels."""

from jims_mower.perception.base import BlindDetector, Detector, GrassObserver
from jims_mower.perception.grass import ColorGrassObserver, grass_fraction, uncut_grass_mask
from jims_mower.perception.hand_signals import HandSignalCurriculum
from jims_mower.perception.mock import MockDetector, detection_from_obstacle
from jims_mower.perception.cv_terrain import classify_terrain_rgb, drain_pixel_fraction
from jims_mower.perception.fuse import fuse_camera_labels, fuse_stamps
from jims_mower.perception.learn import TerrainMLP, load_weights, save_weights
from jims_mower.perception.temporal import DetectionTracklets, HazardHysteresis
from jims_mower.perception.terrain import (
    BlindTerrainObserver,
    HeuristicTerrainObserver,
    LearnedTerrainObserver,
    OracleTerrainObserver,
    TerrainEstimate,
    TerrainObserver,
    terrain_observer_from_mode,
)

__all__ = [
    "BlindDetector",
    "BlindTerrainObserver",
    "ColorGrassObserver",
    "DetectionTracklets",
    "Detector",
    "GrassObserver",
    "HandSignalCurriculum",
    "HazardHysteresis",
    "HeuristicTerrainObserver",
    "LearnedTerrainObserver",
    "MockDetector",
    "OracleTerrainObserver",
    "TerrainEstimate",
    "TerrainMLP",
    "TerrainObserver",
    "classify_terrain_rgb",
    "detection_from_obstacle",
    "drain_pixel_fraction",
    "fuse_camera_labels",
    "fuse_stamps",
    "grass_fraction",
    "load_weights",
    "save_weights",
    "terrain_observer_from_mode",
    "uncut_grass_mask",
]

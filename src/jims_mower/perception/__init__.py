"""Perception plugins: detectors, grass observers, hand-signal labels."""

from jims_mower.perception.base import BlindDetector, Detector, GrassObserver
from jims_mower.perception.classify import HandSignalClassifier, appearance_features, refine_category
from jims_mower.perception.grass import (
    ColorGrassObserver,
    FeatureGrassObserver,
    grass_fraction,
    grass_observer_from_mode,
    uncut_grass_mask,
)
from jims_mower.perception.hand_signals import HandSignalCurriculum
from jims_mower.perception.mock import MockDetector, detection_from_obstacle
from jims_mower.perception.semantic import semantic_raster
from jims_mower.perception.trt import TrtDetector, TrtTerrainObserver, tensorrt_available
from jims_mower.perception.cv_terrain import (
    classify_structure_rgb,
    classify_terrain_rgb,
    drain_pixel_fraction,
)
from jims_mower.perception.grade import PlanarGradeModel, gradients_from_attitude
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
    "FeatureGrassObserver",
    "GrassObserver",
    "HandSignalClassifier",
    "HandSignalCurriculum",
    "HazardHysteresis",
    "HeuristicTerrainObserver",
    "LearnedTerrainObserver",
    "MockDetector",
    "OracleTerrainObserver",
    "TrtDetector",
    "TrtTerrainObserver",
    "TerrainEstimate",
    "TerrainMLP",
    "TerrainObserver",
    "appearance_features",
    "classify_structure_rgb",
    "classify_terrain_rgb",
    "gradients_from_attitude",
    "PlanarGradeModel",
    "detection_from_obstacle",
    "drain_pixel_fraction",
    "fuse_camera_labels",
    "fuse_stamps",
    "grass_fraction",
    "grass_observer_from_mode",
    "load_weights",
    "refine_category",
    "save_weights",
    "semantic_raster",
    "tensorrt_available",
    "terrain_observer_from_mode",
    "uncut_grass_mask",
]

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
from jims_mower.perception.stereo import (
    STEREO_NOTE,
    StereoPair,
    StereoPairError,
    baseline_cm,
    depth_resolution_m,
    disparity_px,
    find_stereo_pair,
    range_from_disparity,
    rasterize_points,
    require_stereo_pair,
    synthetic_stereo_points,
)
from jims_mower.perception.calibration import (
    CalibrationError,
    ExtrinsicsBundle,
    load_extrinsics,
    report_baseline_cm,
    run_gym_acceptance,
    run_lip_fixture,
    tape_vs_ideal_disparity,
    validate_stereo_yaml,
)
from jims_mower.perception.grade import PlanarGradeModel, gradients_from_attitude, paint_planar_grade
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
    "paint_planar_grade",
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
    "STEREO_NOTE",
    "StereoPair",
    "StereoPairError",
    "baseline_cm",
    "CalibrationError",
    "ExtrinsicsBundle",
    "depth_resolution_m",
    "disparity_px",
    "find_stereo_pair",
    "load_extrinsics",
    "range_from_disparity",
    "rasterize_points",
    "report_baseline_cm",
    "require_stereo_pair",
    "run_gym_acceptance",
    "run_lip_fixture",
    "synthetic_stereo_points",
    "tape_vs_ideal_disparity",
    "validate_stereo_yaml",
    "tensorrt_available",
    "terrain_observer_from_mode",
    "uncut_grass_mask",
]

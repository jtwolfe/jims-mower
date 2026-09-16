"""Perception plugins: detectors, grass observers, hand-signal labels."""

from jims_mower.perception.base import BlindDetector, Detector, GrassObserver
from jims_mower.perception.classify import HandSignalClassifier, appearance_features, refine_category
from jims_mower.perception.grass import (
    ClassAwareGrassObserver,
    ColorGrassObserver,
    FeatureGrassObserver,
    classify_coverage_rgb,
    coverage_error,
    cut_grass_mask,
    grass_fraction,
    grass_observer_from_mode,
    paint_lawn_strip_fixture,
    uncut_grass_mask,
)
from jims_mower.perception.hand_signals import HandSignalCurriculum
from jims_mower.perception.mock import MockDetector, detection_from_obstacle
from jims_mower.perception.semantic import semantic_raster
from jims_mower.perception.detect import (
    AppearanceDetector,
    detect_palette_blobs,
    detector_from_mode,
    gym_red_bias_signal,
    paint_kind_blob,
)
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
# Do not import calibration here. It can need ObservedMap; planning.observed
# imports perception.grade, which loads this package. Eager calibration
# is the circular import that broke `jims-mower-mission inspect`.
from jims_mower.perception.grade import PlanarGradeModel, gradients_from_attitude, paint_planar_grade
from jims_mower.perception.elev_fuse import (
    ElevFuseResult,
    MonoDepthPrior,
    NullMonoDepthPrior,
    apply_mono_prior,
    fuse_elev_stereo_tof_imu,
)
from jims_mower.perception.fuse import fuse_camera_labels, fuse_stamps
from jims_mower.perception.learn import TerrainMLP, load_weights, save_weights
from jims_mower.perception.temporal import DetectionTracklets, HazardHysteresis
from jims_mower.perception.onnx_io import (
    OnnxExportError,
    OnnxMLP,
    export_terrain_onnx,
    onnx_available,
    onnxruntime_available,
)
from jims_mower.perception.terrain import (
    BlindTerrainObserver,
    HeuristicTerrainObserver,
    LearnedTerrainObserver,
    OnnxTerrainObserver,
    OracleTerrainObserver,
    TerrainEstimate,
    TerrainObserver,
    normalize_terrain_mode,
    terrain_observer_from_mode,
)

__all__ = [
    "AppearanceDetector",
    "BlindDetector",
    "BlindTerrainObserver",
    "ClassAwareGrassObserver",
    "ColorGrassObserver",
    "ElevFuseResult",
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
    "MonoDepthPrior",
    "NullMonoDepthPrior",
    "OnnxExportError",
    "OnnxMLP",
    "OnnxTerrainObserver",
    "OracleTerrainObserver",
    "TrtDetector",
    "TrtTerrainObserver",
    "TerrainEstimate",
    "TerrainMLP",
    "TerrainObserver",
    "appearance_features",
    "apply_mono_prior",
    "classify_coverage_rgb",
    "classify_structure_rgb",
    "classify_terrain_rgb",
    "coverage_error",
    "cut_grass_mask",
    "fuse_elev_stereo_tof_imu",
    "paint_lawn_strip_fixture",
    "gradients_from_attitude",
    "PlanarGradeModel",
    "paint_planar_grade",
    "detection_from_obstacle",
    "detect_palette_blobs",
    "detector_from_mode",
    "gym_red_bias_signal",
    "paint_kind_blob",
    "drain_pixel_fraction",
    "export_terrain_onnx",
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
    "normalize_terrain_mode",
    "onnx_available",
    "onnxruntime_available",
    "tensorrt_available",
    "terrain_observer_from_mode",
    "uncut_grass_mask",
]

_CALIB_EXPORTS = frozenset(
    {
        "CalibrationError",
        "ExtrinsicsBundle",
        "load_extrinsics",
        "report_baseline_cm",
        "run_gym_acceptance",
        "run_lip_fixture",
        "tape_vs_ideal_disparity",
        "validate_stereo_yaml",
    }
)


def __getattr__(name: str):
    if name in _CALIB_EXPORTS:
        from jims_mower.perception import calibration as _calibration

        return getattr(_calibration, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

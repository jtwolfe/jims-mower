"""Shared label tables for detections and curriculum signals."""

from __future__ import annotations

LIVING_KINDS = frozenset({"person", "dog", "cat", "bird"})
ANIMAL_KINDS = frozenset({"dog", "cat", "bird"})
STATIC_KINDS = frozenset({"tree", "furniture", "toy", "hose", "cord"})
SOFT_KINDS = frozenset({"hose", "cord"})
CUTTER_RISK_KINDS = frozenset({"hose", "cord"})
ALL_KINDS = LIVING_KINDS | STATIC_KINDS

WORLD_LAYOUTS = frozenset(
    {
        "random",
        "suburban",
        "paddock",
        "playground",
        "orchard",
        "terrace",
        "kerb_gutter",
        "swale",
        "golf_rough",
        "golf_fairway",
        "acre_yard",
    }
)
WEATHER_PACKS = frozenset({"clear", "dawn", "dusk", "night", "rain"})

HAND_SIGNALS = ("stop", "go", "follow", "back")
TRAJECTORY_MODES = frozenset({"wander", "patrol", "loop", "line"})
MOVER_DENSITIES = frozenset({"sparse", "default", "busy"})
RECOVERY_MODES = ("idle", "reverse", "pivot", "help")
TOF_COUNTS = frozenset({0, 2, 4})

# Count overlays for world.movers.density. ``default`` leaves YAML counts.
DENSITY_COUNTS = {
    "sparse": {"person": 0, "dog": 1, "cat": 0, "bird": 0},
    "busy": {"person": 2, "dog": 2, "cat": 1, "bird": 1},
}

LABEL_TO_ID = {
    "person": 1,
    "dog": 2,
    "cat": 3,
    "bird": 4,
    "tree": 5,
    "furniture": 6,
    "toy": 7,
    "hose": 8,
    "cord": 9,
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

SIGNAL_TO_ID = {name: i + 1 for i, name in enumerate(HAND_SIGNALS)}
ID_TO_SIGNAL = {v: k for k, v in SIGNAL_TO_ID.items()}

MAX_DETECTIONS = 24
DET_FEATURES = 8  # label, camera, u, v, w, h, conf, signal

# Synthetic palette used by the renderer (and the color-heuristic grass hook).
UNCUT_GRASS_RGB = (46, 140, 58)
LONG_GRASS_RGB = (28, 108, 42)
CUT_GRASS_RGB = (168, 148, 72)
DIRT_RGB = (90, 70, 50)
SKY_RGB = (135, 186, 230)
DRAIN_RGB = (58, 42, 28)
DRAIN_EDGE_RGB = (86, 62, 40)
BANK_RGB = (72, 118, 52)
PUDDLE_RGB = (48, 92, 128)
POND_RGB = (36, 78, 118)
PATH_RGB = (128, 128, 122)
BUNKER_RGB = (210, 180, 120)
BUILDING_RGB = (92, 92, 98)
GARDEN_RGB = (72, 96, 48)
GREEN_RGB = (36, 128, 72)

# Height-field labels (first-class yard features).
TERRAIN_FLAT = 0
TERRAIN_BANK = 1
TERRAIN_DRAIN = 2
TERRAIN_DRAIN_EDGE = 3
TERRAIN_PUDDLE = 4
TERRAIN_PATH = 5
TERRAIN_BUILDING = 6
TERRAIN_BUNKER = 7
TERRAIN_GARDEN = 8
TERRAIN_GREEN = 9
TERRAIN_POND = 10

# Human-structure raster (obs["structure"] / info["structure"]).
STRUCTURE_NONE = 0
STRUCTURE_PATH = 1
STRUCTURE_BUILDING = 2
STRUCTURE_BUNKER = 3
STRUCTURE_GARDEN = 4
STRUCTURE_GREEN = 5
STRUCTURE_POND = 6
STRUCTURE_NAMES = (
    "none",
    "path_paved",
    "building",
    "bunker",
    "garden_bed",
    "green",
    "pond",
)
PATH_COST = 8.0
BUNKER_COST = 12.0

# Observation / safety hazard raster: 0 free, 1 steep, 2 drain lip, 3 channel.
HAZARD_NONE = 0
HAZARD_STEEP = 1
HAZARD_DRAIN_EDGE = 2
HAZARD_DRAIN = 3

GRAVITY_MPS2 = 9.80665

TERRAIN_ADVICE = ("ok", "slow", "stop", "reroute")
MISSION_SCHEMA = "jims_mower.mission.v1"
MISSION_FLOW_SCHEMA = "jims_mower.mission_flow.v1"
MISSION_PHASES = (
    "calibrate_boundary",
    "explore",
    "review",
    "mow",
    "return_home",
    "complete",
    "fault",
    "safe",
)
SAFE_MODES = ("run", "limp", "estop", "safe")
BC_SCHEMA = "jims_mower.bc.v1"
TELEMETRY_SCHEMA = "jims_mower.telemetry.v1"
OWNER_OVERLAY_SCHEMA = "jims_mower.owner.v1"
DATASET_SCHEMA = "jims_mower.dataset.v1"
SURVEY_SCHEMA = "jims_mower.survey.v1"
VIEWER_SCHEMA = "jims_mower.viewer.v1"
LIVE_SCHEMA = "jims_mower.live.v1"
SESSION_SCHEMA = "jims_mower.session.v1"
LIVE_CONTROL_CMDS = (
    "start",
    "pause",
    "resume",
    "speed",
    "start_mow",
    "reexplore",
    "estop",
    "hold",
    "clear",
    "yard",
    "inject",
    "pair",
)
APP_LIVE_PORT = 8766
YARD_PROFILE_SCHEMA = "jims_mower.yard.v1"
MESH_SCHEMA = "jims_mower.mesh.v1"
APP_STATUS_SCHEMA = "jims_mower.app_status.v1"
COVERAGE_MAP_SCHEMA = "jims_mower.coverage.v1"
APP_COMMANDS = ("start", "stop", "return", "estop", "teach")
APP_MISSIONS = ("idle", "mowing", "returning", "teach", "estop")
RADIO_LINKS = ("bluetooth", "wifi", "lora")
SCHEDULE_DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
CURRICULUM_SCHEMA = "jims_mower.curriculum.v1"
SEQUENCE_SCHEMA = "jims_mower.sequence.v1"
OVERLAY_SCHEMA = "jims_mower.overlay.v1"
SEASONS = frozenset({"none", "long_grass", "leaf_clutter"})
DETECTION_CATEGORIES = ("person", "animal", "toy", "static")

# Optional semantic layer (info["semantic"], not a published score).
SEMANTIC_FREE = 0
SEMANTIC_GRASS = 1
SEMANTIC_NON_GRASS = 2
SEMANTIC_DRAIN = 3
SEMANTIC_BANK = 4
SEMANTIC_STATIC = 5
SEMANTIC_PATH = 6
SEMANTIC_BUILDING = 7
SEMANTIC_BUNKER = 8
SEMANTIC_GARDEN = 9
SEMANTIC_NAMES = (
    "free",
    "grass",
    "non_grass",
    "drain",
    "bank",
    "static",
    "path_paved",
    "building",
    "bunker",
    "garden_bed",
)
NEAR_MISS_LIVING_M = 1.5
BC_FEATURE_DIM = 40
DEFAULT_BC_WEIGHTS = "bc_weights.npz"

# WAVE UX-B — faults + radio sim (software only; no RF hardware).
FAULT_SCHEMA = "jims_mower.fault.v1"
RADIO_SCHEMA = "jims_mower.radio.v1"
SELFTEST_SCHEMA = "jims_mower.selftest.v1"
FAULT_CODES = (
    "ok",
    "STUCK",
    "FAULT_IMMOBILISED",
    "TRIMMER_JAM",
    "CAM_BLIND",
    "IMU_FREEZE",
    "GNSS_DROPOUT",
    "RADIO_LOSS",
)
MOTOR_KILL_MODES = frozenset({"cmd_ignored", "encoder_stuck", "open_circuit"})
RADIO_CHANNELS = ("wifi", "bt", "lora")
RADIO_LOSS_ACTIONS = frozenset({"limp_home", "stop_beacon"})

KIND_RGB = {
    "person": (220, 80, 80),
    "dog": (160, 100, 50),
    "cat": (200, 160, 80),
    "bird": (80, 80, 220),
    "tree": (30, 90, 40),
    "furniture": (120, 80, 140),
    "toy": (240, 180, 40),
    "hose": (36, 78, 150),
    "cord": (28, 28, 28),
}

DEFAULT_RADII = {
    "person": 0.25,
    "dog": 0.20,
    "cat": 0.12,
    "bird": 0.08,
    "tree": 0.35,
    "furniture": 0.30,
    "toy": 0.10,
    "hose": 0.06,
    "cord": 0.04,
}

DEFAULT_SPEEDS = {
    "person": 0.35,
    "dog": 0.70,
    "cat": 0.50,
    "bird": 1.10,
}

DEFAULT_HEIGHTS = {
    "person": 0.90,
    "dog": 0.35,
    "cat": 0.20,
    "bird": 1.40,
    "tree": 0.40,
    "furniture": 0.40,
    "toy": 0.08,
    "hose": 0.03,
    "cord": 0.02,
}

# Birds above this height are a trimmer-safety concern but not a body collision.
BIRD_COLLISION_Z_M = 0.60

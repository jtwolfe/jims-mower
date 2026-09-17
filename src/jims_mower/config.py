"""YAML-backed environment configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from jims_mower.constants import (
    COVERAGE_SOURCES,
    DETECTOR_BACKENDS,
    GYM_PAIR_PIN,
    INTERLOCK_SOURCES,
    GRASS_MODES,
    MOTOR_KILL_MODES,
    MOVER_DENSITIES,
    RADIO_CHANNELS,
    RADIO_LOSS_ACTIONS,
    SEASONS,
    TERRAIN_MODES,
    TOF_COUNTS,
    TRAJECTORY_MODES,
    WEATHER_PACKS,
    WORLD_LAYOUTS,
)
from jims_mower.types import CameraSpec

def _discover_default_config() -> Path:
    packaged = Path(__file__).resolve().parent / "data" / "default.yaml"
    if packaged.is_file():
        return packaged
    repo = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    return repo


DEFAULT_CONFIG_PATH = _discover_default_config()

# Built-in gym look-around (body frame: x forward, y left, z up). Used
# when YAML leaves `sensors.cameras` empty.
# front_left / front_right are 40° yaw at ~40 cm — NOT a stereo pair.
# Metric near-field depth wants configs/orin/extrinsics_stereo.yaml
# (6–12 cm baseline, shared yaw/pitch). Front pair is pitched down so
# drain lips sit in the lower image third.
_DEFAULT_RIG = (
    CameraSpec("front", 0.25, 0.00, 0.38, 0.0, -22.0),
    CameraSpec("front_left", 0.20, 0.20, 0.38, 40.0, -18.0),
    CameraSpec("front_right", 0.20, -0.20, 0.38, -40.0, -18.0),
    CameraSpec("rear", -0.25, 0.00, 0.38, 180.0, -12.0),
    CameraSpec("left", 0.00, 0.25, 0.38, 90.0, -12.0),
    CameraSpec("right", 0.00, -0.25, 0.38, -90.0, -12.0),
)

# 4-cam: cardinal. 5-cam: add front_left. 6-cam: full rig.
_COUNT_NAMES = {
    4: ("front", "rear", "left", "right"),
    5: ("front", "front_left", "rear", "left", "right"),
    6: ("front", "front_left", "front_right", "rear", "left", "right"),
}


class ConfigError(ValueError):
    """Invalid gym configuration."""


@dataclass
class DriveConfig:
    """Command → wheel m/s scale. ``measured: false`` until a tape / stopwatch bench.

    Gym: a 1.0 wheel command moves at ``max_wheel_speed_mps * scale``.
    Do not invent Kv. See ``docs/SCALE.md``.
    """

    scale: float = 1.0
    measured: bool = False
    measured_at: str = ""
    notes: str = ""


@dataclass
class TrimmerConfig:
    offset_m: float = 0.32
    radius_m: float = 0.16
    safety_radius_m: float = 1.50
    height_m: float = 0.12
    # Command → RPM scale. ``measured: false`` until a tach bench. No invented Kv.
    scale: float = 1.0
    measured: bool = False
    measured_at: str = ""
    notes: str = ""


@dataclass
class RobotConfig:
    length_m: float = 0.50
    width_m: float = 0.50
    height_m: float = 0.50
    wheelbase_m: float = 0.40
    track_m: float = 0.40
    max_wheel_speed_mps: float = 1.2
    collision_radius_m: float = 0.28
    tip_roll_rad: float = 0.40
    tip_pitch_rad: float = 0.45
    wheel_drop_m: float = 0.08
    steep_slope_rad: float = 0.30
    trimmer: TrimmerConfig = field(default_factory=TrimmerConfig)
    drive: DriveConfig = field(default_factory=DriveConfig)

    def wheel_speed_mps(self) -> float:
        """1.0 command → m/s. Scale is 1.0 until a human measures it."""
        return float(self.max_wheel_speed_mps) * float(self.drive.scale)


@dataclass
class IMUConfig:
    enabled: bool = True
    accel_noise_std: float = 0.05
    gyro_noise_std: float = 0.015
    accel_bias_std: float = 0.02


@dataclass
class GPSConfig:
    enabled: bool = True
    horiz_noise_std_m: float = 1.2
    vert_noise_std_m: float = 2.0
    dropout_prob: float = 0.02
    include_altitude: bool = True


@dataclass
class ToFConfig:
    enabled: bool = True
    count: int = 4  # 0, 2 (FL/FR), or 4 (FL, FR, RL, RR)
    noise_std_m: float = 0.012
    max_range_m: float = 1.2


@dataclass
class CalibrationConfig:
    """Extrinsics provenance. Example YAML stays ``measured: false``.

    See ``docs/CALIBRATION.md``. A human on the rig still has to tape
    the baseline and commit ``extrinsics_stereo_measured.yaml``.
    """

    measured: bool = False
    template: bool = False
    measured_at: str = ""
    tape_baseline_cm: Optional[float] = None
    notes: str = ""


@dataclass
class SensorsConfig:
    width: int = 80
    height: int = 60
    camera_count: int = 6
    fov_deg: float = 70.0
    cameras: list[CameraSpec] = field(default_factory=list)
    imu: IMUConfig = field(default_factory=IMUConfig)
    gps: GPSConfig = field(default_factory=GPSConfig)
    tof: ToFConfig = field(default_factory=ToFConfig)


@dataclass
class BaseGradientConfig:
    """Yard-scale planar grade plus optional long-wavelength undulation.

    ``slope_rad`` is the planar grade (atan of rise/run). ``yaw_rad`` is
    the direction of *ascent* (elevation increases along this heading).
    ``undulation_m`` is a gentle sine roll plus a weak quadratic dish.
    ``slope_pct`` is an optional percent-grade alias (5.0 → ~5%); when
    set it overrides ``slope_rad``.
    """

    slope_rad: float = 0.05
    yaw_rad: float = 0.55
    undulation_m: float = 0.04
    slope_pct: Optional[float] = None

    def effective_slope_rad(self) -> float:
        if self.slope_pct is not None:
            return math.atan(float(self.slope_pct) / 100.0)
        return float(self.slope_rad)


@dataclass
class TerrainConfig:
    enabled: bool = True
    n_drains: int = 2
    n_banks: int = 2
    drain_width_m: float = 0.40
    drain_depth_m: float = 0.16
    drain_length_m: float = 4.0
    drain_side_slope: float = 1.5
    bank_height_m: float = 0.40
    bank_width_m: float = 1.8
    bank_length_m: float = 3.5
    max_slope_rad: float = 0.45
    noise_amp_m: float = 0.015
    keepout_m: float = 1.6
    puddle_radius_m: float = 0.45
    puddle_depth_m: float = 0.04
    base_gradient: BaseGradientConfig = field(default_factory=BaseGradientConfig)
    multi_scale_amp_m: float = 0.0
    swale_amp_m: float = 0.0
    dem_path: Optional[str] = None


@dataclass
class GrassGrowthConfig:
    """Multi-session stub: grow cut grass back and persist the yard mask."""

    enabled: bool = False
    regenerate_frac: float = 0.05
    persist_path: Optional[str] = None


@dataclass
class MoverConfig:
    """Living-agent motion. Trajectories stay cheap (no social-force model)."""

    heading_jitter: float = 0.35
    default_mode: str = "wander"
    density: str = "default"


@dataclass
class WeatherConfig:
    """Time-of-day / weather pack applied by the geometric renderer."""

    pack: str = "clear"
    porch_lights: bool = False
    colour_temperature_k: float = 0.0


@dataclass
class DomainRandomizationConfig:
    """Seeded renderer appearance knobs. Off → pre-1C geometric look."""

    enabled: bool = False
    seed: int = 0
    lighting: bool = True
    colour_jitter: bool = True
    shadow_blobs: bool = True
    motion_blur: bool = False
    camera_dirt: bool = True
    vignette: bool = True
    wet_specular: bool = False


@dataclass
class WorldConfig:
    width_m: float = 12.0
    height_m: float = 12.0
    resolution_m: float = 0.10
    n_people: int = 1
    n_dogs: int = 1
    n_cats: int = 1
    n_birds: int = 1
    n_trees: int = 3
    n_furniture: int = 1
    n_toys: int = 2
    n_hoses: int = 0
    n_cords: int = 0
    n_puddles: int = 0
    layout: str = "random"
    orchard_rows: int = 3
    orchard_cols: int = 4
    season: str = "none"
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    grass: GrassGrowthConfig = field(default_factory=GrassGrowthConfig)
    movers: MoverConfig = field(default_factory=MoverConfig)


@dataclass
class RewardConfig:
    coverage_scale: float = 1.0
    time_penalty: float = 0.01
    collision_living: float = 50.0
    collision_static: float = 20.0
    out_of_bounds: float = 10.0
    completion_bonus: float = 15.0
    completion_threshold: float = 0.95
    tipover: float = 40.0
    drain_drop: float = 30.0
    steep: float = 2.0


@dataclass
class CurriculumConfig:
    hand_signals: bool = False
    signal_hold_steps: int = 40
    hand_signal_classifier: bool = False


@dataclass
class PerceptionConfig:
    terrain_mode: str = "heuristic"  # heuristic | oracle | blind | learned | onnx | trt
    weights_path: str = ""  # optional .npz for LearnedTerrainObserver
    onnx_path: str = ""  # optional sim_only terrain / detector ONNX; unused in CI
    temporal: bool = False  # hysteresis on heuristic; learned defaults on in factory
    grass_mode: str = "color"  # color | feature | class
    coverage_source: str = "gym_grid"  # gym_grid | observer
    semantic: bool = True
    persistent_occupancy: bool = True
    occupancy_decay: float = 0.55
    height_fusion: bool = True
    loop_closure: bool = True
    detector_backend: str = "mock"  # mock | trt | onnx | appearance | blind
    engine_path: str = ""  # TensorRT .engine placeholder; unused in CI
    # PLN-4: auto → detections when appearance/onnx/blind, obstacles when mock
    interlock_source: str = "auto"


@dataclass
class EkfConfig:
    """Process / measurement noise for ``EkfPoseFilter`` (metres, radians)."""

    q_xy: float = 0.04
    q_z: float = 0.08
    q_yaw: float = 0.03
    q_tilt: float = 0.04
    q_odom: float = 0.06
    r_gps_xy: float = 1.2
    r_gps_z: float = 2.0
    r_tilt: float = 0.10
    use_gps_z: bool = True
    gps_gate_m: float = 5.0
    gyro_yaw_mix: float = 0.30


@dataclass
class UncertaintyConfig:
    """Costmap inflation when hazard/slope confidence is low."""

    inflate: float = 1.5
    hazard_boost: float = 4.0
    confidence_floor: float = 0.25


@dataclass
class SafeStateConfig:
    """ESTOP / limp / safe hold. Used by the controller, not a claimed SIL rating."""

    limp_after_stops: int = 8
    safe_after_limp_steps: int = 12
    limp_scale: float = 0.35
    recover_ok_steps: int = 4


@dataclass
class PlannerConfig:
    """Coverage planner + controller knobs (max climb, drain clearance, slow)."""

    max_climb_slope_rad: float = 0.32
    drain_clearance_m: float = 0.40
    slow_speed_factor: float = 0.35
    # Extra scale on KIND_GRADE / look-ahead steep (explore and mow).
    grade_speed_factor: float = 0.50
    # Forward sit-probe (cameras / stereo / ToF elevation), not a lidar stack.
    grade_look_ahead_m: float = 1.10
    grade_look_ahead_samples: int = 6
    strip_spacing_m: float = 0.28
    waypoint_stride_m: float = 0.32
    arrive_radius_m: float = 0.20
    turn_in_place_rad: float = 0.70
    cruise_speed: float = 0.55
    imu_slow_frac: float = 0.55
    imu_stop_frac: float = 0.85
    # Consecutive filtered frames before IMU stop (noise / hill spike).
    imu_stop_hold_steps: int = 3
    imu_tilt_window: int = 5
    # Costmap lethal grade is tip_lethal_frac * min(tip_roll, tip_pitch).
    # Between max_climb and that margin: high contour cost, not blocked.
    tip_lethal_frac: float = 0.95
    contour_cost: float = 12.0
    gps_blend: float = 0.08
    accel_blend: float = 0.10
    max_replans: int = 8
    occupancy_inflate_m: float = 0.20
    pose_filter: str = "ekf"  # ekf | complementary
    ekf: EkfConfig = field(default_factory=EkfConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    living_slow_m: float = 3.0
    living_reroute_m: float = 1.8
    living_stop_m: float = 0.90
    geofence_inflate_m: float = 0.30
    geofence_slow_m: float = 0.80
    geofence_stop_m: float = 0.28
    recovery_trigger: int = 3
    recovery_reverse_steps: int = 6
    recovery_pivot_steps: int = 5
    max_recoveries: int = 2
    safe_state: SafeStateConfig = field(default_factory=SafeStateConfig)
    wet_slope_extra: float = 3.0
    energy_aware_strips: bool = True
    blend_elevation_prior: bool = True
    elevation_prior_weight: float = 0.55
    path_cost: float = 8.0
    bunker_cost: float = 12.0


@dataclass
class MissionConfig:
    """Calibrate → explore → review → mow. Unknown is not assumed mowable."""

    observe_confidence: float = 0.35
    explore_complete: float = 1.0
    explore_no_frontier: float = 0.90
    stamp_radius_m: float = 1.35
    camera_range_m: float = 6.0
    max_calibrate_steps: int = 900
    max_explore_steps: int = 2800
    max_mow_steps: int = 5500
    max_return_steps: int = 500
    snapshot_stride: int = 25
    review_min_closure_m: float = 0.80
    # Wheel-command fraction. Actual ≈ explore_cruise * max_wheel_speed_mps
    # (default 0.45 × 1.2 ≈ 0.54 m/s). Fast CI settings may raise this.
    explore_cruise: float = 0.45
    mow_cruise: float = 0.55
    # 0 = auto from yard size. Large yards use a longer stride so the
    # teach tracker does not chatter on a 200 m fence.
    calibrate_stride_m: float = 0.0
    calibrate_cruise: float = 0.0
    calibrate_arrive_m: float = 0.0
    # 0 = full perimeter lap. Demo profiles set a short confirmation
    # distance then close on the authored keep-in (not a physics cheat).
    calibrate_confirm_m: float = 0.0
    phase_budget_scale: float = 1.0
    # MAP READY beat before auto-mow (steps). Owner Start mow skips this.
    review_hold_steps: int = 20
    # 0 = mow until the plan or max_mow_steps. Demo profiles set a
    # job-cut fraction (cut of planned reachable) that homes the robot.
    mow_complete_frac: float = 0.0
    # 0 = paint with the physical trimmer radius. Coarse demo rasters
    # set this so a strip is visible on 0.50 m cells (not a bigger blade).
    cover_radius_m: float = 0.0
    # IMU tip-stop: skip this many waypoints, then keep painting.
    mow_skip_cluster: int = 4
    mow_stop_cool: int = 10
    # Owner path is always the full taught fence. Demo 30% / short-cap
    # early exits are gone. full_explore stays True so a step cap cannot
    # declare MAP READY while reachable frontiers remain.
    full_explore: bool = True
    full_explore_complete: float = 1.0
    full_explore_steps: int = 4000
    min_explore_steps: int = 8
    # 0 = use YardProfile schedule.min_soc, else runtime.battery.limp_soc.
    min_soc_return: float = 0.0
    charge_resume_soc: float = 0.80
    # Gym charge increment per mission tick at the dock (not a BMS claim).
    gym_charge_soc_per_step: float = 0.12
    # Learned no-go: no progress toward the current frontier / waypoint
    # for this many steps (or collision / tip / slip proxies) stamps a
    # local blockage disk, skips that frontier, and replans.
    blockage_no_progress_steps: int = 16
    blockage_radius_m: float = 0.55
    blockage_cluster_cells: int = 4
    blockage_replan_after: int = 6


@dataclass
class BatteryConfig:
    """Orin-class pack stub unless ``measured: true`` with a filled bench.

    Gym may keep 50 Wh when ``measured`` is false. Flipping measured
    without ``capacity_wh`` / ``charge_time_h`` / ``measured_at`` from
    ``docs/PACK_THERMAL.md`` is a ConfigError — the stub must not look
    like a field claim. ``acre_runtime_h`` is never derived here.
    """

    capacity_wh: Optional[float] = 50.0
    charge_time_h: Optional[float] = None
    measured: bool = False
    template: bool = False
    measured_at: str = ""
    notes: str = ""
    soc: float = 1.0
    idle_w: float = 8.0
    drive_w: float = 25.0
    compute_w: float = 7.0
    trimmer_w: float = 12.0
    limp_soc: float = 0.15
    stop_soc: float = 0.05


@dataclass
class ThermalConfig:
    """First-order thermal RC. Not a board TDP claim.

    ``board_load_c`` stays null until a bench log. ``measured: true``
    requires that number (see ``docs/PACK_THERMAL.md``).
    """

    t_c: float = 45.0
    t_ambient_c: float = 35.0
    t_hot_c: float = 75.0
    t_crit_c: float = 85.0
    tau_s: float = 90.0
    heat_c_per_w: float = 0.35
    board_load_c: Optional[float] = None
    measured: bool = False


@dataclass
class WatchdogConfig:
    """Stop wheels if IMU / vision stamps freeze. Off by default in gym tests.

    Enable on the bench loop via ``runtime.watchdog.enabled: true``
    (see ``configs/orin/bench.yaml``).
    """

    enabled: bool = False
    imu_stall_s: float = 0.40
    vision_stall_s: float = 0.40


@dataclass
class RuntimeCamerasConfig:
    """CSI → ``obs["cameras"]`` adapter. Default empty = gym renderer.

    ``adapter``: ``renderer`` / ``""`` (gym pinhole), ``fake_gst`` /
    ``fake_csi`` (CI / bench), ``gst`` (``GstNvmmAdapter``; raises
    without JetPack). Downsample is ``runtime.capture.downsample_rgb``.
    Prefer field names from ``configs/orin/extrinsics_stereo.yaml``.
    """

    adapter: str = ""


@dataclass
class RuntimeConfig:
    """On-box budget stub. Off by default so short gym tests stay unchanged."""

    enabled: bool = False
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    watchdog: WatchdogConfig = field(default_factory=WatchdogConfig)
    cameras: RuntimeCamerasConfig = field(default_factory=RuntimeCamerasConfig)


@dataclass
class FaultsConfig:
    """Gym / runtime fault injection. Off by default so existing tests stay still."""

    enabled: bool = False
    inject: list = field(default_factory=list)


@dataclass
class RadioLinkConfig:
    bandwidth_bps: float = 1_000_000.0
    range_m: float = 20.0
    drop_prob: float = 0.05


@dataclass
class RadioConfig:
    """Simulated Wi-Fi / BT / LoRa. No real RF hardware."""

    enabled: bool = False
    distance_m: float = 10.0
    heartbeat_timeout_s: float = 2.0
    on_loss: str = "stop_beacon"  # limp_home | stop_beacon
    payload_bytes: int = 48
    wifi: RadioLinkConfig = field(
        default_factory=lambda: RadioLinkConfig(20_000_000.0, 40.0, 0.02)
    )
    bt: RadioLinkConfig = field(
        default_factory=lambda: RadioLinkConfig(1_000_000.0, 12.0, 0.08)
    )
    lora: RadioLinkConfig = field(
        default_factory=lambda: RadioLinkConfig(5_000.0, 2_000.0, 0.15)
    )


@dataclass
class OwnerConfig:
    """Owner-app / live overlay. Gym unit tests leave ``require_pair`` false."""

    require_pair: bool = False
    pair_pin: str = GYM_PAIR_PIN


@dataclass
class EnvConfig:
    dt: float = 0.10
    max_steps: int = 500
    robot: RobotConfig = field(default_factory=RobotConfig)
    sensors: SensorsConfig = field(default_factory=SensorsConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    mission: MissionConfig = field(default_factory=MissionConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    domain_randomization: DomainRandomizationConfig = field(
        default_factory=DomainRandomizationConfig
    )
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    faults: FaultsConfig = field(default_factory=FaultsConfig)
    radio: RadioConfig = field(default_factory=RadioConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    owner: OwnerConfig = field(default_factory=OwnerConfig)

    def resolved_cameras(self) -> list[CameraSpec]:
        """Return the 4–6 camera rig, applying the default FOV when needed."""
        if self.sensors.cameras:
            cams = list(self.sensors.cameras)
        else:
            if self.sensors.camera_count not in _COUNT_NAMES:
                raise ConfigError(
                    f"camera_count must be 4, 5, or 6; got {self.sensors.camera_count}"
                )
            names = _COUNT_NAMES[self.sensors.camera_count]
            by_name = {c.name: c for c in _DEFAULT_RIG}
            cams = [by_name[n] for n in names]
        fov = float(self.sensors.fov_deg)
        return [
            CameraSpec(c.name, c.x, c.y, c.z, c.yaw_deg, c.pitch_deg, c.fov_deg or fov)
            for c in cams
        ]


def default_camera_rig(count: int = 6, fov_deg: float = 70.0) -> list[CameraSpec]:
    if count not in _COUNT_NAMES:
        raise ConfigError(f"camera_count must be 4, 5, or 6; got {count}")
    by_name = {c.name: c for c in _DEFAULT_RIG}
    return [
        CameraSpec(c.name, c.x, c.y, c.z, c.yaw_deg, c.pitch_deg, fov_deg)
        for c in (by_name[n] for n in _COUNT_NAMES[count])
    ]


def field_stereo_cameras() -> list[CameraSpec]:
    """Preferred Orin names from ``extrinsics_stereo.yaml`` (example poses)."""
    from jims_mower.runtime.capture import load_field_camera_specs

    return load_field_camera_specs()


def _finish_config(cfg: EnvConfig) -> EnvConfig:
    cfg = validate_config(cfg)
    if cfg.world.season and cfg.world.season != "none":
        from jims_mower.overlays import apply_season

        apply_season(cfg, cfg.world.season)
    return cfg


def overlay_config(cfg: EnvConfig, data: dict[str, Any]) -> EnvConfig:
    """Merge a mapping onto an existing config and re-validate."""
    if not isinstance(data, dict):
        raise ConfigError("overlay must be a mapping")
    _merge_dataclass(cfg, data)
    return _finish_config(cfg)


def _merge_dataclass(obj: Any, data: dict[str, Any]) -> None:
    valid = {f.name for f in fields(obj)}
    for key, value in data.items():
        if key not in valid:
            raise ConfigError(f"Unknown config key '{key}' for {type(obj).__name__}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge_dataclass(current, value)
        elif key == "cameras" and isinstance(value, list):
            setattr(obj, key, [_camera_from_mapping(item) for item in value])
        else:
            setattr(obj, key, value)


def _camera_from_mapping(item: Any) -> CameraSpec:
    if not isinstance(item, dict):
        raise ConfigError("Each camera entry must be a mapping")
    required = ("name", "x", "y", "z", "yaw_deg", "pitch_deg")
    missing = [k for k in required if k not in item]
    if missing:
        raise ConfigError(f"Camera missing fields: {missing}")
    return CameraSpec(
        name=str(item["name"]),
        x=float(item["x"]),
        y=float(item["y"]),
        z=float(item["z"]),
        yaw_deg=float(item["yaw_deg"]),
        pitch_deg=float(item["pitch_deg"]),
        fov_deg=float(item.get("fov_deg", 0.0) or 0.0),
    )


def validate_config(cfg: EnvConfig) -> EnvConfig:
    if cfg.dt <= 0:
        raise ConfigError("dt must be positive")
    if cfg.max_steps < 1:
        raise ConfigError("max_steps must be >= 1")
    if cfg.robot.wheelbase_m <= 0:
        raise ConfigError("wheelbase_m must be positive")
    if cfg.robot.track_m <= 0:
        raise ConfigError("track_m must be positive")
    if cfg.robot.tip_roll_rad <= 0 or cfg.robot.tip_pitch_rad <= 0:
        raise ConfigError("tip roll/pitch thresholds must be positive")
    if cfg.robot.wheel_drop_m <= 0:
        raise ConfigError("wheel_drop_m must be positive")
    if cfg.robot.steep_slope_rad <= 0:
        raise ConfigError("steep_slope_rad must be positive")
    if cfg.robot.max_wheel_speed_mps <= 0:
        raise ConfigError("max_wheel_speed_mps must be positive")
    if float(cfg.robot.drive.scale) <= 0:
        raise ConfigError("robot.drive.scale must be positive")
    if float(cfg.robot.trimmer.scale) <= 0:
        raise ConfigError("robot.trimmer.scale must be positive")
    if cfg.robot.length_m <= 0 or cfg.robot.width_m <= 0 or cfg.robot.height_m <= 0:
        raise ConfigError("robot body extents must be positive")
    if cfg.robot.trimmer.safety_radius_m <= 0:
        raise ConfigError("trimmer.safety_radius_m must be positive")
    if cfg.world.width_m <= 0 or cfg.world.height_m <= 0:
        raise ConfigError("world extents must be positive")
    if cfg.world.resolution_m <= 0:
        raise ConfigError("resolution_m must be positive")
    terr = cfg.world.terrain
    if terr.n_drains < 0 or terr.n_banks < 0:
        raise ConfigError("terrain feature counts must be >= 0")
    if terr.drain_width_m <= 0 or terr.drain_depth_m <= 0 or terr.drain_length_m <= 0:
        raise ConfigError("drain width/depth/length must be positive")
    if terr.drain_side_slope <= 0:
        raise ConfigError("drain_side_slope must be positive")
    if terr.bank_height_m < 0 or terr.bank_width_m <= 0 or terr.bank_length_m <= 0:
        raise ConfigError("bank extents must be positive (height may be 0)")
    if terr.max_slope_rad <= 0:
        raise ConfigError("max_slope_rad must be positive")
    if terr.noise_amp_m < 0:
        raise ConfigError("noise_amp_m must be >= 0")
    if terr.multi_scale_amp_m < 0:
        raise ConfigError("multi_scale_amp_m must be >= 0")
    if terr.swale_amp_m < 0:
        raise ConfigError("swale_amp_m must be >= 0")
    if terr.puddle_radius_m <= 0 or terr.puddle_depth_m < 0:
        raise ConfigError("puddle_radius_m must be > 0 and puddle_depth_m >= 0")
    grad = terr.base_gradient
    if not 0.0 <= float(grad.slope_rad) < 0.5 * math.pi:
        raise ConfigError("base_gradient.slope_rad must be in [0, π/2)")
    if float(grad.undulation_m) < 0:
        raise ConfigError("base_gradient.undulation_m must be >= 0")
    if grad.slope_pct is not None and float(grad.slope_pct) < 0:
        raise ConfigError("base_gradient.slope_pct must be >= 0")
    if cfg.world.n_hoses < 0 or cfg.world.n_cords < 0 or cfg.world.n_puddles < 0:
        raise ConfigError("hose/cord/puddle counts must be >= 0")
    if cfg.world.layout not in WORLD_LAYOUTS:
        raise ConfigError(
            f"world.layout must be one of {sorted(WORLD_LAYOUTS)}; got {cfg.world.layout!r}"
        )
    if cfg.world.season not in SEASONS:
        raise ConfigError(
            f"world.season must be one of {sorted(SEASONS)}; got {cfg.world.season!r}"
        )
    if cfg.world.orchard_rows < 1 or cfg.world.orchard_cols < 1:
        raise ConfigError("orchard_rows/orchard_cols must be >= 1")
    grass = cfg.world.grass
    if not 0.0 <= grass.regenerate_frac <= 1.0:
        raise ConfigError("world.grass.regenerate_frac must be in [0, 1]")
    movers = cfg.world.movers
    if movers.default_mode not in TRAJECTORY_MODES:
        raise ConfigError(
            f"world.movers.default_mode must be one of {sorted(TRAJECTORY_MODES)}; "
            f"got {movers.default_mode!r}"
        )
    if movers.density not in MOVER_DENSITIES:
        raise ConfigError(
            f"world.movers.density must be one of {sorted(MOVER_DENSITIES)}; "
            f"got {movers.density!r}"
        )
    if movers.heading_jitter < 0:
        raise ConfigError("world.movers.heading_jitter must be >= 0")
    if cfg.weather.pack not in WEATHER_PACKS:
        raise ConfigError(
            f"weather.pack must be one of {sorted(WEATHER_PACKS)}; got {cfg.weather.pack!r}"
        )
    if cfg.weather.colour_temperature_k < 0:
        raise ConfigError("weather.colour_temperature_k must be >= 0")
    if cfg.domain_randomization.seed < 0:
        raise ConfigError("domain_randomization.seed must be >= 0")
    imu = cfg.sensors.imu
    gps = cfg.sensors.gps
    tof = cfg.sensors.tof
    if imu.accel_noise_std < 0 or imu.gyro_noise_std < 0 or imu.accel_bias_std < 0:
        raise ConfigError("IMU noise/bias std must be >= 0")
    if gps.horiz_noise_std_m < 0 or gps.vert_noise_std_m < 0:
        raise ConfigError("GPS noise std must be >= 0")
    if not 0.0 <= gps.dropout_prob <= 1.0:
        raise ConfigError("gps.dropout_prob must be in [0, 1]")
    if tof.noise_std_m < 0 or tof.max_range_m <= 0:
        raise ConfigError("ToF noise must be >= 0 and max_range_m positive")
    if int(tof.count) not in TOF_COUNTS:
        raise ConfigError(f"sensors.tof.count must be one of {sorted(TOF_COUNTS)}; got {tof.count!r}")
    rt = cfg.runtime
    batt = rt.battery
    therm = rt.thermal
    if batt.capacity_wh is None:
        if batt.measured and not batt.template:
            raise ConfigError(
                "runtime.battery.measured requires capacity_wh from the bench"
            )
        if not batt.template:
            batt.capacity_wh = 50.0
    elif float(batt.capacity_wh) <= 0:
        raise ConfigError("runtime.battery.capacity_wh must be positive")
    if batt.charge_time_h is not None and float(batt.charge_time_h) <= 0:
        raise ConfigError("runtime.battery.charge_time_h must be positive when set")
    from jims_mower.pack import PackError, validate_pack_claim

    try:
        validate_pack_claim(batt)
    except PackError as exc:
        raise ConfigError(str(exc)) from exc
    if not 0.0 <= batt.soc <= 1.0:
        raise ConfigError("runtime.battery.soc must be in [0, 1]")
    for name in ("idle_w", "drive_w", "compute_w", "trimmer_w"):
        if float(getattr(batt, name)) < 0.0:
            raise ConfigError(f"runtime.battery.{name} must be >= 0")
    if not 0.0 <= batt.stop_soc <= batt.limp_soc <= 1.0:
        raise ConfigError("runtime.battery stop_soc <= limp_soc must hold in [0, 1]")
    if therm.tau_s <= 0 or therm.heat_c_per_w < 0:
        raise ConfigError("runtime.thermal tau_s must be > 0 and heat_c_per_w >= 0")
    if not (therm.t_ambient_c < therm.t_hot_c <= therm.t_crit_c):
        raise ConfigError("runtime.thermal t_ambient_c < t_hot_c <= t_crit_c")
    if therm.board_load_c is not None and float(therm.board_load_c) < -40.0:
        raise ConfigError("runtime.thermal.board_load_c looks unusable")
    if therm.measured and therm.board_load_c is None:
        raise ConfigError(
            "runtime.thermal.measured requires board_load_c from the bench"
        )
    mode = cfg.perception.terrain_mode
    if mode not in TERRAIN_MODES:
        raise ConfigError(
            "perception.terrain_mode must be "
            "oracle|blind|heuristic|learned|onnx|trt; "
            f"got {mode!r}"
        )
    grass_mode = str(cfg.perception.grass_mode or "color").strip().lower()
    if grass_mode not in GRASS_MODES:
        raise ConfigError(
            "perception.grass_mode must be color|feature|class; "
            f"got {cfg.perception.grass_mode!r}"
        )
    cov_src = str(cfg.perception.coverage_source or "gym_grid").strip().lower()
    if cov_src not in COVERAGE_SOURCES:
        raise ConfigError(
            "perception.coverage_source must be gym_grid|observer; "
            f"got {cfg.perception.coverage_source!r}"
        )
    cfg.perception.coverage_source = cov_src
    if not 0.0 <= float(cfg.perception.occupancy_decay) <= 1.0:
        raise ConfigError("perception.occupancy_decay must be in [0, 1]")
    det_backend = str(cfg.perception.detector_backend or "mock").strip().lower()
    if det_backend not in DETECTOR_BACKENDS:
        raise ConfigError("perception.detector_backend must be mock|trt|onnx|appearance|blind")
    interlock_src = str(cfg.perception.interlock_source or "auto").strip().lower()
    if interlock_src not in INTERLOCK_SOURCES:
        raise ConfigError("perception.interlock_source must be auto|detections|obstacles")
    cfg.perception.interlock_source = interlock_src
    wd = cfg.runtime.watchdog
    if wd.imu_stall_s <= 0 or wd.vision_stall_s <= 0:
        raise ConfigError("runtime.watchdog stall windows must be positive")
    from jims_mower.runtime.capture import adapter_kind

    cam_adapter = adapter_kind(getattr(cfg.runtime.cameras, "adapter", ""))
    if cam_adapter not in {"renderer", "fake_gst", "fake_csi", "gst"}:
        raise ConfigError(
            "runtime.cameras.adapter must be renderer|fake_gst|fake_csi|gst; "
            f"got {cfg.runtime.cameras.adapter!r}"
        )
    cfg.runtime.cameras.adapter = "" if cam_adapter == "renderer" else cam_adapter
    radio = cfg.radio
    if radio.distance_m < 0:
        raise ConfigError("radio.distance_m must be >= 0")
    if radio.heartbeat_timeout_s <= 0:
        raise ConfigError("radio.heartbeat_timeout_s must be positive")
    if radio.payload_bytes < 1:
        raise ConfigError("radio.payload_bytes must be >= 1")
    on_loss = str(radio.on_loss or "").strip().lower()
    if on_loss not in RADIO_LOSS_ACTIONS:
        raise ConfigError(
            f"radio.on_loss must be one of {sorted(RADIO_LOSS_ACTIONS)}; got {radio.on_loss!r}"
        )
    radio.on_loss = on_loss
    owner = cfg.owner
    pin = str(getattr(owner, "pair_pin", GYM_PAIR_PIN) or GYM_PAIR_PIN).strip()
    if not pin:
        raise ConfigError("owner.pair_pin must be a non-empty gym PIN")
    owner.pair_pin = pin
    owner.require_pair = bool(getattr(owner, "require_pair", False))
    for name in RADIO_CHANNELS:
        link = getattr(radio, name)
        if float(link.bandwidth_bps) <= 0 or float(link.range_m) <= 0:
            raise ConfigError(f"radio.{name} bandwidth_bps and range_m must be positive")
        if not 0.0 <= float(link.drop_prob) <= 1.0:
            raise ConfigError(f"radio.{name}.drop_prob must be in [0, 1]")
    if not isinstance(cfg.faults.inject, list):
        raise ConfigError("faults.inject must be a list")
    for item in cfg.faults.inject:
        if not isinstance(item, dict):
            raise ConfigError("each faults.inject entry must be a mapping")
        kind = item.get("kind") or item.get("component")
        if not kind:
            raise ConfigError("faults.inject entry needs kind or component")
        if "mode" in item:
            mode = str(item["mode"]).strip().lower()
            if mode not in MOTOR_KILL_MODES and mode not in RADIO_LOSS_ACTIONS and mode not in {
                "stuck",
                "jam",
                "blind",
                "freeze",
                "dropout",
            }:
                raise ConfigError(f"faults.inject mode {item['mode']!r} is not recognised")
        if int(item.get("at_step", 0)) < 0:
            raise ConfigError("faults.inject at_step must be >= 0")
    plan = cfg.planner
    if plan.max_climb_slope_rad <= 0:
        raise ConfigError("planner.max_climb_slope_rad must be positive")
    if plan.drain_clearance_m < 0:
        raise ConfigError("planner.drain_clearance_m must be >= 0")
    if not 0.0 < plan.slow_speed_factor <= 1.0:
        raise ConfigError("planner.slow_speed_factor must be in (0, 1]")
    if not 0.0 < plan.grade_speed_factor <= 1.0:
        raise ConfigError("planner.grade_speed_factor must be in (0, 1]")
    if float(plan.grade_look_ahead_m) < 0.0:
        raise ConfigError("planner.grade_look_ahead_m must be >= 0")
    if int(plan.grade_look_ahead_samples) < 1:
        raise ConfigError("planner.grade_look_ahead_samples must be >= 1")
    if plan.strip_spacing_m <= 0 or plan.waypoint_stride_m <= 0:
        raise ConfigError("planner strip/waypoint spacing must be positive")
    if plan.arrive_radius_m <= 0:
        raise ConfigError("planner.arrive_radius_m must be positive")
    if plan.cruise_speed <= 0:
        raise ConfigError("planner.cruise_speed must be positive")
    if plan.max_replans < 0:
        raise ConfigError("planner.max_replans must be >= 0")
    if not 0.0 <= plan.gps_blend <= 1.0 or not 0.0 <= plan.accel_blend <= 1.0:
        raise ConfigError("planner gps_blend/accel_blend must be in [0, 1]")
    if not 0.0 < plan.imu_slow_frac <= plan.imu_stop_frac:
        raise ConfigError("planner imu_slow_frac must be in (0, imu_stop_frac]")
    if int(plan.imu_stop_hold_steps) < 1:
        raise ConfigError("planner.imu_stop_hold_steps must be >= 1")
    if int(plan.imu_tilt_window) < 1:
        raise ConfigError("planner.imu_tilt_window must be >= 1")
    if not 0.5 <= float(plan.tip_lethal_frac) <= 1.2:
        raise ConfigError("planner.tip_lethal_frac must be in [0.5, 1.2]")
    if float(plan.contour_cost) < 1.0:
        raise ConfigError("planner.contour_cost must be >= 1")
    kind = str(plan.pose_filter or "").strip().lower()
    if kind not in {"ekf", "complementary", "comp", "stub"}:
        raise ConfigError("planner.pose_filter must be ekf|complementary")
    ekf = plan.ekf
    for name in (
        "q_xy",
        "q_z",
        "q_yaw",
        "q_tilt",
        "q_odom",
        "r_gps_xy",
        "r_gps_z",
        "r_tilt",
        "gps_gate_m",
    ):
        if float(getattr(ekf, name)) < 0.0:
            raise ConfigError(f"planner.ekf.{name} must be >= 0")
    if not 0.0 <= float(ekf.gyro_yaw_mix) <= 1.0:
        raise ConfigError("planner.ekf.gyro_yaw_mix must be in [0, 1]")
    unc = plan.uncertainty
    if unc.inflate < 0.0 or unc.hazard_boost < 0.0:
        raise ConfigError("planner.uncertainty inflate/hazard_boost must be >= 0")
    if not 0.0 <= unc.confidence_floor <= 1.0:
        raise ConfigError("planner.uncertainty.confidence_floor must be in [0, 1]")
    if plan.living_slow_m <= 0 or plan.living_reroute_m <= 0 or plan.living_stop_m <= 0:
        raise ConfigError("planner living radii must be positive")
    if not (plan.living_stop_m <= plan.living_reroute_m <= plan.living_slow_m):
        raise ConfigError("planner living_stop_m <= living_reroute_m <= living_slow_m")
    if plan.geofence_inflate_m < 0 or plan.geofence_slow_m < 0 or plan.geofence_stop_m < 0:
        raise ConfigError("planner geofence inflate/slow/stop must be >= 0")
    if plan.recovery_trigger < 1 or plan.recovery_reverse_steps < 1 or plan.recovery_pivot_steps < 1:
        raise ConfigError("planner recovery trigger/steps must be >= 1")
    if plan.max_recoveries < 0:
        raise ConfigError("planner.max_recoveries must be >= 0")
    safe = plan.safe_state
    if safe.limp_after_stops < 1 or safe.safe_after_limp_steps < 1:
        raise ConfigError("planner.safe_state limp/safe thresholds must be >= 1")
    if not 0.0 < safe.limp_scale <= 1.0:
        raise ConfigError("planner.safe_state.limp_scale must be in (0, 1]")
    if safe.recover_ok_steps < 1:
        raise ConfigError("planner.safe_state.recover_ok_steps must be >= 1")
    if plan.wet_slope_extra < 0:
        raise ConfigError("planner.wet_slope_extra must be >= 0")
    if not 0.0 <= float(plan.elevation_prior_weight) <= 1.0:
        raise ConfigError("planner.elevation_prior_weight must be in [0, 1]")
    if float(plan.path_cost) < 0.0 or float(plan.bunker_cost) < 0.0:
        raise ConfigError("planner path_cost/bunker_cost must be >= 0")
    mission = cfg.mission
    if not 0.0 <= mission.observe_confidence <= 1.0:
        raise ConfigError("mission.observe_confidence must be in [0, 1]")
    if not 0.0 <= mission.explore_complete <= 1.0:
        raise ConfigError("mission.explore_complete must be in [0, 1]")
    if not 0.0 <= mission.explore_no_frontier <= 1.0:
        raise ConfigError("mission.explore_no_frontier must be in [0, 1]")
    if mission.stamp_radius_m <= 0.0 or mission.camera_range_m <= 0.0:
        raise ConfigError("mission stamp/camera range must be positive")
    for name in (
        "max_calibrate_steps",
        "max_explore_steps",
        "max_mow_steps",
        "max_return_steps",
        "snapshot_stride",
    ):
        if int(getattr(mission, name)) < 1:
            raise ConfigError(f"mission.{name} must be >= 1")
    if mission.review_min_closure_m < 0.0:
        raise ConfigError("mission.review_min_closure_m must be >= 0")
    if mission.explore_cruise <= 0.0 or mission.mow_cruise <= 0.0:
        raise ConfigError("mission cruise speeds must be positive")
    if mission.calibrate_stride_m < 0.0 or mission.calibrate_cruise < 0.0:
        raise ConfigError("mission calibrate stride/cruise must be >= 0")
    if mission.calibrate_arrive_m < 0.0 or mission.calibrate_confirm_m < 0.0:
        raise ConfigError("mission calibrate arrive/confirm must be >= 0")
    if not 0.05 <= float(mission.phase_budget_scale) <= 1.0:
        raise ConfigError("mission.phase_budget_scale must be in [0.05, 1]")
    if int(mission.review_hold_steps) < 1:
        raise ConfigError("mission.review_hold_steps must be >= 1")
    if not 0.0 <= float(mission.mow_complete_frac) <= 1.0:
        raise ConfigError("mission.mow_complete_frac must be in [0, 1]")
    if float(mission.cover_radius_m) < 0.0:
        raise ConfigError("mission.cover_radius_m must be >= 0")
    if int(mission.mow_skip_cluster) < 1:
        raise ConfigError("mission.mow_skip_cluster must be >= 1")
    if int(mission.mow_stop_cool) < 0:
        raise ConfigError("mission.mow_stop_cool must be >= 0")
    if not 0.0 <= float(mission.full_explore_complete) <= 1.0:
        raise ConfigError("mission.full_explore_complete must be in [0, 1]")
    if int(mission.full_explore_steps) < 1:
        raise ConfigError("mission.full_explore_steps must be >= 1")
    if int(mission.min_explore_steps) < 0:
        raise ConfigError("mission.min_explore_steps must be >= 0")
    if not 0.0 <= float(mission.min_soc_return) <= 1.0:
        raise ConfigError("mission.min_soc_return must be in [0, 1]")
    if not 0.0 <= float(mission.charge_resume_soc) <= 1.0:
        raise ConfigError("mission.charge_resume_soc must be in [0, 1]")
    if not 0.0 < float(mission.gym_charge_soc_per_step) <= 1.0:
        raise ConfigError("mission.gym_charge_soc_per_step must be in (0, 1]")
    if int(mission.blockage_no_progress_steps) < 2:
        raise ConfigError("mission.blockage_no_progress_steps must be >= 2")
    if float(mission.blockage_radius_m) <= 0.0:
        raise ConfigError("mission.blockage_radius_m must be > 0")
    if int(mission.blockage_cluster_cells) < 1:
        raise ConfigError("mission.blockage_cluster_cells must be >= 1")
    if int(mission.blockage_replan_after) < 1:
        raise ConfigError("mission.blockage_replan_after must be >= 1")
    if cfg.sensors.width < 8 or cfg.sensors.height < 8:
        raise ConfigError("camera resolution must be at least 8x8")
    cams = cfg.resolved_cameras()
    n = len(cams)
    if n not in (4, 5, 6):
        raise ConfigError(f"Need 4–6 cameras; got {n}")
    if cfg.sensors.cameras and cfg.sensors.camera_count not in (0, n):
        if cfg.sensors.camera_count not in (4, 5, 6):
            raise ConfigError(
                f"camera_count must be 4, 5, or 6; got {cfg.sensors.camera_count}"
            )
    elif not cfg.sensors.cameras and cfg.sensors.camera_count not in (4, 5, 6):
        raise ConfigError(
            f"camera_count must be 4, 5, or 6; got {cfg.sensors.camera_count}"
        )
    names = [c.name for c in cams]
    if len(names) != len(set(names)):
        raise ConfigError(f"Camera names must be unique: {names}")
    cal = cfg.calibration
    if cal.tape_baseline_cm is not None and float(cal.tape_baseline_cm) <= 0.0:
        raise ConfigError("calibration.tape_baseline_cm must be positive when set")
    return cfg


def load_config(source: Optional[Union[str, Path, dict, EnvConfig]] = None) -> EnvConfig:
    """Load config from a path, mapping, or existing object (defaults if None)."""
    if isinstance(source, EnvConfig):
        return _finish_config(source)
    cfg = EnvConfig()
    if source is None:
        path = DEFAULT_CONFIG_PATH
        if path.is_file():
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            if not isinstance(data, dict):
                raise ConfigError("Config YAML must be a mapping")
            _merge_dataclass(cfg, data)
        return _finish_config(cfg)
    if isinstance(source, dict):
        _merge_dataclass(cfg, source)
        return _finish_config(cfg)
    path = Path(source)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError("Config YAML must be a mapping")
    _merge_dataclass(cfg, data)
    return _finish_config(cfg)

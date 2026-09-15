"""YAML-backed environment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from jims_mower.constants import (
    MOVER_DENSITIES,
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

# Built-in rig (body frame: x forward, y left, z up). Used when YAML
# leaves `sensors.cameras` empty.
# Front pair pitched a bit more down so drain lips sit in the lower image third.
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
class TrimmerConfig:
    offset_m: float = 0.32
    radius_m: float = 0.16
    safety_radius_m: float = 1.50
    height_m: float = 0.12


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


@dataclass
class PerceptionConfig:
    terrain_mode: str = "heuristic"  # heuristic | oracle | blind | learned
    weights_path: str = ""  # optional .npz for LearnedTerrainObserver
    temporal: bool = False  # hysteresis on heuristic; learned defaults on in factory


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
    strip_spacing_m: float = 0.28
    waypoint_stride_m: float = 0.32
    arrive_radius_m: float = 0.20
    turn_in_place_rad: float = 0.70
    cruise_speed: float = 0.55
    imu_slow_frac: float = 0.55
    imu_stop_frac: float = 0.85
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


@dataclass
class BatteryConfig:
    """Orin-class pack stub (watt-hours / watts are class-scale, not measured)."""

    capacity_wh: float = 50.0
    soc: float = 1.0
    idle_w: float = 8.0
    drive_w: float = 25.0
    compute_w: float = 7.0
    trimmer_w: float = 12.0
    limp_soc: float = 0.15
    stop_soc: float = 0.05


@dataclass
class ThermalConfig:
    """First-order thermal RC. Not a board TDP claim."""

    t_c: float = 45.0
    t_ambient_c: float = 35.0
    t_hot_c: float = 75.0
    t_crit_c: float = 85.0
    tau_s: float = 90.0
    heat_c_per_w: float = 0.35


@dataclass
class RuntimeConfig:
    """On-box budget stub. Off by default so short gym tests stay unchanged."""

    enabled: bool = False
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)


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
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    domain_randomization: DomainRandomizationConfig = field(
        default_factory=DomainRandomizationConfig
    )
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

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


def overlay_config(cfg: EnvConfig, data: dict[str, Any]) -> EnvConfig:
    """Merge a mapping onto an existing config and re-validate."""
    if not isinstance(data, dict):
        raise ConfigError("overlay must be a mapping")
    _merge_dataclass(cfg, data)
    return validate_config(cfg)


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
    if terr.puddle_radius_m <= 0 or terr.puddle_depth_m < 0:
        raise ConfigError("puddle_radius_m must be > 0 and puddle_depth_m >= 0")
    if cfg.world.n_hoses < 0 or cfg.world.n_cords < 0 or cfg.world.n_puddles < 0:
        raise ConfigError("hose/cord/puddle counts must be >= 0")
    if cfg.world.layout not in WORLD_LAYOUTS:
        raise ConfigError(
            f"world.layout must be one of {sorted(WORLD_LAYOUTS)}; got {cfg.world.layout!r}"
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
    if batt.capacity_wh <= 0:
        raise ConfigError("runtime.battery.capacity_wh must be positive")
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
    mode = cfg.perception.terrain_mode
    if mode not in {"oracle", "blind", "heuristic", "learned"}:
        raise ConfigError(
            f"perception.terrain_mode must be oracle|blind|heuristic|learned; got {mode!r}"
        )
    plan = cfg.planner
    if plan.max_climb_slope_rad <= 0:
        raise ConfigError("planner.max_climb_slope_rad must be positive")
    if plan.drain_clearance_m < 0:
        raise ConfigError("planner.drain_clearance_m must be >= 0")
    if not 0.0 < plan.slow_speed_factor <= 1.0:
        raise ConfigError("planner.slow_speed_factor must be in (0, 1]")
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
    return cfg


def load_config(source: Optional[Union[str, Path, dict, EnvConfig]] = None) -> EnvConfig:
    """Load config from a path, mapping, or existing object (defaults if None)."""
    if isinstance(source, EnvConfig):
        return validate_config(source)
    cfg = EnvConfig()
    if source is None:
        path = DEFAULT_CONFIG_PATH
        if path.is_file():
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            if not isinstance(data, dict):
                raise ConfigError("Config YAML must be a mapping")
            _merge_dataclass(cfg, data)
        return validate_config(cfg)
    if isinstance(source, dict):
        _merge_dataclass(cfg, source)
        return validate_config(cfg)
    path = Path(source)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError("Config YAML must be a mapping")
    _merge_dataclass(cfg, data)
    return validate_config(cfg)

"""YAML-backed environment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, Union

import yaml

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
_DEFAULT_RIG = (
    CameraSpec("front", 0.25, 0.00, 0.38, 0.0, -18.0),
    CameraSpec("front_left", 0.20, 0.20, 0.38, 40.0, -15.0),
    CameraSpec("front_right", 0.20, -0.20, 0.38, -40.0, -15.0),
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
    max_wheel_speed_mps: float = 1.2
    collision_radius_m: float = 0.28
    trimmer: TrimmerConfig = field(default_factory=TrimmerConfig)


@dataclass
class SensorsConfig:
    width: int = 80
    height: int = 60
    camera_count: int = 6
    fov_deg: float = 70.0
    cameras: list[CameraSpec] = field(default_factory=list)


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


@dataclass
class RewardConfig:
    coverage_scale: float = 1.0
    time_penalty: float = 0.01
    collision_living: float = 50.0
    collision_static: float = 20.0
    out_of_bounds: float = 10.0
    completion_bonus: float = 15.0
    completion_threshold: float = 0.95


@dataclass
class CurriculumConfig:
    hand_signals: bool = False
    signal_hold_steps: int = 40


@dataclass
class EnvConfig:
    dt: float = 0.10
    max_steps: int = 500
    robot: RobotConfig = field(default_factory=RobotConfig)
    sensors: SensorsConfig = field(default_factory=SensorsConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)

    def resolved_cameras(self) -> list[CameraSpec]:
        """Return the 4–6 camera rig, applying the default FOV when needed."""
        if self.sensors.cameras:
            cams = list(self.sensors.cameras)
        else:
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

"""Named yard scenarios: WAVE 1C EnvConfig library plus WAVE 1A DSL.

WAVE 1C files under ``configs/scenarios/`` are raw ``EnvConfig`` YAML
(``suburban``, ``rural_paddock``, …). ``load_scenario(name)`` always returns
``EnvConfig``.

WAVE 1A authored yards use ``kind: scenario`` (weather flags, geofence,
explicit drains/banks/obstacles). Those keep distinct filenames
(``paddock``, ``night_dawn``, ``wet_slope``) and load through the DSL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from jims_mower.config import (
    DEFAULT_CONFIG_PATH,
    ConfigError,
    EnvConfig,
    load_config,
    overlay_config,
    validate_config,
)
from jims_mower.terrain import BankFeature, DrainFeature
from jims_mower.types import Obstacle
from jims_mower.world import place_obstacle

ENV_OVERLAY_KEYS = frozenset(
    {
        "dt",
        "max_steps",
        "robot",
        "sensors",
        "world",
        "reward",
        "curriculum",
        "perception",
        "planner",
        "domain_randomization",
    }
)

SCENARIO_HINT_KEYS = frozenset(
    {"weather", "geofence", "drains", "banks", "obstacles", "env", "base"}
)

LIGHTING_FLAGS = frozenset({"night", "dawn", "day"})

_SKIP_STEMS = frozenset({"schema", "README"})


class ScenarioError(ConfigError):
    """Invalid scenario YAML."""


@dataclass(frozen=True)
class WeatherFlags:
    """Lighting / surface flags. Night and dawn are mutually exclusive."""

    night: bool = False
    dawn: bool = False
    wet: bool = False

    @property
    def lighting(self) -> str:
        if self.night:
            return "night"
        if self.dawn:
            return "dawn"
        return "day"


@dataclass
class Scenario:
    """Loaded scenario: env overlay plus authored yard extras."""

    name: str
    description: str = ""
    weather: WeatherFlags = field(default_factory=WeatherFlags)
    geofence: list[tuple[float, float]] = field(default_factory=list)
    drains: list[DrainFeature] = field(default_factory=list)
    banks: list[BankFeature] = field(default_factory=list)
    obstacles: list[Obstacle] = field(default_factory=list)
    config: EnvConfig = field(default_factory=EnvConfig)
    path: Optional[Path] = None

    def to_env_config(self) -> EnvConfig:
        return self.config


def scenario_dir() -> Path:
    repo = Path(__file__).resolve().parents[2] / "configs" / "scenarios"
    if repo.is_dir():
        return repo
    return Path(__file__).resolve().parent / "data" / "scenarios"


def _discover_scenario_dirs() -> list[Path]:
    dirs: list[Path] = []
    packaged = Path(__file__).resolve().parent / "data" / "scenarios"
    repo = Path(__file__).resolve().parents[2] / "configs" / "scenarios"
    for candidate in (repo, packaged):
        if candidate.is_dir() and candidate not in dirs:
            dirs.append(candidate)
    return dirs


def scenario_search_dirs() -> list[Path]:
    return _discover_scenario_dirs()


def looks_like_scenario(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    if str(data.get("kind", "")).strip().lower() == "scenario":
        return True
    if data.get("name") and any(k in data for k in SCENARIO_HINT_KEYS):
        return True
    return False


def _resolve_base(name: Optional[str]) -> Optional[Union[str, Path]]:
    if not name or str(name).strip() in {"", "default"}:
        return None
    key = str(name).strip()
    path = Path(key)
    if path.is_file():
        return path
    if key in {"steep_yard", "steep-yard"}:
        repo = Path(__file__).resolve().parents[2] / "configs" / "steep_yard.yaml"
        if repo.is_file():
            return repo
    repo_cfg = Path(__file__).resolve().parents[2] / "configs" / f"{key}.yaml"
    if repo_cfg.is_file():
        return repo_cfg
    if DEFAULT_CONFIG_PATH.is_file() and key == "default":
        return DEFAULT_CONFIG_PATH
    raise ScenarioError(f"Unknown scenario base {name!r}")


def _xy_pair(item: Any, *, field: str) -> tuple[float, float]:
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return float(item[0]), float(item[1])
    if isinstance(item, dict) and "x" in item and "y" in item:
        return float(item["x"]), float(item["y"])
    raise ScenarioError(f"{field} entries must be [x, y] or {{x, y}}")


def _parse_geofence(raw: Any) -> list[tuple[float, float]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ScenarioError("geofence must be a list of [x, y] vertices")
    poly = [_xy_pair(p, field="geofence") for p in raw]
    if poly and len(poly) < 3:
        raise ScenarioError("geofence needs at least 3 vertices")
    return poly


def _parse_weather(raw: Any) -> WeatherFlags:
    if raw is None:
        return WeatherFlags()
    if not isinstance(raw, dict):
        raise ScenarioError("weather must be a mapping")
    night = bool(raw.get("night", False))
    dawn = bool(raw.get("dawn", False))
    wet = bool(raw.get("wet", False))
    lighting = raw.get("lighting")
    if lighting is not None:
        key = str(lighting).strip().lower()
        if key not in LIGHTING_FLAGS:
            raise ScenarioError(f"weather.lighting must be day|dawn|night; got {lighting!r}")
        night = key == "night"
        dawn = key == "dawn"
    if night and dawn:
        raise ScenarioError("weather.night and weather.dawn cannot both be true")
    extra = set(raw) - {"night", "dawn", "wet", "lighting"}
    if extra:
        raise ScenarioError(f"Unknown weather keys: {sorted(extra)}")
    return WeatherFlags(night=night, dawn=dawn, wet=wet)


def _apply_weather_flags(cfg: EnvConfig, weather: WeatherFlags) -> None:
    """Map 1A flags onto the 1C ``weather.pack`` so listed DSL yards validate."""
    if weather.night:
        cfg.weather.pack = "night"
    elif weather.dawn:
        cfg.weather.pack = "dawn"
    elif weather.wet:
        cfg.weather.pack = "rain"
    if weather.wet:
        cfg.domain_randomization.wet_specular = True


def _parse_drain(item: Any) -> DrainFeature:
    if not isinstance(item, dict):
        raise ScenarioError("Each drain must be a mapping")
    required = ("x0", "y0", "x1", "y1")
    missing = [k for k in required if k not in item]
    if missing:
        raise ScenarioError(f"Drain missing fields: {missing}")
    return DrainFeature(
        x0=float(item["x0"]),
        y0=float(item["y0"]),
        x1=float(item["x1"]),
        y1=float(item["y1"]),
        width_m=float(item.get("width_m", 0.40)),
        depth_m=float(item.get("depth_m", 0.16)),
        side_slope=float(item.get("side_slope", 1.5)),
        kind=str(item.get("kind", "drain")),
    )


def _parse_bank(item: Any) -> BankFeature:
    if not isinstance(item, dict):
        raise ScenarioError("Each bank must be a mapping")
    required = ("x0", "y0", "x1", "y1")
    missing = [k for k in required if k not in item]
    if missing:
        raise ScenarioError(f"Bank missing fields: {missing}")
    return BankFeature(
        x0=float(item["x0"]),
        y0=float(item["y0"]),
        x1=float(item["x1"]),
        y1=float(item["y1"]),
        width_m=float(item.get("width_m", 1.8)),
        height_m=float(item.get("height_m", 0.40)),
        kind=str(item.get("kind", "bank")),
    )


def _parse_obstacle(item: Any) -> Obstacle:
    if not isinstance(item, dict):
        raise ScenarioError("Each obstacle must be a mapping")
    if "kind" not in item or "x" not in item or "y" not in item:
        raise ScenarioError("Obstacle needs kind, x, y")
    try:
        return place_obstacle(
            str(item["kind"]),
            float(item["x"]),
            float(item["y"]),
            radius=float(item["radius"]) if "radius" in item else None,
            heading=float(item.get("heading", 0.0)),
            name=str(item.get("name", "")),
            z=float(item["z"]) if "z" in item else None,
        )
    except ValueError as exc:
        raise ScenarioError(str(exc)) from exc


def _env_overlay(data: dict[str, Any]) -> dict[str, Any]:
    overlay = {k: data[k] for k in ENV_OVERLAY_KEYS if k in data}
    nested = data.get("env")
    if nested is None:
        return overlay
    if not isinstance(nested, dict):
        raise ScenarioError("env overlay must be a mapping")
    merged = dict(overlay)
    for key, value in nested.items():
        if key not in ENV_OVERLAY_KEYS:
            raise ScenarioError(f"Unknown env overlay key '{key}'")
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def parse_scenario(data: dict[str, Any], *, path: Optional[Path] = None) -> Scenario:
    if not isinstance(data, dict):
        raise ScenarioError("Scenario YAML must be a mapping")
    name = str(data.get("name") or "").strip()
    if not name:
        raise ScenarioError("Scenario needs a non-empty name")
    weather = _parse_weather(data.get("weather"))
    geofence = _parse_geofence(data.get("geofence"))
    drains = [_parse_drain(d) for d in (data.get("drains") or [])]
    banks = [_parse_bank(b) for b in (data.get("banks") or [])]
    obstacles = [_parse_obstacle(o) for o in (data.get("obstacles") or [])]
    try:
        cfg = load_config(_resolve_base(data.get("base")))
        overlay = _env_overlay(data)
        if overlay:
            overlay_config(cfg, overlay)
        _apply_weather_flags(cfg, weather)
        validate_config(cfg)
    except ConfigError as exc:
        if isinstance(exc, ScenarioError):
            raise
        raise ScenarioError(str(exc)) from exc
    return Scenario(
        name=name,
        description=str(data.get("description") or ""),
        weather=weather,
        geofence=geofence,
        drains=drains,
        banks=banks,
        obstacles=obstacles,
        config=cfg,
        path=path,
    )


def resolve_scenario_path(name: str) -> Optional[Path]:
    """Resolve a scenario name (``suburban``) or path to a YAML file."""
    raw = Path(name)
    if raw.is_file():
        return raw
    stem = name[:-5] if name.endswith(".yaml") else name
    stem = Path(stem).name
    for folder in _discover_scenario_dirs():
        candidate = folder / f"{stem}.yaml"
        if candidate.is_file():
            return candidate
    return None


def scenario_path(name: str) -> Path:
    raw = Path(name)
    if raw.is_file():
        return raw
    path = scenario_dir() / f"{name}.yaml"
    if path.is_file():
        return path
    resolved = resolve_scenario_path(name)
    if resolved is not None:
        return resolved
    raise ConfigError(f"Scenario not found: {name} ({path})")


def list_scenarios() -> list[str]:
    """Registered scenario names (bundled YAML stems, excluding schema)."""
    names: set[str] = set()
    for folder in _discover_scenario_dirs():
        for path in folder.glob("*.yaml"):
            if path.name.startswith("_") or path.stem in _SKIP_STEMS:
                continue
            names.add(path.stem)
    return sorted(names)


def load_dsl_scenario(source: Union[str, Path, dict]) -> Scenario:
    """Load a WAVE 1A ``kind: scenario`` yard (not a 1C EnvConfig file)."""
    if isinstance(source, dict):
        return parse_scenario(source)
    path = Path(source)
    if not path.is_file():
        resolved = resolve_scenario_path(str(source))
        if resolved is None:
            raise ScenarioError(f"Scenario not found: {source}")
        path = resolved
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ScenarioError("Scenario YAML must be a mapping")
    if not looks_like_scenario(data):
        raise ScenarioError(f"Not a scenario DSL file: {path}")
    return parse_scenario(data, path=path)


def load_source(
    source: Optional[Union[str, Path, dict, EnvConfig, Scenario]] = None,
) -> tuple[EnvConfig, Optional[Scenario]]:
    """Load an env config, detecting scenario YAML when present."""
    if isinstance(source, Scenario):
        return source.config, source
    if isinstance(source, EnvConfig):
        return source, None
    if source is None:
        return load_config(None), None
    if isinstance(source, dict):
        if looks_like_scenario(source):
            scn = parse_scenario(source)
            return scn.config, scn
        return load_config(source), None
    path = Path(source)
    if path.is_file():
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if looks_like_scenario(data):
            scn = parse_scenario(data, path=path)
            return scn.config, scn
        return load_config(path), None
    resolved = resolve_scenario_path(str(source))
    if resolved is not None:
        return load_source(resolved)
    base = _resolve_base(str(source))
    if base is not None:
        return load_config(base), None
    return load_config(source), None


def load_scenario(name: str) -> EnvConfig:
    """Load ``suburban`` or a path under ``configs/scenarios/`` as EnvConfig."""
    return load_source(scenario_path(name))[0]


def empty_scenario(cfg: Optional[EnvConfig] = None) -> Scenario:
    return Scenario(name="", config=cfg or EnvConfig())

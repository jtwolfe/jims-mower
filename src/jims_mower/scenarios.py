"""Load named WAVE 1C yard scenarios from ``configs/scenarios/``."""

from __future__ import annotations

from pathlib import Path

from jims_mower.config import ConfigError, EnvConfig, load_config


def scenario_dir() -> Path:
    repo = Path(__file__).resolve().parents[2] / "configs" / "scenarios"
    if repo.is_dir():
        return repo
    return Path(__file__).resolve().parent / "data" / "scenarios"


def list_scenarios() -> list[str]:
    root = scenario_dir()
    if not root.is_dir():
        return []
    return sorted(path.stem for path in root.glob("*.yaml"))


def scenario_path(name: str) -> Path:
    raw = Path(name)
    if raw.is_file():
        return raw
    path = scenario_dir() / f"{name}.yaml"
    if not path.is_file():
        raise ConfigError(f"Scenario not found: {name} ({path})")
    return path


def load_scenario(name: str) -> EnvConfig:
    """Load ``suburban`` or a path under ``configs/scenarios/``."""
    return load_config(scenario_path(name))

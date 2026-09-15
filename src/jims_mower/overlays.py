"""Seasonal scenario overlays: long grass / leaf clutter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import yaml

from jims_mower.config import EnvConfig, overlay_config
from jims_mower.constants import OVERLAY_SCHEMA, SEASONS
from jims_mower.types import Obstacle
from jims_mower.world import place_obstacle


class OverlayError(ValueError):
    """Invalid seasonal overlay."""


def overlay_dir() -> Path:
    repo = Path(__file__).resolve().parents[2] / "configs" / "overlays"
    if repo.is_dir():
        return repo
    return Path(__file__).resolve().parent / "data" / "overlays"


def apply_season(cfg: EnvConfig, season: str) -> EnvConfig:
    """Mutate an env config for long grass or leaf clutter."""
    key = (season or "none").strip().lower()
    if key not in SEASONS:
        raise OverlayError(f"season must be one of {sorted(SEASONS)}; got {season!r}")
    cfg.world.season = key
    if key == "long_grass":
        cfg.world.grass.enabled = True
        cfg.world.grass.regenerate_frac = max(float(cfg.world.grass.regenerate_frac), 0.22)
    elif key == "leaf_clutter":
        cfg.world.n_toys = max(int(cfg.world.n_toys), 8)
    return cfg


def leaf_clutter_obstacles(
    rng,
    width_m: float,
    height_m: float,
    n: int = 8,
) -> list[Obstacle]:
    """Extra static toys as leaf piles. Deterministic given ``rng``."""
    out: list[Obstacle] = []
    for i in range(max(0, int(n))):
        x = float(rng.uniform(0.8, max(width_m - 0.8, 1.0)))
        y = float(rng.uniform(0.8, max(height_m - 0.8, 1.0)))
        out.append(place_obstacle("toy", x, y, name=f"leaf_{i}"))
    return out


def load_overlay(source: Union[str, Path, dict]) -> dict[str, Any]:
    if isinstance(source, dict):
        data = source
    else:
        path = Path(source)
        if not path.is_file():
            resolved = overlay_dir() / f"{source}.yaml"
            if not resolved.is_file():
                raise OverlayError(f"overlay not found: {source}")
            path = resolved
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise OverlayError("overlay YAML must be a mapping")
    schema = str(data.get("schema") or "").strip()
    if schema and schema != OVERLAY_SCHEMA:
        raise OverlayError(f"unsupported overlay schema {schema!r}")
    return data


def apply_overlay(cfg: EnvConfig, source: Union[str, Path, dict]) -> EnvConfig:
    data = load_overlay(source)
    season = str(data.get("season") or data.get("name") or "none")
    env = data.get("env") if isinstance(data.get("env"), dict) else {}
    merged = {k: v for k, v in data.items() if k in {"world", "weather", "planner", "perception", "curriculum"}}
    merged.update(env)
    if merged:
        overlay_config(cfg, merged)
    if season in SEASONS:
        apply_season(cfg, season)
    return cfg


def resolve_overlay(name: Optional[str]) -> Optional[Path]:
    if not name:
        return None
    path = Path(name)
    if path.is_file():
        return path
    candidate = overlay_dir() / f"{name}.yaml"
    return candidate if candidate.is_file() else None

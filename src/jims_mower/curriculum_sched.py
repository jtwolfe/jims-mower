"""Curriculum schedule: flat → suburban → wet → night."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import yaml

from jims_mower.constants import CURRICULUM_SCHEMA

DEFAULT_STAGES = (
    {"name": "flat", "scenario": "flat"},
    {"name": "suburban", "scenario": "suburban"},
    {"name": "wet", "scenario": "wet_slope"},
    {"name": "night", "scenario": "night_dawn"},
)


def curriculum_path(name: Optional[str] = None) -> Path:
    if name:
        raw = Path(name)
        if raw.is_file():
            return raw
    repo = Path(__file__).resolve().parents[2] / "configs" / "curriculum" / "flat_to_night.yaml"
    if repo.is_file():
        return repo
    return Path(__file__).resolve().parent / "data" / "curriculum" / "flat_to_night.yaml"


def load_curriculum(source: Optional[Path] = None) -> dict[str, Any]:
    path = curriculum_path(str(source) if source else None)
    if path.is_file():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        data = {
            "schema": CURRICULUM_SCHEMA,
            "name": "flat_to_night",
            "stages": [dict(s) for s in DEFAULT_STAGES],
        }
    if not isinstance(data, dict):
        raise ValueError("curriculum YAML must be a mapping")
    schema = str(data.get("schema") or CURRICULUM_SCHEMA)
    if schema != CURRICULUM_SCHEMA:
        raise ValueError(f"unsupported curriculum schema {schema!r}")
    stages = data.get("stages") or list(DEFAULT_STAGES)
    if not isinstance(stages, list) or not stages:
        raise ValueError("curriculum.stages must be a non-empty list")
    cleaned = []
    for item in stages:
        if not isinstance(item, dict) or "scenario" not in item:
            raise ValueError("each stage needs a scenario")
        cleaned.append(
            {
                "name": str(item.get("name") or item["scenario"]),
                "scenario": str(item["scenario"]),
                "steps": int(item.get("steps") or 0),
            }
        )
    return {
        "schema": schema,
        "name": str(data.get("name") or "flat_to_night"),
        "stages": cleaned,
    }


def stage_names(payload: Optional[dict[str, Any]] = None) -> list[str]:
    data = payload or load_curriculum()
    return [str(s["name"]) for s in data["stages"]]


def next_stage(current: Optional[str], payload: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
    data = payload or load_curriculum()
    stages = data["stages"]
    if current is None:
        return dict(stages[0])
    names = [str(s["name"]) for s in stages]
    if current not in names:
        return dict(stages[0])
    idx = names.index(current)
    if idx + 1 >= len(stages):
        return None
    return dict(stages[idx + 1])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inspect the WAVE 4 curriculum schedule")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    payload = load_curriculum(args.config)
    text = json.dumps(payload, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

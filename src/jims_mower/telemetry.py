"""Headless telemetry JSON: coverage, tip rate, drain entries, living near-misses."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional, Union

from jims_mower.constants import NEAR_MISS_LIVING_M, TELEMETRY_SCHEMA, TERRAIN_ADVICE
from jims_mower.episode import EpisodeReader
from jims_mower.metrics import EpisodeScorecard


def summarize_telemetry(
    *,
    steps: list[dict[str, Any]],
    seed: Optional[int] = None,
    scenario: str = "",
    policy: str = "",
    near_miss_m: float = NEAR_MISS_LIVING_M,
) -> dict[str, Any]:
    """Counts only. ``not_a_benchmark`` is always true."""
    n = len(steps)
    tips = 0
    drains = 0
    near_misses = 0
    min_person = math.inf
    advice = {k: 0 for k in TERRAIN_ADVICE}
    living = {k: 0 for k in TERRAIN_ADVICE}
    coverage = 0.0
    for rec in steps:
        info = rec.get("info") if isinstance(rec.get("info"), dict) else rec
        if info.get("tipover"):
            tips += 1
        if info.get("drain_drop"):
            drains += 1
        person = info.get("nearest_person_m")
        if person is not None and math.isfinite(float(person)):
            d = float(person)
            min_person = min(min_person, d)
            if d <= near_miss_m:
                near_misses += 1
        key = str(info.get("terrain_advice") or rec.get("terrain_advice") or "ok")
        if key not in advice:
            advice[key] = 0
        advice[key] += 1
        live = str(info.get("living_advice") or "ok")
        if live not in living:
            living[live] = 0
        living[live] += 1
        if info.get("coverage_fraction") is not None:
            coverage = float(info["coverage_fraction"])
        elif rec.get("coverage_fraction") is not None:
            coverage = float(rec["coverage_fraction"])
    return {
        "schema": TELEMETRY_SCHEMA,
        "seed": seed,
        "scenario": scenario,
        "policy": policy,
        "steps": n,
        "coverage_pct": 100.0 * coverage,
        "tip_count": tips,
        "tip_rate": (tips / n) if n else 0.0,
        "drain_entries": drains,
        "living_near_misses": near_misses,
        "living_near_miss_min_m": None if math.isinf(min_person) else float(min_person),
        "near_miss_threshold_m": float(near_miss_m),
        "terrain_advice": advice,
        "living_advice": living,
        "not_a_benchmark": True,
    }


def telemetry_from_episode(
    episode_dir: Union[str, Path],
    *,
    near_miss_m: float = NEAR_MISS_LIVING_M,
) -> dict[str, Any]:
    reader = EpisodeReader(episode_dir)
    steps = []
    for rec in reader.steps:
        blob = dict(rec.get("info") or {})
        blob.setdefault("terrain_advice", (rec.get("info") or {}).get("terrain_advice"))
        steps.append({"info": blob, "coverage_fraction": blob.get("coverage_fraction")})
    return summarize_telemetry(
        steps=steps,
        seed=reader.manifest.get("seed"),
        scenario=str(reader.manifest.get("scenario") or ""),
        policy=str(reader.manifest.get("policy") or ""),
        near_miss_m=near_miss_m,
    )


def telemetry_from_scorecard(card: EpisodeScorecard) -> dict[str, Any]:
    return {
        "schema": TELEMETRY_SCHEMA,
        "seed": card.seed,
        "scenario": card.scenario,
        "policy": card.policy,
        "steps": card.steps,
        "coverage_pct": card.coverage_pct,
        "tip_count": card.tip_count,
        "tip_rate": (card.tip_count / card.steps) if card.steps else 0.0,
        "drain_entries": card.drain_entries,
        "living_near_misses": 0 if card.near_miss_person_m is None else int(
            card.near_miss_person_m <= NEAR_MISS_LIVING_M
        ),
        "living_near_miss_min_m": card.near_miss_person_m,
        "near_miss_threshold_m": NEAR_MISS_LIVING_M,
        "terrain_advice": dict(card.terrain_advice),
        "not_a_benchmark": True,
    }


def telemetry_from_demo_summary(path: Union[str, Path]) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    log = payload.get("log") or []
    return summarize_telemetry(
        steps=log,
        seed=payload.get("seed"),
        scenario=str(payload.get("scenario") or ""),
        policy=str(payload.get("policy") or ""),
    )


def write_telemetry(path: Union[str, Path], payload: dict[str, Any]) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dest

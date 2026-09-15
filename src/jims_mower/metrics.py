"""Episode scorecards: coverage, tips, drains, near-miss, advice histogram."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from jims_mower.constants import TERRAIN_ADVICE
from jims_mower.env import MowerEnv
from jims_mower.planning import TerrainPolicy

POLICIES = ("terrain", "scripted", "random")


@dataclass
class EpisodeScorecard:
    """One-episode metrics. Counts are integers, not claimed detector scores."""

    seed: int
    scenario: str
    policy: str
    steps: int
    coverage_pct: float
    tip_count: int
    drain_entries: int
    near_miss_person_m: Optional[float]
    terrain_advice: dict[str, int]
    terminated: bool
    truncated: bool
    collision: Optional[str] = None
    success: bool = False
    out_of_bounds: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.near_miss_person_m is not None and math.isinf(self.near_miss_person_m):
            payload["near_miss_person_m"] = None
        return payload


def _scripted_action() -> np.ndarray:
    return np.array([0.55, 0.50, 1.0], dtype=np.float32)


def _random_action(env: MowerEnv, rng: np.random.Generator) -> np.ndarray:
    action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
    action[2] = 1.0
    return action


def evaluate_episode(
    env: MowerEnv,
    *,
    seed: int,
    steps: int,
    policy: str = "terrain",
    close: bool = False,
) -> EpisodeScorecard:
    """Run one episode and return a scorecard. Does not invent mAP / FPS."""
    name = (policy or "terrain").strip().lower()
    if name not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}; got {policy!r}")
    obs, info = env.reset(seed=seed)
    terrain_policy: Optional[TerrainPolicy] = None
    rng = np.random.default_rng(seed)
    if name == "terrain":
        terrain_policy = TerrainPolicy(env.cfg)
        terrain_policy.reset(obs, info)

    advice = Counter({k: 0 for k in TERRAIN_ADVICE})
    tip_count = 0
    drain_entries = 0
    near_miss = math.inf
    steps_run = 0
    terminated = False
    truncated = False
    collision = None
    success = False
    oob = False

    def _consume(info_i: dict[str, Any]) -> None:
        nonlocal tip_count, drain_entries, near_miss, collision, success, oob
        key = str(info_i.get("terrain_advice") or "ok")
        if key not in advice:
            advice[key] = 0
        advice[key] += 1
        if info_i.get("tipover"):
            tip_count += 1
        if info_i.get("drain_drop"):
            drain_entries += 1
        person_m = info_i.get("nearest_person_m")
        if person_m is not None and not math.isinf(float(person_m)):
            near_miss = min(near_miss, float(person_m))
        if info_i.get("collision"):
            collision = info_i.get("collision")
        success = bool(info_i.get("success"))
        oob = bool(info_i.get("out_of_bounds"))

    _consume(info)
    # Reset advice is the spawn state; do not count it as a control step.
    advice = Counter({k: 0 for k in TERRAIN_ADVICE})
    tip_count = 0
    drain_entries = 0

    for _ in range(max(0, int(steps))):
        if name == "scripted":
            action = _scripted_action()
        elif name == "random":
            action = _random_action(env, rng)
        else:
            assert terrain_policy is not None
            action = terrain_policy.act(obs, info)
        obs, _reward, terminated, truncated, info = env.step(action)
        steps_run += 1
        _consume(info)
        if terminated or truncated:
            break

    coverage_pct = 100.0 * float(info.get("coverage_fraction") or 0.0)
    card = EpisodeScorecard(
        seed=int(seed),
        scenario=str(info.get("scenario") or (env.scenario.name if env.scenario else "")),
        policy=name,
        steps=steps_run,
        coverage_pct=coverage_pct,
        tip_count=tip_count,
        drain_entries=drain_entries,
        near_miss_person_m=None if math.isinf(near_miss) else float(near_miss),
        terrain_advice={k: int(advice.get(k, 0)) for k in TERRAIN_ADVICE},
        terminated=bool(terminated),
        truncated=bool(truncated),
        collision=collision if isinstance(collision, str) else None,
        success=success,
        out_of_bounds=oob,
    )
    if close:
        env.close()
    return card


def write_scorecard(path: Path, card: EpisodeScorecard) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card.to_dict(), indent=2), encoding="utf-8")


def summarize_scorecards(cards: list[EpisodeScorecard]) -> dict[str, Any]:
    n = len(cards)
    tips = sum(c.tip_count for c in cards)
    drains = sum(c.drain_entries for c in cards)
    coverages = [c.coverage_pct for c in cards]
    advice = Counter()
    for card in cards:
        advice.update(card.terrain_advice)
    return {
        "n_episodes": n,
        "n_tips": tips,
        "n_drain_entries": drains,
        "mean_coverage_pct": float(sum(coverages) / n) if n else 0.0,
        "max_coverage_pct": float(max(coverages)) if coverages else 0.0,
        "terrain_advice": {k: int(advice.get(k, 0)) for k in TERRAIN_ADVICE},
        "episodes": [c.to_dict() for c in cards],
    }

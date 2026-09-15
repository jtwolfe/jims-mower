"""Lightweight overnight regression farm: N seeds × M scenarios.

Writes a scorecard summary JSON. Exits non-zero when tip / drain totals
exceed the configured thresholds. ``--dry-run`` only resolves the matrix.

PR CI should not invoke the full farm; use the optional nightly workflow.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from jims_mower.env import MowerEnv
from jims_mower.metrics import EpisodeScorecard, evaluate_episode, summarize_scorecards
from jims_mower.scenarios import list_scenarios, load_source

DEFAULT_MAX_TIPS = 0
DEFAULT_MAX_DRAINS = 0


class FarmThresholdError(RuntimeError):
    """Tip or drain totals exceeded the farm gates."""


def _parse_csv_ints(raw: str) -> list[int]:
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if not parts:
        raise ValueError("need at least one seed")
    return [int(p) for p in parts]


def _parse_scenario_list(raw: Optional[str]) -> list[str]:
    if not raw:
        names = list_scenarios()
        if not names:
            raise ValueError("no bundled scenarios found")
        return names
    return [p.strip() for p in str(raw).split(",") if p.strip()]


def plan_matrix(seeds: list[int], scenarios: list[str]) -> list[dict[str, object]]:
    return [{"seed": s, "scenario": sc} for sc in scenarios for s in seeds]


def run_farm(
    out_dir: Path,
    *,
    seeds: list[int],
    scenarios: list[str],
    steps: int = 40,
    policy: str = "terrain",
    cameras: Optional[int] = None,
    dry_run: bool = False,
    max_tips: int = DEFAULT_MAX_TIPS,
    max_drain_entries: int = DEFAULT_MAX_DRAINS,
    terrain_observer: Optional[str] = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = plan_matrix(seeds, scenarios)
    summary: dict = {
        "dry_run": bool(dry_run),
        "steps": int(steps),
        "policy": policy,
        "seeds": list(seeds),
        "scenarios": list(scenarios),
        "n_planned": len(matrix),
        "matrix": matrix,
        "thresholds": {"max_tips": int(max_tips), "max_drain_entries": int(max_drain_entries)},
        "breached": False,
    }
    if dry_run:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (out_dir / "matrix.json").write_text(json.dumps(matrix, indent=2), encoding="utf-8")
        return summary

    cards: list[EpisodeScorecard] = []
    for item in matrix:
        seed = int(item["seed"])
        name = str(item["scenario"])
        cfg, scenario = load_source(name)
        if cameras is not None:
            cfg.sensors.camera_count = int(cameras)
            cfg.sensors.cameras = []
        if terrain_observer:
            cfg.perception.terrain_mode = terrain_observer.strip().lower()
        env = MowerEnv(config=cfg, scenario=scenario, render_mode=None)
        card = evaluate_episode(env, seed=seed, steps=steps, policy=policy, close=True)
        if not card.scenario:
            card.scenario = name
        cards.append(card)
        write_path = out_dir / f"{name}_seed{seed}.json"
        write_path.write_text(json.dumps(card.to_dict(), indent=2), encoding="utf-8")

    stats = summarize_scorecards(cards)
    breached = stats["n_tips"] > max_tips or stats["n_drain_entries"] > max_drain_entries
    summary.update(stats)
    summary["breached"] = bool(breached)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if breached:
        raise FarmThresholdError(
            f"farm gates breached: tips={stats['n_tips']} (max {max_tips}), "
            f"drains={stats['n_drain_entries']} (max {max_drain_entries})"
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="jims-mower overnight farm (headless scorecards)")
    p.add_argument("--out", type=Path, default=Path("farm_out"))
    p.add_argument("--seeds", type=str, default="0,1,5", help="comma-separated ints")
    p.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help="comma-separated names (default: all bundled scenarios)",
    )
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--policy", default="terrain")
    p.add_argument("--cameras", type=int, default=4, help="4–6; default 4 to keep the farm light")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-tips", type=int, default=DEFAULT_MAX_TIPS)
    p.add_argument("--max-drains", type=int, default=DEFAULT_MAX_DRAINS)
    p.add_argument(
        "--terrain-observer",
        choices=("heuristic", "oracle", "blind", "learned"),
        default=None,
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        summary = run_farm(
            args.out,
            seeds=_parse_csv_ints(args.seeds),
            scenarios=_parse_scenario_list(args.scenarios),
            steps=args.steps,
            policy=args.policy,
            cameras=args.cameras,
            dry_run=args.dry_run,
            max_tips=args.max_tips,
            max_drain_entries=args.max_drains,
            terrain_observer=args.terrain_observer,
        )
    except FarmThresholdError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
    print(
        f"farm {'dry-run ' if summary.get('dry_run') else ''}"
        f"planned={summary['n_planned']} "
        f"tips={summary.get('n_tips', 0)} drains={summary.get('n_drain_entries', 0)} "
        f"→ {args.out / 'summary.json'}"
    )


if __name__ == "__main__":
    main()

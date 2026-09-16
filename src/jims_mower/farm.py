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

from jims_mower.constants import TERRAIN_MODE_CLI
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


def load_quarantine(path: Optional[Path]) -> set[str]:
    if path is None or not Path(path).is_file():
        return set()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        names = raw.get("quarantined") or raw.get("scenarios") or []
    elif isinstance(raw, list):
        names = raw
    else:
        names = []
    return {str(n) for n in names}


def write_quarantine(path: Path, names: set[str], reasons: dict[str, str]) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(
            {
                "quarantined": sorted(names),
                "reasons": {k: reasons[k] for k in sorted(reasons)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


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
    flake_budget: int = 1,
    quarantine_path: Optional[Path] = None,
    quarantine: Optional[list[str]] = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = plan_matrix(seeds, scenarios)
    quarantined = load_quarantine(quarantine_path)
    if quarantine:
        quarantined.update(str(n) for n in quarantine)
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
        "flake_budget": int(flake_budget),
        "quarantined": sorted(quarantined),
        "skipped": [],
        "flakes": [],
    }
    if dry_run:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (out_dir / "matrix.json").write_text(json.dumps(matrix, indent=2), encoding="utf-8")
        return summary

    cards: list[EpisodeScorecard] = []
    flakes: dict[str, int] = {}
    flake_reasons: dict[str, str] = {}
    q_path = Path(quarantine_path) if quarantine_path is not None else out_dir / "quarantine.json"
    for item in matrix:
        seed = int(item["seed"])
        name = str(item["scenario"])
        if name in quarantined:
            summary["skipped"].append(
                {"seed": seed, "scenario": name, "reason": "quarantined"}
            )
            continue
        try:
            cfg, scenario = load_source(name)
            if cameras is not None:
                cfg.sensors.camera_count = int(cameras)
                cfg.sensors.cameras = []
            if terrain_observer:
                from jims_mower.perception.terrain import normalize_terrain_mode

                cfg.perception.terrain_mode = normalize_terrain_mode(terrain_observer)
            env = MowerEnv(config=cfg, scenario=scenario, render_mode=None)
            card = evaluate_episode(env, seed=seed, steps=steps, policy=policy, close=True)
        except Exception as exc:  # noqa: BLE001 — farm records flakes instead of silent skip
            flakes[name] = flakes.get(name, 0) + 1
            flake_reasons[name] = f"{type(exc).__name__}: {exc}"
            summary["flakes"].append(
                {"seed": seed, "scenario": name, "error": flake_reasons[name]}
            )
            if flakes[name] >= int(flake_budget):
                quarantined.add(name)
                write_quarantine(q_path, quarantined, flake_reasons)
            continue
        if not card.scenario:
            card.scenario = name
        cards.append(card)
        write_path = out_dir / f"{name}_seed{seed}.json"
        write_path.write_text(json.dumps(card.to_dict(), indent=2), encoding="utf-8")
    summary["quarantined"] = sorted(quarantined)
    if quarantined:
        write_quarantine(q_path, quarantined, flake_reasons)

    stats = summarize_scorecards(cards) if cards else {
        "n_episodes": 0,
        "n_tips": 0,
        "n_drain_entries": 0,
    }
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
        choices=TERRAIN_MODE_CLI,
        default=None,
    )
    p.add_argument(
        "--flake-budget",
        type=int,
        default=1,
        help="quarantine a scenario after this many exceptions (not a silent skip)",
    )
    p.add_argument(
        "--quarantine",
        type=str,
        default="",
        help="comma-separated scenario names to skip (recorded as quarantined)",
    )
    p.add_argument(
        "--quarantine-file",
        type=Path,
        default=None,
        help="JSON list of quarantined scenarios (read/write)",
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
            flake_budget=args.flake_budget,
            quarantine_path=args.quarantine_file,
            quarantine=[p.strip() for p in str(args.quarantine).split(",") if p.strip()],
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

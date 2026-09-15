"""Multi-yard coverage sequence (paddock then suburban, …)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from jims_mower.constants import SEQUENCE_SCHEMA
from jims_mower.env import MowerEnv
from jims_mower.metrics import EpisodeScorecard, evaluate_episode, summarize_scorecards
from jims_mower.scenarios import load_source


DEFAULT_YARDS = ("paddock", "suburban")


def load_sequence(source: Optional[Path] = None) -> dict[str, Any]:
    if source is None:
        return {
            "schema": SEQUENCE_SCHEMA,
            "name": "paddock_then_suburban",
            "yards": list(DEFAULT_YARDS),
        }
    path = Path(source)
    if not path.is_file():
        bundled = Path(__file__).resolve().parents[2] / "configs" / "sequences" / path.name
        if not bundled.is_file():
            bundled = Path(__file__).resolve().parent / "data" / "sequences" / path.name
        if bundled.is_file():
            path = bundled
        else:
            raise FileNotFoundError(f"sequence file not found: {source}")
    if path.suffix in {".json"}:
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("sequence file must be a mapping")
    schema = str(data.get("schema") or SEQUENCE_SCHEMA)
    if schema != SEQUENCE_SCHEMA:
        raise ValueError(f"unsupported sequence schema {schema!r}")
    yards = [str(y) for y in (data.get("yards") or DEFAULT_YARDS)]
    if not yards:
        raise ValueError("sequence.yards must be non-empty")
    data["schema"] = schema
    data["yards"] = yards
    return data


def run_sequence(
    out_dir: Path,
    *,
    yards: list[str],
    steps: int = 12,
    seed: int = 0,
    cameras: int = 4,
    policy: str = "terrain",
    dry_run: bool = False,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "schema": SEQUENCE_SCHEMA,
        "yards": list(yards),
        "steps": int(steps),
        "seed": int(seed),
        "policy": policy,
        "dry_run": bool(dry_run),
        "n_planned": len(yards),
        "cards": [],
    }
    if dry_run:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    cards: list[EpisodeScorecard] = []
    for i, name in enumerate(yards):
        cfg, scenario = load_source(name)
        cfg.sensors.camera_count = int(cameras)
        cfg.sensors.cameras = []
        env = MowerEnv(config=cfg, scenario=scenario, render_mode=None)
        card = evaluate_episode(env, seed=seed + i, steps=steps, policy=policy, close=True)
        if not card.scenario:
            card.scenario = name
        cards.append(card)
        (out_dir / f"{i:02d}_{name}.json").write_text(
            json.dumps(card.to_dict(), indent=2), encoding="utf-8"
        )
    stats = summarize_scorecards(cards)
    summary.update(stats)
    summary["cards"] = [c.to_dict() for c in cards]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a multi-yard coverage sequence")
    p.add_argument("--out", type=Path, default=Path("sequence_out"))
    p.add_argument("--yards", type=str, default="paddock,suburban")
    p.add_argument("--sequence", type=Path, default=None)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cameras", type=int, default=4)
    p.add_argument("--policy", default="terrain")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.sequence is not None:
        data = load_sequence(args.sequence)
        yards = list(data["yards"])
    else:
        yards = [p.strip() for p in str(args.yards).split(",") if p.strip()]
    summary = run_sequence(
        args.out,
        yards=yards,
        steps=args.steps,
        seed=args.seed,
        cameras=args.cameras,
        policy=args.policy,
        dry_run=args.dry_run,
    )
    print(
        f"sequence {'dry-run ' if summary.get('dry_run') else ''}"
        f"yards={summary['yards']} → {args.out / 'summary.json'}"
    )


if __name__ == "__main__":
    main()

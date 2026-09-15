"""Design-study ablations: camera count, ToF count, IMU noise on frozen seeds.

Reports **tip** and **drain-entry** rates from the gym physics scorecard.
That is a wheel-in-channel / tip-over count, not detector recall and not
mAP. There is no FPS column.

``--dry-run`` only writes the planned matrix (PR CI). A live sweep is a
laptop / overnight job.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Optional

from jims_mower.config import overlay_config
from jims_mower.env import MowerEnv
from jims_mower.metrics import EpisodeScorecard, evaluate_episode, summarize_scorecards
from jims_mower.scenarios import load_source

# Frozen seeds used by the heuristic planner / farm. Do not invent new ones
# in CI without updating tests.
DEFAULT_SEEDS = (0, 1, 7)
DEFAULT_CAMERAS = (4, 5, 6)
DEFAULT_TOF = (0, 2, 4)
# Multipliers on YAML ``sensors.imu.accel_noise_std`` / ``gyro_noise_std``.
DEFAULT_IMU_SCALES = (1.0, 4.0)
DEFAULT_SCENARIO = "steep_yard"


def _parse_csv_ints(raw: str) -> list[int]:
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if not parts:
        raise ValueError("need at least one integer")
    return [int(p) for p in parts]


def _parse_csv_floats(raw: str) -> list[float]:
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if not parts:
        raise ValueError("need at least one number")
    return [float(p) for p in parts]


def apply_ablation(
    cfg: Any,
    *,
    cameras: int,
    tof_count: int,
    imu_scale: float,
) -> Any:
    """Mutate a loaded config for one ablation cell."""
    overlay = {
        "sensors": {
            "camera_count": int(cameras),
            "cameras": [],
            "tof": {"count": int(tof_count), "enabled": int(tof_count) > 0},
            "imu": {
                "accel_noise_std": float(cfg.sensors.imu.accel_noise_std) * float(imu_scale),
                "gyro_noise_std": float(cfg.sensors.imu.gyro_noise_std) * float(imu_scale),
            },
        }
    }
    return overlay_config(cfg, overlay)


def plan_study_matrix(
    *,
    seeds: list[int],
    cameras: list[int],
    tof_counts: list[int],
    imu_scales: list[float],
    scenario: str,
) -> list[dict[str, Any]]:
    matrix: list[dict[str, Any]] = []
    for cam in cameras:
        for tof in tof_counts:
            for scale in imu_scales:
                for seed in seeds:
                    matrix.append(
                        {
                            "scenario": scenario,
                            "seed": int(seed),
                            "cameras": int(cam),
                            "tof_count": int(tof),
                            "imu_scale": float(scale),
                        }
                    )
    return matrix


def _cell_key(item: dict[str, Any]) -> tuple:
    return (
        int(item["cameras"]),
        int(item["tof_count"]),
        float(item["imu_scale"]),
        str(item["scenario"]),
    )


def summarize_cells(cards: list[EpisodeScorecard], matrix: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (cameras, tof, imu_scale) with tip / drain-entry rates."""
    grouped: dict[tuple, list[EpisodeScorecard]] = {}
    extras: dict[tuple, dict[str, Any]] = {}
    for item, card in zip(matrix, cards):
        key = _cell_key(item)
        grouped.setdefault(key, []).append(card)
        extras[key] = {
            "cameras": int(item["cameras"]),
            "tof_count": int(item["tof_count"]),
            "imu_scale": float(item["imu_scale"]),
            "scenario": str(item["scenario"]),
        }
    rows: list[dict[str, Any]] = []
    for key, group in grouped.items():
        n = len(group)
        tips = sum(c.tip_count for c in group)
        drains = sum(c.drain_entries for c in group)
        tipped = sum(1 for c in group if c.tip_count > 0)
        drain_hit = sum(1 for c in group if c.drain_entries > 0)
        cover = [c.coverage_pct for c in group]
        meta = extras[key]
        rows.append(
            {
                **meta,
                "n_episodes": n,
                "n_tips": tips,
                "n_drain_entries": drains,
                "tip_rate": (tipped / n) if n else 0.0,
                "drain_entry_rate": (drain_hit / n) if n else 0.0,
                "mean_coverage_pct": (sum(cover) / n) if n else 0.0,
                "seeds": [c.seed for c in group],
                "not_mAP": True,
                "not_a_benchmark": True,
                "fps_claim": None,
            }
        )
    rows.sort(key=lambda r: (r["cameras"], r["tof_count"], r["imu_scale"]))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scenario",
        "cameras",
        "tof_count",
        "imu_scale",
        "n_episodes",
        "n_tips",
        "n_drain_entries",
        "tip_rate",
        "drain_entry_rate",
        "mean_coverage_pct",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Design study — tip / drain-entry rates",
        "",
        "Honest gym physics counts on **frozen seeds**. `tip_rate` is the",
        "fraction of episodes with a tip-over. `drain_entry_rate` is the",
        "fraction of episodes with a wheel in a channel (`drain_drop`).",
        "These are **not** detector recall, mAP, or FPS.",
        "",
        f"- scenario: `{summary.get('scenario')}`",
        f"- seeds: {summary.get('seeds')}",
        f"- steps: {summary.get('steps')}",
        f"- policy: `{summary.get('policy')}`",
        f"- dry_run: {summary.get('dry_run')}",
        f"- n_planned: {summary.get('n_planned')}",
        "",
        "| cameras | ToF | IMU× | n | tips | drains | tip_rate | drain_entry_rate | coverage % |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.get("cells") or []:
        lines.append(
            "| {cameras} | {tof_count} | {imu_scale} | {n_episodes} | {n_tips} | "
            "{n_drain_entries} | {tip_rate:.3f} | {drain_entry_rate:.3f} | "
            "{mean_coverage_pct:.2f} |".format(**row)
        )
    if not summary.get("cells"):
        lines.append("| — | — | — | 0 | — | — | — | — | — |")
        lines.append("")
        lines.append("Dry-run: no episodes executed. Re-run without `--dry-run` for counts.")
    lines.append("")
    lines.append("`fps_claim`: null. `not_a_benchmark`: true.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_study(
    out_dir: Path,
    *,
    seeds: list[int],
    cameras: list[int],
    tof_counts: list[int],
    imu_scales: list[float],
    scenario: str = DEFAULT_SCENARIO,
    steps: int = 20,
    policy: str = "terrain",
    dry_run: bool = False,
    terrain_observer: Optional[str] = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = plan_study_matrix(
        seeds=seeds,
        cameras=cameras,
        tof_counts=tof_counts,
        imu_scales=imu_scales,
        scenario=scenario,
    )
    summary: dict[str, Any] = {
        "dry_run": bool(dry_run),
        "steps": int(steps),
        "policy": policy,
        "scenario": scenario,
        "seeds": list(seeds),
        "cameras": list(cameras),
        "tof_counts": list(tof_counts),
        "imu_scales": list(imu_scales),
        "n_planned": len(matrix),
        "matrix": matrix,
        "cells": [],
        "not_a_benchmark": True,
        "fps_claim": None,
        "map_claim": None,
    }
    if dry_run:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (out_dir / "matrix.json").write_text(json.dumps(matrix, indent=2), encoding="utf-8")
        write_csv(out_dir / "report.csv", [])
        write_markdown(out_dir / "report.md", summary)
        return summary

    cards: list[EpisodeScorecard] = []
    for item in matrix:
        cfg, scen = load_source(scenario)
        cfg = apply_ablation(
            cfg,
            cameras=int(item["cameras"]),
            tof_count=int(item["tof_count"]),
            imu_scale=float(item["imu_scale"]),
        )
        if terrain_observer:
            cfg.perception.terrain_mode = terrain_observer.strip().lower()
        env = MowerEnv(config=cfg, scenario=scen, render_mode=None)
        card = evaluate_episode(
            env, seed=int(item["seed"]), steps=steps, policy=policy, close=True
        )
        if not card.scenario:
            card.scenario = scenario
        card.extra = {
            "cameras": int(item["cameras"]),
            "tof_count": int(item["tof_count"]),
            "imu_scale": float(item["imu_scale"]),
        }
        cards.append(card)

    cells = summarize_cells(cards, matrix)
    stats = summarize_scorecards(cards)
    summary.update(stats)
    summary["cells"] = cells
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "matrix.json").write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    write_csv(out_dir / "report.csv", cells)
    write_markdown(out_dir / "report.md", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="jims-mower design-study ablations (honest tip/drain rates)")
    p.add_argument("--out", type=Path, default=Path("study_out"))
    p.add_argument("--seeds", type=str, default="0,1,7")
    p.add_argument("--cameras", type=str, default="4,5,6")
    p.add_argument("--tof", type=str, default="0,2,4", help="ToF corner counts: 0, 2, or 4")
    p.add_argument("--imu-scales", type=str, default="1,4", help="multipliers on IMU white-noise std")
    p.add_argument("--scenario", type=str, default=DEFAULT_SCENARIO)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--policy", default="terrain")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--terrain-observer",
        choices=("heuristic", "oracle", "blind"),
        default=None,
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        summary = run_study(
            args.out,
            seeds=_parse_csv_ints(args.seeds),
            cameras=_parse_csv_ints(args.cameras),
            tof_counts=_parse_csv_ints(args.tof),
            imu_scales=_parse_csv_floats(args.imu_scales),
            scenario=args.scenario,
            steps=args.steps,
            policy=args.policy,
            dry_run=args.dry_run,
            terrain_observer=args.terrain_observer,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
    print(
        f"study {'dry-run ' if summary.get('dry_run') else ''}"
        f"planned={summary['n_planned']} "
        f"tips={summary.get('n_tips', '—')} drains={summary.get('n_drain_entries', '—')} "
        f"→ {args.out / 'report.md'}"
    )


if __name__ == "__main__":
    main()

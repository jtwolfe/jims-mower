"""Design-study dry-run and a single-cell live ablation."""

from __future__ import annotations

import json
from pathlib import Path

from jims_mower.studies import apply_ablation, main, plan_study_matrix, run_study
from jims_mower.scenarios import load_source


def test_plan_study_matrix() -> None:
    matrix = plan_study_matrix(
        seeds=[0, 7],
        cameras=[4, 6],
        tof_counts=[0, 4],
        imu_scales=[1.0, 4.0],
        scenario="steep_yard",
    )
    assert len(matrix) == 16
    assert matrix[0]["cameras"] == 4
    assert matrix[0]["tof_count"] == 0


def test_apply_ablation_tof_and_imu() -> None:
    cfg, _scen = load_source("steep_yard")
    base_imu = cfg.sensors.imu.accel_noise_std
    out = apply_ablation(cfg, cameras=5, tof_count=2, imu_scale=4.0)
    assert len(out.resolved_cameras()) == 5
    assert out.sensors.tof.count == 2
    assert out.sensors.tof.enabled is True
    assert out.sensors.imu.accel_noise_std == base_imu * 4.0

    cfg0, _ = load_source("steep_yard")
    off = apply_ablation(cfg0, cameras=4, tof_count=0, imu_scale=1.0)
    assert off.sensors.tof.count == 0
    assert off.sensors.tof.enabled is False


def test_study_dry_run(tmp_path: Path) -> None:
    summary = run_study(
        tmp_path,
        seeds=[0, 1, 7],
        cameras=[4, 5, 6],
        tof_counts=[0, 2, 4],
        imu_scales=[1.0, 4.0],
        scenario="steep_yard",
        steps=20,
        dry_run=True,
    )
    assert summary["dry_run"] is True
    assert summary["n_planned"] == 3 * 3 * 2 * 3
    assert summary["fps_claim"] is None
    assert summary["map_claim"] is None
    assert (tmp_path / "report.md").is_file()
    assert (tmp_path / "report.csv").is_file()
    assert (tmp_path / "matrix.json").is_file()
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "not" in md.lower() and "mAP" in md
    assert "FPS" in md or "fps" in md.lower()


def test_study_cli_dry_run(tmp_path: Path) -> None:
    main(
        [
            "--dry-run",
            "--seeds",
            "0",
            "--cameras",
            "4,6",
            "--tof",
            "0,4",
            "--imu-scales",
            "1",
            "--out",
            str(tmp_path),
        ]
    )
    payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert payload["n_planned"] == 4


def test_study_tiny_live(tmp_path: Path) -> None:
    summary = run_study(
        tmp_path,
        seeds=[0],
        cameras=[4],
        tof_counts=[4],
        imu_scales=[1.0],
        scenario="playground",
        steps=2,
        policy="scripted",
        dry_run=False,
        terrain_observer="oracle",
    )
    assert summary["n_episodes"] == 1
    assert summary["cells"]
    row = summary["cells"][0]
    assert row["not_mAP"] is True
    assert row["fps_claim"] is None
    assert (tmp_path / "report.csv").is_file()

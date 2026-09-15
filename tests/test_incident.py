"""Incident replay viewer + telemetry JSON."""

from __future__ import annotations

import json
from pathlib import Path

from jims_mower.episode import record_episode
from jims_mower.incident import write_incident_viewer
from jims_mower.incident_cli import main as incident_main
from jims_mower.telemetry import telemetry_from_episode
from jims_mower.telemetry_cli import main as telemetry_main


def _tiny() -> dict:
    return {
        "dt": 0.1,
        "max_steps": 20,
        "sensors": {"width": 32, "height": 24, "camera_count": 4, "fov_deg": 70.0},
        "world": {
            "width_m": 8.0,
            "height_m": 8.0,
            "resolution_m": 0.20,
            "n_people": 0,
            "n_dogs": 0,
            "n_cats": 0,
            "n_birds": 0,
            "n_trees": 0,
            "n_furniture": 0,
            "n_toys": 0,
            "terrain": {"enabled": False},
        },
        "perception": {"terrain_mode": "oracle"},
        "robot": {
            "trimmer": {"safety_radius_m": 1.2, "offset_m": 0.32, "radius_m": 0.16}
        },
    }


def test_incident_viewer_and_telemetry(tmp_path: Path) -> None:
    ep = tmp_path / "ep"
    record_episode(ep, steps=3, seed=2, config=_tiny(), cameras=4, policy="scripted")
    view = tmp_path / "view"
    index = write_incident_viewer(ep, view)
    assert index["n_frames"] >= 1
    assert (view / "index.html").is_file()
    assert (view / "index.json").is_file()
    html = (view / "index.html").read_text(encoding="utf-8")
    assert "advice" in html
    assert list(view.joinpath("frames").glob("*.png"))
    tel = telemetry_from_episode(ep)
    assert tel["not_a_benchmark"] is True
    assert tel["steps"] >= 1
    assert "coverage_pct" in tel
    assert "tip_rate" in tel
    assert "drain_entries" in tel
    assert "living_near_misses" in tel
    assert tel["tip_count"] == 0


def test_incident_and_telemetry_cli(tmp_path: Path) -> None:
    ep = tmp_path / "cli-ep"
    record_episode(ep, steps=2, seed=1, config=_tiny(), cameras=4, policy="scripted")
    view = tmp_path / "cli-view"
    incident_main([str(ep), "--out", str(view)])
    assert (view / "index.html").is_file()
    tel = tmp_path / "tel.json"
    telemetry_main([str(ep), "--out", str(tel)])
    payload = json.loads(tel.read_text(encoding="utf-8"))
    assert payload["not_a_benchmark"] is True
    assert payload["schema"].startswith("jims_mower.telemetry")

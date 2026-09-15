"""Record / replay roundtrip: offline planner match + env action playback."""

from __future__ import annotations

import json
from pathlib import Path

from jims_mower.contract import CONTRACT_VERSION
from jims_mower.episode import record_episode, replay_env, replay_offline
from jims_mower.record_cli import main as record_main
from jims_mower.replay_cli import main as replay_main


def _tiny() -> dict:
    return {
        "dt": 0.1,
        "max_steps": 40,
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


def test_record_replay_roundtrip(tmp_path: Path) -> None:
    ep = tmp_path / "ep"
    result = record_episode(ep, steps=4, seed=3, config=_tiny(), cameras=4, policy="terrain")
    assert result["n_steps"] >= 1
    assert (ep / "manifest.json").is_file()
    assert (ep / "steps.jsonl").is_file()
    assert (ep / "scorecard.json").is_file()
    manifest = json.loads((ep / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["contract_version"] == CONTRACT_VERSION
    assert manifest["n_steps"] == result["n_steps"]
    card = json.loads((ep / "scorecard.json").read_text(encoding="utf-8"))
    assert card["not_a_benchmark"] is True
    assert card["fps_claim"] is None
    assert "frames" in (ep / "steps.jsonl").read_text(encoding="utf-8")

    offline = replay_offline(ep)
    assert offline["n_steps"] == result["n_steps"]
    assert offline["max_abs_action_delta"] < 1e-5

    played = replay_env(ep)
    assert played["n_steps"] >= 1
    assert "final_coverage_fraction" in played


def test_record_replay_cli(tmp_path: Path) -> None:
    ep = tmp_path / "cli-ep"
    record_main(
        [
            "--out",
            str(ep),
            "--steps",
            "2",
            "--seed",
            "1",
            "--cameras",
            "4",
            "--policy",
            "scripted",
        ]
    )
    assert (ep / "manifest.json").is_file()
    out = tmp_path / "cli-rep"
    replay_main([str(ep), "--mode", "offline", "--out", str(out)])
    assert (out / "replay.json").is_file()
    replay_main([str(ep), "--mode", "env"])

"""Multiprocessing / in-process bridge + record/replay compatibility."""

from __future__ import annotations

import json
from pathlib import Path

from jims_mower.episode import record_episode
from jims_mower.runtime.bridge import MultiprocessBridge, replay_through_bridge
from jims_mower.runtime.export_trt import run_export


def _tiny() -> dict:
    return {
        "dt": 0.1,
        "max_steps": 20,
        "sensors": {"width": 16, "height": 12, "camera_count": 4},
        "world": {
            "width_m": 8.0,
            "height_m": 8.0,
            "resolution_m": 0.25,
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
    }


def test_inprocess_bridge_publishes() -> None:
    from jims_mower.env import MowerEnv

    env = MowerEnv(config=_tiny(), render_mode=None)
    obs, _info = env.reset(seed=2)
    with MultiprocessBridge(use_process=False) as bridge:
        msgs = bridge.publish_obs(obs, stamp_s=0.0)
    kinds = {m.kind for m in msgs}
    assert "ImuSample" in kinds
    assert "CameraFrame" in kinds
    env.close()


def test_bridge_replays_recorded_episode(tmp_path: Path) -> None:
    ep = tmp_path / "ep"
    result = record_episode(ep, steps=3, seed=4, config=_tiny(), cameras=4, policy="scripted")
    assert result["n_steps"] >= 1
    summary = replay_through_bridge(ep, use_process=False, out_path=tmp_path / "bridge.json")
    assert summary["n_observations"] >= 1
    assert summary["n_messages"] >= 4
    assert summary["fps_claim"] is None
    assert summary["not_a_benchmark"] is True
    assert summary["kinds"]["ImuSample"] >= 1
    assert (tmp_path / "bridge.json").is_file()
    payload = json.loads((tmp_path / "bridge.json").read_text(encoding="utf-8"))
    assert payload["contract_version"]


def test_multiprocess_bridge_roundtrip() -> None:
    from jims_mower.env import MowerEnv

    env = MowerEnv(config=_tiny(), render_mode=None)
    obs, _info = env.reset(seed=5)
    with MultiprocessBridge(use_process=True) as bridge:
        msgs = bridge.publish_obs(obs, stamp_s=0.2)
    assert any(m.kind == "GpsFix" for m in msgs)
    env.close()


def test_tensorrt_placeholder_dry_run(tmp_path: Path) -> None:
    out = tmp_path / "trt.json"
    payload = run_export(onnx=None, engine=tmp_path / "x.engine", dry_run=True, out=out)
    assert payload["fps_claim"] is None
    assert payload["map_claim"] is None
    assert payload["iou_claim"] is None
    assert payload["field_ready"] is False
    assert payload["not_a_benchmark"] is True
    assert payload["dry_run"] is True
    assert "trtexec" in payload["command"][0]

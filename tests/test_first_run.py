"""First-run teach → YardProfile → live Start (no full-acre mow)."""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from pathlib import Path

from jims_mower.app.live_backend import LiveBackend
from jims_mower.app.server import make_server
from jims_mower.constants import LIVE_SCHEMA
from jims_mower.live import LiveSession, owner_copy_for, robot_status_for
from jims_mower.profile import (
    YardProfile,
    apply_profile_to_scenario,
    keep_in_usable,
    load_yard_profile,
)
from jims_mower.scenarios import load_source


def _tiny_keep_in() -> list[list[float]]:
    return [[0.8, 0.8], [4.8, 0.8], [4.8, 3.8], [0.8, 3.8]]


def _scribble_keep_in() -> list[list[float]]:
    return [[35.0 + 0.04 * i, 29.0 + 0.03 * ((-1) ** i)] for i in range(97)]


def _acre_session(tmp_path: Path, *, first_run: bool = True) -> LiveSession:
    return LiveSession(
        config="acre_yard_demo",
        fast=False,
        speed="max",
        steps=80,
        seed=3,
        cameras=4,
        out_dir=tmp_path / "first-run-acre",
        cam_stride=80,
        map_stride=8,
        first_run=first_run,
        yard_path=tmp_path / "profile.json",
    )


def _session(tmp_path: Path, *, first_run: bool = True) -> LiveSession:
    return LiveSession(
        config="mission_tiny",
        fast=True,
        speed="max",
        steps=80,
        seed=2,
        cameras=4,
        out_dir=tmp_path / "first-run",
        cam_stride=80,
        map_stride=4,
        first_run=first_run,
        yard_path=tmp_path / "profile.json",
    )


def _json(host: str, port: int, method: str, path: str, body=None, timeout: float = 6.0):
    conn = HTTPConnection(host, port, timeout=timeout)
    payload = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    conn.request(method, path, body=payload, headers=headers)
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    data = json.loads(raw.decode("utf-8")) if raw else {}
    return resp.status, data


def test_apply_profile_keeps_acre_world() -> None:
    cfg, scenario = load_source("acre_yard_demo")
    width = float(cfg.world.width_m)
    keep_in = [(4.0, 4.0), (30.0, 4.0), (30.0, 24.0), (4.0, 24.0)]
    profile = YardProfile(
        name="taught_acre",
        width_m=12.0,
        height_m=12.0,
        resolution_m=0.20,
        keep_in=keep_in,
        home={"x": 6.0, "y": 6.0, "theta": 0.0},
    )
    apply_profile_to_scenario(scenario, profile, resize_world=False)
    assert float(scenario.config.world.width_m) == width
    assert list(scenario.geofence)[0][0] == 4.0


def test_teach_save_start_skips_calibrate(tmp_path: Path) -> None:
    session = _session(tmp_path)
    backend = LiveBackend(
        session=session,
        reset=True,
        first_run=True,
        yard_path=tmp_path / "profile.json",
        out_dir=tmp_path / "first-run",
    )
    try:
        idle = session.snapshot()
        assert idle["phase"] == "calibrate_boundary"
        assert idle["taught"] is False
        assert idle["needs_teach"] is True

        saved = session.control("save_yard", keep_in=_tiny_keep_in())
        assert saved["ok"] is True
        assert saved["taught"] is True
        assert saved["job_state"] == "idle"
        assert "taught" in (saved["owner_copy"] or "").lower()
        dest = tmp_path / "profile.json"
        assert dest.is_file()
        loaded = load_yard_profile(dest)
        assert len(loaded.keep_in) >= 3
        assert loaded.width_m == session.env.cfg.world.width_m

        session.control("pair")
        started = session.control("start")
        assert started["ok"] is True
        assert started["phase"] == "explore"
        assert started["taught"] is True
        assert len(started.get("keep_in") or []) >= 3
        session.control("pause")
        for _ in range(6):
            session.step_once()
        after = session.snapshot()
        assert after["phase"] in {"explore", "review", "mow"}
        assert after["phase"] != "calibrate_boundary"

        status = backend.status()
        assert status["taught"] is True
        assert status["needs_teach"] is False
        assert status["first_run"] is True
    finally:
        session.control("pause")
        backend.close()


def test_teach_paused_then_start_reaches_explore_not_idle_calibrate(tmp_path: Path) -> None:
    """Regression: teach-complete sets paused; Start must still skip calibrate.

    Old bug: Start treated paused-after-teach like a mid-job resume, left
    phase at calibrate_boundary on a dirty env, then died after 1–2 steps
    while the owner pill read IDLE and copy said Calibrating.
    """
    session = _session(tmp_path)
    session.reset()
    session.job_state = "teach"
    session._taught_env_dirty = True
    for _ in range(6):
        session.step_once()
    session.job_state = "paused"
    saved = session.control("save_yard", keep_in=_tiny_keep_in())
    assert saved["ok"] is True
    assert saved["taught"] is True
    assert saved["job_state"] == "idle"

    started = session.control("start")
    try:
        assert started["ok"] is True
        assert started["phase"] == "explore"
        assert started["job_state"] == "running"
        assert started["done"] is False
        assert started["phase"] != "calibrate_boundary"
        assert session.env is not None
        assert int(session.env._steps) == 0
        copy = owner_copy_for(started["job_state"], started["phase"], taught=True)
        pill = robot_status_for(
            paired=True,
            job_state=started["job_state"],
            done=bool(started["done"]),
        )
        assert "calibrat" not in copy.lower()
        assert pill == "live"
        assert pill != "idle" or started["phase"] != "calibrate_boundary"

        session.control("pause")
        session.policy.clear_owner_hold()
        maps = [float(started.get("map_pct") or 0.0)]
        last = started
        pose0 = (started.get("pose") or {}).copy()
        for _ in range(10):
            last = session.step_once()
            maps.append(float(last.get("map_pct") or 0.0))
            assert last["phase"] in {"explore", "review", "mow", "return_home", "complete"}
            assert last["phase"] != "calibrate_boundary"
            assert not (last["done"] and last["phase"] == "calibrate_boundary")
            copy = owner_copy_for(last["job_state"], last["phase"], taught=True)
            pill = robot_status_for(
                paired=True,
                job_state=last["job_state"],
                done=bool(last["done"]),
            )
            assert not (pill == "idle" and "calibrat" in copy.lower())
        assert last["phase"] in {"explore", "review", "mow"}
        pose1 = last.get("pose") or {}
        moved = abs(float(pose1.get("x", 0.0)) - float(pose0.get("x", 0.0))) + abs(
            float(pose1.get("y", 0.0)) - float(pose0.get("y", 0.0))
        )
        assert max(maps) > maps[0] or moved > 0.05 or last["phase"] in {"review", "mow"}
    finally:
        session.control("pause")
        session.close()


def test_load_saved_profile_then_start(tmp_path: Path) -> None:
    keep = _tiny_keep_in()
    profile = YardProfile(
        name="disk_yard",
        width_m=6.0,
        height_m=5.0,
        resolution_m=0.25,
        keep_in=[(float(x), float(y)) for x, y in keep],
        home={"x": 1.4, "y": 1.4, "theta": 0.0},
    )
    dest = tmp_path / "saved.json"
    from jims_mower.profile import write_yard_profile

    write_yard_profile(dest, profile)
    session = _session(tmp_path, first_run=False)
    session.reset()
    loaded = session.control("load_yard", path=str(dest))
    assert loaded["ok"] is True
    assert loaded["taught"] is True
    started = session.control("start")
    assert started["phase"] == "explore"
    session.control("pause")
    session.close()


def test_snap_home_outside_keep_in_moves_inside(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.reset()
    saved = session.control(
        "save_yard",
        keep_in=_tiny_keep_in(),
        home={"x": -4.0, "y": 20.0, "theta": 0.3},
    )
    assert saved["ok"] is True
    home = session.yard_profile.home
    xs = [p[0] for p in session.yard_profile.keep_in]
    ys = [p[1] for p in session.yard_profile.keep_in]
    assert min(xs) <= float(home["x"]) <= max(xs)
    assert min(ys) <= float(home["y"]) <= max(ys)
    session.close()


def test_demo_start_without_teach_still_calibrates(tmp_path: Path) -> None:
    session = _session(tmp_path, first_run=False)
    session.reset()
    started = session.control("start")
    assert started["phase"] == "calibrate_boundary"
    assert started["taught"] is False
    session.control("pause")
    session.close()


def test_teach_drive_records_trail_then_save(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.reset()
    session.job_state = "teach"
    for _ in range(5):
        session.step_once()
    assert session.teach_policy is not None
    assert len(session.teach_policy.trail) >= 2
    saved = session.control("save_yard")
    assert saved["ok"] is True
    assert saved["taught"] is True
    assert (tmp_path / "profile.json").is_file()
    session.close()


def test_app_first_run_http_contract(tmp_path: Path) -> None:
    session = _session(tmp_path)
    backend = LiveBackend(
        session=session,
        reset=True,
        first_run=True,
        yard_path=tmp_path / "profile.json",
        out_dir=tmp_path / "first-run",
    )
    httpd = make_server(backend, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    try:
        code, status = _json(host, int(port), "GET", "/status")
        assert code == 200
        assert status["first_run"] is True
        assert status["needs_teach"] is True

        code, paired = _json(host, int(port), "POST", "/command", {"cmd": "pair"})
        assert paired["paired"] is True

        code, saved = _json(
            host,
            int(port),
            "POST",
            "/api/live/control",
            {"cmd": "save_yard", "keep_in": _tiny_keep_in()},
        )
        assert code == 200
        assert saved["ok"] is True
        assert saved["schema"] == LIVE_SCHEMA
        assert saved["taught"] is True

        code, started = _json(host, int(port), "POST", "/api/live/control", {"cmd": "start"})
        assert started["ok"] is True
        assert started["phase"] == "explore"
        _json(host, int(port), "POST", "/api/live/control", {"cmd": "pause"})
    finally:
        httpd.shutdown()
        httpd.server_close()
        backend.close()


def test_scribble_keep_in_is_repaired_on_acre(tmp_path: Path) -> None:
    session = _acre_session(tmp_path)
    session.reset()
    saved = session.control("save_yard", keep_in=_scribble_keep_in())
    assert saved["ok"] is True
    assert saved.get("keep_in_repaired") is True
    keep = session.yard_profile.keep_in
    assert keep_in_usable(keep, 70.0, 58.0)
    xs = [p[0] for p in keep]
    ys = [p[1] for p in keep]
    assert max(xs) - min(xs) > 20.0
    assert max(ys) - min(ys) > 20.0
    loaded = load_yard_profile(tmp_path / "profile.json")
    assert keep_in_usable(loaded.keep_in, loaded.width_m, loaded.height_m)
    session.close()


def test_short_teach_drive_save_start_explores_acre(tmp_path: Path) -> None:
    """Teach a few metres, Save, Start — must explore, not instant MAP READY."""
    session = _acre_session(tmp_path)
    session.reset()
    session.job_state = "teach"
    session._taught_env_dirty = True
    for _ in range(8):
        session.step_once()
    saved = session.control("save_yard")
    assert saved["ok"] is True
    assert keep_in_usable(session.yard_profile.keep_in, 70.0, 58.0)

    session._reset_for_taught_job()
    session.job_state = "running"
    if session.policy is not None:
        session.policy.clear_owner_hold()
    started = session.snapshot()
    try:
        assert started["phase"] == "explore"
        assert started["taught"] is True
        assert "hold — safe" not in (started.get("owner_copy") or "").lower()
        maps = [float(started.get("map_pct") or 0.0)]
        last = started
        explore_steps = 0
        for _ in range(24):
            last = session.step_once()
            maps.append(float(last.get("map_pct") or 0.0))
            assert last["phase"] != "calibrate_boundary"
            copy = (last.get("owner_copy") or "").lower()
            assert "hold — safe" not in copy
            if last["phase"] == "explore":
                explore_steps += 1
                assert last.get("fence_unusable") is not True
            if last["phase"] == "review":
                assert last.get("needs_reteach") is not True or last.get("can_start_mow") is False
                assert "re-teach" in copy or "map ready" in copy
                break
        assert explore_steps >= 8 or max(maps) > maps[0] + 0.01
        assert last["phase"] in {"explore", "review", "mow"}
        if last["phase"] == "review":
            assert int(last.get("n_waypoints") or 0) > 0 or last.get("needs_reteach") is True
            assert last["phase"] != "safe"
        summary = (tmp_path / "first-run-acre" / "session_summary.json")
        if summary.is_file():
            card = json.loads(summary.read_text(encoding="utf-8"))
            assert card.get("phase") in {"explore", "review", "mow", "return_home", "complete"}
    finally:
        session.close()


def test_empty_mow_plan_is_reteach_not_hold_safe() -> None:
    copy = owner_copy_for("running", "review", fence_unusable=True)
    assert "re-teach" in copy.lower()
    assert "hold — safe" not in copy.lower()
    assert owner_copy_for("running", "review") == "Map ready — start mow?"
    assert owner_copy_for("running", "safe") == "Hold — safe."
    assert owner_copy_for("estop", "review") == "E-STOP — hold."
    assert "paddle" in owner_copy_for("estop", "review", hw_estop=True).lower()


def test_taught_live_job_completes_to_idle(tmp_path: Path) -> None:
    session = LiveSession(
        config="mission_tiny",
        fast=True,
        speed="max",
        steps=720,
        seed=3,
        cameras=4,
        out_dir=tmp_path / "taught-done",
        cam_stride=80,
        map_stride=8,
        yard_path=tmp_path / "profile.json",
    )
    saved = session.control("save_yard", keep_in=_tiny_keep_in())
    assert saved["ok"] is True
    assert saved["taught"] is True
    session._reset_for_next_job()
    assert session.policy is not None
    assert session.policy.phase.value == "explore"
    session.policy.settings.review_hold_steps = 2
    last = session.run_n(680)
    session.close()
    phases = {row.get("phase") for row in session.poses}
    assert "mow" in phases
    cuts = [float(row.get("actual_coverage_fraction") or 0.0) for row in session.poses]
    assert max(cuts) > 0.03
    assert last["phase"] not in {"safe", "fault"}
    assert last["done"] or last["phase"] in {"return_home", "complete"}
    if last["done"] or last["phase"] == "complete":
        assert last["job_state"] == "idle"
        assert last["taught"] is True
        card = last.get("session_summary") or {}
        assert card.get("schema")
        assert float(card.get("cut_pct") or last.get("cut_pct") or 0.0) > 0.03
        assert "Done" in (last.get("owner_copy") or "")
        assert last.get("keep_in")

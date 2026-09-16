"""First-run teach → YardProfile → live Start (no full-acre mow)."""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from pathlib import Path

from jims_mower.app.live_backend import LiveBackend
from jims_mower.app.server import make_server
from jims_mower.constants import LIVE_SCHEMA
from jims_mower.live import LiveSession
from jims_mower.profile import YardProfile, apply_profile_to_scenario, load_yard_profile
from jims_mower.scenarios import load_source


def _tiny_keep_in() -> list[list[float]]:
    return [[0.8, 0.8], [4.8, 0.8], [4.8, 3.8], [0.8, 3.8]]


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

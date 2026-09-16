"""Phone app owns a live job: control contract over LiveSession."""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from pathlib import Path

from jims_mower.app.cli import build_parser
from jims_mower.app.live_backend import LiveBackend
from jims_mower.app.server import make_server
from jims_mower.constants import APP_LIVE_PORT, APP_STATUS_SCHEMA, LIVE_SCHEMA
from jims_mower.live import LiveSession
from jims_mower.owner_cli import build_parser as owner_parser


def _serve_live(tmp_path: Path):
    session = LiveSession(
        config="mission_tiny",
        fast=True,
        speed="max",
        steps=2000,
        seed=2,
        cameras=4,
        out_dir=tmp_path / "app-live",
        cam_stride=80,
        map_stride=4,
    )
    backend = LiveBackend(session=session, reset=True)
    httpd = make_server(backend, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    return httpd, host, int(port), backend


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


def test_app_live_cli_and_owner_flag() -> None:
    app = build_parser()
    args = app.parse_args(["--live", "--config", "acre_yard_demo", "--speed", "5"])
    assert args.live is True
    assert args.config == "acre_yard_demo"
    assert args.speed == "5"
    help_text = app.format_help()
    assert "--live" in help_text
    assert "acre_yard_demo" in help_text
    owner = owner_parser()
    ohelp = owner.format_help()
    assert "--live" in ohelp
    assert str(APP_LIVE_PORT) in ohelp or "8766" in ohelp


def test_app_live_control_contract(tmp_path: Path) -> None:
    httpd, host, port, backend = _serve_live(tmp_path)
    try:
        code, status = _json(host, port, "GET", "/status")
        assert code == 200
        assert status["schema"] == APP_STATUS_SCHEMA
        assert status["backend"] == "live"
        assert status["live"] is True
        assert status["state"]["job_state"] == "idle"
        assert status["robot"] == "pairing"
        assert status["radio_path"]["chips"]
        labels = {c["label"] for c in status["radio_path"]["chips"]}
        assert labels == {"BT teach", "Wi-Fi map", "LoRa sparse"}
        assert "Yard unknown" in (status["owner_copy"] or "")

        code, paired = _json(host, port, "POST", "/command", {"cmd": "pair"})
        assert code == 200
        assert paired["paired"] is True
        assert paired["robot"] == "idle"

        code, started = _json(host, port, "POST", "/api/live/control", {"cmd": "start", "speed": "5"})
        assert code == 200
        assert started["ok"] is True
        assert started["schema"] == LIVE_SCHEMA
        assert started["job_state"] == "running"
        assert started["speed"] == 5.0

        code, status = _json(host, port, "GET", "/status")
        assert status["robot"] == "live"
        assert status["state"]["job_state"] == "running"

        code, paused = _json(host, port, "POST", "/api/live/control", {"cmd": "pause"})
        assert paused["job_state"] == "paused"

        code, resumed = _json(host, port, "POST", "/command", {"cmd": "start"})
        assert resumed["state"]["job_state"] == "running"

        code, sped = _json(host, port, "POST", "/api/live/control", {"cmd": "speed", "speed": "max"})
        assert sped["speed"] == 0.0

        code, snap = _json(host, port, "GET", "/api/live/snapshot")
        assert snap["schema"] == LIVE_SCHEMA
        assert "owner_copy" in snap
        assert snap["radio_path"]["simulated"] is True

        conn = HTTPConnection(host, port, timeout=6.0)
        conn.request("GET", "/api/live?n=1")
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8")
        conn.close()
        assert resp.status == 200
        assert LIVE_SCHEMA in raw

        code, stuck = _json(host, port, "POST", "/api/live/control", {"cmd": "inject", "kind": "stuck"})
        assert any(f.get("code") == "STUCK" for f in stuck.get("faults") or [])
        assert "Stuck" in (stuck.get("owner_copy") or "")

        code, sos = _json(host, port, "POST", "/api/live/control", {"cmd": "inject", "kind": "sos"})
        assert any(f.get("code") == "FAULT_IMMOBILISED" for f in sos.get("faults") or [])
        assert sos.get("faults")[0].get("retrieve") is True
        assert "immobilised" in (sos.get("owner_copy") or "").lower()

        code, status = _json(host, port, "GET", "/status")
        assert status["robot"] == "fault"

        code, estop = _json(host, port, "POST", "/api/live/control", {"cmd": "estop"})
        assert estop["job_state"] == "estop"
        assert estop["estop"] is True

        conn = HTTPConnection(host, port, timeout=6.0)
        conn.request("GET", "/")
        resp = conn.getresponse()
        html = resp.read().decode("utf-8")
        conn.close()
        assert resp.status == 200
        assert "Jim's Mower" in html
        assert "/static/app.js" in html
        assert "robot-pill" in html

        conn = HTTPConnection(host, port, timeout=6.0)
        conn.request("GET", "/static/app.js")
        js = conn.getresponse().read().decode("utf-8")
        conn.close()
        assert "/api/live/control" in js
        assert "Start job" in js
        assert "Start mow" in js
        assert "Inject SOS" in js

        conn = HTTPConnection(host, port, timeout=6.0)
        conn.request("GET", "/viewer")
        viewer = conn.getresponse().read().decode("utf-8")
        conn.close()
        assert "World Viewer" in viewer
        assert "/viewer/app.js" in viewer

        code, bad = _json(host, port, "POST", "/api/live/control", {"cmd": "dance"})
        assert bad["ok"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()
        backend.close()

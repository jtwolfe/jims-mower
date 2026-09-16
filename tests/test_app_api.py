"""Owner-app HTTP API smoke (memory backend)."""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from pathlib import Path

import pytest

from jims_mower.app.backend import MemoryBackend, make_backend
from jims_mower.app.cli import build_parser
from jims_mower.app.server import make_server
from jims_mower.constants import APP_STATUS_SCHEMA, MESH_SCHEMA, YARD_PROFILE_SCHEMA
from jims_mower.yard_profile import default_yard_profile


def _serve(backend=None):
    backend = backend or MemoryBackend(default_yard_profile())
    httpd = make_server(backend, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    return httpd, host, int(port), backend


def _json(host: str, port: int, method: str, path: str, body=None, timeout: float = 4.0):
    conn = HTTPConnection(host, port, timeout=timeout)
    payload = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    conn.request(method, path, body=payload, headers=headers)
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    data = json.loads(raw.decode("utf-8")) if raw else {}
    return resp.status, data


def test_status_yard_maps_and_commands() -> None:
    httpd, host, port, backend = _serve()
    try:
        code, status = _json(host, port, "GET", "/status")
        assert code == 200
        assert status["schema"] == APP_STATUS_SCHEMA
        assert "pose" in status and "battery" in status
        assert "state" in status and "radio" in status
        assert status["state"]["mission"] == "idle"
        assert status["radio"]["link"] == "lora"

        code, yard = _json(host, port, "GET", "/yard")
        assert code == 200
        assert yard["schema"] == YARD_PROFILE_SCHEMA
        assert len(yard["keep_in"]) >= 3

        yard["home"] = {"x": 3.5, "y": 2.5, "theta": 0.2}
        code, saved = _json(host, port, "PUT", "/yard", yard)
        assert code == 200
        assert saved["home"]["x"] == pytest.approx(3.5)

        code, after = _json(host, port, "POST", "/command", {"cmd": "start"})
        assert code == 200
        assert after["state"]["mission"] == "mowing"
        backend.tick()
        code, moved = _json(host, port, "GET", "/status")
        assert moved["coverage_pct"] >= 0.0

        code, mesh = _json(host, port, "GET", "/map/mesh")
        assert code == 200
        assert mesh["not_slam"] is True
        assert mesh["schema"] == MESH_SCHEMA
        assert mesh["vertex_count"] >= 3
        assert mesh["ux_a_href"] == "/viewer"

        code, cov = _json(host, port, "GET", "/map/coverage")
        assert code == 200
        assert cov["rows"] * cov["cols"] == len(cov["values"])

        code, stopped = _json(host, port, "POST", "/command", {"command": "stop"})
        assert stopped["state"]["mission"] == "idle"

        code, estop = _json(host, port, "POST", "/command", {"cmd": "estop", "reason": "test"})
        assert estop["state"]["mission"] == "estop"
        assert estop["faults"]
        code, bad_ret = _json(host, port, "POST", "/command", {"cmd": "return"})
        assert code == 400
        code, resumed = _json(host, port, "POST", "/command", {"cmd": "start"})
        assert resumed["state"]["mission"] == "mowing"
        code, taught = _json(host, port, "POST", "/command", {"cmd": "teach"})
        assert taught["state"]["mission"] == "teach"
    finally:
        httpd.shutdown()
        httpd.server_close()
        backend.close()


def test_bad_command_and_yard() -> None:
    httpd, host, port, backend = _serve()
    try:
        code, body = _json(host, port, "POST", "/command", {"cmd": "dance"})
        assert code == 400
        assert "unknown command" in body["error"]
        code, body = _json(host, port, "PUT", "/yard", {"schema": "nope"})
        assert code == 400
    finally:
        httpd.shutdown()
        httpd.server_close()
        backend.close()


def test_sse_and_static_shell() -> None:
    httpd, host, port, backend = _serve()
    try:
        conn = HTTPConnection(host, port, timeout=4.0)
        conn.request("GET", "/events?n=2")
        resp = conn.getresponse()
        assert resp.status == 200
        assert "text/event-stream" in resp.getheader("Content-Type", "")
        raw = resp.read().decode("utf-8")
        conn.close()
        assert raw.count("data: ") == 2
        assert APP_STATUS_SCHEMA in raw

        conn = HTTPConnection(host, port, timeout=4.0)
        conn.request("GET", "/")
        resp = conn.getresponse()
        html = resp.read().decode("utf-8")
        conn.close()
        assert resp.status == 200
        assert "Jim's Mower" in html
        assert "/static/app.js" in html

        conn = HTTPConnection(host, port, timeout=4.0)
        conn.request("GET", "/healthz")
        resp = conn.getresponse()
        health = json.loads(resp.read().decode("utf-8"))
        conn.close()
        assert health["ok"] is True

        conn = HTTPConnection(host, port, timeout=4.0)
        conn.request("GET", "/viewer")
        resp = conn.getresponse()
        html = resp.read().decode("utf-8")
        conn.close()
        assert resp.status == 200
        assert "three" in html
        assert "/viewer/app.js" in html

        code, manifest = _json(host, port, "GET", "/api/manifest")
        assert code == 200
        assert manifest["mesh_json"] == "yard.json"
        assert manifest["profile"] == "profile.json"
        assert manifest["vertex_count"] >= 3

        code, profile = _json(host, port, "GET", "/data/profile.json")
        assert code == 200
        assert profile["schema"] == YARD_PROFILE_SCHEMA
        code, mesh_json = _json(host, port, "GET", "/data/yard.json")
        assert mesh_json["schema"] == MESH_SCHEMA
        assert mesh_json["vertex_count"] >= 3
    finally:
        httpd.shutdown()
        httpd.server_close()
        backend.close()


def test_put_yard_persists(tmp_path: Path) -> None:
    dest = tmp_path / "yard.json"
    backend = MemoryBackend(default_yard_profile(), yard_path=dest)
    httpd, host, port, _ = _serve(backend)
    try:
        yard = default_yard_profile().as_dict()
        yard["name"] = "persisted"
        code, _ = _json(host, port, "PUT", "/yard", yard)
        assert code == 200
        saved = json.loads(dest.read_text(encoding="utf-8"))
        assert saved["name"] == "persisted"
    finally:
        httpd.shutdown()
        httpd.server_close()
        backend.close()


def test_cli_parser_and_make_backend() -> None:
    parser = build_parser()
    args = parser.parse_args(["--backend", "memory", "--port", "9001", "--host", "127.0.0.1"])
    assert args.backend == "memory"
    assert args.port == 9001
    live_args = parser.parse_args(["--live", "--fast"])
    assert live_args.live is True
    assert live_args.fast is True
    assert "--live" in parser.format_help()
    backend = make_backend(kind="memory")
    status = backend.status()
    assert status["schema"] == APP_STATUS_SCHEMA
    backend.close()


def test_ux_b_fault_and_radio_overlay() -> None:
    from jims_mower.app.backend import _overlay_radio_sim, _radio_sim_from_info, _ux_b_faults

    info = {
        "radio_enabled": True,
        "radio_channel": "bt",
        "radio_lost": False,
        "fault": {"code": "FAULT_IMMOBILISED", "component": "drive_left", "retrieve": True},
    }
    radio = _overlay_radio_sim({"link": "lora", "ok": True}, _radio_sim_from_info(info))
    assert radio["link"] == "bluetooth"
    assert radio["sim"]["radio_channel"] == "bt"
    faults = _ux_b_faults(info, [])
    assert faults[0]["code"] == "FAULT_IMMOBILISED"
    assert faults[0]["retrieve"] is True
    lost = _overlay_radio_sim({"link": "lora", "ok": True}, _radio_sim_from_info({**info, "radio_lost": True}))
    assert lost["link"] == "none"


def test_make_backend_rejects_unknown() -> None:
    from jims_mower.yard_profile import YardProfileError

    with pytest.raises(YardProfileError):
        make_backend(kind="cloud")

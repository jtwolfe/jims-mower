"""Served owner-phone JS/HTML must expose the manual-phase controls.

PR #43 added backend flags (can_explore / can_mow / can_return,
explore_reason, full_explore, area_legend) but the live phone missed
them when the trio sat below Start job or used easy-to-miss labels.
This contract is Playwright-free: parse static files for button ids
and labels, then check /status + the live frame expose the same flags.
"""

from __future__ import annotations

import json
import re
import threading
from http.client import HTTPConnection
from pathlib import Path

from jims_mower.app.live_backend import LiveBackend
from jims_mower.app.server import STATIC_CACHE_CONTROL, UI_BUILD, make_server, static_dir
from jims_mower.constants import APP_STATUS_SCHEMA, LIVE_SCHEMA
from jims_mower.live import LiveSession
from jims_mower.viewer import static_dir as viewer_static_dir


BUTTON_IDS = (
    "cmd-explore",
    "cmd-mow",
    "cmd-return",
    "full-explore",
    "inj-soc",
    "explore-reason",
    "area-legend",
    "phase-row",
    "cmd-reset",
)

OWNER_LABELS = (
    "Explore",
    "Mow",
    "Return home",
    "Full explore",
    "Low battery",
    "Area types",
    "Grass",
    "Mow this",
    "Path",
    "Sand",
    "Building",
    "Water",
    "Drain",
    "Beds",
    "Keep-out",
    "Blocked / no-go learned",
    "Reset",
    "Tip risk — reversing",
    "Steep grade — contouring",
)

STATUS_FLAGS = (
    "can_explore",
    "can_mow",
    "can_return",
    "explore_reason",
    "full_explore",
    "area_legend",
    "areas_url",
)


def _app_js() -> str:
    return (static_dir() / "app.js").read_text(encoding="utf-8")


def _app_html() -> str:
    return (static_dir() / "index.html").read_text(encoding="utf-8")


def test_static_app_js_has_manual_phase_controls() -> None:
    js = _app_js()
    for btn_id in BUTTON_IDS:
        assert f'id="{btn_id}"' in js or f"#{btn_id}" in js, btn_id
    for label in OWNER_LABELS:
        assert label in js, label
    assert 'id="cmd-explore"' in js
    assert 'id="cmd-mow"' in js
    assert 'id="cmd-return"' in js
    assert ">Return home<" in js
    assert ">Explore<" in js
    assert ">Mow<" in js
    assert ">Full explore<" in js or "Full explore</strong>" in js
    assert ">Low battery<" in js
    assert "low_soc" in js
    assert "can_explore" in js
    assert "can_mow" in js
    assert "can_return" in js
    assert "full_explore" in js
    assert "/api/live/control" in js
    assert re.search(r'id="inj-soc"', js)
    # Not gated behind Start mow / hidden-until-MAP-READY.
    assert "phase-row" in js
    assert "Manual phases" in js
    assert 'id="cmd-reset"' in js
    assert ">Reset<" in js
    assert "Tip risk — reversing" in js
    assert "Steep grade — contouring" in js
    assert "advanced-card" in js
    assert f'window.JIMS_UI_BUILD = "{UI_BUILD}"' in js
    assert "UI build " in js
    assert "stale cached app.js" in js


def test_static_html_and_css_wire_cache_bust_and_legends() -> None:
    html = _app_html()
    assert f"/static/app.js?v={UI_BUILD}" in html
    assert f"/static/app.css?v={UI_BUILD}" in html
    assert f"/static/viewer.js?v={UI_BUILD}" in html
    assert f"UI build {UI_BUILD}" in html
    assert "owner-ui-2" not in html
    css = (static_dir() / "app.css").read_text(encoding="utf-8")
    assert "phase-card" in css
    assert "area-legend" in css
    assert "has-areas" in css
    assert ".ui-build" in css
    viewer = (viewer_static_dir() / "index.html").read_text(encoding="utf-8")
    assert 'id="btn-explore"' in viewer
    assert 'id="btn-return"' in viewer
    assert "Return home" in viewer
    assert 'id="btn-full-explore"' in viewer
    assert "Full explore" in viewer
    assert f"?v={UI_BUILD}" in viewer


def _serve_live(tmp_path: Path):
    session = LiveSession(
        config="mission_tiny",
        fast=True,
        speed="max",
        steps=400,
        seed=2,
        cameras=4,
        out_dir=tmp_path / "ui-contract",
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


def test_status_and_live_frame_expose_owner_ui_flags(tmp_path: Path) -> None:
    httpd, host, port, backend = _serve_live(tmp_path)
    try:
        conn = HTTPConnection(host, port, timeout=6.0)
        conn.request("GET", "/static/app.js")
        js_resp = conn.getresponse()
        js = js_resp.read().decode("utf-8")
        js_cc = js_resp.getheader("Cache-Control") or ""
        conn.close()
        assert ">Return home<" in js
        assert "Manual phases" in js
        assert "Full explore" in js
        assert "Area types" in js
        assert ">Low battery<" in js
        assert "JIMS_UI_BUILD" in js
        assert f'window.JIMS_UI_BUILD = "{UI_BUILD}"' in js
        assert 'id="cmd-explore"' in js
        assert "no-store" in js_cc
        assert "must-revalidate" in js_cc
        assert js_cc == STATIC_CACHE_CONTROL

        conn = HTTPConnection(host, port, timeout=6.0)
        conn.request("GET", "/")
        html_resp = conn.getresponse()
        html = html_resp.read().decode("utf-8")
        html_cc = html_resp.getheader("Cache-Control") or ""
        conn.close()
        assert f"/static/app.js?v={UI_BUILD}" in html
        assert f"UI build {UI_BUILD}" in html
        assert "no-store" in html_cc
        assert "must-revalidate" in html_cc

        conn = HTTPConnection(host, port, timeout=6.0)
        conn.request("GET", "/viewer")
        viewer_resp = conn.getresponse()
        viewer = viewer_resp.read().decode("utf-8")
        conn.close()
        assert f"/viewer/app.js?v={UI_BUILD}" in viewer
        assert f"/viewer/style.css?v={UI_BUILD}" in viewer
        assert 'src="/app.js' not in viewer

        code, unpaired = _json(host, port, "GET", "/status")
        assert code == 200
        assert unpaired["schema"] == APP_STATUS_SCHEMA
        for key in STATUS_FLAGS:
            assert key in unpaired, key

        code, paired = _json(host, port, "POST", "/command", {"cmd": "pair"})
        assert code == 200
        assert paired["paired"] is True
        for key in STATUS_FLAGS:
            assert key in paired, key
        assert paired["can_explore"] is True
        assert isinstance(paired["explore_reason"], dict)
        assert isinstance(paired["area_legend"], list)
        legend_ids = {row.get("id") for row in paired["area_legend"]}
        if legend_ids:
            assert {"grass", "path", "sand", "building", "water", "drain", "beds", "keepout", "fog"} <= legend_ids

        code, started = _json(host, port, "POST", "/api/live/control", {"cmd": "start", "speed": "5"})
        assert code == 200
        assert started["ok"] is True
        assert started["schema"] == LIVE_SCHEMA
        for key in STATUS_FLAGS:
            assert key in started, key
        assert started["can_explore"] is True
        assert started.get("can_return") in {True, False}

        code, snap = _json(host, port, "GET", "/api/live/snapshot")
        assert snap["schema"] == LIVE_SCHEMA
        for key in STATUS_FLAGS:
            assert key in snap, key

        code, status = _json(host, port, "GET", "/status")
        assert status["live"] is True
        for key in STATUS_FLAGS:
            assert key in status, key
        assert status["can_explore"] is True

        code, explored = _json(host, port, "POST", "/api/live/control", {"cmd": "explore"})
        assert explored["ok"] is True
        code, returned = _json(host, port, "POST", "/api/live/control", {"cmd": "return"})
        assert returned["ok"] is True
        code, full = _json(host, port, "POST", "/api/live/control", {"cmd": "full_explore", "enabled": True})
        assert full["ok"] is True
        assert full.get("full_explore") is True
        code, reset = _json(host, port, "POST", "/api/live/control", {"cmd": "reset"})
        assert reset["ok"] is True
        assert reset.get("cmd") == "reset"
        assert reset.get("job_state") == "idle"
        assert reset.get("kept_blockages") is True
        code, low = _json(host, port, "POST", "/api/live/control", {"cmd": "inject", "kind": "low_soc", "soc": 0.12})
        assert low["ok"] is True
    finally:
        httpd.shutdown()
        httpd.server_close()
        backend.close()

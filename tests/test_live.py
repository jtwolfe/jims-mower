"""Live mission session: wall-clock runner + SSE owner protocol."""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from pathlib import Path

import numpy as np

from jims_mower.constants import LIVE_SCHEMA, VIEWER_SCHEMA
from jims_mower.live import (
    LiveSession,
    build_parser,
    coarsen2d,
    owner_copy_for,
    parse_speed,
    resolve_live_config,
)
from jims_mower.planning.observed import ObservedMap, fog_rgba
from jims_mower.viewer import serve_viewer


def test_parse_speed_and_config() -> None:
    assert parse_speed("1") == 1.0
    assert parse_speed("2x") == 2.0
    assert parse_speed("5") == 5.0
    assert parse_speed("max") == 0.0
    assert parse_speed(0) == 0.0
    assert resolve_live_config(None, fast=False) == "acre_yard"
    assert resolve_live_config(None, fast=True) == "mission_tiny"
    assert resolve_live_config("acre_yard", fast=True) == "mission_tiny"
    assert resolve_live_config("golf_rough", fast=False) == "golf_rough"
    help_text = build_parser().format_help()
    assert "--speed" in help_text
    assert "acre_yard" in help_text
    assert "--fast" in help_text
    assert "--phase-budget" in help_text
    assert "--calibrate-stride" in help_text
    assert "acre_yard_demo" in help_text


def test_fog_rgba_unknown_opaque_observed_clear() -> None:
    omap = ObservedMap.empty(3.0, 3.0, 0.25)
    fog = fog_rgba(omap.observed)
    assert fog.shape[2] == 4
    assert int(fog[:, :, 3].min()) >= 200
    omap.stamp_disk(0.6, 0.6, 0.45, explored=True)
    punched = fog_rgba(omap.observed)
    assert int((punched[:, :, 3] == 0).sum()) > 0
    assert int(punched[:, :, 3].max()) >= 200


def test_coarsen2d_caps_acre_side() -> None:
    big = np.zeros((116, 140), dtype=np.uint8)
    small = coarsen2d(big, max_side=96)
    assert max(small.shape) <= 96


def test_live_session_grows_observed_and_streams_phase(tmp_path: Path) -> None:
    session = LiveSession(
        config="mission_tiny",
        fast=True,
        speed="max",
        steps=36,
        seed=3,
        cameras=4,
        out_dir=tmp_path / "live",
        cam_stride=80,
        map_stride=2,
    )
    start = session.reset()
    assert start["schema"] == LIVE_SCHEMA
    assert start["live"] is True
    assert start["phase"] == "calibrate_boundary"
    assert start["job_state"] == "idle"
    assert "unknown" in start["owner_copy"].lower()
    assert start["owner_mode"] == "observed_terrain"
    assert start["map_pct"] < 1.0
    start_obs = int(start["n_observed"])
    assert start_obs > 0
    assert session.fog_png_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert session.observed_png_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    last = session.run_n(36)
    assert last["step"] >= 20
    assert int(last["n_observed"]) >= start_obs
    phases = {row.get("phase") for row in session.poses}
    assert "calibrate_boundary" in phases
    assert last["phase"] in {
        "calibrate_boundary",
        "explore",
        "review",
        "mow",
        "return_home",
        "complete",
    }
    assert (tmp_path / "live" / "viewer.json").is_file()
    manifest = json.loads((tmp_path / "live" / "viewer.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == VIEWER_SCHEMA
    assert manifest["live"] is True
    assert manifest.get("owner_mode") == "observed_terrain"
    assert session.observed_mesh_bytes()
    mesh = json.loads(session.observed_mesh_bytes().decode("utf-8"))
    assert mesh.get("kind") == "observed"
    assert "honesty" in mesh
    session.close()


def _serve_live(tmp_path: Path):
    session = LiveSession(
        config="mission_tiny",
        fast=True,
        speed="max",
        steps=24,
        seed=2,
        cameras=4,
        out_dir=tmp_path / "http",
        cam_stride=80,
        map_stride=2,
    )
    session.reset()
    session.run_n(16)
    httpd = serve_viewer(session.out_dir, host="127.0.0.1", port=0, session=session)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    return httpd, host, int(port), session


def _get(host: str, port: int, path: str, timeout: float = 6.0):
    conn = HTTPConnection(host, port, timeout=timeout)
    conn.request("GET", path)
    resp = conn.getresponse()
    raw = resp.read()
    headers = dict(resp.getheaders())
    status = resp.status
    conn.close()
    return status, headers, raw


def test_live_http_sse_and_assets(tmp_path: Path) -> None:
    httpd, host, port, session = _serve_live(tmp_path)
    try:
        code, _headers, raw = _get(host, port, "/api/manifest")
        assert code == 200
        manifest = json.loads(raw.decode("utf-8"))
        assert manifest["live"] is True
        assert manifest["owner_mode"] == "observed_terrain"

        code, _headers, raw = _get(host, port, "/api/live/snapshot")
        assert code == 200
        snap = json.loads(raw.decode("utf-8"))
        assert snap["schema"] == LIVE_SCHEMA
        assert "phase" in snap and "pose" in snap
        assert "map_pct" in snap
        assert snap["observed_url"].startswith("/api/live/observed.png")
        assert snap["fog_url"].startswith("/api/live/fog.png")
        assert snap["observed_mesh_url"].startswith("/api/live/observed_mesh.json")

        code, headers, raw = _get(host, port, "/api/live?n=2")
        assert code == 200
        ctype = headers.get("Content-Type") or headers.get("content-type") or ""
        assert "text/event-stream" in ctype
        text = raw.decode("utf-8")
        assert text.count("data: ") == 2
        assert LIVE_SCHEMA in text
        frame = json.loads(text.split("data: ", 1)[1].split("\n\n", 1)[0])
        assert frame["phase"] in {
            "calibrate_boundary",
            "explore",
            "review",
            "mow",
            "return_home",
            "complete",
        }

        code, _headers, fog = _get(host, port, "/api/live/fog.png")
        assert code == 200
        assert fog[:8] == b"\x89PNG\r\n\x1a\n"

        code, _headers, observed = _get(host, port, "/api/live/observed.png")
        assert code == 200
        assert observed[:8] == b"\x89PNG\r\n\x1a\n"

        code, _headers, mesh_raw = _get(host, port, "/api/live/observed_mesh.json")
        assert code == 200
        mesh = json.loads(mesh_raw.decode("utf-8"))
        assert mesh.get("kind") == "observed"

        cam = (snap.get("cameras") or ["front"])[0]
        code, _headers, jpeg = _get(host, port, f"/api/live/cam/{cam}")
        assert code == 200
        assert jpeg[:2] == b"\xff\xd8" or jpeg[:8] == b"\x89PNG\r\n\x1a\n"

        code, _headers, html = _get(host, port, "/")
        assert code == 200
        page = html.decode("utf-8")
        assert "World Viewer" in page
        assert "tog-fog" in page
        assert "owner-bar" in page
        assert "btn-job-start" in page
        assert "btn-estop" in page
    finally:
        httpd.shutdown()
        httpd.server_close()
        session.close()


def test_live_cli_prepare_only(tmp_path: Path) -> None:
    from jims_mower.live import main as live_main

    out = tmp_path / "prep"
    live_main(
        [
            "--fast",
            "--speed",
            "max",
            "--steps",
            "12",
            "--prepare-only",
            "--out",
            str(out),
            "--cameras",
            "4",
        ]
    )
    assert (out / "viewer.json").is_file()
    manifest = json.loads((out / "viewer.json").read_text(encoding="utf-8"))
    assert manifest["live"] is True
    assert (out / "maps" / "fog.png").is_file()
    assert (out / "maps" / "observed.png").is_file()
    assert (out / "maps" / "observed_mesh.json").is_file()


def test_owner_copy_reads_like_a_product() -> None:
    assert owner_copy_for("idle", "calibrate_boundary") == "Yard unknown — start a job when ready."
    assert owner_copy_for("running", "calibrate_boundary") == "Calibrating boundary…"
    assert owner_copy_for("running", "explore") == "Exploring unknown yard…"
    assert owner_copy_for("running", "review") == "Map ready — start mow?"
    assert owner_copy_for("running", "mow") == "Mowing…"
    assert owner_copy_for("estop", "mow") == "E-STOP — hold."


def _post(host: str, port: int, path: str, payload: dict, timeout: float = 6.0):
    raw = json.dumps(payload).encode("utf-8")
    conn = HTTPConnection(host, port, timeout=timeout)
    conn.request("POST", path, body=raw, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    body = resp.read()
    status = resp.status
    conn.close()
    return status, body


def test_live_control_http_contract(tmp_path: Path) -> None:
    session = LiveSession(
        config="mission_tiny",
        fast=True,
        speed="max",
        steps=40,
        seed=2,
        cameras=4,
        out_dir=tmp_path / "ctrl",
        cam_stride=80,
        map_stride=4,
    )
    session.reset()
    assert session.snapshot()["job_state"] == "idle"
    httpd = serve_viewer(session.out_dir, host="127.0.0.1", port=0, session=session)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    try:
        code, raw = _post(host, int(port), "/api/live/control", {"cmd": "start", "speed": "5"})
        assert code == 200
        body = json.loads(raw.decode("utf-8"))
        assert body["ok"] is True
        assert body["job_state"] == "running"
        assert body["speed"] == 5.0

        code, raw = _post(host, int(port), "/api/live/control", {"cmd": "pause"})
        assert code == 200
        assert json.loads(raw.decode("utf-8"))["job_state"] == "paused"

        code, raw = _post(host, int(port), "/api/live/control", {"cmd": "resume"})
        assert json.loads(raw.decode("utf-8"))["job_state"] == "running"

        code, raw = _post(host, int(port), "/api/live/control", {"cmd": "speed", "speed": "max"})
        assert json.loads(raw.decode("utf-8"))["speed"] == 0.0

        code, raw = _post(host, int(port), "/api/live/control", {"cmd": "estop"})
        estop = json.loads(raw.decode("utf-8"))
        assert estop["ok"] is True
        assert estop["job_state"] == "estop"
        assert estop["estop"] is True

        code, raw = _post(host, int(port), "/api/live/control", {"cmd": "nope"})
        assert json.loads(raw.decode("utf-8"))["ok"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()
        session.close()


def test_live_tiny_reaches_map_ready(tmp_path: Path) -> None:
    session = LiveSession(
        config="mission_tiny",
        fast=True,
        speed="max",
        steps=380,
        seed=3,
        cameras=4,
        out_dir=tmp_path / "ready",
        cam_stride=80,
        map_stride=8,
    )
    last = session.run_n(380)
    phases = {row.get("phase") for row in session.poses}
    assert "explore" in phases
    assert "review" in phases or last["phase"] in {"review", "mow", "return_home", "complete"}
    assert last["session_summary"]["schema"].startswith("jims_mower.session")
    assert "map_pct" in last["session_summary"]
    assert "cut_pct" in last["session_summary"]
    assert "skips" in last["session_summary"]
    session.close()
    if last["done"] or last["phase"] in {"return_home", "complete"}:
        assert (tmp_path / "ready" / "session_summary.json").is_file()

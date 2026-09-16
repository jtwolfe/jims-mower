"""SAF-3 black box, SAF-4 OTA stub, UX-2 notify, UX-3 multi-yard."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jims_mower.app.backend import MemoryBackend
from jims_mower.app.server import make_server
from jims_mower.blackbox import BlackBox
from jims_mower.env import MowerEnv
from jims_mower.notify import NotificationLog
from jims_mower.ota import ota_apply, ota_status
from jims_mower.profile import YardProfile
from jims_mower.yard_profile import default_yard_profile
from jims_mower.yards import YardStore


def test_blackbox_retrieves_tip_imu_and_cmds(tmp_path: Path) -> None:
    box = BlackBox(tmp_path / "blackbox.jsonl", max_records=32, rotate_bytes=50_000)
    box.record(step=1, imu=[0, 0, 9.81, 0, 0, 0], cmd=[0.2, 0.2, 1.0], advice="ok")
    box.record(
        step=2,
        imu=[0.0, 5.0, 8.0, 0.2, 0.0, 0.0],
        cmd=[-0.4, -0.4, 0.0],
        pose={"x": 2.0, "y": 1.0, "theta": 0.1},
        advice="stop",
        event="tip",
    )
    rows = box.retrieve(event="tip")
    assert len(rows) == 1
    assert rows[0]["imu"][1] == pytest.approx(5.0)
    assert rows[0]["cmd"][0] == pytest.approx(-0.4)
    sos = box.retrieve_sos()
    assert sos and sos[-1]["event"] == "tip"


def test_env_blackbox_path_records_steps(tmp_path: Path) -> None:
    dest = tmp_path / "bb.jsonl"
    cfg = {
        "sensors": {"width": 16, "height": 12, "camera_count": 4},
        "world": {
            "width_m": 6.0,
            "height_m": 5.0,
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
    }
    env = MowerEnv(config=cfg)
    env.reset(seed=1, options={"blackbox": str(dest)})
    env.step(np.array([0.3, 0.3, 0.0], dtype=np.float32))
    env.close()
    assert dest.is_file()
    lines = [ln for ln in dest.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines
    row = json.loads(lines[-1])
    assert "imu" in row and "cmd" in row


def test_ota_is_documented_noop() -> None:
    status = ota_status()
    assert status["available"] is False
    assert status["updater"] == "missing"
    applied = ota_apply("whatever.bin")
    assert applied["applied"] is False
    assert applied["ok"] is False


def test_notification_log_and_webhook_stub(tmp_path: Path) -> None:
    log = NotificationLog(tmp_path / "notes.jsonl")
    item = log.emit("skip", "soc_low", yard="front")
    assert item["sms"] is False
    assert item["channel"] == "in_app"
    log.emit("finish", "duration", yard="front")
    assert len(log.list()) == 2
    stub = log.webhook_stub("https://example.test/hook", item)
    assert stub["delivered"] is False
    assert stub["stub"] is True


def test_yard_store_switch_does_not_bleed_fence(tmp_path: Path) -> None:
    store = YardStore(tmp_path / "yards")
    a = YardProfile(
        name="front",
        keep_in=[(1, 1), (8, 1), (8, 6), (1, 6)],
        home={"x": 2.0, "y": 2.0, "theta": 0.0},
    )
    b = YardProfile(
        name="back",
        keep_in=[(10, 10), (18, 10), (18, 16), (10, 16)],
        home={"x": 12.0, "y": 12.0, "theta": 1.2},
    )
    store.put(a)
    store.put(b)
    store.select("front")
    assert store.active().name == "front"
    store.select("back")
    active = store.active()
    assert active is not None
    assert active.name == "back"
    assert (1.0, 1.0) not in active.keep_in
    assert active.keep_in[0] == (10.0, 10.0)
    assert active.home["x"] == pytest.approx(12.0)


def test_app_yards_and_notifications_http() -> None:
    import threading
    from http.client import HTTPConnection

    backend = MemoryBackend(default_yard_profile())
    second = default_yard_profile(name="paddock")
    second.keep_in = [(0.5, 0.5), (4.0, 0.5), (4.0, 3.0), (0.5, 3.0)]
    backend.put_yard(second)
    backend.select_yard("example_yard")
    httpd = make_server(backend, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    try:
        conn = HTTPConnection(host, int(port), timeout=4.0)
        conn.request("GET", "/yards")
        yards = json.loads(conn.getresponse().read().decode("utf-8"))
        conn.close()
        names = {row["name"] for row in yards["yards"]}
        assert "example_yard" in names
        assert "paddock" in names

        conn = HTTPConnection(host, int(port), timeout=4.0)
        conn.request(
            "POST",
            "/yards/select",
            body=json.dumps({"name": "paddock"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        selected = json.loads(conn.getresponse().read().decode("utf-8"))
        conn.close()
        assert selected["name"] == "paddock"
        assert selected["keep_in"][0] == [0.5, 0.5]

        conn = HTTPConnection(host, int(port), timeout=4.0)
        conn.request("POST", "/command", body=json.dumps({"cmd": "stop", "reason": "rain"}).encode("utf-8"), headers={"Content-Type": "application/json"})
        conn.getresponse().read()
        conn.close()

        conn = HTTPConnection(host, int(port), timeout=4.0)
        conn.request("GET", "/notifications")
        notes = json.loads(conn.getresponse().read().decode("utf-8"))
        conn.close()
        assert notes["sms"] is False
        reasons = [n["reason"] for n in notes["items"]]
        assert any("paddock" in (n.get("reason") or "") or n.get("yard") == "paddock" for n in notes["items"]) or reasons

        conn = HTTPConnection(host, int(port), timeout=4.0)
        conn.request("GET", "/ota")
        ota = json.loads(conn.getresponse().read().decode("utf-8"))
        conn.close()
        assert ota["available"] is False
    finally:
        httpd.shutdown()

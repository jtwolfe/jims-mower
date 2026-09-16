"""UX-5 pairing state machine + UX-4 radio sim honesty. No BlueZ / no metres."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from jims_mower.app.backend import MemoryBackend
from jims_mower.app.live_backend import LiveBackend
from jims_mower.constants import DEFAULT_SCHEDULE_TIMEZONE, GYM_PAIR_PIN, PAIR_STATES
from jims_mower.live import LiveSession
from jims_mower.pairing import PairingError, PairingMachine, parse_pairing
from jims_mower.radio import RadioSim
from jims_mower.schedule import FrozenClock, ScheduleEngine, ScheduleGates, ScheduleSpec
from jims_mower.yard_profile import YardProfileError, default_yard_profile


def test_pairing_states_and_pin() -> None:
    machine = PairingMachine(require_pair=True, pin=GYM_PAIR_PIN)
    assert machine.state == "unpaired"
    assert machine.can_start is False
    assert machine.refuse_start() == "not_paired"
    failed = machine.request_pair("0000")
    assert failed.ok is False
    assert failed.state == "failed"
    assert machine.state == "failed"
    ok = machine.request_pair(GYM_PAIR_PIN)
    assert ok.ok is True
    assert machine.state == "paired"
    assert machine.can_start is True
    machine.unpair()
    assert machine.state == "unpaired"
    machine.request_pair()
    machine.lose(reason="radio_lost")
    assert machine.state == "lost"
    assert machine.can_start is False
    blob = machine.as_info()
    assert blob["on_lost"] == "hold_safe"
    assert blob["simulated"] is True
    assert set(PAIR_STATES) >= {machine.state}


def test_parse_pairing_bool_legacy() -> None:
    assert parse_pairing(True)["state"] == "paired"
    assert parse_pairing(False)["state"] == "unpaired"
    with pytest.raises(PairingError):
        parse_pairing({"state": "bonded"})


def test_memory_start_blocked_until_pair() -> None:
    backend = MemoryBackend(default_yard_profile(), require_pair=True)
    assert backend.status()["pairing"]["state"] == "unpaired"
    assert backend.status()["radio"]["rf_claim"] is None
    with pytest.raises(YardProfileError, match="not paired"):
        backend.command("start")
    paired = backend.command("pair", pin=GYM_PAIR_PIN)
    assert paired["paired"] is True
    assert paired["pairing"]["state"] == "paired"
    started = backend.command("start")
    assert started["state"]["mission"] == "mowing"
    backend.pairing.lose(reason="radio_lost")
    backend._hold_safe_unlocked("radio_lost")
    held = backend.status()
    assert held["state"]["mission"] == "idle"
    assert held["pairing"]["state"] == "lost"
    assert held["state"]["machine"] != "estop"
    with pytest.raises(YardProfileError, match="not paired"):
        backend.command("start")


def test_memory_unpair_holds_safe() -> None:
    backend = MemoryBackend(default_yard_profile(), require_pair=True)
    backend.command("pair")
    backend.command("start")
    after = backend.command("unpair")
    assert after["pairing"]["state"] == "unpaired"
    assert after["state"]["mission"] == "idle"
    assert after["state"]["machine"] != "estop"


def test_radio_far_fence_no_metre_claim() -> None:
    sim = RadioSim(enabled=True)
    sim.lose_link("wifi")
    sim.lose_link("bt")
    pause = sim.route_command("pause", paired=True)
    assert pause.ok is True
    assert pause.channel == "lora"
    start = sim.route_command("start", paired=True)
    assert start.ok is True
    assert start.channel == "lora"
    refused = sim.route_command("start", paired=False)
    assert refused.ok is False
    assert refused.reason == "not_paired"
    resume = sim.route_command("resume", paired=True)
    assert resume.ok is False
    status = sim.owner_status()
    assert status["rf_claim"] is None
    assert status["transport"] == "lora"
    info = sim.as_info()
    assert info["rf_claim"] is None
    assert "radio_distance_m" not in info
    dumped = str(status)
    assert "range_m" not in dumped
    assert "rssi" not in dumped.lower()


def test_live_require_pair_start_and_radio_lost(tmp_path) -> None:
    session = LiveSession(
        config="mission_tiny",
        fast=True,
        speed="max",
        steps=80,
        seed=2,
        cameras=4,
        out_dir=tmp_path / "pair-live",
        require_pair=True,
    )
    try:
        session.reset()
        blocked = session.control("start")
        assert blocked["ok"] is False
        assert "not paired" in (blocked.get("error") or "")
        assert session.pairing.state == "unpaired"
        paired = session.control("pair", pin=GYM_PAIR_PIN)
        assert paired["ok"] is True
        assert paired["paired"] is True
        started = session.control("start")
        assert started["ok"] is True
        assert started["job_state"] == "running"
        lost = session.control("inject", kind="radio_lost")
        assert session.pairing.state == "lost"
        assert lost["job_state"] == "hold"
        assert lost["radio_path"]["rf_claim"] is None
        again = session.control("start")
        assert again["ok"] is False
    finally:
        session.close()


def test_live_backend_owner_defaults_require_pair(tmp_path) -> None:
    session = LiveSession(
        config="mission_tiny",
        fast=True,
        speed="max",
        steps=40,
        seed=1,
        cameras=4,
        out_dir=tmp_path / "pair-backend",
    )
    backend = LiveBackend(session=session, reset=True, out_dir=tmp_path / "pair-backend")
    try:
        status = backend.status()
        assert status["require_pair"] is True
        assert status["paired"] is False
        assert status["radio"]["rf_claim"] is None
        refused = backend.command("start")
        assert refused["paired"] is False
        assert refused["state"]["job_state"] != "running"
        backend.command("pair", pin=GYM_PAIR_PIN)
        started = backend.command("start")
        assert started["paired"] is True
        assert started["state"]["job_state"] == "running"
    finally:
        backend.close()


def test_brisbane_0900_is_not_utc_0900() -> None:
    assert DEFAULT_SCHEDULE_TIMEZONE == "Australia/Brisbane"
    spec = ScheduleSpec(
        enabled=True,
        days=("mon",),
        start_local="09:00",
        duration_min=60,
        timezone="Australia/Brisbane",
        min_soc=0.25,
        skip_rain=True,
        arm_window_min=15,
    )
    # 2026-09-13 23:00 UTC = Monday 09:00 Australia/Brisbane (UTC+10, no DST).
    brisbane = ScheduleEngine(
        spec, clock=FrozenClock(datetime(2026, 9, 13, 23, 0, tzinfo=timezone.utc))
    )
    assert brisbane.evaluate(ScheduleGates(soc=0.9)).action == "arm"
    utc = ScheduleEngine(
        spec, clock=FrozenClock(datetime(2026, 9, 14, 9, 0, tzinfo=timezone.utc))
    )
    assert utc.evaluate(ScheduleGates(soc=0.9)).action != "arm"
    assert ScheduleSpec().timezone == "Australia/Brisbane"

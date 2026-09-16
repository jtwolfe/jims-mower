"""Weekly schedule engine: arm, skip gates, duration stop. Not a cloud calendar."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from jims_mower.app.backend import MemoryBackend
from jims_mower.constants import SCHEDULE_SCHEMA, YARD_PROFILE_SCHEMA
from jims_mower.profile import ProfileError, parse_yard_profile
from jims_mower.pack import GYM_STUB_CAPACITY_WH
from jims_mower.schedule import (
    FrozenClock,
    ScheduleEngine,
    ScheduleGates,
    ScheduleHook,
    ScheduleSpec,
    next_window_start,
)
from jims_mower.yard_profile import default_yard_profile, validate_yard_profile


def _monday_0900() -> datetime:
    return datetime(2026, 9, 14, 9, 0, tzinfo=timezone.utc)


def _spec(**kwargs) -> ScheduleSpec:
    payload = dict(
        enabled=True,
        days=("mon", "wed", "fri"),
        start_local="09:00",
        duration_min=60,
        timezone="UTC",
        min_soc=0.25,
        skip_rain=True,
        arm_window_min=15,
    )
    payload.update(kwargs)
    return ScheduleSpec(**payload)


def test_next_window_same_morning() -> None:
    spec = _spec()
    now = _monday_0900()
    nxt = next_window_start(now, spec)
    assert nxt is not None
    assert nxt == now


def test_arm_in_window() -> None:
    engine = ScheduleEngine(_spec(), clock=FrozenClock(_monday_0900()))
    decision = engine.evaluate(ScheduleGates(soc=0.8))
    assert decision.action == "arm"
    assert decision.window_id == "2026-09-14T09:00"
    again = engine.evaluate(ScheduleGates(soc=0.8))
    assert again.action == "idle"


def test_soc_gate_exposes_configured_capacity() -> None:
    hook = ScheduleHook()
    hook.engine = ScheduleEngine(_spec(), clock=FrozenClock(_monday_0900()))
    hook.poll(ScheduleGates(soc=0.10, capacity_wh=200.0, pack_measured=True))
    status = hook.status_dict()
    assert status["reason"] == "soc_low"
    assert status["capacity_wh"] == pytest.approx(200.0)
    assert status["pack_measured"] is True
    assert status["remaining_wh"] == pytest.approx(20.0)
    assert status["soc"] == pytest.approx(0.10)


def test_memory_backend_battery_is_unmeasured_stub() -> None:
    backend = MemoryBackend(default_yard_profile())
    battery = backend.status()["battery"]
    assert battery["measured"] is False
    assert battery["capacity_wh"] == pytest.approx(GYM_STUB_CAPACITY_WH)
    assert battery["acre_runtime_h"] is None
    assert battery["not_a_power_trace"] is True


def test_skip_soc_rain_fault_estop() -> None:
    clock = FrozenClock(_monday_0900())
    for gates, reason in (
        (ScheduleGates(soc=0.10), "soc_low"),
        (ScheduleGates(soc=0.9, rain=True), "rain"),
        (ScheduleGates(soc=0.9, fault=True), "fault"),
        (ScheduleGates(soc=0.9, estop=True), "estop"),
    ):
        engine = ScheduleEngine(_spec(), clock=clock)
        decision = engine.evaluate(gates)
        assert decision.action == "skip", reason
        assert decision.reason == reason


def test_already_running_does_not_consume_window() -> None:
    engine = ScheduleEngine(_spec(), clock=FrozenClock(_monday_0900()))
    held = engine.evaluate(ScheduleGates(soc=0.9, running=True))
    assert held.action == "hold"
    assert held.reason == "already_running"
    later = engine.evaluate(ScheduleGates(soc=0.9, running=False))
    assert later.action == "arm"


def test_disabled_and_no_days() -> None:
    idle = ScheduleEngine(_spec(enabled=False), clock=FrozenClock(_monday_0900())).evaluate()
    assert idle.action == "idle"
    assert idle.reason == "disabled"
    empty = ScheduleEngine(_spec(days=()), clock=FrozenClock(_monday_0900())).evaluate()
    assert empty.reason == "no_days"


def test_duration_stop() -> None:
    clock = FrozenClock(_monday_0900())
    engine = ScheduleEngine(_spec(duration_min=30), clock=clock)
    assert engine.evaluate(ScheduleGates(soc=0.9)).action == "arm"
    clock.advance(minutes=31)
    stop = engine.evaluate(ScheduleGates(soc=0.9, running=True))
    assert stop.action == "stop"
    assert stop.reason == "duration"


def test_timezone_utc_vs_offset() -> None:
    clock = FrozenClock(_monday_0900())
    utc = ScheduleEngine(_spec(timezone="UTC"), clock=clock)
    assert utc.peek().next_run is not None
    with pytest.raises(ValueError, match="IANA"):
        ScheduleEngine(_spec(timezone="Not/AZone")).now()


def test_memory_backend_arms_and_skips() -> None:
    backend = MemoryBackend(default_yard_profile())
    clock = FrozenClock(_monday_0900())
    backend.set_clock(clock)
    yard = backend.get_yard()
    yard["schedule"] = _spec().as_dict()
    backend.put_yard(parse_yard_profile(yard))
    status = backend.status()
    assert status["schedule"]["schema"] == SCHEDULE_SCHEMA
    assert status["state"]["mission"] == "mowing"
    assert status["schedule"]["action"] == "arm"

    backend.command("stop")
    backend.rain = True
    clock.advance(days=2)  # Wednesday 09:00
    clock.set(datetime(2026, 9, 16, 9, 0, tzinfo=timezone.utc))
    skipped = backend.status()
    assert skipped["state"]["mission"] == "idle"
    assert skipped["schedule"]["action"] == "skip"
    assert skipped["schedule"]["reason"] == "rain"


def test_memory_backend_soc_and_estop_skip() -> None:
    backend = MemoryBackend(default_yard_profile())
    backend.set_clock(FrozenClock(_monday_0900()))
    yard = backend.get_yard()
    yard["schedule"] = _spec().as_dict()
    backend.put_yard(parse_yard_profile(yard))
    backend.soc = 0.10
    low = backend.status()
    assert low["schedule"]["reason"] == "soc_low"
    assert low["state"]["mission"] == "idle"

    backend2 = MemoryBackend(default_yard_profile())
    backend2.set_clock(FrozenClock(_monday_0900()))
    backend2.put_yard(parse_yard_profile(yard))
    halted = backend2.command("estop", reason="test")
    assert halted["state"]["mission"] == "estop"
    assert halted["schedule"]["reason"] == "estop"


def test_profile_schedule_fields_roundtrip() -> None:
    raw = default_yard_profile().as_dict()
    raw["schedule"] = _spec().as_dict()
    loaded = parse_yard_profile(raw)
    assert loaded.schedule["timezone"] == "UTC"
    assert loaded.schedule["min_soc"] == pytest.approx(0.25)
    assert loaded.schedule["skip_rain"] is True
    assert "stub" not in loaded.schedule["note"]
    with pytest.raises(ProfileError, match="IANA"):
        validate_yard_profile(dict(raw, schedule={**raw["schedule"], "timezone": "Mars/Phobos"}))
    with pytest.raises(ProfileError, match="min_soc"):
        validate_yard_profile(dict(raw, schedule={**raw["schedule"], "min_soc": 1.5}))
    old = dict(raw["schedule"])
    old.pop("timezone")
    old.pop("min_soc")
    old["note"] = "stub — not a scheduler"
    migrated = parse_yard_profile(dict(raw, schema=YARD_PROFILE_SCHEMA, schedule=old))
    assert migrated.schedule["timezone"] == "Australia/Brisbane"
    assert "stub" not in migrated.schedule["note"]

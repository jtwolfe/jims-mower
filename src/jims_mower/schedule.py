"""Weekly yard schedule that actually arms / stops owner jobs.

``YardProfile.schedule`` used to be a stored stub. This module evaluates
the same document against a clock (wall or frozen/sim) and the live
gates: SOC, rain flag, latched fault / ESTOP. It does **not** claim a
cloud calendar, push notifications, or a weather service — rain is an
explicit flag from the env / owner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Any, Callable, Optional, Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from jims_mower.constants import (
    DEFAULT_SCHEDULE_TIMEZONE,
    SCHEDULE_DAYS,
    SCHEDULE_SCHEMA,
    SCHEDULE_SKIP_REASONS,
)

DEFAULT_NOTE = "weekly window — engine arms start when due"
DEFAULT_TIMEZONE = DEFAULT_SCHEDULE_TIMEZONE
_OLD_STUB_NOTE = "stub — not a scheduler"
_DAY_INDEX = {name: i for i, name in enumerate(SCHEDULE_DAYS)}


class Clock(Protocol):
    """Aware ``now`` in the schedule timezone."""

    def now(self, tz: tzinfo) -> datetime: ...


class WallClock:
    """Host wall clock. Used by the live / app backends."""

    def now(self, tz: tzinfo) -> datetime:
        return datetime.now(tz)


class FrozenClock:
    """Injectable clock for tests and sim-time advances."""

    def __init__(self, instant: datetime) -> None:
        if instant.tzinfo is None:
            instant = instant.replace(tzinfo=timezone.utc)
        self.instant = instant

    def now(self, tz: tzinfo) -> datetime:
        return self.instant.astimezone(tz)

    def set(self, instant: datetime) -> None:
        if instant.tzinfo is None:
            instant = instant.replace(tzinfo=self.instant.tzinfo or timezone.utc)
        self.instant = instant

    def advance(self, **kwargs: Any) -> datetime:
        self.instant = self.instant + timedelta(**kwargs)
        return self.instant


def resolve_timezone(name: str) -> tzinfo:
    key = str(name or DEFAULT_TIMEZONE).strip() or DEFAULT_TIMEZONE
    if key.lower() == "local":
        found = datetime.now().astimezone().tzinfo
        return found if found is not None else timezone.utc
    try:
        return ZoneInfo(key)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(
            f"schedule.timezone must be 'local' or an IANA name "
            f"(default {DEFAULT_TIMEZONE}); got {name!r}"
        ) from exc


def parse_hhmm(raw: str) -> tuple[int, int]:
    text = str(raw or "").strip()
    hour_s, minute_s = text.split(":", 1)
    return int(hour_s), int(minute_s)


@dataclass(frozen=True)
class ScheduleSpec:
    """Validated weekly window stored on ``YardProfile.schedule``."""

    enabled: bool = False
    days: tuple[str, ...] = ()
    start_local: str = "09:00"
    duration_min: int = 60
    timezone: str = DEFAULT_TIMEZONE
    min_soc: float = 0.25
    skip_rain: bool = True
    arm_window_min: int = 15
    note: str = DEFAULT_NOTE

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "days": list(self.days),
            "start_local": str(self.start_local),
            "duration_min": int(self.duration_min),
            "timezone": str(self.timezone),
            "min_soc": float(self.min_soc),
            "skip_rain": bool(self.skip_rain),
            "arm_window_min": int(self.arm_window_min),
            "note": str(self.note or DEFAULT_NOTE),
        }

    def tzinfo(self) -> tzinfo:
        return resolve_timezone(self.timezone)


# Backward-compatible name used by older imports.
ScheduleStub = ScheduleSpec


@dataclass(frozen=True)
class ScheduleGates:
    """Live conditions the engine must pass before arming a job."""

    soc: float = 1.0
    rain: bool = False
    fault: bool = False
    estop: bool = False
    running: bool = False
    # Configured pack Wh (OrinBudget / runtime.battery). SOC gate is
    # still a fraction of this capacity. Not an acre-runtime claim.
    capacity_wh: Optional[float] = None
    pack_measured: bool = False


@dataclass
class ScheduleDecision:
    """One evaluation of the weekly window."""

    action: str = "idle"  # idle | arm | skip | stop | hold
    reason: Optional[str] = None
    next_run: Optional[datetime] = None
    window_id: Optional[str] = None
    consumed: bool = False

    def as_status(self, spec: ScheduleSpec) -> dict[str, Any]:
        nxt = self.next_run
        return {
            "schema": SCHEDULE_SCHEMA,
            "enabled": bool(spec.enabled),
            "days": list(spec.days),
            "start_local": spec.start_local,
            "duration_min": int(spec.duration_min),
            "timezone": spec.timezone,
            "min_soc": float(spec.min_soc),
            "skip_rain": bool(spec.skip_rain),
            "arm_window_min": int(spec.arm_window_min),
            "action": self.action,
            "reason": self.reason,
            "skip_reason": self.reason if self.action in {"skip", "hold", "idle"} else None,
            "next_run": nxt.isoformat() if nxt is not None else None,
            "next_run_local": _format_local(nxt) if nxt is not None else None,
            "window_id": self.window_id,
            "armed": self.action == "arm",
            "note": spec.note,
            "not_a_cloud_calendar": True,
        }


def spec_from_mapping(raw: Optional[dict[str, Any]]) -> ScheduleSpec:
    if raw is None:
        return ScheduleSpec()
    if not isinstance(raw, dict):
        raise ValueError("schedule must be a mapping")
    return ScheduleSpec(
        enabled=bool(raw.get("enabled", False)),
        days=tuple(str(d).strip().lower()[:3] for d in (raw.get("days") or [])),
        start_local=str(raw.get("start_local") or "09:00"),
        duration_min=int(raw.get("duration_min", 60)),
        timezone=str(raw.get("timezone") or DEFAULT_TIMEZONE),
        min_soc=float(raw.get("min_soc", 0.25)),
        skip_rain=bool(raw.get("skip_rain", True)),
        arm_window_min=int(raw.get("arm_window_min", 15)),
        note=str(raw.get("note") or DEFAULT_NOTE),
    )


def window_id_for(start: datetime) -> str:
    return f"{start.date().isoformat()}T{start.strftime('%H:%M')}"


def next_window_start(now: datetime, spec: ScheduleSpec) -> Optional[datetime]:
    """Next start that is still armable (now inside the window, or later)."""
    if not spec.enabled or not spec.days:
        return None
    hour, minute = parse_hhmm(spec.start_local)
    arm = timedelta(minutes=max(int(spec.arm_window_min), 0))
    for offset in range(0, 8):
        day = (now + timedelta(days=offset)).date()
        name = SCHEDULE_DAYS[day.weekday()]
        if name not in spec.days:
            continue
        start = datetime(day.year, day.month, day.day, hour, minute, tzinfo=now.tzinfo)
        if now > start + arm:
            continue
        return start
    return None


def current_window_start(now: datetime, spec: ScheduleSpec) -> Optional[datetime]:
    start = next_window_start(now, spec)
    if start is None:
        return None
    arm = timedelta(minutes=max(int(spec.arm_window_min), 0))
    if start <= now <= start + arm:
        return start
    return None


def _format_local(instant: datetime) -> str:
    return instant.strftime("%a %Y-%m-%d %H:%M")


def _gate_reason(spec: ScheduleSpec, gates: ScheduleGates) -> Optional[str]:
    if gates.estop:
        return "estop"
    if gates.fault:
        return "fault"
    if spec.skip_rain and gates.rain:
        return "rain"
    if float(gates.soc) < float(spec.min_soc):
        return "soc_low"
    if gates.running:
        return "already_running"
    return None


class ScheduleEngine:
    """Evaluate a ``YardProfile.schedule`` against a clock + gates.

    Each weekly window is consumed at most once (arm or skip). A skip
    during the arm window does not retry until the next scheduled day.
    ``already_running`` does not consume the window, so an owner stop
    inside the window can still let the engine arm.
    """

    def __init__(
        self,
        spec: Optional[ScheduleSpec] = None,
        *,
        clock: Optional[Clock] = None,
    ) -> None:
        self.spec = spec or ScheduleSpec()
        self.clock: Clock = clock or WallClock()
        self._consumed: set[str] = set()
        self._last_skip: dict[str, str] = {}
        self._armed_window: Optional[str] = None
        self._armed_at: Optional[datetime] = None
        self.last = ScheduleDecision()

    @classmethod
    def from_mapping(
        cls,
        raw: Optional[dict[str, Any]],
        *,
        clock: Optional[Clock] = None,
    ) -> "ScheduleEngine":
        return cls(spec_from_mapping(raw), clock=clock)

    def replace_spec(self, raw: Optional[dict[str, Any]]) -> None:
        self.spec = spec_from_mapping(raw)

    def now(self) -> datetime:
        return self.clock.now(self.spec.tzinfo())

    def peek(self, gates: Optional[ScheduleGates] = None) -> ScheduleDecision:
        return self._evaluate(gates or ScheduleGates(), consume=False)

    def evaluate(self, gates: Optional[ScheduleGates] = None) -> ScheduleDecision:
        decision = self._evaluate(gates or ScheduleGates(), consume=True)
        self.last = decision
        return decision

    def mark_stopped(self) -> None:
        self._armed_window = None
        self._armed_at = None

    def _evaluate(self, gates: ScheduleGates, *, consume: bool) -> ScheduleDecision:
        spec = self.spec
        now = self.now()
        nxt = next_window_start(now, spec)
        if not spec.enabled:
            stop = self._maybe_stop(now, gates, consume=consume)
            if stop is not None:
                return stop
            return ScheduleDecision(action="idle", reason="disabled", next_run=nxt)
        if not spec.days:
            return ScheduleDecision(action="idle", reason="no_days", next_run=None)

        stop = self._maybe_stop(now, gates, consume=consume)
        if stop is not None:
            return stop

        window = current_window_start(now, spec)
        if window is None:
            return ScheduleDecision(action="idle", reason=None, next_run=nxt)

        wid = window_id_for(window)
        if wid in self._consumed:
            skipped = self._last_skip.get(wid)
            return ScheduleDecision(
                action="skip" if skipped else "idle",
                reason=skipped,
                next_run=next_window_start(window + timedelta(minutes=spec.arm_window_min) + timedelta(seconds=1), spec),
                window_id=wid,
            )

        blocked = _gate_reason(spec, gates)
        if blocked == "already_running":
            return ScheduleDecision(
                action="hold",
                reason="already_running",
                next_run=window,
                window_id=wid,
            )
        if blocked is not None:
            if consume:
                self._consumed.add(wid)
                self._last_skip[wid] = blocked
            return ScheduleDecision(
                action="skip",
                reason=blocked,
                next_run=next_window_start(window + timedelta(minutes=spec.arm_window_min) + timedelta(seconds=1), spec),
                window_id=wid,
                consumed=consume,
            )

        if consume:
            self._consumed.add(wid)
            self._armed_window = wid
            self._armed_at = now
        return ScheduleDecision(
            action="arm",
            reason=None,
            next_run=window,
            window_id=wid,
            consumed=consume,
        )

    def _maybe_stop(
        self,
        now: datetime,
        gates: ScheduleGates,
        *,
        consume: bool,
    ) -> Optional[ScheduleDecision]:
        if self._armed_at is None or int(self.spec.duration_min) <= 0:
            if consume and not gates.running:
                self.mark_stopped()
            return None
        deadline = self._armed_at + timedelta(minutes=int(self.spec.duration_min))
        if now < deadline:
            return None
        if not gates.running:
            if consume:
                self.mark_stopped()
            return None
        wid = self._armed_window
        if consume:
            self.mark_stopped()
        nxt = next_window_start(now, self.spec)
        return ScheduleDecision(action="stop", reason="duration", next_run=nxt, window_id=wid)


@dataclass
class ScheduleHook:
    """Per-backend helper: keep one engine, poll, and expose ``/status``."""

    engine: ScheduleEngine = field(default_factory=ScheduleEngine)
    last: ScheduleDecision = field(default_factory=ScheduleDecision)
    started_by_schedule: bool = False
    last_gates: Optional[ScheduleGates] = None

    @classmethod
    def from_profile_schedule(
        cls,
        raw: Optional[dict[str, Any]],
        *,
        clock: Optional[Clock] = None,
    ) -> "ScheduleHook":
        return cls(engine=ScheduleEngine.from_mapping(raw, clock=clock))

    def set_clock(self, clock: Clock) -> None:
        self.engine.clock = clock

    def sync(self, raw: Optional[dict[str, Any]]) -> None:
        self.engine.replace_spec(raw)

    def poll(
        self,
        gates: ScheduleGates,
        *,
        start: Optional[Callable[[], None]] = None,
        stop: Optional[Callable[[], None]] = None,
        spec: Optional[dict[str, Any]] = None,
    ) -> ScheduleDecision:
        if spec is not None:
            self.sync(spec)
        decision = self.engine.evaluate(gates)
        self.last = decision
        self.last_gates = gates
        if decision.action == "arm" and start is not None:
            start()
            self.started_by_schedule = True
        elif decision.action == "stop" and self.started_by_schedule and stop is not None:
            stop()
            self.started_by_schedule = False
        if decision.action == "stop":
            self.started_by_schedule = False
        if not gates.running and decision.action in {"idle", "skip"}:
            self.started_by_schedule = False
        return decision

    def status_dict(self, spec: Optional[ScheduleSpec] = None) -> dict[str, Any]:
        use = spec or self.engine.spec
        blob = self.last.as_status(use)
        if blob.get("next_run") is None:
            nxt = next_window_start(self.engine.now(), use)
            blob["next_run"] = nxt.isoformat() if nxt is not None else None
            blob["next_run_local"] = _format_local(nxt) if nxt is not None else None
        blob["started_by_schedule"] = bool(self.started_by_schedule)
        gates = self.last_gates
        if gates is not None:
            from jims_mower.pack import remaining_wh

            blob["soc"] = float(gates.soc)
            blob["capacity_wh"] = None if gates.capacity_wh is None else float(gates.capacity_wh)
            blob["pack_measured"] = bool(gates.pack_measured)
            blob["remaining_wh"] = remaining_wh(gates.soc, gates.capacity_wh)
        return blob


def gates_from_owner_state(
    *,
    soc: float,
    rain: bool,
    faults: Any,
    mission: str,
    machine: str,
    capacity_wh: Optional[float] = None,
    pack_measured: bool = False,
) -> ScheduleGates:
    blocking = False
    if isinstance(faults, list):
        for item in faults:
            if not isinstance(item, dict):
                continue
            code = str(item.get("code") or "ok").strip()
            if code and code.lower() not in {"ok"}:
                blocking = True
                break
    mission_l = str(mission or "idle").strip().lower()
    machine_l = str(machine or "run").strip().lower()
    running = mission_l in {
        "mowing",
        "returning",
        "teach",
        "review",
        "explore",
        "calibrate_boundary",
        "mapping",
    } or mission_l == "running"
    hw_estop = False
    if isinstance(faults, list):
        hw_estop = any(
            isinstance(item, dict) and str(item.get("code") or "") == "HW_ESTOP"
            for item in faults
        )
    estop = machine_l == "estop" or mission_l == "estop" or hw_estop
    return ScheduleGates(
        soc=float(soc),
        rain=bool(rain),
        fault=blocking and not estop,
        estop=estop,
        running=running and not estop,
        capacity_wh=None if capacity_wh is None else float(capacity_wh),
        pack_measured=bool(pack_measured),
    )


def rain_from_weather(weather: Any) -> bool:
    if isinstance(weather, dict):
        if weather.get("wet") or weather.get("rain"):
            return True
        pack = str(weather.get("pack") or weather.get("lighting") or "").strip().lower()
        return pack == "rain"
    if isinstance(weather, str):
        return weather.strip().lower() in {"rain", "wet"}
    return False


def assert_skip_reason(reason: Optional[str]) -> Optional[str]:
    if reason is None:
        return None
    if reason not in SCHEDULE_SKIP_REASONS and reason != "duration":
        return reason
    return reason

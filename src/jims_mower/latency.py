"""Host-side camera → plan → cmd timing with optional injected delays.

This is a budget harness, not a board benchmark. Numbers are wall-clock
times on the machine that ran the loop, plus any ``sleep`` you asked for.
They are **not** Jetson Orin Nano FPS and must not be quoted as such.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

# Design-against comments for an Orin Nano *class* board. Not measurements.
ORIN_NANO_CLASS_NOTES = (
    "Host timings + injected delays only. No onboard FPS / mAP claims. "
    "Orin Nano class budget to design against: camera+preprocess typically "
    "tens of milliseconds; keep the numpy costmap/planner in-process; "
    "wheel-command publish is small versus cameras. Do not treat this "
    "scorecard as a device benchmark."
)


def inject_delay(seconds: float) -> None:
    delay = float(seconds)
    if delay > 0.0:
        time.sleep(delay)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def summarize_ms(samples_s: list[float]) -> dict[str, float]:
    xs = [1000.0 * float(v) for v in samples_s]
    if not xs:
        return {"n": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "n": len(xs),
        "mean_ms": float(sum(xs) / len(xs)),
        "p50_ms": _percentile(xs, 50.0),
        "p95_ms": _percentile(xs, 95.0),
        "max_ms": float(max(xs)),
    }


@dataclass
class LatencyDelays:
    camera_s: float = 0.0
    plan_s: float = 0.0
    cmd_s: float = 0.0

    @classmethod
    def from_ms(
        cls,
        camera_ms: float = 0.0,
        plan_ms: float = 0.0,
        cmd_ms: float = 0.0,
    ) -> "LatencyDelays":
        return cls(camera_ms / 1000.0, plan_ms / 1000.0, cmd_ms / 1000.0)


@dataclass
class CycleTiming:
    t_start: float
    t_camera: float = 0.0
    t_plan: float = 0.0
    t_cmd: float = 0.0

    @property
    def camera_to_plan_s(self) -> float:
        return max(0.0, self.t_plan - self.t_camera)

    @property
    def plan_to_cmd_s(self) -> float:
        return max(0.0, self.t_cmd - self.t_plan)

    @property
    def camera_to_cmd_s(self) -> float:
        return max(0.0, self.t_cmd - self.t_camera)


@dataclass
class LatencyHarness:
    """Mark camera / plan / cmd instants; optionally sleep injected delays."""

    delays: LatencyDelays = field(default_factory=LatencyDelays)
    cycles: list[CycleTiming] = field(default_factory=list)
    _current: Optional[CycleTiming] = None

    def start_cycle(self) -> None:
        self._current = CycleTiming(t_start=time.perf_counter())

    def after_camera(self) -> None:
        inject_delay(self.delays.camera_s)
        if self._current is not None:
            self._current.t_camera = time.perf_counter()

    def after_plan(self) -> None:
        inject_delay(self.delays.plan_s)
        if self._current is not None:
            self._current.t_plan = time.perf_counter()

    def after_cmd(self) -> None:
        inject_delay(self.delays.cmd_s)
        if self._current is not None:
            self._current.t_cmd = time.perf_counter()
            self.cycles.append(self._current)
            self._current = None

    def stats(self) -> dict[str, Any]:
        return {
            "camera_to_plan": summarize_ms([c.camera_to_plan_s for c in self.cycles]),
            "plan_to_cmd": summarize_ms([c.plan_to_cmd_s for c in self.cycles]),
            "camera_to_cmd": summarize_ms([c.camera_to_cmd_s for c in self.cycles]),
        }

    def scorecard(self, extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": "1",
            "not_a_benchmark": True,
            "fps_claim": None,
            "platform_assumption": "Jetson Orin Nano class",
            "notes": ORIN_NANO_CLASS_NOTES,
            "injected_delay_s": {
                "camera": float(self.delays.camera_s),
                "plan": float(self.delays.plan_s),
                "cmd": float(self.delays.cmd_s),
            },
            "timings_ms": self.stats(),
        }
        if extra:
            payload.update(extra)
        return payload

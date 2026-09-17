"""Grade-aware IMU tip classification.

IMU tip-stop is a *chassis attitude* trip. A climbable hill tilts the
camera and the IMU the same way — that is not a tip. This module:

* reads tilt from accel and/or fused pose
* distinguishes climbable grade from tip-risk
* holds ``stop`` across a few frames so a noise spike does not reverse-loop

Physics still owns true tip-over (``terrain_hazards`` / episode terminate).
Planner lethal cells are only those above the tip-safe margin — see
``docs/TIP_HILL.md``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from jims_mower.constants import GRAVITY_MPS2
from jims_mower.planning.fusion import attitude_from_accel

KIND_OK = "ok"
KIND_GRADE = "grade"
KIND_TIP = "tip"

OWNER_TIP_RISK = "Tip risk — reversing"
OWNER_STEEP_GRADE = "Steep grade — contouring"


@dataclass(frozen=True)
class TiltClass:
    """One-frame (or filtered) tilt decision."""

    advice: str
    kind: str
    roll: float
    pitch: float

    @property
    def owner_copy(self) -> str:
        if self.kind == KIND_TIP:
            return OWNER_TIP_RISK
        if self.kind == KIND_GRADE:
            return OWNER_STEEP_GRADE
        return ""


def read_tilt(
    imu: np.ndarray,
    *,
    pose_pitch: float = 0.0,
    pose_roll: float = 0.0,
) -> tuple[float, float]:
    """Accel tilt *or* fused/true pitch-roll. Worse-case per axis."""
    roll, pitch = float(pose_roll), float(pose_pitch)
    imu_arr = np.asarray(imu, dtype=np.float32).reshape(-1)
    spec = float(np.linalg.norm(imu_arr[:3])) if imu_arr.size >= 3 else GRAVITY_MPS2
    if abs(spec - GRAVITY_MPS2) < 0.75:
        roll_a, pitch_a = attitude_from_accel(imu_arr)
        roll = roll_a if abs(roll_a) >= abs(roll) else roll
        pitch = pitch_a if abs(pitch_a) >= abs(pitch) else pitch
    return float(roll), float(pitch)


def tip_lethal_slope_rad(
    *,
    tip_roll_rad: float,
    tip_pitch_rad: float,
    tip_lethal_frac: float = 0.95,
) -> float:
    """Costmap lethal grade: just under the software tip trip, not max_climb."""
    frac = float(tip_lethal_frac)
    return float(frac) * min(float(tip_roll_rad), float(tip_pitch_rad))


def classify_tilt(
    roll: float,
    pitch: float,
    *,
    tip_roll_rad: float,
    tip_pitch_rad: float,
    slow_frac: float,
    stop_frac: float,
    max_climb_slope_rad: Optional[float] = None,
    tip_lethal_frac: float = 0.95,
) -> TiltClass:
    """Map chassis attitude onto ok / slow / reroute / stop.

    When ``max_climb_slope_rad`` is set (mission / coverage path):

    * attitude inside the climb cap → climbable **grade** (``slow``, not stop)
    * between climb and tip-lethal → **contour** (``reroute``)
    * at/above tip-lethal or the full tip trip → **tip** (``stop``)

    When ``max_climb_slope_rad`` is omitted, keep the legacy fractional
    trips so unit tests and older callers still see stop at ``stop_frac``.
    """
    roll_a = float(roll)
    pitch_a = float(pitch)
    abs_r, abs_p = abs(roll_a), abs(pitch_a)
    tip_r = float(tip_roll_rad)
    tip_p = float(tip_pitch_rad)
    slow_r = float(slow_frac) * tip_r
    slow_p = float(slow_frac) * tip_p
    stop_r = float(stop_frac) * tip_r
    stop_p = float(stop_frac) * tip_p
    lethal_r = float(tip_lethal_frac) * tip_r
    lethal_p = float(tip_lethal_frac) * tip_p

    # Always trip at the physics tip — do not wait for a filter on this path.
    if abs_r >= tip_r or abs_p >= tip_p:
        return TiltClass("stop", KIND_TIP, roll_a, pitch_a)

    if max_climb_slope_rad is None:
        if abs_r >= stop_r or abs_p >= stop_p:
            return TiltClass("stop", KIND_TIP, roll_a, pitch_a)
        if abs_r >= slow_r or abs_p >= slow_p:
            return TiltClass("slow", KIND_GRADE, roll_a, pitch_a)
        return TiltClass("ok", KIND_OK, roll_a, pitch_a)

    climb = float(max_climb_slope_rad)
    # True tip margin (near software trip), including a hard side-roll.
    if abs_r >= lethal_r or abs_p >= lethal_p:
        return TiltClass("stop", KIND_TIP, roll_a, pitch_a)

    in_climb = abs_r <= climb and abs_p <= climb
    if in_climb:
        if abs_r >= slow_r or abs_p >= slow_p:
            return TiltClass("slow", KIND_GRADE, roll_a, pitch_a)
        return TiltClass("ok", KIND_OK, roll_a, pitch_a)

    # Past the climb cap but not tip-lethal: prefer contour, do not reverse.
    if abs_r >= slow_r or abs_p >= slow_p:
        return TiltClass("reroute", KIND_GRADE, roll_a, pitch_a)
    return TiltClass("ok", KIND_OK, roll_a, pitch_a)


class TipHoldFilter:
    """Median-window tilt + consecutive-frame hold before emitting stop.

    A single IMU spike on a gentle hill must not enter reverse / skip.
    Attitude at or above the full tip trip still emits immediately.
    """

    def __init__(self, *, window: int = 5, hold_steps: int = 3) -> None:
        self.window = max(1, int(window))
        self.hold_steps = max(1, int(hold_steps))
        self._rolls: deque[float] = deque(maxlen=self.window)
        self._pitches: deque[float] = deque(maxlen=self.window)
        self._stop_hold = 0
        self.last = TiltClass("ok", KIND_OK, 0.0, 0.0)

    def reset(self) -> None:
        self._rolls.clear()
        self._pitches.clear()
        self._stop_hold = 0
        self.last = TiltClass("ok", KIND_OK, 0.0, 0.0)

    def update(
        self,
        sample: TiltClass,
        *,
        tip_roll_rad: float,
        tip_pitch_rad: float,
    ) -> TiltClass:
        self._rolls.append(float(sample.roll))
        self._pitches.append(float(sample.pitch))
        roll = float(np.median(np.asarray(self._rolls, dtype=np.float32)))
        pitch = float(np.median(np.asarray(self._pitches, dtype=np.float32)))
        # Re-classify on the filtered attitude using the sample's kind/advice
        # as a hint only when the median still agrees.
        filtered = TiltClass(sample.advice, sample.kind, roll, pitch)
        # Immediate physics-near tip: do not debounce.
        if abs(roll) >= float(tip_roll_rad) or abs(pitch) >= float(tip_pitch_rad):
            self._stop_hold = 0
            self.last = TiltClass("stop", KIND_TIP, roll, pitch)
            return self.last
        if sample.kind == KIND_TIP and sample.advice == "stop":
            self._stop_hold += 1
            if self._stop_hold >= self.hold_steps:
                self.last = TiltClass("stop", KIND_TIP, roll, pitch)
                return self.last
            # Hold: treat as grade-slow so we do not reverse-loop on a spike.
            self.last = TiltClass("slow", KIND_GRADE, roll, pitch)
            return self.last
        self._stop_hold = 0
        self.last = TiltClass(filtered.advice, sample.kind, roll, pitch)
        return self.last


def owner_copy_for_tilt(kind: str, advice: str = "") -> str:
    """Owner line when tip vs climbable grade is distinguishable."""
    if kind == KIND_TIP and advice in {"", "stop"}:
        return OWNER_TIP_RISK
    if kind == KIND_GRADE:
        return OWNER_STEEP_GRADE
    return ""

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

import math
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from jims_mower.constants import GRAVITY_MPS2
from jims_mower.kinematics import sit_on_height_fn
from jims_mower.planning.fusion import attitude_from_accel
from jims_mower.perception.stereo import height_sampler_from_raster
from jims_mower.types import Pose

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
    urgent: bool = False

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
    * past the climb cap → **tip** ``stop`` (urgent). A ridge can jump
      ~0.10 rad in one physics step; waiting for ``imu_stop_frac`` lets
      pitch run through 0.38 → 0.47 and actually tip over.
    * the costmap still *contours* mapped cells between climb and the
      tip-safe margin — that is planning, not chassis IMU.

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

    # Always trip at the physics tip — do not wait for a filter on this path.
    if abs_r >= tip_r or abs_p >= tip_p:
        return TiltClass("stop", KIND_TIP, roll_a, pitch_a, urgent=True)

    if max_climb_slope_rad is None:
        if abs_r >= stop_r or abs_p >= stop_p:
            return TiltClass("stop", KIND_TIP, roll_a, pitch_a)
        if abs_r >= slow_r or abs_p >= slow_p:
            return TiltClass("slow", KIND_GRADE, roll_a, pitch_a)
        return TiltClass("ok", KIND_OK, roll_a, pitch_a)

    climb = float(max_climb_slope_rad)
    in_climb = abs_r <= climb and abs_p <= climb
    if in_climb:
        # The original bug: IMU stop == max_climb on the acre demo, so a
        # climbable face looked like a tip. Stay slow, do not reverse.
        if abs_r >= slow_r or abs_p >= slow_p:
            return TiltClass("slow", KIND_GRADE, roll_a, pitch_a)
        return TiltClass("ok", KIND_OK, roll_a, pitch_a)
    # Past the climb cap: stop. A ridge can jump 0.10 rad in one physics
    # step; waiting for imu_stop_frac lets pitch run through 0.38 → 0.47.
    if abs_r >= slow_r or abs_p >= slow_p:
        return TiltClass("stop", KIND_TIP, roll_a, pitch_a, urgent=True)
    return TiltClass("reroute", KIND_GRADE, roll_a, pitch_a)


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
        filtered = TiltClass(sample.advice, sample.kind, roll, pitch)
        # Immediate physics-near tip: do not debounce the raw *or* median
        # sample. A real tip after level driving must not look like a spike.
        if sample.urgent or (
            abs(sample.roll) >= float(tip_roll_rad)
            or abs(sample.pitch) >= float(tip_pitch_rad)
            or abs(roll) >= float(tip_roll_rad)
            or abs(pitch) >= float(tip_pitch_rad)
        ):
            self._stop_hold = 0
            self.last = TiltClass("stop", KIND_TIP, sample.roll, sample.pitch, urgent=True)
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


def grade_aware_cruise(
    base: float,
    *,
    advice: str,
    tilt_kind: str,
    slow_speed_factor: float,
    grade_speed_factor: float,
) -> float:
    """Scale a wheel-command cruise for explore / steep / unknown faces.

    ``base`` is the existing cruise fraction (× ``max_wheel_speed_mps``
    ≈ m/s). Climbable grade uses ``grade_speed_factor``; other ``slow``
    advice uses ``slow_speed_factor``. Does not change tip radians.
    """
    cruise = float(base)
    if tilt_kind == KIND_GRADE:
        cruise *= float(grade_speed_factor)
    elif advice == "slow":
        cruise *= float(slow_speed_factor)
    return max(0.0, cruise)


def probe_forward_grade(
    pose: Pose,
    sample_z: Callable[[float, float], float],
    *,
    length_m: float,
    track_m: float,
    look_ahead_m: float = 1.10,
    n_samples: int = 4,
    tip_roll_rad: float = 0.40,
    tip_pitch_rad: float = 0.45,
    slow_frac: float = 0.55,
    stop_frac: float = 0.85,
    max_climb_slope_rad: Optional[float] = None,
    tip_lethal_frac: float = 0.95,
) -> TiltClass:
    """Sit-model pitch/roll if the chassis were translated forward.

    Same ``atan2`` seating as ``sit_on_terrain`` — not a rolling rigid
    body. Used so the controller can slow or contour *before* the seated
    pose crosses climb → tip. Does not emit owner tip-stop by itself.
    """
    reach = max(float(look_ahead_m), 0.0)
    samples = max(1, int(n_samples))
    if reach <= 1e-6:
        seated = sit_on_height_fn(pose, sample_z, length_m, track_m)
        return classify_tilt(
            seated.roll,
            seated.pitch,
            tip_roll_rad=tip_roll_rad,
            tip_pitch_rad=tip_pitch_rad,
            slow_frac=slow_frac,
            stop_frac=stop_frac,
            max_climb_slope_rad=max_climb_slope_rad,
            tip_lethal_frac=tip_lethal_frac,
        )
    heading = float(pose.theta)
    c, s = math.cos(heading), math.sin(heading)
    worst = classify_tilt(
        0.0,
        0.0,
        tip_roll_rad=tip_roll_rad,
        tip_pitch_rad=tip_pitch_rad,
        slow_frac=slow_frac,
        stop_frac=stop_frac,
        max_climb_slope_rad=max_climb_slope_rad,
        tip_lethal_frac=tip_lethal_frac,
    )
    rank = {"ok": 0, "slow": 1, "reroute": 2, "stop": 3}
    start = min(0.28, reach)
    for dist in np.linspace(start, reach, samples):
        ghost = Pose(
            pose.x + float(dist) * c,
            pose.y + float(dist) * s,
            pose.theta,
        )
        seated = sit_on_height_fn(ghost, sample_z, length_m, track_m)
        got = classify_tilt(
            seated.roll,
            seated.pitch,
            tip_roll_rad=tip_roll_rad,
            tip_pitch_rad=tip_pitch_rad,
            slow_frac=slow_frac,
            stop_frac=stop_frac,
            max_climb_slope_rad=max_climb_slope_rad,
            tip_lethal_frac=tip_lethal_frac,
        )
        if rank.get(got.advice, 0) > rank.get(worst.advice, 0):
            worst = got
        elif rank.get(got.advice, 0) == rank.get(worst.advice, 0) and (
            abs(got.pitch) + abs(got.roll) > abs(worst.pitch) + abs(worst.roll)
        ):
            worst = got
    return worst


def look_ahead_from_elevation(
    pose: Pose,
    elevation: Optional[np.ndarray],
    *,
    resolution_m: float,
    length_m: float,
    track_m: float,
    look_ahead_m: float = 1.10,
    n_samples: int = 4,
    tip_roll_rad: float = 0.40,
    tip_pitch_rad: float = 0.45,
    slow_frac: float = 0.55,
    stop_frac: float = 0.85,
    max_climb_slope_rad: Optional[float] = None,
    tip_lethal_frac: float = 0.95,
) -> Optional[TiltClass]:
    """Forward sit-probe on an observer / stereo+ToF elevation raster."""
    if elevation is None:
        return None
    arr = np.asarray(elevation)
    if arr.ndim != 2 or arr.size == 0:
        return None
    sample_z = height_sampler_from_raster(arr, resolution_m=resolution_m)
    return probe_forward_grade(
        pose,
        sample_z,
        length_m=length_m,
        track_m=track_m,
        look_ahead_m=look_ahead_m,
        n_samples=n_samples,
        tip_roll_rad=tip_roll_rad,
        tip_pitch_rad=tip_pitch_rad,
        slow_frac=slow_frac,
        stop_frac=stop_frac,
        max_climb_slope_rad=max_climb_slope_rad,
        tip_lethal_frac=tip_lethal_frac,
    )


def look_ahead_advice(ahead: Optional[TiltClass]) -> str:
    """Map a sit-probe to controller advice. Ahead tip-risk is reroute, not stop."""
    if ahead is None or ahead.kind == KIND_OK:
        return "ok"
    if ahead.kind == KIND_TIP:
        return "reroute"
    if ahead.kind == KIND_GRADE:
        return "slow"
    return "ok"


def merge_look_ahead_kind(current_kind: str, ahead: Optional[TiltClass]) -> str:
    """Look-ahead tip-risk is a contour, not an IMU reverse / no-go stamp."""
    if ahead is None or ahead.kind == KIND_OK:
        return current_kind
    if current_kind == KIND_TIP:
        return current_kind
    if ahead.kind in {KIND_GRADE, KIND_TIP}:
        return KIND_GRADE
    return current_kind

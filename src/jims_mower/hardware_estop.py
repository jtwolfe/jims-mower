"""Hardware ESTOP / rail kill — independent of Python ``SafeStateMachine``.

Sim model of a paddle that drops **traction and trimmer rails** without
the control loop. Policy output and software ESTOP ``clear()`` cannot
soft-override a latched paddle. Only the documented hardware reset
(``reset()``) restores the rails.

Not a claimed SIL rating. Not a physical paddle. Gym ``env.reset()``
starts a new trial with the paddle released (sim power-cycle), which
is **not** the field reset procedure — see ``docs/ESTOP.md``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

# Owner / inject aliases. ``reset`` / ``clear`` are the documented HW reset.
PADDLE_HIT_KINDS = frozenset({"hw_estop", "paddle", "hardware_estop", "estop_paddle"})
PADDLE_RESET_MODES = frozenset({"reset", "clear", "hw_reset", "release"})


def is_hw_estop_kind(kind: str) -> bool:
    return str(kind or "").strip().lower() in PADDLE_HIT_KINDS


def is_hw_reset_mode(mode: str) -> bool:
    return str(mode or "").strip().lower() in PADDLE_RESET_MODES


class HardwareEstop:
    """Latch that forces wheel + trimmer rails to zero underneath policy."""

    def __init__(self) -> None:
        self.latched = False
        self.reason: Optional[str] = None
        self.last_event: Optional[str] = None
        self.last_applied = (0.0, 0.0, 0.0)

    def reset(self) -> None:
        """Documented hardware reset: paddle released, then explicit reset.

        Software ESTOP ``SafeStateMachine.clear()`` must not call this.
        """
        was = self.latched
        self.latched = False
        self.reason = None
        if was:
            self.last_event = "hw_estop_reset"

    def hit(self, reason: str = "paddle") -> None:
        """Paddle struck — latch rails open. Independent of Python."""
        self.latched = True
        self.reason = str(reason or "paddle")
        self.last_event = "hw_estop_latched"

    def apply(
        self, left: float, right: float, requested: bool
    ) -> tuple[float, float, bool]:
        """Last rail filter. Latched → wheels and trimmer are dead."""
        if self.latched:
            self.last_applied = (0.0, 0.0, 0.0)
            return 0.0, 0.0, False
        trim = 1.0 if requested else 0.0
        self.last_applied = (float(left), float(right), trim)
        return float(left), float(right), bool(requested)

    def filter_action(self, action: np.ndarray) -> np.ndarray:
        """Zero a ``[left, right, trimmer]`` command when the paddle is latched."""
        act = np.asarray(action, dtype=np.float32).reshape(-1)
        if act.size < 3:
            act = np.pad(act, (0, 3 - act.size))
        if self.latched:
            return np.zeros(3, dtype=np.float32)
        return act[:3].copy()

    def as_info(self) -> dict[str, Any]:
        return {
            "hw_estop": bool(self.latched),
            "hw_estop_latched": bool(self.latched),
            "hw_estop_reason": self.reason,
            "hw_estop_rails": "dead" if self.latched else "live",
            "hw_estop_event": self.last_event,
            "hw_estop_applied": [float(v) for v in self.last_applied],
        }

    def as_fault(self) -> Optional[dict[str, Any]]:
        """Owner-fault row when the paddle is latched. Not a retrieve SOS."""
        if not self.latched:
            return None
        return {
            "code": "HW_ESTOP",
            "detail": self.reason or "paddle latched — rails dead",
            "retrieve": False,
            "kind": "hardware",
        }


def estop_kind(*, software: bool, hardware: bool) -> Optional[str]:
    """Distinguish software latch vs hardware paddle in status / events."""
    if hardware and software:
        return "both"
    if hardware:
        return "hardware"
    if software:
        return "software"
    return None

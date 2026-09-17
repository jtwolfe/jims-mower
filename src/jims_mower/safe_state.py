"""ESTOP / limp / safe-state machine used by the coverage controller.

Software contract only — not a claimed SIL / hardware rating. A latched
ESTOP zeros wheels and the trimmer until an operator ``clear()``. Limp
scales cruise after repeated ``stop`` advice; exhausted limp becomes a
SAFE hold (call-for-help).

The hardware paddle is a **separate** latch (``HardwareEstop``) on the
rails. ``clear()`` must not restore motion while that paddle is latched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from jims_mower.constants import SAFE_MODES

_ADVICE_RANK = {name: i for i, name in enumerate(("ok", "slow", "reroute", "stop"))}


@dataclass(frozen=True)
class SafeCommand:
    mode: str
    scale: float
    hold: bool
    trimmer_allowed: bool
    help_requested: bool
    reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "scale": self.scale,
            "hold": self.hold,
            "trimmer_allowed": self.trimmer_allowed,
            "help_requested": self.help_requested,
            "reason": self.reason,
        }


class SafeStateMachine:
    """RUN → LIMP → SAFE, with ESTOP latching over everything."""

    def __init__(
        self,
        *,
        limp_after_stops: int = 8,
        safe_after_limp_steps: int = 12,
        limp_scale: float = 0.35,
        recover_ok_steps: int = 4,
    ) -> None:
        if limp_after_stops < 1 or safe_after_limp_steps < 1 or recover_ok_steps < 1:
            raise ValueError("safe-state thresholds must be >= 1")
        if not 0.0 < limp_scale <= 1.0:
            raise ValueError("limp_scale must be in (0, 1]")
        self.limp_after_stops = int(limp_after_stops)
        self.safe_after_limp_steps = int(safe_after_limp_steps)
        self.limp_scale = float(limp_scale)
        self.recover_ok_steps = int(recover_ok_steps)
        self.mode = "run"
        self.last_reason: Optional[str] = None
        self._stop_ticks = 0
        self._limp_ticks = 0
        self._ok_ticks = 0

    @classmethod
    def from_config(cls, cfg: Any) -> "SafeStateMachine":
        return cls(
            limp_after_stops=int(getattr(cfg, "limp_after_stops", 8)),
            safe_after_limp_steps=int(getattr(cfg, "safe_after_limp_steps", 12)),
            limp_scale=float(getattr(cfg, "limp_scale", 0.35)),
            recover_ok_steps=int(getattr(cfg, "recover_ok_steps", 4)),
        )

    def reset(self) -> None:
        self.mode = "run"
        self.last_reason = None
        self._stop_ticks = 0
        self._limp_ticks = 0
        self._ok_ticks = 0

    def request_estop(self, reason: str = "estop") -> None:
        self.mode = "estop"
        self.last_reason = reason
        self._ok_ticks = 0

    def enter_safe(self, reason: str = "safe hold") -> None:
        if self.mode == "estop":
            return
        self.mode = "safe"
        self.last_reason = reason

    def clear(self) -> None:
        """Operator reset: ESTOP / SAFE / LIMP → RUN."""
        self.reset()

    def clear_if_not_estop(self) -> None:
        """Hand-signal ``go`` / recovery resume. Does not unlatch ESTOP."""
        if self.mode == "estop":
            return
        self.reset()

    def tick(
        self,
        *,
        advice: str = "ok",
        estop: bool = False,
        tipover: bool = False,
        drain_drop: bool = False,
        help_requested: bool = False,
    ) -> SafeCommand:
        if estop:
            self.request_estop("software/hardware estop")
            return self.command()
        if self.mode == "estop":
            return self.command()
        if help_requested:
            self.enter_safe("call-for-help")
            return self.command()
        if tipover:
            self.enter_safe("tip-over — immobilised")
            return self.command()
        if self.mode == "safe":
            return self.command()

        rank = _ADVICE_RANK.get(advice if advice in _ADVICE_RANK else "ok", 0)
        severe = drain_drop or rank >= _ADVICE_RANK["stop"]
        if self.mode == "run":
            if severe:
                self._stop_ticks += 1
                self._ok_ticks = 0
                if self._stop_ticks >= self.limp_after_stops:
                    self.mode = "limp"
                    self._limp_ticks = 0
                    self.last_reason = "repeated stop — limp"
            else:
                self._stop_ticks = 0
                self._ok_ticks += 1
        elif self.mode == "limp":
            self._limp_ticks += 1
            if severe:
                self._ok_ticks = 0
                if self._limp_ticks >= self.safe_after_limp_steps:
                    self.enter_safe("limp exhausted — safe hold")
            else:
                self._ok_ticks += 1
                if self._ok_ticks >= self.recover_ok_steps:
                    self.mode = "run"
                    self._stop_ticks = 0
                    self._limp_ticks = 0
                    self.last_reason = None
        return self.command()

    def command(self) -> SafeCommand:
        if self.mode not in SAFE_MODES:
            self.mode = "run"
        if self.mode == "estop":
            return SafeCommand("estop", 0.0, True, False, True, self.last_reason)
        if self.mode == "safe":
            return SafeCommand("safe", 0.0, True, False, True, self.last_reason)
        if self.mode == "limp":
            return SafeCommand(
                "limp",
                self.limp_scale,
                False,
                False,
                False,
                self.last_reason,
            )
        return SafeCommand("run", 1.0, False, True, False, self.last_reason)

    def apply(self, action: np.ndarray) -> np.ndarray:
        """Scale or zero a ``[left, right, trimmer]`` command."""
        cmd = self.command()
        out = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        if out.size < 3:
            padded = np.zeros(3, dtype=np.float32)
            padded[: out.size] = out
            out = padded
        if cmd.hold:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        out[0] *= cmd.scale
        out[1] *= cmd.scale
        if not cmd.trimmer_allowed:
            out[2] = 0.0
        return out[:3]


def estop_requested(obs: Optional[dict[str, Any]] = None, info: Optional[dict[str, Any]] = None) -> bool:
    """True when the gym / owner latches a software ESTOP this step."""
    for blob in (info, obs):
        if not isinstance(blob, dict):
            continue
        raw = blob.get("estop")
        if raw is True or raw == 1 or raw == "estop":
            return True
        if isinstance(raw, (np.ndarray, list, tuple)) and len(raw) and bool(raw[0]):
            return True
    return False

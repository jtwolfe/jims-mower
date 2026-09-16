"""Simulated Bluetooth pairing state machine. No BlueZ.

Owner phone flow is Pair → Start. When ``require_pair`` is true (live
owner / field-dryrun overlay; gym unit tests may leave it false), Start
is refused until ``state == paired``.

States: unpaired → pairing → paired / failed / lost.

Unpair returns to ``unpaired``. Injected radio-lost sets ``lost``.
Both **hold safe** (job → hold, not ESTOP) if a job is running. ESTOP
stays the paddle / owner ESTOP path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from jims_mower.constants import GYM_PAIR_PIN, PAIR_STATES, PAIRING_SCHEMA


class PairingError(ValueError):
    """Invalid pairing request or persisted state."""


def default_pairing(*, state: str = "unpaired") -> dict[str, Any]:
    key = str(state or "unpaired").strip().lower()
    if key not in PAIR_STATES:
        key = "unpaired"
    return {
        "schema": PAIRING_SCHEMA,
        "state": key,
        "simulated": True,
        "not_bluez": True,
        "note": "sim pair — no BlueZ / no RF hardware",
    }


def parse_pairing(raw: Any) -> dict[str, Any]:
    if raw is None:
        return default_pairing()
    if isinstance(raw, bool):
        return default_pairing(state="paired" if raw else "unpaired")
    if not isinstance(raw, dict):
        raise PairingError("pairing must be a mapping")
    state = str(raw.get("state") or "unpaired").strip().lower()
    if state not in PAIR_STATES:
        raise PairingError(f"pairing.state must be one of {PAIR_STATES}; got {state!r}")
    blob = default_pairing(state=state)
    if raw.get("note"):
        blob["note"] = str(raw["note"])
    return blob


@dataclass
class PairResult:
    ok: bool
    state: str
    reason: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {"ok": bool(self.ok), "state": self.state, "reason": self.reason}


@dataclass
class PairingMachine:
    """Explicit pair request → optional gym PIN → persistable state."""

    require_pair: bool = False
    pin: str = GYM_PAIR_PIN
    state: str = "unpaired"
    last_reason: Optional[str] = None
    history: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        key = str(self.state or "unpaired").strip().lower()
        if key not in PAIR_STATES:
            raise PairingError(f"pairing.state must be one of {PAIR_STATES}; got {key!r}")
        self.state = key
        self.pin = str(self.pin or GYM_PAIR_PIN)
        self.history = list(self.history or [])
        if not self.history:
            self.history.append(self.state)

    @classmethod
    def from_profile(
        cls,
        raw: Any,
        *,
        require_pair: bool = False,
        pin: str = GYM_PAIR_PIN,
    ) -> "PairingMachine":
        blob = parse_pairing(raw)
        return cls(require_pair=bool(require_pair), pin=str(pin or GYM_PAIR_PIN), state=blob["state"])

    @property
    def is_paired(self) -> bool:
        return self.state == "paired"

    @property
    def can_start(self) -> bool:
        if not self.require_pair:
            return True
        return self.is_paired

    def refuse_start(self) -> Optional[str]:
        if self.can_start:
            return None
        return "not_paired"

    def _set(self, state: str, *, reason: Optional[str] = None) -> None:
        key = str(state).strip().lower()
        if key not in PAIR_STATES:
            raise PairingError(f"pairing.state must be one of {PAIR_STATES}; got {key!r}")
        self.state = key
        self.last_reason = reason
        self.history.append(key)

    def request_pair(self, pin: Optional[str] = None) -> PairResult:
        """Explicit pair request. Optional PIN must match the gym PIN if given."""
        self._set("pairing", reason="request")
        offered = None if pin is None else str(pin).strip()
        if offered:
            if offered != str(self.pin):
                self._set("failed", reason="pin_mismatch")
                return PairResult(False, "failed", "pin_mismatch")
        self._set("paired", reason="paired")
        return PairResult(True, "paired", None)

    def force_paired(self) -> None:
        """Restore a persisted paired session without a new PIN prompt."""
        self._set("paired", reason="restored")

    def unpair(self) -> PairResult:
        self._set("unpaired", reason="unpair")
        return PairResult(True, "unpaired", "unpair")

    def lose(self, *, reason: str = "radio_lost") -> PairResult:
        """Radio-lost inject. Distinct from unpair; Start still refused."""
        self._set("lost", reason=reason)
        return PairResult(True, "lost", reason)

    def as_info(self) -> dict[str, Any]:
        blob = default_pairing(state=self.state)
        blob.update(
            {
                "require_pair": bool(self.require_pair),
                "paired": bool(self.is_paired),
                "can_start": bool(self.can_start),
                "reason": self.last_reason,
                "pin_required": False,
                "gym_pin": str(self.pin),
                "on_lost": "hold_safe",
                "note": (
                    "sim pair — no BlueZ. Unpair / radio-lost holds the job safe "
                    "(not ESTOP). Re-pair before Start when require_pair is true."
                ),
            }
        )
        return blob

    def persist(self) -> dict[str, Any]:
        return default_pairing(state=self.state)

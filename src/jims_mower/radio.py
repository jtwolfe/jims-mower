"""Simulated BT / Wi-Fi / LoRa links. No real RF hardware.

Command preference is Wi-Fi → BT → LoRa. Each link has a bandwidth,
range, and drop probability. Heartbeat loss can limp-home or stop+beacon
(config). Figures are class-scale stubs, not measured air data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from jims_mower.constants import (
    RADIO_CHANNELS,
    RADIO_LOSS_ACTIONS,
    RADIO_SCHEMA,
)


@dataclass(frozen=True)
class RadioLink:
    """One simulated bearer."""

    name: str
    bandwidth_bps: float
    range_m: float
    drop_prob: float

    def reachable(self, distance_m: float) -> bool:
        return float(distance_m) <= float(self.range_m)

    def attempt(self, rng: np.random.Generator, distance_m: float) -> bool:
        if not self.reachable(distance_m):
            return False
        if not 0.0 <= self.drop_prob <= 1.0:
            return False
        return float(rng.random()) >= float(self.drop_prob)


def default_links() -> dict[str, RadioLink]:
    """Class-scale stubs — not a field RF survey."""
    return {
        "wifi": RadioLink("wifi", bandwidth_bps=20_000_000.0, range_m=40.0, drop_prob=0.02),
        "bt": RadioLink("bt", bandwidth_bps=1_000_000.0, range_m=12.0, drop_prob=0.08),
        "lora": RadioLink("lora", bandwidth_bps=5_000.0, range_m=2_000.0, drop_prob=0.15),
    }


def links_from_config(cfg: Any) -> dict[str, RadioLink]:
    defaults = default_links()
    if cfg is None:
        return defaults
    out = dict(defaults)
    for name in RADIO_CHANNELS:
        raw = getattr(cfg, name, None)
        if raw is None:
            continue
        base = out[name]
        out[name] = RadioLink(
            name=name,
            bandwidth_bps=float(getattr(raw, "bandwidth_bps", base.bandwidth_bps)),
            range_m=float(getattr(raw, "range_m", base.range_m)),
            drop_prob=float(getattr(raw, "drop_prob", base.drop_prob)),
        )
    return out


@dataclass
class RadioDelivery:
    ok: bool
    channel: Optional[str]
    attempts: list[str] = field(default_factory=list)
    reason: Optional[str] = None
    latency_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "channel": self.channel,
            "attempts": list(self.attempts),
            "reason": self.reason,
            "latency_s": float(self.latency_s),
        }


class RadioSim:
    """Command / heartbeat router over simulated bearers."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        distance_m: float = 10.0,
        heartbeat_timeout_s: float = 2.0,
        on_loss: str = "stop_beacon",
        links: Optional[dict[str, RadioLink]] = None,
        seed: int = 0,
        payload_bytes: int = 48,
    ) -> None:
        action = str(on_loss or "stop_beacon").strip().lower()
        if action not in RADIO_LOSS_ACTIONS:
            raise ValueError(
                f"radio on_loss must be one of {sorted(RADIO_LOSS_ACTIONS)}; got {on_loss!r}"
            )
        self.enabled = bool(enabled)
        self.distance_m = float(distance_m)
        self.heartbeat_timeout_s = float(heartbeat_timeout_s)
        self.on_loss = action
        self.links = links or default_links()
        self.payload_bytes = int(payload_bytes)
        self.rng = np.random.default_rng(int(seed))
        self._since_ok_s = 0.0
        self.lost = False
        self.last_channel: Optional[str] = None
        self.last_delivery: Optional[RadioDelivery] = None

    @classmethod
    def from_config(cls, cfg: Any, *, seed: int = 0) -> "RadioSim":
        radio = getattr(cfg, "radio", cfg)
        if radio is None:
            return cls(enabled=False, seed=seed)
        return cls(
            enabled=bool(getattr(radio, "enabled", False)),
            distance_m=float(getattr(radio, "distance_m", 10.0)),
            heartbeat_timeout_s=float(getattr(radio, "heartbeat_timeout_s", 2.0)),
            on_loss=str(getattr(radio, "on_loss", "stop_beacon")),
            links=links_from_config(radio),
            seed=seed,
            payload_bytes=int(getattr(radio, "payload_bytes", 48)),
        )

    def reset(self, *, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        self._since_ok_s = 0.0
        self.lost = False
        self.last_channel = None
        self.last_delivery = None

    def send(
        self,
        *,
        distance_m: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
        payload_bytes: Optional[int] = None,
    ) -> RadioDelivery:
        """Try Wi-Fi, then BT, then LoRa. First success wins."""
        dist = float(self.distance_m if distance_m is None else distance_m)
        gen = self.rng if rng is None else rng
        nbytes = int(self.payload_bytes if payload_bytes is None else payload_bytes)
        attempts: list[str] = []
        for name in RADIO_CHANNELS:
            link = self.links[name]
            attempts.append(name)
            if link.attempt(gen, dist):
                latency = float(nbytes * 8.0 / max(link.bandwidth_bps, 1.0))
                delivery = RadioDelivery(
                    True, name, attempts, reason=None, latency_s=latency
                )
                self.last_channel = name
                self.last_delivery = delivery
                return delivery
        delivery = RadioDelivery(False, None, attempts, reason="all_channels_failed")
        self.last_channel = None
        self.last_delivery = delivery
        return delivery

    def tick(
        self,
        dt: float,
        *,
        distance_m: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
        send_heartbeat: bool = True,
    ) -> dict[str, Any]:
        """Age the heartbeat. When enabled, a failed window latches ``lost``."""
        if not self.enabled:
            self.lost = False
            return self.as_info()
        if send_heartbeat:
            delivery = self.send(distance_m=distance_m, rng=rng)
            if delivery.ok:
                self._since_ok_s = 0.0
                self.lost = False
                return self.as_info()
        self._since_ok_s += float(dt)
        if self._since_ok_s >= self.heartbeat_timeout_s:
            self.lost = True
        return self.as_info()

    def as_info(self) -> dict[str, Any]:
        delivery = self.last_delivery.to_dict() if self.last_delivery else None
        return {
            "schema": RADIO_SCHEMA,
            "radio_enabled": bool(self.enabled),
            "radio_channel": self.last_channel,
            "radio_lost": bool(self.lost),
            "radio_on_loss": self.on_loss,
            "radio_distance_m": float(self.distance_m),
            "radio_since_ok_s": float(self._since_ok_s),
            "radio_delivery": delivery,
            "not_rf_hardware": True,
        }

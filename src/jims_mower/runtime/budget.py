"""Orin-class battery / thermal *stub* — limp the policy when hot or empty.

This is a first-order energy and heat budget for sim, not a measured
Orin Nano power trace and not a claimed TDP. Default capacity is the
50 Wh gym stub. When ``runtime.battery.measured`` is true, SOC drain
uses the configured bench Wh. Still not a BMS. Do not treat
``temp_c`` or ``soc`` as board telemetry. No acre-runtime claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

# Advice strings match the planner vocabulary (ok / slow / stop).
BUDGET_OK = "ok"
BUDGET_LIMP = "slow"
BUDGET_STOP = "stop"


@dataclass
class OrinBudget:
    """Discrete-time SOC + first-order thermal RC toward ambient."""

    capacity_wh: float = 50.0
    charge_time_h: Optional[float] = None
    measured: bool = False
    soc: float = 1.0
    idle_w: float = 8.0
    drive_w: float = 25.0
    compute_w: float = 7.0
    trimmer_w: float = 12.0
    limp_soc: float = 0.15
    stop_soc: float = 0.05
    t_c: float = 45.0
    t_ambient_c: float = 35.0
    t_hot_c: float = 75.0
    t_crit_c: float = 85.0
    tau_s: float = 90.0
    heat_c_per_w: float = 0.35
    enabled: bool = True

    def reset(self, *, soc: Optional[float] = None, t_c: Optional[float] = None) -> None:
        if soc is not None:
            self.soc = float(np.clip(soc, 0.0, 1.0))
        if t_c is not None:
            self.t_c = float(t_c)

    def step(
        self,
        dt: float,
        action: Optional[np.ndarray] = None,
        *,
        n_cameras: int = 4,
    ) -> str:
        """Advance energy and temperature. Returns ``ok`` / ``slow`` / ``stop``."""
        if not self.enabled:
            return BUDGET_OK
        dt = max(float(dt), 1e-6)
        act = np.zeros(3, dtype=np.float32) if action is None else np.asarray(action, dtype=np.float32).reshape(-1)
        left = float(act[0]) if act.size > 0 else 0.0
        right = float(act[1]) if act.size > 1 else 0.0
        trimmer = float(act[2]) if act.size > 2 else 0.0
        drive = 0.5 * (abs(left) + abs(right))
        cams = max(int(n_cameras), 0) / 6.0
        power_w = (
            self.idle_w
            + self.drive_w * drive
            + self.compute_w * cams
            + self.trimmer_w * (1.0 if trimmer > 0.5 else 0.0)
        )
        if self.capacity_wh > 0.0:
            d_soc = (power_w * dt / 3600.0) / self.capacity_wh
            self.soc = float(np.clip(self.soc - d_soc, 0.0, 1.0))
        tau = max(self.tau_s, 1e-3)
        # First-order: heat from draw, cool toward ambient.
        self.t_c = float(
            self.t_c
            + dt
            * (
                self.heat_c_per_w * (power_w / max(self.idle_w, 1.0))
                - (self.t_c - self.t_ambient_c) / tau
            )
        )
        return self.advice()

    def advice(self) -> str:
        if not self.enabled:
            return BUDGET_OK
        if self.soc <= self.stop_soc or self.t_c >= self.t_crit_c:
            return BUDGET_STOP
        if self.soc <= self.limp_soc or self.t_c >= self.t_hot_c:
            return BUDGET_LIMP
        return BUDGET_OK

    def reason(self) -> str:
        if not self.enabled:
            return "disabled"
        parts: list[str] = []
        if self.soc <= self.stop_soc:
            parts.append("battery_empty")
        elif self.soc <= self.limp_soc:
            parts.append("battery_low")
        if self.t_c >= self.t_crit_c:
            parts.append("thermal_crit")
        elif self.t_c >= self.t_hot_c:
            parts.append("thermal_hot")
        return ",".join(parts) if parts else "ok"

    def as_info(self) -> dict[str, Any]:
        from jims_mower.pack import remaining_wh

        cap = float(self.capacity_wh) if self.capacity_wh else None
        return {
            "battery_soc": float(self.soc),
            "thermal_c": float(self.t_c),
            "capacity_wh": cap,
            "charge_time_h": None if self.charge_time_h is None else float(self.charge_time_h),
            "remaining_wh": remaining_wh(self.soc, cap),
            "pack_measured": bool(self.measured),
            "budget_advice": self.advice(),
            "budget_reason": self.reason(),
            "budget_enabled": bool(self.enabled),
            "not_a_power_trace": True,
            "acre_runtime_h": None,
        }


def budget_advice(info: dict[str, Any]) -> str:
    raw = str(info.get("budget_advice") or BUDGET_OK)
    if raw in {BUDGET_OK, BUDGET_LIMP, BUDGET_STOP}:
        return raw
    return BUDGET_OK


def budget_from_config(cfg: Any) -> OrinBudget:
    """Build a budget from ``EnvConfig.runtime`` (or defaults)."""
    runtime = getattr(cfg, "runtime", None)
    if runtime is None:
        return OrinBudget(enabled=False)
    batt = getattr(runtime, "battery", runtime)
    therm = getattr(runtime, "thermal", runtime)
    cap = getattr(batt, "capacity_wh", 50.0)
    charge = getattr(batt, "charge_time_h", None)
    return OrinBudget(
        capacity_wh=50.0 if cap is None else float(cap),
        charge_time_h=None if charge is None else float(charge),
        measured=bool(getattr(batt, "measured", False)),
        soc=float(getattr(batt, "soc", 1.0)),
        idle_w=float(getattr(batt, "idle_w", 8.0)),
        drive_w=float(getattr(batt, "drive_w", 25.0)),
        compute_w=float(getattr(batt, "compute_w", 7.0)),
        trimmer_w=float(getattr(batt, "trimmer_w", 12.0)),
        limp_soc=float(getattr(batt, "limp_soc", 0.15)),
        stop_soc=float(getattr(batt, "stop_soc", 0.05)),
        t_c=float(getattr(therm, "t_c", getattr(therm, "start_c", 45.0))),
        t_ambient_c=float(getattr(therm, "t_ambient_c", 35.0)),
        t_hot_c=float(getattr(therm, "t_hot_c", 75.0)),
        t_crit_c=float(getattr(therm, "t_crit_c", 85.0)),
        tau_s=float(getattr(therm, "tau_s", 90.0)),
        heat_c_per_w=float(getattr(therm, "heat_c_per_w", 0.35)),
        enabled=bool(getattr(runtime, "enabled", True)),
    )

"""Pack / thermal bench provenance — procedure hooks, not a BMS.

Gym may keep the 50 Wh stub when ``runtime.battery.measured`` is false.
Flipping ``measured: true`` without a filled bench (capacity, charge
time, date) is a config error so a silent default cannot look like a
field claim.

See ``docs/PACK_THERMAL.md``. No acre-runtime claim lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

GYM_STUB_CAPACITY_WH = 50.0
PACK_SCHEMA = "jims_mower.pack.v1"
LABEL_STUB = "STUB"
LABEL_TEMPLATE = "TEMPLATE"
LABEL_MEASURED = "MEASURED"

CHECKLIST = (
    "Full-charge the pack (charger terminate / BMS full).",
    "Discharge through a known load; integrate I·dt (Ah) and V·I·dt (Wh).",
    "Write capacity_wh from the discharge Wh (not the HARDWARE_DESIGN model).",
    "Time empty → charger terminate; write charge_time_h.",
    "Log board °C under a representative load; write thermal.board_load_c.",
    "Set runtime.battery.measured: true, template: false, fill measured_at.",
    "Do not claim acre runtime. acre_runtime_h stays null.",
)


class PackError(ValueError):
    """Measured pack claim is missing bench numbers or is still a stub."""


def pack_template_path() -> Path:
    """Copy-this MEASURED template. Placeholders, ``template: true``."""
    packaged = Path(__file__).resolve().parent / "data" / "orin" / "pack_measured.template.yaml"
    if packaged.is_file():
        return packaged
    return Path(__file__).resolve().parents[2] / "configs" / "orin" / "pack_measured.template.yaml"


@dataclass(frozen=True)
class PackMeta:
    measured: bool
    template: bool
    capacity_wh: Optional[float]
    charge_time_h: Optional[float]
    measured_at: str
    notes: str
    board_load_c: Optional[float]
    thermal_measured: bool

    @property
    def kind(self) -> str:
        if self.template:
            return "template"
        if self.measured:
            return "measured"
        return "stub"

    @property
    def label(self) -> str:
        if self.template:
            return LABEL_TEMPLATE
        if self.measured:
            return LABEL_MEASURED
        return LABEL_STUB


def _opt_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def meta_from_battery(
    batt: Any,
    *,
    board_load_c: Optional[float] = None,
    thermal_measured: bool = False,
) -> PackMeta:
    return PackMeta(
        measured=bool(getattr(batt, "measured", False)),
        template=bool(getattr(batt, "template", False)),
        capacity_wh=_opt_float(getattr(batt, "capacity_wh", None)),
        charge_time_h=_opt_float(getattr(batt, "charge_time_h", None)),
        measured_at=str(getattr(batt, "measured_at", "") or ""),
        notes=str(getattr(batt, "notes", "") or ""),
        board_load_c=_opt_float(board_load_c),
        thermal_measured=bool(thermal_measured),
    )


def meta_from_config(cfg: Any) -> PackMeta:
    runtime = getattr(cfg, "runtime", cfg)
    batt = getattr(runtime, "battery", runtime)
    therm = getattr(runtime, "thermal", None)
    board = getattr(therm, "board_load_c", None) if therm is not None else None
    thermal_measured = bool(getattr(therm, "measured", False)) if therm is not None else False
    return meta_from_battery(batt, board_load_c=board, thermal_measured=thermal_measured)


def validate_pack_claim(batt: Any, *, template_ok: bool = True) -> None:
    """Refuse ``measured: true`` that still looks like the gym stub.

    Templates may keep null capacity / charge time. A committed measured
    pack must have capacity_wh, charge_time_h, and measured_at from a
    filled bench — not the silent 50 Wh default.
    """
    template = bool(getattr(batt, "template", False))
    measured = bool(getattr(batt, "measured", False))
    if template:
        if not template_ok:
            raise PackError("pack template cannot be treated as a measured field claim")
        return
    if not measured:
        return
    cap = getattr(batt, "capacity_wh", None)
    charge = getattr(batt, "charge_time_h", None)
    measured_at = str(getattr(batt, "measured_at", "") or "").strip()
    notes = str(getattr(batt, "notes", "") or "").strip()
    missing: list[str] = []
    if cap is None:
        missing.append("capacity_wh")
    if charge is None:
        missing.append("charge_time_h")
    if not measured_at:
        missing.append("measured_at")
    if missing:
        raise PackError(
            "runtime.battery.measured: true requires bench fields "
            f"{', '.join(missing)}; do not flip measured on the "
            f"{GYM_STUB_CAPACITY_WH:g} Wh gym stub"
        )
    if float(cap) == GYM_STUB_CAPACITY_WH and not notes:
        raise PackError(
            "runtime.battery.measured: true with capacity_wh="
            f"{GYM_STUB_CAPACITY_WH:g} (gym stub) needs notes that this "
            "Wh came from the bench, not the default"
        )


def remaining_wh(soc: float, capacity_wh: Optional[float]) -> Optional[float]:
    """SOC fraction × configured capacity. Not an acre-runtime claim."""
    if capacity_wh is None:
        return None
    return float(soc) * float(capacity_wh)


def battery_status_block(
    *,
    soc: float,
    temp_c: float,
    info: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Owner ``/status`` battery object. ``acre_runtime_h`` stays null."""
    blob = info if isinstance(info, dict) else {}
    cap = blob.get("capacity_wh", GYM_STUB_CAPACITY_WH)
    charge = blob.get("charge_time_h")
    measured = bool(blob.get("pack_measured", False))
    cap_f = None if cap is None else float(cap)
    rem = remaining_wh(soc, cap_f)
    return {
        "soc": float(soc),
        "temp_c": float(temp_c),
        "capacity_wh": cap_f,
        "charge_time_h": None if charge is None else float(charge),
        "remaining_wh": rem,
        "measured": measured,
        "not_a_power_trace": True,
        "acre_runtime_h": None,
    }


def load_pack_template() -> Any:
    """Load the MEASURED template (placeholders)."""
    from jims_mower.config import load_config

    return load_config(pack_template_path())

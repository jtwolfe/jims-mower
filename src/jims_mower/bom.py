"""BOM freeze — part *classes* from HARDWARE_DESIGN §9.

No SKUs, no prices. Every numeric assumption is marked. Load the repo
file at ``configs/hardware/bom.yaml``. See ``docs/FAB_CHECKLIST.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

BOM_SCHEMA = "jims_mower.bom.v1"
BUY_MAKE = frozenset({"buy", "make", "buy_or_make"})
NUMBER_STATUS = frozenset({"assumption", "measured", "class", "target"})
FORBIDDEN_KEYS = frozenset(
    {
        "sku",
        "skus",
        "part_number",
        "mpn",
        "price",
        "price_usd",
        "cost",
        "cost_usd",
        "vendor_sku",
    }
)
REQUIRED_ROW = (
    "id",
    "qty",
    "class",
    "buy_make",
    "verify_before_spin",
    "open_questions",
    "number_status",
)


class BomError(ValueError):
    """BOM freeze file is missing, priced, or unmarked."""


@dataclass(frozen=True)
class BomRow:
    id: str
    qty: Any
    class_name: str
    buy_make: str
    verify_before_spin: str
    open_questions: str
    number_status: str
    feeds: str = ""
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "qty": self.qty,
            "class": self.class_name,
            "buy_make": self.buy_make,
            "verify_before_spin": self.verify_before_spin,
            "open_questions": self.open_questions,
            "number_status": self.number_status,
            "feeds": self.feeds,
            "notes": self.notes,
        }


@dataclass
class BomFreeze:
    schema: str
    source: str
    measured: bool
    note: str
    rows: list[BomRow]
    path: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source": self.source,
            "measured": self.measured,
            "note": self.note,
            "classes": [row.as_dict() for row in self.rows],
        }


def bom_path() -> Path:
    packaged = Path(__file__).resolve().parent / "data" / "hardware" / "bom.yaml"
    if packaged.is_file():
        return packaged
    return Path(__file__).resolve().parents[2] / "configs" / "hardware" / "bom.yaml"


def _forbid_prices(node: Any, *, where: str) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            low = str(key).strip().lower()
            if low in FORBIDDEN_KEYS:
                raise BomError(
                    f"{where}: {key!r} is forbidden — BOM freeze is part "
                    "classes only (no SKUs / prices)"
                )
            _forbid_prices(value, where=f"{where}.{key}")
    elif isinstance(node, list):
        for i, item in enumerate(node):
            _forbid_prices(item, where=f"{where}[{i}]")


def _row_from_mapping(raw: dict[str, Any], *, index: int) -> BomRow:
    missing = [k for k in REQUIRED_ROW if k not in raw or raw[k] in (None, "")]
    if missing:
        raise BomError(f"classes[{index}] missing {missing}")
    buy = str(raw["buy_make"]).strip().lower()
    if buy not in BUY_MAKE:
        raise BomError(
            f"classes[{index}].buy_make must be buy|make|buy_or_make; got {raw['buy_make']!r}"
        )
    status = str(raw["number_status"]).strip().lower()
    if status not in NUMBER_STATUS:
        raise BomError(
            f"classes[{index}].number_status must be assumption|measured|class|target; "
            f"got {raw['number_status']!r}"
        )
    return BomRow(
        id=str(raw["id"]),
        qty=raw["qty"],
        class_name=str(raw["class"]),
        buy_make=buy,
        verify_before_spin=str(raw["verify_before_spin"]),
        open_questions=str(raw["open_questions"]),
        number_status=status,
        feeds=str(raw.get("feeds") or ""),
        notes=str(raw.get("notes") or ""),
    )


def load_bom(path: Optional[Path] = None) -> BomFreeze:
    target = Path(path) if path is not None else bom_path()
    if not target.is_file():
        raise BomError(f"BOM file not found: {target}")
    data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise BomError("BOM YAML must be a mapping")
    _forbid_prices(data, where="bom")
    schema = str(data.get("schema") or "")
    if schema != BOM_SCHEMA:
        raise BomError(f"BOM schema must be {BOM_SCHEMA}; got {schema!r}")
    rows_raw = data.get("classes")
    if not isinstance(rows_raw, list) or not rows_raw:
        raise BomError("BOM needs a non-empty classes list")
    rows = []
    for i, item in enumerate(rows_raw):
        if not isinstance(item, dict):
            raise BomError(f"classes[{i}] must be a mapping")
        _forbid_prices(item, where=f"classes[{i}]")
        rows.append(_row_from_mapping(item, index=i))
    measured = bool(data.get("measured", False))
    if measured and any(r.number_status != "measured" for r in rows):
        raise BomError(
            "bom.measured: true requires every class number_status=measured "
            "(hang / bench first)"
        )
    return BomFreeze(
        schema=schema,
        source=str(data.get("source") or ""),
        measured=measured,
        note=str(data.get("note") or ""),
        rows=rows,
        path=target,
    )

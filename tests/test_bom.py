"""BOM freeze: part classes only. No SKUs, no prices, no invented mass."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from jims_mower.bom import BOM_SCHEMA, BomError, load_bom

_ROOT = Path(__file__).resolve().parents[1]
_BOM = _ROOT / "configs" / "hardware" / "bom.yaml"


def test_bom_freeze_loads() -> None:
    bom = load_bom(_BOM)
    assert bom.schema == BOM_SCHEMA
    assert bom.measured is False
    ids = [row.id for row in bom.rows]
    assert "chassis" in ids
    assert "pack" in ids
    assert "estop_paddle" in ids
    assert all(row.number_status in {"assumption", "measured", "class", "target"} for row in bom.rows)
    assert not any(row.number_status == "measured" for row in bom.rows)


def test_bom_forbids_sku_and_price() -> None:
    raw = yaml.safe_load(_BOM.read_text(encoding="utf-8"))
    raw["classes"][0]["sku"] = "FAKE-123"
    with pytest.raises(BomError, match="SKU"):
        from jims_mower.bom import _forbid_prices

        _forbid_prices(raw, where="bom")


def test_packaged_bom_matches_repo_ids() -> None:
    repo = load_bom(_BOM)
    packaged = load_bom()
    assert [r.id for r in repo.rows] == [r.id for r in packaged.rows]


def test_measured_bom_requires_every_row_measured(tmp_path: Path) -> None:
    raw = yaml.safe_load(_BOM.read_text(encoding="utf-8"))
    raw["measured"] = True
    path = tmp_path / "bom.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(BomError, match="measured"):
        load_bom(path)

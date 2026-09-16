"""Pack / thermal provenance. No invented Wh or acre runtime."""

from __future__ import annotations

from pathlib import Path

import pytest

from jims_mower.config import ConfigError, load_config
from jims_mower.pack import (
    GYM_STUB_CAPACITY_WH,
    LABEL_STUB,
    LABEL_TEMPLATE,
    battery_status_block,
    load_pack_template,
    meta_from_config,
    pack_template_path,
    remaining_wh,
    validate_pack_claim,
)
from jims_mower.runtime.budget import budget_from_config

_ROOT = Path(__file__).resolve().parents[1]
_TEMPLATE = _ROOT / "configs" / "orin" / "pack_measured.template.yaml"


def test_default_pack_is_gym_stub() -> None:
    cfg = load_config()
    meta = meta_from_config(cfg)
    assert meta.label == LABEL_STUB
    assert meta.measured is False
    assert meta.capacity_wh == pytest.approx(GYM_STUB_CAPACITY_WH)
    assert meta.charge_time_h is None
    assert remaining_wh(0.5, meta.capacity_wh) == pytest.approx(25.0)


def test_pack_template_is_placeholders() -> None:
    cfg = load_config(_TEMPLATE)
    assert cfg.runtime.battery.measured is True
    assert cfg.runtime.battery.template is True
    assert cfg.runtime.battery.capacity_wh is None
    assert cfg.runtime.battery.charge_time_h is None
    meta = meta_from_config(cfg)
    assert meta.label == LABEL_TEMPLATE
    validate_pack_claim(cfg.runtime.battery)
    with pytest.raises(Exception, match="template"):
        validate_pack_claim(cfg.runtime.battery, template_ok=False)


def test_packaged_template_matches_repo() -> None:
    packaged = load_pack_template()
    repo = load_config(_TEMPLATE)
    assert packaged.runtime.battery.template is True
    assert repo.runtime.battery.template is True
    assert pack_template_path().is_file()


def test_measured_without_charge_time_is_refused() -> None:
    with pytest.raises(ConfigError, match="charge_time_h"):
        load_config(
            {
                "runtime": {
                    "battery": {
                        "measured": True,
                        "capacity_wh": 200.0,
                        "measured_at": "2026-09-16",
                    }
                }
            }
        )


def test_measured_50wh_needs_notes() -> None:
    with pytest.raises(ConfigError, match="gym stub"):
        load_config(
            {
                "runtime": {
                    "battery": {
                        "measured": True,
                        "capacity_wh": 50.0,
                        "charge_time_h": 1.0,
                        "measured_at": "2026-09-16",
                    }
                }
            }
        )
    cfg = load_config(
        {
            "runtime": {
                "battery": {
                    "measured": True,
                    "capacity_wh": 50.0,
                    "charge_time_h": 1.0,
                    "measured_at": "2026-09-16",
                    "notes": "bench 50 Wh dummy pack, not the gym default",
                }
            }
        }
    )
    assert cfg.runtime.battery.measured is True


def test_budget_reads_configured_capacity() -> None:
    cfg = load_config(
        {
            "runtime": {
                "enabled": True,
                "battery": {
                    "measured": True,
                    "capacity_wh": 200.0,
                    "charge_time_h": 2.0,
                    "measured_at": "2026-01-01",
                    "notes": "unit-test fixture",
                    "soc": 1.0,
                },
            }
        }
    )
    budget = budget_from_config(cfg)
    assert budget.capacity_wh == pytest.approx(200.0)
    assert budget.measured is True
    assert budget.charge_time_h == pytest.approx(2.0)
    info = budget.as_info()
    assert info["capacity_wh"] == pytest.approx(200.0)
    assert info["pack_measured"] is True
    assert info["acre_runtime_h"] is None
    assert info["not_a_power_trace"] is True


def test_thermal_measured_requires_board_load() -> None:
    with pytest.raises(ConfigError, match="board_load_c"):
        load_config({"runtime": {"thermal": {"measured": True}}})


def test_status_block_never_claims_acre_runtime() -> None:
    block = battery_status_block(soc=0.4, temp_c=41.0)
    assert block["capacity_wh"] == pytest.approx(GYM_STUB_CAPACITY_WH)
    assert block["measured"] is False
    assert block["acre_runtime_h"] is None
    assert block["remaining_wh"] == pytest.approx(20.0)

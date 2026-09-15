"""Owner phone overlay: geofence + plan HTML stub."""

from __future__ import annotations

from pathlib import Path

from jims_mower.owner import export_owner_overlay
from jims_mower.owner_cli import main as owner_main


def test_owner_overlay_html(tmp_path: Path) -> None:
    dest = tmp_path / "overlay.html"
    payload = export_owner_overlay(dest, config="geofence_movers", seed=3, cameras=4)
    assert dest.is_file()
    assert dest.with_suffix(".json").is_file()
    html = dest.read_text(encoding="utf-8")
    assert "Yard overlay" in html
    assert "<svg" in html
    assert "geofence" in html.lower() or "keep" in html.lower() or "polygon" in html
    assert payload["not_a_benchmark"] is True
    assert payload["schema"].startswith("jims_mower.owner")
    assert len(payload.get("geofence", {}).get("keep_in") or []) >= 3


def test_owner_cli(tmp_path: Path) -> None:
    dest = tmp_path / "phone.html"
    owner_main(["--out", str(dest), "--config", "geofence_movers", "--cameras", "4", "--seed", "1"])
    assert dest.is_file()

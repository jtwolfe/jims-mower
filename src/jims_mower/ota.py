"""SAF-4 OTA — documented no-op stub.

UX-C mentions optional Wi-Fi for OTA. There is no updater, no signed
image, and no A/B slot in this repo. Do not pretend a pull happened.
"""

from __future__ import annotations

from typing import Any

from jims_mower.constants import OTA_SCHEMA


def ota_status() -> dict[str, Any]:
    return {
        "schema": OTA_SCHEMA,
        "available": False,
        "updater": "missing",
        "signed_image": False,
        "ab_slot": None,
        "last_attempt": None,
        "note": "SAF-4 honest no-op — no A/B slot, no signed image, no updater",
    }


def ota_apply(_image: str = "") -> dict[str, Any]:
    """Refuse to apply. Callers must not treat this as success."""
    status = ota_status()
    status["ok"] = False
    status["applied"] = False
    status["error"] = "OTA updater is a documented no-op stub"
    return status

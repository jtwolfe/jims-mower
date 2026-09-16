"""Multi-yard store (UX-3): switch YardProfiles without fence bleed.

One active profile at a time. Switching replaces keep-in / home /
schedule wholesale — yesterday's fence vertices must not leak.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

from jims_mower.constants import YARD_STORE_SCHEMA
from jims_mower.profile import ProfileError, YardProfile, parse_yard_profile, write_yard_profile


class YardStore:
    """Directory of ``<name>.json`` YardProfiles plus an active pointer."""

    def __init__(self, root: Union[str, Path]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._active: Optional[str] = None
        pointer = self.root / "active.json"
        if pointer.is_file():
            try:
                blob = json.loads(pointer.read_text(encoding="utf-8"))
                name = str((blob or {}).get("active") or "").strip()
                if name:
                    self._active = name
            except json.JSONDecodeError:
                self._active = None

    def _path(self, name: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name.strip())
        if not safe:
            raise ProfileError("yard name is empty")
        return self.root / f"{safe}.json"

    def put(self, profile: YardProfile) -> Path:
        dest = self._path(profile.name)
        write_yard_profile(dest, profile)
        if self._active is None:
            self.select(profile.name)
        return dest

    def get(self, name: str) -> YardProfile:
        path = self._path(name)
        if not path.is_file():
            raise ProfileError(f"yard not found: {name}")
        return parse_yard_profile(json.loads(path.read_text(encoding="utf-8")))

    def list(self) -> list[dict[str, Any]]:
        rows = []
        for path in sorted(self.root.glob("*.json")):
            if path.name == "active.json":
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                profile = parse_yard_profile(raw)
            except (json.JSONDecodeError, ProfileError, OSError):
                continue
            rows.append(
                {
                    "name": profile.name,
                    "active": profile.name == self._active,
                    "keep_in_vertices": len(profile.keep_in),
                    "home": dict(profile.home),
                    "origin": profile.origin.as_dict(),
                }
            )
        return rows

    def select(self, name: str) -> YardProfile:
        profile = self.get(name)
        self._active = profile.name
        pointer = self.root / "active.json"
        pointer.write_text(
            json.dumps({"schema": YARD_STORE_SCHEMA, "active": profile.name}, indent=2),
            encoding="utf-8",
        )
        return profile

    def active(self) -> Optional[YardProfile]:
        if not self._active:
            return None
        try:
            return self.get(self._active)
        except ProfileError:
            return None

    def as_info(self) -> dict[str, Any]:
        return {
            "schema": YARD_STORE_SCHEMA,
            "active": self._active,
            "yards": self.list(),
            "note": "switch replaces keep-in/home — no fence bleed",
        }

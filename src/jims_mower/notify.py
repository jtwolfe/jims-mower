"""Owner notifications (UX-2 / SCH-5) — in-app list + webhook stub.

Not SMS. Not a push vendor. A local channel the phone shell can poll,
plus an optional HTTP webhook that is a **stub** (never claims delivery
unless a caller records a test sink).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import urlparse

from jims_mower.constants import NOTIFY_SCHEMA

NOTIFY_KINDS = (
    "skip",
    "finish",
    "pause",
    "resume",
    "fault",
    "sos",
    "schedule",
    "info",
)


class NotificationLog:
    """In-memory + optional JSONL list of skip / finish reasons."""

    def __init__(self, dest: Optional[Union[str, Path]] = None, *, max_items: int = 200) -> None:
        self.path = Path(dest) if dest else None
        self.max_items = max(8, int(max_items))
        self._items: list[dict[str, Any]] = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.is_file():
                self._load()

    def emit(
        self,
        kind: str,
        reason: str,
        *,
        yard: str = "",
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        key = str(kind or "info").strip().lower()
        if key not in NOTIFY_KINDS:
            key = "info"
        item = {
            "schema": NOTIFY_SCHEMA,
            "kind": key,
            "reason": str(reason or ""),
            "yard": str(yard or ""),
            "ts": datetime.now(timezone.utc).isoformat(),
            "sms": False,
            "push": False,
            "channel": "in_app",
        }
        if extra:
            item["extra"] = dict(extra)
        self._items.append(item)
        if len(self._items) > self.max_items:
            self._items = self._items[-self.max_items :]
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(item) + "\n")
        return item

    def list(self, *, last_n: int = 50) -> list[dict[str, Any]]:
        n = max(0, int(last_n))
        return list(self._items[-n:] if n else self._items)

    def webhook_stub(self, url: str, item: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Do not POST. Record that a webhook was requested and return a stub."""
        parsed = urlparse(str(url or ""))
        return {
            "schema": NOTIFY_SCHEMA,
            "delivered": False,
            "stub": True,
            "url_host": parsed.hostname or "",
            "kind": (item or {}).get("kind"),
            "note": "webhook stub — no SMS / no vendor push",
        }

    def _load(self) -> None:
        assert self.path is not None
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                blob = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(blob, dict):
                rows.append(blob)
        self._items = rows[-self.max_items :]

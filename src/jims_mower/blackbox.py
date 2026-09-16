"""Always-on rotating incident black box (SAF-3 gym/bench path).

Not a flight-recorder certification. After a tip / immobilised event the
owner can retrieve recent IMU samples + wheel commands from disk.
OTA is a separate honest no-op (``jims_mower.ota``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

from jims_mower.constants import BLACKBOX_SCHEMA

DEFAULT_MAX_RECORDS = 4000
DEFAULT_ROTATE_BYTES = 1_000_000


class BlackBox:
    """Append-only JSONL, rotate when the file grows past ``rotate_bytes``."""

    def __init__(
        self,
        dest: Union[str, Path],
        *,
        max_records: int = DEFAULT_MAX_RECORDS,
        rotate_bytes: int = DEFAULT_ROTATE_BYTES,
    ) -> None:
        self.path = Path(dest)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_records = max(32, int(max_records))
        self.rotate_bytes = max(1024, int(rotate_bytes))
        self._count = 0
        if not self.path.is_file():
            self.path.write_text("", encoding="utf-8")

    def record(
        self,
        *,
        step: int,
        imu: Any,
        cmd: Any,
        pose: Optional[dict[str, Any]] = None,
        advice: str = "ok",
        event: str = "",
        fault: Optional[dict[str, Any]] = None,
    ) -> None:
        row = {
            "schema": BLACKBOX_SCHEMA,
            "step": int(step),
            "imu": _as_list(imu),
            "cmd": _as_list(cmd),
            "pose": dict(pose) if pose else {},
            "advice": str(advice or "ok"),
            "event": str(event or ""),
            "fault": dict(fault) if fault else {},
        }
        line = json.dumps(row, separators=(",", ":")) + "\n"
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line)
        self._count += 1
        if self.path.stat().st_size >= self.rotate_bytes or self._count >= self.max_records:
            self._rotate()

    def _rotate(self) -> None:
        prev = self.path.with_suffix(self.path.suffix + ".1")
        if prev.is_file():
            prev.unlink()
        if self.path.is_file():
            self.path.replace(prev)
        self.path.write_text("", encoding="utf-8")
        self._count = 0

    def retrieve(
        self,
        *,
        event: str = "",
        last_n: int = 64,
    ) -> list[dict[str, Any]]:
        """Return recent records, optionally filtered by ``event`` (e.g. tip)."""
        rows = list(self._iter_files())
        if event:
            key = str(event).strip().lower()
            rows = [r for r in rows if str(r.get("event") or "").lower() == key]
        if last_n > 0:
            rows = rows[-int(last_n) :]
        return rows

    def retrieve_sos(self, *, last_n: int = 64) -> list[dict[str, Any]]:
        """IMU + cmds around tip / immobilised — the SAF-2 retrieve path."""
        rows = list(self._iter_files())
        keep = []
        for row in rows:
            ev = str(row.get("event") or "").lower()
            code = str((row.get("fault") or {}).get("code") or "").lower()
            if ev in {"tip", "tipover", "immobilised", "sos"} or "immobil" in code:
                keep.append(row)
        if not keep:
            keep = rows
        return keep[-int(last_n) :]

    def _iter_files(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        prev = self.path.with_suffix(self.path.suffix + ".1")
        for path in (prev, self.path):
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    blob = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(blob, dict):
                    rows.append(blob)
        return rows


def _as_list(value: Any) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return [float(x) for x in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    return [float(value)]

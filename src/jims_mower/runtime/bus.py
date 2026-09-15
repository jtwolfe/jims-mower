"""In-process queues for versioned runtime-contract messages."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from jims_mower.contract import MESSAGE_KINDS, validate_payload

DEFAULT_MAXLEN = 32


@dataclass
class BusMessage:
    """One contract payload plus the bus it arrived on."""

    kind: str
    payload: dict[str, Any]
    bus: str = "inproc"
    topic: str = ""

    def validate(self) -> dict[str, Any]:
        return validate_payload(self.kind, self.payload)


class InProcessBus:
    """Named deques of ``BusMessage``. No sockets, no extra deps.

    Topics default to the contract kinds (``ImuSample``, ``GpsFix``, …).
    ``publish`` validates against ``jims_mower.contract`` before enqueue.
    """

    def __init__(
        self,
        *,
        maxlen: int = DEFAULT_MAXLEN,
        topics: Optional[Iterable[str]] = None,
    ) -> None:
        names = tuple(topics) if topics is not None else MESSAGE_KINDS
        self.maxlen = int(maxlen)
        self._queues: dict[str, deque[BusMessage]] = {
            name: deque(maxlen=self.maxlen) for name in names
        }

    def topics(self) -> list[str]:
        return list(self._queues)

    def publish(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        bus: str = "inproc",
        topic: Optional[str] = None,
    ) -> BusMessage:
        checked = validate_payload(kind, payload)
        dest = topic or kind
        if dest not in self._queues:
            self._queues[dest] = deque(maxlen=self.maxlen)
        msg = BusMessage(kind=kind, payload=checked, bus=bus, topic=dest)
        self._queues[dest].append(msg)
        return msg

    def get(self, topic: str) -> Optional[BusMessage]:
        q = self._queues.get(topic)
        if not q:
            return None
        return q.popleft()

    def drain(self, topic: Optional[str] = None) -> list[BusMessage]:
        if topic is None:
            out: list[BusMessage] = []
            for name in list(self._queues):
                out.extend(self.drain(name))
            return out
        q = self._queues.get(topic)
        if not q:
            return []
        items = list(q)
        q.clear()
        return items

    def qsize(self, topic: str) -> int:
        q = self._queues.get(topic)
        return 0 if q is None else len(q)

"""Optional hand-signal curriculum labels (stop / go / follow / back)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from jims_mower.constants import HAND_SIGNALS
from jims_mower.types import Obstacle


class HandSignalCurriculum:
    """Assigns a held signal to each person. Disabled → all None."""

    def __init__(self, enabled: bool, hold_steps: int = 40) -> None:
        self.enabled = bool(enabled)
        self.hold_steps = max(1, int(hold_steps))
        self._age = 0

    def assign(self, obstacles: list[Obstacle], rng: np.random.Generator) -> None:
        self._age = 0
        if not self.enabled:
            for obst in obstacles:
                obst.hand_signal = None
            return
        for obst in obstacles:
            if obst.kind == "person":
                obst.hand_signal = str(rng.choice(HAND_SIGNALS))
            else:
                obst.hand_signal = None

    def maybe_rotate(self, obstacles: list[Obstacle], rng: np.random.Generator) -> None:
        if not self.enabled:
            return
        self._age += 1
        if self._age < self.hold_steps:
            return
        self.assign(obstacles, rng)

    @staticmethod
    def nearest_person_signal(obstacles: list[Obstacle], xy: tuple[float, float]) -> Optional[str]:
        best = None
        best_d = float("inf")
        for obst in obstacles:
            if obst.kind != "person" or not obst.hand_signal:
                continue
            d = (obst.x - xy[0]) ** 2 + (obst.y - xy[1]) ** 2
            if d < best_d:
                best_d = d
                best = obst.hand_signal
        return best

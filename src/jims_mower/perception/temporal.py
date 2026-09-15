"""Temporal filters: hazard hysteresis / decay, and moving-object tracklets.

Sim stubs. Hysteresis stops one-frame flashes from blocking the planner;
decay forgets cells that cameras stop confirming. Tracklets associate
person/dog detections across frames by world XY — not a published MOT
score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from jims_mower.constants import LIVING_KINDS
from jims_mower.types import Detection

TRACKED_LABELS = frozenset({"person", "dog"})


class HazardHysteresis:
    """Per-cell evidence: confirm after hits, decay when unseen."""

    def __init__(
        self,
        *,
        confirm: float = 0.90,
        decay: float = 0.62,
        forget: float = 0.20,
        hit_boost: float = 1.0,
        shape: Optional[tuple[int, int]] = None,
    ) -> None:
        self.confirm = float(confirm)
        self.decay = float(decay)
        self.forget = float(forget)
        self.hit_boost = float(hit_boost)
        self._evidence: Optional[np.ndarray] = None
        self._label: Optional[np.ndarray] = None
        self._latched: Optional[np.ndarray] = None
        if shape is not None:
            self._ensure(shape)

    def reset(self) -> None:
        self._evidence = None
        self._label = None
        self._latched = None

    def _ensure(self, shape: tuple[int, int]) -> None:
        if self._evidence is None or self._evidence.shape != shape:
            self._evidence = np.zeros(shape, dtype=np.float32)
            self._label = np.zeros(shape, dtype=np.float32)
            self._latched = np.zeros(shape, dtype=bool)

    def update(
        self,
        incoming: np.ndarray,
        confidence: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Blend a new hazard raster. Returns the filtered map (copy)."""
        haz = np.asarray(incoming, dtype=np.float32)
        self._ensure(haz.shape)
        assert self._evidence is not None and self._label is not None
        if confidence is None:
            conf = (haz > 0).astype(np.float32)
        else:
            conf = np.asarray(confidence, dtype=np.float32)
            if conf.shape != haz.shape:
                raise ValueError("confidence must match incoming hazard")
        assert self._latched is not None
        seen = (haz > 0) & (conf > 0)
        self._evidence *= np.float32(self.decay)
        boost = np.where(seen, np.maximum(conf, 0.0) * self.hit_boost, 0.0)
        self._evidence += boost.astype(np.float32)
        self._label = np.where(seen, np.maximum(self._label, haz), self._label)
        self._latched = self._latched | (self._evidence >= self.confirm)
        keep = self._evidence >= self.forget
        # Once latched, hold until evidence decays below forget.
        emit = self._latched & keep
        out = np.where(emit, self._label, 0.0).astype(np.float32)
        self._label = np.where(keep, self._label, 0.0)
        self._evidence = np.where(keep, self._evidence, 0.0)
        self._latched = self._latched & keep
        return out.copy()


@dataclass
class Tracklet:
    track_id: int
    label: str
    x: float
    y: float
    hits: int
    missed: int
    confidence: float
    camera: str = ""

    def as_dict(self) -> dict:
        return {
            "id": self.track_id,
            "label": self.label,
            "x": self.x,
            "y": self.y,
            "hits": self.hits,
            "missed": self.missed,
            "confidence": self.confidence,
            "camera": self.camera,
        }


@dataclass
class DetectionTracklets:
    """Greedy nearest-neighbour association for person/dog detections."""

    max_dist_m: float = 1.4
    max_missed: int = 6
    _next_id: int = 1
    tracks: list[Tracklet] = field(default_factory=list)

    def reset(self) -> None:
        self.tracks = []
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Tracklet]:
        candidates: list[Detection] = []
        for det in detections:
            if det.label not in TRACKED_LABELS and det.label not in LIVING_KINDS:
                continue
            if det.label not in TRACKED_LABELS:
                continue
            if det.world_xy is None:
                continue
            candidates.append(det)

        assigned: set[int] = set()
        used_tracks: set[int] = set()
        # Greedy: strongest detections first.
        ordered = sorted(candidates, key=lambda d: -float(d.confidence))
        for det in ordered:
            assert det.world_xy is not None
            dx, dy = det.world_xy
            best_i = -1
            best_d = self.max_dist_m
            for i, tr in enumerate(self.tracks):
                if i in used_tracks or tr.label != det.label:
                    continue
                dist = float(np.hypot(tr.x - dx, tr.y - dy))
                if dist < best_d:
                    best_d = dist
                    best_i = i
            if best_i >= 0:
                tr = self.tracks[best_i]
                tr.x, tr.y = float(dx), float(dy)
                tr.hits += 1
                tr.missed = 0
                tr.confidence = float(det.confidence)
                tr.camera = det.camera
                used_tracks.add(best_i)
                assigned.add(id(det))
            else:
                self.tracks.append(
                    Tracklet(
                        track_id=self._next_id,
                        label=det.label,
                        x=float(dx),
                        y=float(dy),
                        hits=1,
                        missed=0,
                        confidence=float(det.confidence),
                        camera=det.camera,
                    )
                )
                self._next_id += 1
                assigned.add(id(det))

        for i, tr in enumerate(self.tracks):
            if i not in used_tracks:
                tr.missed += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return list(self.tracks)

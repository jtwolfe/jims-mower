"""Dataclasses shared across the gym."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    theta: float


@dataclass(frozen=True)
class CameraSpec:
    name: str
    x: float
    y: float
    z: float
    yaw_deg: float
    pitch_deg: float
    fov_deg: float = 70.0


@dataclass
class Obstacle:
    kind: str
    x: float
    y: float
    radius: float
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    heading: float = 0.0
    hand_signal: Optional[str] = None
    name: str = ""

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class Detection:
    label: str
    camera: str
    bbox: tuple[int, int, int, int]
    confidence: float
    world_xy: Optional[tuple[float, float]] = None
    hand_signal: Optional[str] = None
    category: str = ""
    depth_m: float = 0.0

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "camera": self.camera,
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "world_xy": list(self.world_xy) if self.world_xy is not None else None,
            "hand_signal": self.hand_signal,
            "category": self.category,
            "depth_m": self.depth_m,
        }


@dataclass
class PerceptionContext:
    pose: Pose
    cameras: list[CameraSpec]
    obstacles: list[Obstacle] = field(default_factory=list)
    image_size: tuple[int, int] = (80, 60)
    hand_signals_enabled: bool = False

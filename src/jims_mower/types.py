"""Dataclasses shared across the gym."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from jims_mower.constants import CUTTER_RISK_KINDS, SOFT_KINDS


@dataclass
class Trajectory:
    """Simple living-agent path. ``wander`` is the random-walk default."""

    mode: str = "wander"
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    speed_mps: float = 0.0
    index: int = 0
    direction: int = 1


@dataclass(frozen=True)
class Pose:
    """Planar yaw plus contact attitude on the height field.

    ``z`` is the mean wheel-contact elevation. ``pitch`` is nose-up about
    body +y; ``roll`` is left-side-up about body +x. On flat ground both
    are zero and this reduces to ``(x, y, theta)``.
    """

    x: float
    y: float
    theta: float
    z: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


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
    length_m: float = 0.0
    soft: bool = False
    cutter_risk: bool = False
    trajectory: Optional[Trajectory] = None

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)

    @property
    def visual_z(self) -> float:
        """Height used for the projected blob. Birds stay airborne."""
        if self.kind == "bird":
            return self.z
        return 0.5 * self.z

    @property
    def is_soft(self) -> bool:
        return self.soft or self.kind in SOFT_KINDS

    @property
    def is_cutter_risk(self) -> bool:
        return self.cutter_risk or self.kind in CUTTER_RISK_KINDS

    def segment_ends(self) -> tuple[float, float, float, float]:
        """World-XY endpoints for elongated clutter (hose / cord)."""
        import math

        half = 0.5 * max(self.length_m, 0.0)
        if half <= 1e-6:
            return self.x, self.y, self.x, self.y
        dx = half * math.cos(self.heading)
        dy = half * math.sin(self.heading)
        return self.x - dx, self.y - dy, self.x + dx, self.y + dy


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
    imu: Optional[object] = None
    gps: Optional[object] = None
    terrain: Optional[object] = None
    map_shape: tuple[int, int] = (1, 1)
    resolution_m: float = 0.10
    world_size: tuple[float, float] = (12.0, 12.0)
    steep_slope_rad: float = 0.30
    tof: Optional[object] = None
    length_m: float = 0.50
    track_m: float = 0.40
    chassis_hover_m: float = 0.06
    hand_signal_classifier: bool = False

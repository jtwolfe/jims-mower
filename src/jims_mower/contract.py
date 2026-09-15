"""Runtime message contract (versioned dataclasses + JSON Schema).

These types are the on-box / log interchange for camera, IMU, GNSS, ToF,
detections, terrain maps, plans, and wheel commands. The gym observation
dict is a flattened form of the same fields. No extra JSON Schema library:
``validate_payload`` checks ``version`` plus required keys.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional

CONTRACT_VERSION = "1"

MESSAGE_KINDS = (
    "CameraFrame",
    "ImuSample",
    "GpsFix",
    "TofArray",
    "DetectionSet",
    "TerrainMaps",
    "Plan",
    "WheelCommand",
)


def _require_version(payload: dict[str, Any]) -> str:
    raw = payload.get("version")
    if raw is None:
        raise ValueError("message missing version")
    return str(raw)


@dataclass
class CameraFrame:
    name: str
    width: int
    height: int
    encoding: str = "rgb8"
    stamp_s: float = 0.0
    frame_id: str = "camera"
    version: str = CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraFrame":
        _require_version(data)
        return cls(
            name=str(data["name"]),
            width=int(data["width"]),
            height=int(data["height"]),
            encoding=str(data.get("encoding", "rgb8")),
            stamp_s=float(data.get("stamp_s", 0.0)),
            frame_id=str(data.get("frame_id", "camera")),
            version=str(data.get("version", CONTRACT_VERSION)),
        )


@dataclass
class ImuSample:
    accel_mps2: tuple[float, float, float]
    gyro_radps: tuple[float, float, float]
    stamp_s: float = 0.0
    frame_id: str = "imu"
    version: str = CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImuSample":
        _require_version(data)
        accel = tuple(float(v) for v in data["accel_mps2"])
        gyro = tuple(float(v) for v in data["gyro_radps"])
        if len(accel) != 3 or len(gyro) != 3:
            raise ValueError("ImuSample accel/gyro must be length 3")
        return cls(
            accel_mps2=(accel[0], accel[1], accel[2]),
            gyro_radps=(gyro[0], gyro[1], gyro[2]),
            stamp_s=float(data.get("stamp_s", 0.0)),
            frame_id=str(data.get("frame_id", "imu")),
            version=str(data.get("version", CONTRACT_VERSION)),
        )


@dataclass
class GpsFix:
    x: float
    y: float
    z: float = 0.0
    valid: float = 1.0
    stamp_s: float = 0.0
    frame_id: str = "gps"
    version: str = CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GpsFix":
        _require_version(data)
        return cls(
            x=float(data["x"]),
            y=float(data["y"]),
            z=float(data.get("z", 0.0)),
            valid=float(data.get("valid", 1.0)),
            stamp_s=float(data.get("stamp_s", 0.0)),
            frame_id=str(data.get("frame_id", "gps")),
            version=str(data.get("version", CONTRACT_VERSION)),
        )


@dataclass
class TofArray:
    ranges_m: tuple[float, float, float, float]
    stamp_s: float = 0.0
    frame_id: str = "tof"
    version: str = CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TofArray":
        _require_version(data)
        ranges = tuple(float(v) for v in data["ranges_m"])
        if len(ranges) != 4:
            raise ValueError("TofArray.ranges_m must be FL, FR, RL, RR")
        return cls(
            ranges_m=(ranges[0], ranges[1], ranges[2], ranges[3]),
            stamp_s=float(data.get("stamp_s", 0.0)),
            frame_id=str(data.get("frame_id", "tof")),
            version=str(data.get("version", CONTRACT_VERSION)),
        )


@dataclass
class DetectionRecord:
    label: str
    camera: str
    bbox: tuple[int, int, int, int]
    confidence: float
    world_xy: Optional[tuple[float, float]] = None
    hand_signal: Optional[str] = None
    category: str = ""
    depth_m: float = 0.0


@dataclass
class DetectionSet:
    detections: list[DetectionRecord] = field(default_factory=list)
    stamp_s: float = 0.0
    version: str = CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DetectionSet":
        _require_version(data)
        dets: list[DetectionRecord] = []
        for item in data.get("detections") or []:
            bbox = tuple(int(v) for v in item["bbox"])
            if len(bbox) != 4:
                raise ValueError("detection bbox must be [u, v, w, h]")
            world = item.get("world_xy")
            dets.append(
                DetectionRecord(
                    label=str(item["label"]),
                    camera=str(item["camera"]),
                    bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    confidence=float(item["confidence"]),
                    world_xy=(float(world[0]), float(world[1])) if world else None,
                    hand_signal=item.get("hand_signal"),
                    category=str(item.get("category") or ""),
                    depth_m=float(item.get("depth_m") or 0.0),
                )
            )
        return cls(
            detections=dets,
            stamp_s=float(data.get("stamp_s", 0.0)),
            version=str(data.get("version", CONTRACT_VERSION)),
        )


@dataclass
class TerrainMaps:
    elevation: str = "elevation"
    slope: str = "slope"
    hazard: str = "hazard"
    confidence: str = "confidence"
    resolution_m: float = 0.10
    width_m: float = 12.0
    height_m: float = 12.0
    source: str = "heuristic"
    stamp_s: float = 0.0
    version: str = CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TerrainMaps":
        _require_version(data)
        return cls(
            elevation=str(data.get("elevation", "elevation")),
            slope=str(data.get("slope", "slope")),
            hazard=str(data.get("hazard", "hazard")),
            confidence=str(data.get("confidence", "confidence")),
            resolution_m=float(data.get("resolution_m", 0.10)),
            width_m=float(data.get("width_m", 12.0)),
            height_m=float(data.get("height_m", 12.0)),
            source=str(data.get("source", "heuristic")),
            stamp_s=float(data.get("stamp_s", 0.0)),
            version=str(data.get("version", CONTRACT_VERSION)),
        )


@dataclass
class Plan:
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    index: int = 0
    n_segments: int = 0
    advice: str = "ok"
    stamp_s: float = 0.0
    version: str = CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Plan":
        _require_version(data)
        wps = []
        for item in data.get("waypoints") or []:
            if isinstance(item, dict):
                wps.append((float(item["x"]), float(item["y"])))
            else:
                wps.append((float(item[0]), float(item[1])))
        return cls(
            waypoints=wps,
            index=int(data.get("index", 0)),
            n_segments=int(data.get("n_segments", 0)),
            advice=str(data.get("advice", "ok")),
            stamp_s=float(data.get("stamp_s", 0.0)),
            version=str(data.get("version", CONTRACT_VERSION)),
        )


@dataclass
class WheelCommand:
    left: float
    right: float
    trimmer: float = 0.0
    stamp_s: float = 0.0
    version: str = CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WheelCommand":
        _require_version(data)
        return cls(
            left=float(data["left"]),
            right=float(data["right"]),
            trimmer=float(data.get("trimmer", 0.0)),
            stamp_s=float(data.get("stamp_s", 0.0)),
            version=str(data.get("version", CONTRACT_VERSION)),
        )


_SCHEMA_COMMON = {
    "version": {"type": "string"},
    "stamp_s": {"type": "number"},
}

JSON_SCHEMAS: dict[str, dict[str, Any]] = {
    "CameraFrame": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "CameraFrame",
        "type": "object",
        "required": ["version", "name", "width", "height"],
        "properties": {
            **_SCHEMA_COMMON,
            "name": {"type": "string"},
            "width": {"type": "integer", "minimum": 1},
            "height": {"type": "integer", "minimum": 1},
            "encoding": {"type": "string"},
            "frame_id": {"type": "string"},
        },
    },
    "ImuSample": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ImuSample",
        "type": "object",
        "required": ["version", "accel_mps2", "gyro_radps"],
        "properties": {
            **_SCHEMA_COMMON,
            "accel_mps2": {"type": "array", "minItems": 3, "maxItems": 3},
            "gyro_radps": {"type": "array", "minItems": 3, "maxItems": 3},
            "frame_id": {"type": "string"},
        },
    },
    "GpsFix": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "GpsFix",
        "type": "object",
        "required": ["version", "x", "y"],
        "properties": {
            **_SCHEMA_COMMON,
            "x": {"type": "number"},
            "y": {"type": "number"},
            "z": {"type": "number"},
            "valid": {"type": "number", "minimum": 0, "maximum": 1},
            "frame_id": {"type": "string"},
        },
    },
    "TofArray": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "TofArray",
        "type": "object",
        "required": ["version", "ranges_m"],
        "properties": {
            **_SCHEMA_COMMON,
            "ranges_m": {"type": "array", "minItems": 4, "maxItems": 4},
            "frame_id": {"type": "string"},
        },
    },
    "DetectionSet": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "DetectionSet",
        "type": "object",
        "required": ["version", "detections"],
        "properties": {
            **_SCHEMA_COMMON,
            "detections": {"type": "array"},
        },
    },
    "TerrainMaps": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "TerrainMaps",
        "type": "object",
        "required": ["version", "resolution_m"],
        "properties": {
            **_SCHEMA_COMMON,
            "elevation": {"type": "string"},
            "slope": {"type": "string"},
            "hazard": {"type": "string"},
            "confidence": {"type": "string"},
            "resolution_m": {"type": "number"},
            "width_m": {"type": "number"},
            "height_m": {"type": "number"},
            "source": {"type": "string"},
        },
    },
    "Plan": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Plan",
        "type": "object",
        "required": ["version", "waypoints"],
        "properties": {
            **_SCHEMA_COMMON,
            "waypoints": {"type": "array"},
            "index": {"type": "integer"},
            "n_segments": {"type": "integer"},
            "advice": {"type": "string"},
        },
    },
    "WheelCommand": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "WheelCommand",
        "type": "object",
        "required": ["version", "left", "right"],
        "properties": {
            **_SCHEMA_COMMON,
            "left": {"type": "number"},
            "right": {"type": "number"},
            "trimmer": {"type": "number"},
        },
    },
}

_LOADERS = {
    "CameraFrame": CameraFrame.from_dict,
    "ImuSample": ImuSample.from_dict,
    "GpsFix": GpsFix.from_dict,
    "TofArray": TofArray.from_dict,
    "DetectionSet": DetectionSet.from_dict,
    "TerrainMaps": TerrainMaps.from_dict,
    "Plan": Plan.from_dict,
    "WheelCommand": WheelCommand.from_dict,
}


def schema_for(kind: str) -> dict[str, Any]:
    if kind not in JSON_SCHEMAS:
        raise KeyError(f"unknown contract kind {kind!r}")
    return JSON_SCHEMAS[kind]


def validate_payload(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Required-field check against the JSON Schema (no third-party validator)."""
    schema = schema_for(kind)
    if not isinstance(payload, dict):
        raise ValueError(f"{kind} payload must be a mapping")
    missing = [key for key in schema.get("required", []) if key not in payload]
    if missing:
        raise ValueError(f"{kind} missing fields: {missing}")
    _require_version(payload)
    return _LOADERS[kind](payload).to_dict()


def contract_field_names(kind: str) -> list[str]:
    cls = {
        "CameraFrame": CameraFrame,
        "ImuSample": ImuSample,
        "GpsFix": GpsFix,
        "TofArray": TofArray,
        "DetectionSet": DetectionSet,
        "TerrainMaps": TerrainMaps,
        "Plan": Plan,
        "WheelCommand": WheelCommand,
    }[kind]
    return [f.name for f in fields(cls)]

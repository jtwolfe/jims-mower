"""Unified fault injection for the gym / runtime (WAVE UX-B).

``FaultBus`` is the single place that can kill a drive motor, jam the
trimmer, blank cameras, freeze the IMU, or force a GNSS dropout.

A **dead drive motor** latches ``FAULT_IMMOBILISED``: both wheels and the
trimmer go to zero so a live wheel cannot drag the chassis across a
drain. That is distinct from **stuck** (lip / channel), which still
runs reverse → pivot → call-for-help.

No claimed SIL rating. Software contract only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np

from jims_mower.constants import (
    FAULT_CODES,
    FAULT_SCHEMA,
    MOTOR_KILL_MODES,
)

# Components the bus can name in info["fault"].
DRIVE_LEFT = "drive_left"
DRIVE_RIGHT = "drive_right"
TRIMMER = "trimmer"
CAMERA = "camera"
IMU = "imu"
GNSS = "gnss"
CHASSIS = "chassis"

CODE_OK = "ok"
CODE_STUCK = "STUCK"
CODE_IMMOBILISED = "FAULT_IMMOBILISED"
CODE_TRIMMER_JAM = "TRIMMER_JAM"
CODE_CAM_BLIND = "CAM_BLIND"
CODE_IMU_FREEZE = "IMU_FREEZE"
CODE_GNSS_DROPOUT = "GNSS_DROPOUT"
CODE_RADIO_LOSS = "RADIO_LOSS"

_KIND_ALIASES = {
    "motor_left": DRIVE_LEFT,
    "left": DRIVE_LEFT,
    "drive_left": DRIVE_LEFT,
    "motor_right": DRIVE_RIGHT,
    "right": DRIVE_RIGHT,
    "drive_right": DRIVE_RIGHT,
    "trimmer": TRIMMER,
    "trimmer_jam": TRIMMER,
    "cam_blind": CAMERA,
    "camera": CAMERA,
    "cameras": CAMERA,
    "imu": IMU,
    "imu_freeze": IMU,
    "gnss": GNSS,
    "gps": GNSS,
    "gnss_dropout": GNSS,
    "stuck": CHASSIS,
    "radio": "radio",
    "radio_loss": "radio",
}


def normalize_component(kind: str) -> str:
    key = str(kind or "").strip().lower()
    if key not in _KIND_ALIASES:
        raise ValueError(
            f"unknown fault component {kind!r}; expected one of {sorted(_KIND_ALIASES)}"
        )
    return _KIND_ALIASES[key]


def normalize_motor_mode(mode: str) -> str:
    key = str(mode or "open_circuit").strip().lower()
    if key not in MOTOR_KILL_MODES:
        raise ValueError(
            f"motor kill mode must be one of {sorted(MOTOR_KILL_MODES)}; got {mode!r}"
        )
    return key


def pose_dict(pose: Any) -> dict[str, float]:
    if pose is None:
        return {"x": 0.0, "y": 0.0, "theta": 0.0, "z": 0.0, "pitch": 0.0, "roll": 0.0}
    if isinstance(pose, dict):
        return {
            "x": float(pose.get("x", 0.0)),
            "y": float(pose.get("y", 0.0)),
            "theta": float(pose.get("theta", 0.0)),
            "z": float(pose.get("z", 0.0)),
            "pitch": float(pose.get("pitch", 0.0)),
            "roll": float(pose.get("roll", 0.0)),
        }
    return {
        "x": float(getattr(pose, "x", 0.0)),
        "y": float(getattr(pose, "y", 0.0)),
        "theta": float(getattr(pose, "theta", 0.0)),
        "z": float(getattr(pose, "z", 0.0)),
        "pitch": float(getattr(pose, "pitch", 0.0)),
        "roll": float(getattr(pose, "roll", 0.0)),
    }


@dataclass
class FaultReport:
    """Structured ``info["fault"]`` payload (SOS / retrieve UX hook)."""

    code: str = CODE_OK
    component: Optional[str] = None
    pose: dict[str, float] = field(default_factory=pose_dict)
    retrieve: bool = False
    mode: Optional[str] = None
    reason: Optional[str] = None
    schema: str = FAULT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "code": self.code,
            "component": self.component,
            "pose": dict(self.pose),
            "retrieve": bool(self.retrieve),
            "mode": self.mode,
            "reason": self.reason,
        }


@dataclass
class ScheduledFault:
    at_step: int
    component: str
    mode: str = "open_circuit"
    cameras: tuple[str, ...] = ()


class FaultBus:
    """Gym / runtime fault injector. Off until something is scheduled or injected."""

    def __init__(self) -> None:
        self._schedule: list[ScheduledFault] = []
        self._left_dead: Optional[str] = None
        self._right_dead: Optional[str] = None
        self._trimmer_jam = False
        self._cam_blind: Optional[tuple[str, ...]] = None  # () = all cameras
        self._imu_freeze = False
        self._gnss_dropout = False
        self._stuck = False
        self._radio_loss: Optional[str] = None
        self._frozen_imu: Optional[np.ndarray] = None
        self._encoder_left = 0.0
        self._encoder_right = 0.0
        self._step = 0
        self.last_applied = (0.0, 0.0, 0.0)

    @classmethod
    def from_config(cls, cfg: Any) -> "FaultBus":
        bus = cls()
        faults = getattr(cfg, "faults", None)
        if faults is None:
            return bus
        if not bool(getattr(faults, "enabled", False)):
            return bus
        for item in getattr(faults, "inject", None) or []:
            bus.schedule_item(item)
        return bus

    def reset(self) -> None:
        schedule = list(self._schedule)
        self.__init__()
        self._schedule = schedule

    def clear(self) -> None:
        self._schedule.clear()
        self.reset()

    def schedule_item(self, item: Any) -> None:
        if isinstance(item, ScheduledFault):
            self._schedule.append(item)
            return
        if not isinstance(item, dict):
            raise ValueError("fault inject item must be a mapping")
        kind = item.get("kind") or item.get("component") or item.get("code")
        cameras = item.get("cameras") or ()
        if isinstance(cameras, str):
            cameras = (cameras,)
        self._schedule.append(
            ScheduledFault(
                at_step=int(item.get("at_step", 0)),
                component=normalize_component(str(kind)),
                mode=str(item.get("mode") or "open_circuit"),
                cameras=tuple(str(c) for c in cameras),
            )
        )

    def inject(
        self,
        kind: str,
        *,
        mode: str = "open_circuit",
        cameras: Optional[Iterable[str]] = None,
        at_step: Optional[int] = None,
    ) -> None:
        """Inject now, or schedule for ``at_step`` (env step index)."""
        component = normalize_component(kind)
        cam_tuple = tuple(str(c) for c in cameras) if cameras is not None else ()
        if at_step is not None:
            self._schedule.append(
                ScheduledFault(
                    at_step=int(at_step),
                    component=component,
                    mode=mode,
                    cameras=cam_tuple,
                )
            )
            return
        self._activate(component, mode, cam_tuple)

    def _activate(self, component: str, mode: str, cameras: tuple[str, ...]) -> None:
        if component in {DRIVE_LEFT, DRIVE_RIGHT}:
            mode = normalize_motor_mode(mode)
            if component == DRIVE_LEFT:
                self._left_dead = mode
            else:
                self._right_dead = mode
            return
        if component == TRIMMER:
            self._trimmer_jam = True
            return
        if component == CAMERA:
            self._cam_blind = cameras
            return
        if component == IMU:
            self._imu_freeze = True
            return
        if component == GNSS:
            self._gnss_dropout = True
            return
        if component == CHASSIS:
            self._stuck = True
            return
        if component == "radio":
            self._radio_loss = str(mode or "stop_beacon")
            return
        raise ValueError(f"cannot activate component {component!r}")

    def tick(self, step: int) -> None:
        self._step = int(step)
        for item in self._schedule:
            if item.at_step == self._step:
                self._activate(item.component, item.mode, item.cameras)

    @property
    def left_dead(self) -> bool:
        return self._left_dead is not None

    @property
    def right_dead(self) -> bool:
        return self._right_dead is not None

    @property
    def drive_dead(self) -> bool:
        return self.left_dead or self.right_dead

    @property
    def immobilised(self) -> bool:
        return self.drive_dead

    @property
    def stuck(self) -> bool:
        return bool(self._stuck) and not self.immobilised

    @property
    def trimmer_jammed(self) -> bool:
        return bool(self._trimmer_jam)

    @property
    def imu_frozen(self) -> bool:
        return bool(self._imu_freeze)

    @property
    def gnss_forced_dropout(self) -> bool:
        return bool(self._gnss_dropout)

    @property
    def cam_blind(self) -> bool:
        return self._cam_blind is not None

    @property
    def retrieve(self) -> bool:
        return self.immobilised or (
            self._radio_loss is not None and self._radio_loss != "limp_home"
        )

    def apply_drive(self, left: float, right: float, requested: bool) -> tuple[float, float, bool]:
        """Filter a wheel / trimmer command. Dead motor → both wheels + trimmer zero."""
        left_cmd = float(left)
        right_cmd = float(right)
        trim = bool(requested)
        if self._left_dead is not None:
            if self._left_dead != "encoder_stuck":
                self._encoder_left = 0.0
            left_cmd = 0.0
        else:
            self._encoder_left = float(left)
        if self._right_dead is not None:
            if self._right_dead != "encoder_stuck":
                self._encoder_right = 0.0
            right_cmd = 0.0
        else:
            self._encoder_right = float(right)
        if self._trimmer_jam:
            trim = False
        if self.immobilised:
            # Do not drag on the live wheel — zero the chassis.
            left_cmd = 0.0
            right_cmd = 0.0
            trim = False
        self.last_applied = (left_cmd, right_cmd, 1.0 if trim else 0.0)
        return left_cmd, right_cmd, trim

    def apply_cameras(self, images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._cam_blind is None:
            return images
        names = set(self._cam_blind)
        out = {}
        for name, frame in images.items():
            arr = np.asarray(frame)
            if not names or name in names:
                out[name] = np.zeros_like(arr)
            else:
                out[name] = arr
        return out

    def apply_imu(self, imu: np.ndarray) -> np.ndarray:
        sample = np.asarray(imu, dtype=np.float32).reshape(-1).copy()
        if self._imu_freeze:
            if self._frozen_imu is None:
                self._frozen_imu = sample
            return self._frozen_imu.copy()
        self._frozen_imu = sample
        return sample

    def apply_gps(self, gps: np.ndarray) -> np.ndarray:
        arr = np.asarray(gps, dtype=np.float32).reshape(-1).copy()
        if arr.size < 4:
            padded = np.zeros(4, dtype=np.float32)
            padded[: arr.size] = arr
            arr = padded
        if self._gnss_dropout:
            arr[3] = 0.0
        return arr[:4]

    def note_radio_loss(self, on_loss: str = "stop_beacon") -> None:
        self._radio_loss = str(on_loss or "stop_beacon")

    def clear_radio_loss(self) -> None:
        self._radio_loss = None

    def report(self, pose: Any = None, *, gnss_dropped: bool = False) -> FaultReport:
        pose_d = pose_dict(pose)
        if self.immobilised:
            side = DRIVE_LEFT if self.left_dead else DRIVE_RIGHT
            mode = self._left_dead if self.left_dead else self._right_dead
            return FaultReport(
                code=CODE_IMMOBILISED,
                component=side,
                pose=pose_d,
                retrieve=True,
                mode=mode,
                reason="dead drive motor — hold, do not drag",
            )
        if self._radio_loss is not None and self._radio_loss != "limp_home":
            return FaultReport(
                code=CODE_RADIO_LOSS,
                component="radio",
                pose=pose_d,
                retrieve=True,
                mode=self._radio_loss,
                reason="radio heartbeat lost — stop + beacon",
            )
        if self._radio_loss == "limp_home":
            return FaultReport(
                code=CODE_RADIO_LOSS,
                component="radio",
                pose=pose_d,
                retrieve=False,
                mode="limp_home",
                reason="radio heartbeat lost — limp home",
            )
        if self._stuck:
            return FaultReport(
                code=CODE_STUCK,
                component=CHASSIS,
                pose=pose_d,
                retrieve=False,
                reason="stuck — recovery reverse / pivot / help still allowed",
            )
        if self._trimmer_jam:
            return FaultReport(
                code=CODE_TRIMMER_JAM,
                component=TRIMMER,
                pose=pose_d,
                retrieve=False,
                reason="trimmer jam — head disabled",
            )
        if self._cam_blind is not None:
            return FaultReport(
                code=CODE_CAM_BLIND,
                component=CAMERA,
                pose=pose_d,
                retrieve=False,
                reason="camera blind — black frames",
            )
        if self._imu_freeze:
            return FaultReport(
                code=CODE_IMU_FREEZE,
                component=IMU,
                pose=pose_d,
                retrieve=False,
                reason="IMU freeze — last sample held",
            )
        if self._gnss_dropout or gnss_dropped:
            return FaultReport(
                code=CODE_GNSS_DROPOUT,
                component=GNSS,
                pose=pose_d,
                retrieve=False,
                reason="GNSS dropout — valid=0",
            )
        return FaultReport(code=CODE_OK, component=None, pose=pose_d, retrieve=False)

    def as_info(self, pose: Any = None, *, gnss_dropped: bool = False) -> dict[str, Any]:
        report = self.report(pose, gnss_dropped=gnss_dropped)
        blob = report.to_dict()
        blob["encoder"] = [float(self._encoder_left), float(self._encoder_right)]
        blob["applied"] = [float(v) for v in self.last_applied]
        blob["immobilised"] = bool(self.immobilised)
        blob["stuck"] = bool(self.stuck)
        if report.code not in FAULT_CODES and report.code != CODE_OK:
            blob["code"] = CODE_OK
        return blob


def fault_is_retrieve(fault: Optional[dict[str, Any]]) -> bool:
    if not isinstance(fault, dict):
        return False
    return bool(fault.get("retrieve")) or str(fault.get("code") or "") == CODE_IMMOBILISED


def fault_is_immobilised(fault: Optional[dict[str, Any]]) -> bool:
    if not isinstance(fault, dict):
        return False
    return bool(fault.get("immobilised")) or str(fault.get("code") or "") == CODE_IMMOBILISED

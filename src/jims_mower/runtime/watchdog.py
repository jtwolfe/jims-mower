"""Stop wheels if IMU or vision stalls.

Software watchdog for the on-box loop. Counts repeated identical IMU /
camera frames; when a stream is frozen longer than the stall window the
command is zeroed. Not a hardware E-stop and not a claimed SIL rating.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class WatchdogConfig:
    enabled: bool = False
    imu_stall_s: float = 0.40
    vision_stall_s: float = 0.40


class SensorWatchdog:
    """Latch a hold when IMU or cameras stop changing."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        imu_stall_s: float = 0.40,
        vision_stall_s: float = 0.40,
        dt: float = 0.10,
    ) -> None:
        self.enabled = bool(enabled)
        self.imu_stall_s = float(imu_stall_s)
        self.vision_stall_s = float(vision_stall_s)
        self.dt = float(dt)
        self._last_imu: Optional[np.ndarray] = None
        self._last_vision: Optional[np.ndarray] = None
        self._last_imu_stamp: Optional[float] = None
        self._last_vision_stamp: Optional[float] = None
        self._imu_frozen_s = 0.0
        self._vision_frozen_s = 0.0
        self.stalled = False
        self.reason = "ok"

    @classmethod
    def from_config(cls, cfg: object, dt: float = 0.10) -> "SensorWatchdog":
        runtime = getattr(cfg, "runtime", cfg)
        wd = getattr(runtime, "watchdog", None)
        if wd is None:
            return cls(enabled=False, dt=dt)
        return cls(
            enabled=bool(getattr(wd, "enabled", False)),
            imu_stall_s=float(getattr(wd, "imu_stall_s", 0.40)),
            vision_stall_s=float(getattr(wd, "vision_stall_s", 0.40)),
            dt=dt,
        )

    def reset(self) -> None:
        self._last_imu = None
        self._last_vision = None
        self._last_imu_stamp = None
        self._last_vision_stamp = None
        self._imu_frozen_s = 0.0
        self._vision_frozen_s = 0.0
        self.stalled = False
        self.reason = "ok"

    def _vision_fingerprint(self, cameras: Optional[dict[str, np.ndarray]]) -> Optional[np.ndarray]:
        if not cameras:
            return None
        chunks = []
        for name in sorted(cameras):
            frame = np.asarray(cameras[name])
            if frame.size == 0:
                continue
            # Mean per channel — cheap and stable for "did the image change?"
            chunks.append(frame.reshape(-1, frame.shape[-1] if frame.ndim == 3 else 1).mean(axis=0))
        if not chunks:
            return None
        return np.concatenate(chunks).astype(np.float32)

    def observe(
        self,
        imu: Optional[np.ndarray],
        cameras: Optional[dict[str, np.ndarray]],
        *,
        dt: Optional[float] = None,
        imu_stamp_s: Optional[float] = None,
        vision_stamp_s: Optional[float] = None,
    ) -> str:
        if not self.enabled:
            self.stalled = False
            self.reason = "ok"
            return self.reason
        step = float(self.dt if dt is None else dt)
        if imu_stamp_s is not None:
            stamp = float(imu_stamp_s)
            if self._last_imu_stamp is not None and abs(stamp - self._last_imu_stamp) < 1e-12:
                self._imu_frozen_s += step
            else:
                self._imu_frozen_s = 0.0
            self._last_imu_stamp = stamp
        else:
            imu_arr = None if imu is None else np.asarray(imu, dtype=np.float32).reshape(-1)
            if imu_arr is None or imu_arr.size < 6:
                self._imu_frozen_s += step
            elif self._last_imu is not None and np.allclose(imu_arr, self._last_imu, atol=1e-6):
                self._imu_frozen_s += step
            else:
                self._imu_frozen_s = 0.0
                if imu_arr is not None:
                    self._last_imu = imu_arr.copy()
        if vision_stamp_s is not None:
            stamp = float(vision_stamp_s)
            if self._last_vision_stamp is not None and abs(stamp - self._last_vision_stamp) < 1e-12:
                self._vision_frozen_s += step
            else:
                self._vision_frozen_s = 0.0
            self._last_vision_stamp = stamp
        else:
            vis = self._vision_fingerprint(cameras)
            if vis is None:
                self._vision_frozen_s += step
            elif self._last_vision is not None and np.allclose(vis, self._last_vision, atol=1e-4):
                self._vision_frozen_s += step
            else:
                self._vision_frozen_s = 0.0
                self._last_vision = vis
        imu_stall = self._imu_frozen_s >= self.imu_stall_s
        vis_stall = self._vision_frozen_s >= self.vision_stall_s
        if imu_stall and vis_stall:
            self.reason = "imu_vision_stall"
        elif imu_stall:
            self.reason = "imu_stall"
        elif vis_stall:
            self.reason = "vision_stall"
        else:
            self.reason = "ok"
        self.stalled = self.reason != "ok"
        return self.reason

    def filter_action(self, action: np.ndarray) -> np.ndarray:
        """Zero wheels + trimmer when stalled."""
        act = np.asarray(action, dtype=np.float32).reshape(-1)
        if act.size < 3:
            act = np.pad(act, (0, 3 - act.size))
        if self.stalled:
            return np.zeros(3, dtype=np.float32)
        return act[:3].copy()

    def as_info(self) -> dict:
        return {
            "watchdog_stalled": bool(self.stalled),
            "watchdog_reason": self.reason,
            "watchdog_enabled": bool(self.enabled),
        }

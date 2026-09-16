"""On-box control loop (RT-7). Never imports ``jims_mower.renderer``.

Loads bench / Orin YAML, Fake* or Gst adapters, the sensor watchdog,
hardware ESTOP, and an optional black-box path. This is the process a
flashed Orin can run without a laptop gym. It is **not** SIL, not real
CSI, and not a field scorecard.

CLI: ``jims-mower-onbox``. Systemd example: ``deploy/jims-mower.service``.
See ``docs/ONBOX.md``. ESTOP stays hardware.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from jims_mower.blackbox import BlackBox
from jims_mower.config import EnvConfig, load_config
from jims_mower.hardware_estop import HardwareEstop
from jims_mower.runtime.capture import (
    StampClock,
    adapter_kind,
    names_from_specs,
)
from jims_mower.runtime.drivers import FakeCsiDriver, level_rest_imu
from jims_mower.runtime.watchdog import SensorWatchdog

RENDERER_MODULE = "jims_mower.renderer"


class OnboxError(RuntimeError):
    """On-box loop imported the gym renderer or was misconfigured."""


def renderer_loaded() -> bool:
    return RENDERER_MODULE in sys.modules


def assert_no_renderer() -> None:
    """Hard guard: the on-box process must never have imported the gym renderer."""
    if renderer_loaded():
        raise OnboxError(
            "on-box loop imported jims_mower.renderer — that is the laptop gym. "
            "Use jims-mower-onbox / OnboxLoop, not MowerEnv, on the Orin."
        )


def _build_adapter(cfg: EnvConfig):
    names = names_from_specs(cfg.resolved_cameras())
    width = int(cfg.sensors.width)
    height = int(cfg.sensors.height)
    kind = adapter_kind(getattr(cfg.runtime.cameras, "adapter", ""))
    if kind in {"", "renderer"}:
        kind = "fake_csi"
    if kind in {"fake_csi", "fake_gst"}:
        return FakeCsiDriver(names=names, width=width, height=height, dt=cfg.dt)
    if kind == "gst":
        from jims_mower.runtime.gstreamer import GstNvmmAdapter

        return GstNvmmAdapter(names, width=width, height=height, dt=cfg.dt)
    raise OnboxError(
        f"on-box cameras.adapter must be fake_csi|fake_gst|gst; got {kind!r}"
    )


class OnboxLoop:
    """One control tick: grab adapters → watchdog → HW ESTOP → black box.

    No kinematics gym, no renderer, no god-view. Fake adapters are the
    CI / first-boot stand-in. ``fps_claim`` stays null.
    """

    def __init__(
        self,
        config: Optional[Union[str, Path, dict, EnvConfig]] = None,
        *,
        blackbox: Optional[Union[str, Path]] = None,
    ) -> None:
        if isinstance(config, EnvConfig):
            self.cfg = config
        elif config is None:
            self.cfg = load_config("configs/orin/bench.yaml")
        else:
            self.cfg = load_config(config)
        self.watchdog = SensorWatchdog.from_config(self.cfg, dt=self.cfg.dt)
        if not self.watchdog.enabled:
            self.watchdog.enabled = True
        self.hw_estop = HardwareEstop()
        dest = blackbox
        if dest is None:
            dest = Path("blackbox.jsonl")
        self.blackbox = BlackBox(dest)
        self.adapter = _build_adapter(self.cfg)
        self._imu_clock = StampClock(dt=self.cfg.dt)
        self._steps = 0
        self._freeze_imu = False
        self._freeze_vision = False
        self._last_imu = level_rest_imu()
        self._last_images: dict[str, np.ndarray] = {}
        self._imu_stamp_s = 0.0
        self._vision_stamp_s = 0.0
        self.fps_claim = None

    @classmethod
    def from_config(
        cls,
        config: Optional[Union[str, Path, dict, EnvConfig]] = "configs/orin/bench.yaml",
        *,
        blackbox: Optional[Union[str, Path]] = None,
    ) -> "OnboxLoop":
        cfg = load_config(config)
        return cls(cfg, blackbox=blackbox)

    def hit_hw_estop(self, reason: str = "paddle") -> None:
        self.hw_estop.hit(reason)

    def reset_hw_estop(self) -> None:
        self.hw_estop.reset()

    def freeze_imu_stamp(self) -> None:
        """Test hook: stop advancing the IMU stamp (watchdog stall)."""
        self._freeze_imu = True

    def freeze_vision_stamp(self) -> None:
        """Test hook: reuse the last camera grab so vision stamps freeze."""
        self._freeze_vision = True

    def step(self, action: Optional[np.ndarray] = None) -> dict[str, Any]:
        act = np.asarray(
            [0.0, 0.0, 0.0] if action is None else action, dtype=np.float32
        ).reshape(-1)
        if act.size < 3:
            act = np.pad(act, (0, 3 - int(act.size)))
        act = act[:3].copy()

        if self._freeze_vision and self._last_images:
            images = self._last_images
            vision_stamp = self._vision_stamp_s
        else:
            grab = self.adapter.grab()
            images = dict(grab.cameras)
            vision_stamp = float(grab.stamp_s)
            self._last_images = images
            self._vision_stamp_s = vision_stamp

        if self._freeze_imu:
            imu_stamp = self._imu_stamp_s
        else:
            imu_stamp = self._imu_clock.next()
            self._imu_stamp_s = imu_stamp
        imu = level_rest_imu()
        self._last_imu = imu

        self.watchdog.observe(
            imu,
            images,
            dt=self.cfg.dt,
            imu_stamp_s=imu_stamp,
            vision_stamp_s=vision_stamp,
        )
        filtered = self.watchdog.filter_action(act)
        filtered = self.hw_estop.filter_action(filtered)
        self._steps += 1
        event = ""
        if self.hw_estop.latched:
            event = "hw_estop"
        elif self.watchdog.stalled:
            event = "watchdog"
        self.blackbox.record(
            step=self._steps,
            imu=imu,
            cmd=filtered,
            pose={},
            advice="hold" if self.watchdog.stalled or self.hw_estop.latched else "ok",
            event=event,
            fault=self.hw_estop.as_fault() or {},
        )
        info = {
            "step": self._steps,
            "cmd": [float(v) for v in filtered],
            "requested": [float(v) for v in act],
            "cameras": images,
            "imu": imu,
            "imu_stamp_s": float(imu_stamp),
            "vision_stamp_s": float(vision_stamp),
            "fps_claim": None,
            "map_claim": None,
            "not_sil": True,
            "renderer_imported": renderer_loaded(),
            **self.watchdog.as_info(),
            **self.hw_estop.as_info(),
        }
        return info

    def run(
        self,
        *,
        steps: int = 0,
        hold: bool = False,
        dt: Optional[float] = None,
        action: Optional[np.ndarray] = None,
    ) -> list[dict[str, Any]]:
        """Run ``steps`` ticks, or hold until SIGINT when ``hold`` is set."""
        out: list[dict[str, Any]] = []
        period = float(self.cfg.dt if dt is None else dt)
        n = max(0, int(steps))
        i = 0
        while True:
            info = self.step(action)
            out.append(info)
            i += 1
            if hold:
                time.sleep(max(period, 0.0))
                continue
            if i >= n:
                break
        return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Jim's Mower on-box loop (no gym renderer). "
            "Hardware ESTOP is still the paddle — this process cannot replace it."
        )
    )
    p.add_argument(
        "--config",
        default="configs/orin/bench.yaml",
        help="bench / Orin YAML (default: configs/orin/bench.yaml)",
    )
    p.add_argument(
        "--blackbox",
        type=Path,
        default=Path("blackbox.jsonl"),
        help="rotating JSONL path (SAF-3)",
    )
    p.add_argument("--steps", type=int, default=4, help="control ticks then exit")
    p.add_argument(
        "--hold",
        action="store_true",
        help="run until SIGINT (systemd). ESTOP is still hardware.",
    )
    p.add_argument(
        "--action",
        default="0,0,0",
        help="left,right,trimmer hold command (default 0,0,0)",
    )
    return p


def _parse_action(raw: str) -> np.ndarray:
    parts = [float(p) for p in str(raw).split(",")]
    while len(parts) < 3:
        parts.append(0.0)
    return np.array(parts[:3], dtype=np.float32)


def main(argv: Optional[list[str]] = None) -> int:
    assert_no_renderer()
    args = build_parser().parse_args(argv)
    loop = OnboxLoop.from_config(args.config, blackbox=args.blackbox)
    action = _parse_action(args.action)
    try:
        if args.hold:
            print(
                "jims-mower-onbox hold — watchdog + HW ESTOP rails. "
                "Paddle is still hardware. Ctrl-C to stop.",
                flush=True,
            )
            loop.run(hold=True, action=action)
        else:
            rows = loop.run(steps=max(1, int(args.steps)), action=action)
            last = rows[-1]
            print(
                f"on-box {len(rows)} steps  watchdog={last.get('watchdog_reason')}  "
                f"hw_estop={last.get('hw_estop')}  blackbox={args.blackbox}  "
                f"renderer={last.get('renderer_imported')}"
            )
    except KeyboardInterrupt:
        print("on-box stopped (SIGINT). ESTOP is still the paddle.", flush=True)
    assert_no_renderer()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

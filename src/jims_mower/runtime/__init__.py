"""Orin-class runtime stubs: fake buses, multiprocessing bridge, budget.

Default install stays stdlib + gym deps. ROS 2 is an optional extra
(``pip install -e ".[ros2]"``) whose node stubs import-guard ``rclpy``.
CI uses the in-process / multiprocessing bridge, not ROS 2 or ZMQ.

``bridge`` is imported lazily so ``jims_mower.env`` can use ``OrinBudget``
without a circular import through record/replay.
"""

from jims_mower.runtime.budget import OrinBudget, budget_advice, budget_from_config
from jims_mower.runtime.bus import InProcessBus
from jims_mower.runtime.capture import (
    CaptureResult,
    downsample_rgb,
    field_rig_yaml_path,
    load_field_camera_specs,
    names_from_specs,
)
from jims_mower.runtime.gstreamer import FakeGstAdapter, GstNvmmAdapter, gstreamer_available
from jims_mower.faults import FaultBus
from jims_mower.radio import RadioSim
from jims_mower.hardware_estop import HardwareEstop
from jims_mower.runtime.watchdog import SensorWatchdog
from jims_mower.runtime.drivers import (
    FakeCsiDriver,
    FakeGnssDriver,
    FakeImuDriver,
    FakeTofDriver,
    SensorRig,
    level_rest_imu,
    publish_obs,
)

__all__ = [
    "CaptureResult",
    "FaultBus",
    "FakeCsiDriver",
    "FakeGnssDriver",
    "FakeGstAdapter",
    "FakeImuDriver",
    "FakeTofDriver",
    "GstNvmmAdapter",
    "HardwareEstop",
    "InProcessBus",
    "MultiprocessBridge",
    "OrinBudget",
    "RadioSim",
    "SensorRig",
    "SensorWatchdog",
    "budget_advice",
    "budget_from_config",
    "downsample_rgb",
    "field_rig_yaml_path",
    "gstreamer_available",
    "level_rest_imu",
    "load_field_camera_specs",
    "names_from_specs",
    "publish_obs",
    "replay_through_bridge",
]


def __getattr__(name: str):
    if name in {"MultiprocessBridge", "replay_through_bridge"}:
        from jims_mower.runtime import bridge as _bridge

        return getattr(_bridge, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

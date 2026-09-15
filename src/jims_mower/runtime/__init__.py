"""Orin-class runtime stubs: fake buses, multiprocessing bridge, budget.

Default install stays stdlib + gym deps. ROS 2 is an optional extra
(``pip install -e ".[ros2]"``) whose node stubs import-guard ``rclpy``.
CI uses the in-process / multiprocessing bridge, not ROS 2 or ZMQ.

``bridge`` is imported lazily so ``jims_mower.env`` can use ``OrinBudget``
without a circular import through record/replay.
"""

from jims_mower.runtime.budget import OrinBudget, budget_advice, budget_from_config
from jims_mower.runtime.bus import InProcessBus
from jims_mower.runtime.gstreamer import FakeGstAdapter, GstNvmmAdapter, gstreamer_available
from jims_mower.runtime.watchdog import SensorWatchdog
from jims_mower.runtime.drivers import (
    FakeCsiDriver,
    FakeGnssDriver,
    FakeImuDriver,
    FakeTofDriver,
    SensorRig,
    publish_obs,
)

__all__ = [
    "FakeCsiDriver",
    "FakeGnssDriver",
    "FakeGstAdapter",
    "FakeImuDriver",
    "FakeTofDriver",
    "GstNvmmAdapter",
    "InProcessBus",
    "MultiprocessBridge",
    "OrinBudget",
    "SensorRig",
    "SensorWatchdog",
    "budget_advice",
    "budget_from_config",
    "gstreamer_available",
    "publish_obs",
    "replay_through_bridge",
]


def __getattr__(name: str):
    if name in {"MultiprocessBridge", "replay_through_bridge"}:
        from jims_mower.runtime import bridge as _bridge

        return getattr(_bridge, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Orin-class runtime stubs: fake buses, multiprocessing bridge, budget.

Default install stays stdlib + gym deps. ROS 2 is an optional extra
(``pip install -e ".[ros2]"``) whose node stubs import-guard ``rclpy``.
CI uses the in-process / multiprocessing bridge, not ROS 2 or ZMQ.
"""

from jims_mower.runtime.bridge import MultiprocessBridge, replay_through_bridge
from jims_mower.runtime.budget import OrinBudget, budget_advice
from jims_mower.runtime.bus import InProcessBus
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
    "FakeImuDriver",
    "FakeTofDriver",
    "InProcessBus",
    "MultiprocessBridge",
    "OrinBudget",
    "SensorRig",
    "budget_advice",
    "publish_obs",
    "replay_through_bridge",
]

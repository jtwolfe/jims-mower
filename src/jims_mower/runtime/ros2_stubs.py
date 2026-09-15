"""Optional ROS 2 *node stubs* — not started in CI.

Install a ROS 2 distro (Humble / Jazzy) on the Orin and then::

    pip install -e ".[ros2]"

The extra does **not** pull ``rclpy`` from PyPI (that wheel is not the
distro's). This module import-guards ``rclpy`` so ``import
jims_mower.runtime.ros2_stubs`` is safe on a laptop.

Topic names match ``docs/runtime_contract.md`` kinds. Payloads stay the
versioned JSON dicts; there is no ``sensor_msgs`` conversion here.
"""

from __future__ import annotations

from typing import Any, Optional

from jims_mower.contract import MESSAGE_KINDS

# Documented topic map. A real node would qos=sensor_data on these.
ROS2_TOPICS: dict[str, str] = {
    "CameraFrame": "/jims_mower/camera",
    "ImuSample": "/jims_mower/imu",
    "GpsFix": "/jims_mower/gps",
    "TofArray": "/jims_mower/tof",
    "DetectionSet": "/jims_mower/detections",
    "TerrainMaps": "/jims_mower/maps",
    "Plan": "/jims_mower/plan",
    "WheelCommand": "/jims_mower/cmd_vel",
}


def ros2_available() -> bool:
    try:
        import rclpy  # noqa: F401
    except ImportError:
        return False
    return True


class Ros2NotInstalled(RuntimeError):
    """Raised when a stub is asked to spin without a ROS 2 distro."""


class ContractNodeStub:
    """One publisher + one subscriber per contract kind. Does not spin.

    ``create()`` raises :class:`Ros2NotInstalled` unless ``rclpy`` imported.
    Use :class:`jims_mower.runtime.bridge.MultiprocessBridge` in CI.
    """

    def __init__(self, kind: str, topic: Optional[str] = None) -> None:
        if kind not in MESSAGE_KINDS:
            raise KeyError(f"unknown contract kind {kind!r}")
        self.kind = kind
        self.topic = topic or ROS2_TOPICS[kind]
        self._node = None

    def create(self) -> None:
        if not ros2_available():
            raise Ros2NotInstalled(
                "rclpy is not importable. Install a ROS 2 distro on the "
                "Orin, or use MultiprocessBridge (default, no extra deps)."
            )
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String

        if not rclpy.ok():
            rclpy.init()
        self._node = Node(f"jims_mower_{self.kind.lower()}")
        self._pub = self._node.create_publisher(String, self.topic, 10)
        self._sub = self._node.create_subscription(String, self.topic, self._on_msg, 10)

    def _on_msg(self, msg: Any) -> None:
        return None

    def publish_json(self, payload: dict[str, Any]) -> None:
        if self._node is None:
            raise Ros2NotInstalled("call create() first (requires rclpy)")
        import json
        from std_msgs.msg import String

        out = String()
        out.data = json.dumps(payload)
        self._pub.publish(out)


def documented_topics() -> dict[str, str]:
    return dict(ROS2_TOPICS)

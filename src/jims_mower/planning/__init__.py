"""Terrain-aware coverage planning, pose fusion stub, and zero-turn control."""

from jims_mower.planning.controller import TerrainPolicy, tracking_action
from jims_mower.planning.costmap import Costmap, build_costmap
from jims_mower.planning.coverage import CoveragePlan, plan_coverage, shortest_path
from jims_mower.planning.fusion import (
    ComplementaryPoseFilter,
    EkfNoise,
    EkfPoseFilter,
    attitude_from_accel,
    make_pose_filter,
)

__all__ = [
    "ComplementaryPoseFilter",
    "Costmap",
    "CoveragePlan",
    "EkfNoise",
    "EkfPoseFilter",
    "TerrainPolicy",
    "attitude_from_accel",
    "build_costmap",
    "make_pose_filter",
    "plan_coverage",
    "shortest_path",
    "tracking_action",
]

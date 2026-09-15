"""Terrain-aware coverage planning, pose fusion stub, and zero-turn control."""

from jims_mower.planning.controller import (
    TerrainPolicy,
    combine_advice,
    observed_hand_signal,
    tracking_action,
)
from jims_mower.safe_state import SafeStateMachine
from jims_mower.planning.costmap import Costmap, build_costmap
from jims_mower.planning.coverage import (
    CoveragePlan,
    choose_strip_orientation,
    connected_components,
    plan_coverage,
    reachable_mask,
    shortest_path,
)
from jims_mower.planning.explore import ExplorePlan, explore_costmap, plan_explore
from jims_mower.planning.observed import ObservedMap, downsample_frontiers, frontiers
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
    "ExplorePlan",
    "ObservedMap",
    "SafeStateMachine",
    "TerrainPolicy",
    "attitude_from_accel",
    "build_costmap",
    "choose_strip_orientation",
    "combine_advice",
    "connected_components",
    "downsample_frontiers",
    "explore_costmap",
    "frontiers",
    "make_pose_filter",
    "observed_hand_signal",
    "plan_coverage",
    "plan_explore",
    "reachable_mask",
    "shortest_path",
    "tracking_action",
]

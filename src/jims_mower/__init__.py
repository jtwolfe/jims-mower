"""Jim's Mower: Gymnasium env for a camera-driven zero-turn string-trimmer.

Package import is **lazy**. Eagerly importing ``MowerEnv`` would pull
``jims_mower.renderer``, which the on-box loop must never load. Gym
registration stays a string entry point.
"""

from __future__ import annotations

from gymnasium.envs.registration import register, registry

from jims_mower.config import EnvConfig, load_config
from jims_mower.pack import GYM_STUB_CAPACITY_WH

__version__ = "0.1.0"

_ENV_ID = "jims_mower/Mower-v0"

_LAZY = {
    "EkfPoseFilter": "jims_mower.planning.fusion",
    "EpisodeScorecard": "jims_mower.metrics",
    "FaultBus": "jims_mower.faults",
    "GeofenceSpec": "jims_mower.geofence",
    "HardwareEstop": "jims_mower.hardware_estop",
    "MowerEnv": "jims_mower.env",
    "OrinBudget": "jims_mower.runtime.budget",
    "RadioSim": "jims_mower.radio",
    "SafeStateMachine": "jims_mower.safe_state",
    "Scenario": "jims_mower.scenarios",
    "TerrainPolicy": "jims_mower.planning.controller",
    "YardProfile": "jims_mower.profile",
    "build_costmap": "jims_mower.planning.costmap",
    "evaluate_episode": "jims_mower.metrics",
    "integrate_pose": "jims_mower.kinematics",
    "list_scenarios": "jims_mower.scenarios",
    "living_advice": "jims_mower.safety",
    "load_mission": "jims_mower.mission",
    "load_scenario": "jims_mower.scenarios",
    "load_source": "jims_mower.scenarios",
    "load_yard_profile": "jims_mower.profile",
    "plan_coverage": "jims_mower.planning.coverage",
    "save_mission": "jims_mower.mission",
    "sit_on_terrain": "jims_mower.kinematics",
    "terrain_hazards": "jims_mower.safety",
    "trail_to_polygon": "jims_mower.profile",
    "trimmer_interlock": "jims_mower.safety",
}


def _register() -> None:
    if _ENV_ID not in registry:
        register(
            id=_ENV_ID,
            entry_point="jims_mower.env:MowerEnv",
            max_episode_steps=500,
        )


_register()


def __getattr__(name: str):
    if name in _LAZY:
        from importlib import import_module

        return getattr(import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EkfPoseFilter",
    "EnvConfig",
    "EpisodeScorecard",
    "FaultBus",
    "GYM_STUB_CAPACITY_WH",
    "GeofenceSpec",
    "HardwareEstop",
    "MowerEnv",
    "YardProfile",
    "OrinBudget",
    "RadioSim",
    "SafeStateMachine",
    "Scenario",
    "TerrainPolicy",
    "build_costmap",
    "evaluate_episode",
    "integrate_pose",
    "list_scenarios",
    "living_advice",
    "load_config",
    "load_mission",
    "load_scenario",
    "load_source",
    "load_yard_profile",
    "plan_coverage",
    "save_mission",
    "sit_on_terrain",
    "terrain_hazards",
    "trail_to_polygon",
    "trimmer_interlock",
    "__version__",
]

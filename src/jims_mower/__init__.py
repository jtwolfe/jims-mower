"""Jim's Mower: Gymnasium env for a camera-driven zero-turn string-trimmer."""

from __future__ import annotations

from gymnasium.envs.registration import register, registry

from jims_mower.config import EnvConfig, load_config
from jims_mower.env import MowerEnv
from jims_mower.metrics import EpisodeScorecard, evaluate_episode
from jims_mower.scenarios import Scenario, list_scenarios, load_scenario, load_source
from jims_mower.kinematics import integrate_pose, sit_on_terrain
from jims_mower.planning import TerrainPolicy, build_costmap, plan_coverage
from jims_mower.planning.fusion import EkfPoseFilter
from jims_mower.safety import terrain_hazards, trimmer_interlock

__version__ = "0.1.0"

_ENV_ID = "jims_mower/Mower-v0"


def _register() -> None:
    if _ENV_ID not in registry:
        register(
            id=_ENV_ID,
            entry_point="jims_mower.env:MowerEnv",
            max_episode_steps=500,
        )


_register()

__all__ = [
    "EkfPoseFilter",
    "EnvConfig",
    "EpisodeScorecard",
    "MowerEnv",
    "Scenario",
    "TerrainPolicy",
    "build_costmap",
    "evaluate_episode",
    "integrate_pose",
    "list_scenarios",
    "load_config",
    "load_scenario",
    "load_source",
    "plan_coverage",
    "sit_on_terrain",
    "terrain_hazards",
    "trimmer_interlock",
    "__version__",
]

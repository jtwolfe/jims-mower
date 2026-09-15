"""BEV debugger composite shape."""

from __future__ import annotations

import numpy as np

from jims_mower.bev import render_bev
from jims_mower.env import MowerEnv
from jims_mower.planning import TerrainPolicy
from tests.conftest import tiny_config_dict


def test_bev_composite_has_panels_and_insets() -> None:
    cfg = tiny_config_dict()
    cfg["world"]["terrain"] = {"enabled": True, "n_drains": 1, "n_banks": 0}
    env = MowerEnv(config=cfg, render_mode="rgb_array")
    obs, info = env.reset(seed=2)
    policy = TerrainPolicy(env.cfg)
    policy.reset(obs, info)
    bev = render_bev(
        env,
        obs,
        waypoints=policy.waypoints,
        waypoint_index=policy.index,
        costmap=policy.plan.costmap if policy.plan else None,
        cameras=obs["cameras"],
        panel_size=80,
    )
    assert bev.ndim == 3 and bev.shape[2] == 3
    assert bev.shape[1] == 80 * 3
    assert bev.shape[0] > 80  # camera strip
    assert bev.dtype == np.uint8
    env.close()

"""BEV debugger composite shape."""

from __future__ import annotations

import numpy as np

from jims_mower.bev import render_bev
from jims_mower.env import MowerEnv
from jims_mower.planning import TerrainPolicy


def _tiny() -> dict:
    return {
        "dt": 0.1,
        "max_steps": 40,
        "sensors": {"width": 32, "height": 24, "camera_count": 4, "fov_deg": 70.0},
        "world": {
            "width_m": 8.0,
            "height_m": 8.0,
            "resolution_m": 0.20,
            "n_people": 1,
            "n_dogs": 0,
            "n_cats": 0,
            "n_birds": 0,
            "n_trees": 1,
            "n_furniture": 0,
            "n_toys": 0,
            "terrain": {"enabled": True, "n_drains": 1, "n_banks": 0},
        },
    }


def test_bev_composite_has_panels_and_insets() -> None:
    cfg = _tiny()
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

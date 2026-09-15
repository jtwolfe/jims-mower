"""Keep-in / keep-out polygons: safety, costmap, scenario YAML, demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jims_mower.env import MowerEnv
from jims_mower.geofence import (
    GeofenceSpec,
    allowed_xy,
    geofence_advice,
    rasterize_geofence,
)
from jims_mower.planning.costmap import build_costmap
from jims_mower.safety import in_yard
from jims_mower.scenarios import load_dsl_scenario, parse_scenario
from jims_mower.types import Pose


KEEP_IN = [(1.0, 1.0), (7.0, 1.0), (7.0, 7.0), (1.0, 7.0)]
KEEP_OUT = [[(3.0, 3.0), (5.0, 3.0), (5.0, 5.0), (3.0, 5.0)]]


def test_keepout_rejects_interior_point() -> None:
    spec = GeofenceSpec(keep_in=KEEP_IN, keep_out=KEEP_OUT)
    assert allowed_xy(2.0, 2.0, spec)
    assert not allowed_xy(4.0, 4.0, spec)
    assert not allowed_xy(8.0, 4.0, spec)


def test_in_yard_keepout() -> None:
    spec = GeofenceSpec(keep_in=KEEP_IN, keep_out=KEEP_OUT)
    assert in_yard(Pose(2.0, 2.0, 0.0), 8.0, 8.0, 0.2, spec=spec)
    assert not in_yard(Pose(4.0, 4.0, 0.0), 8.0, 8.0, 0.2, spec=spec)


def test_geofence_pre_touch_slows() -> None:
    spec = GeofenceSpec(keep_in=KEEP_IN)
    # Heading +x toward the east fence from just inside.
    pose = Pose(6.6, 4.0, 0.0)
    assert geofence_advice(pose, spec, slow_m=0.8, stop_m=0.15, look_ahead_m=0.55) in {
        "slow",
        "stop",
    }
    mid = Pose(4.0, 4.0, 0.0)
    assert geofence_advice(mid, spec, slow_m=0.8, stop_m=0.15) == "ok"


def test_costmap_blocks_outside_keep_in() -> None:
    hazard = np.zeros((20, 20), dtype=np.float32)
    slope = np.zeros_like(hazard)
    spec = GeofenceSpec(keep_in=[(0.6, 0.6), (3.4, 0.6), (3.4, 3.4), (0.6, 3.4)])
    cm = build_costmap(
        hazard,
        slope,
        resolution_m=0.2,
        width_m=4.0,
        height_m=4.0,
        max_climb_slope_rad=0.32,
        drain_clearance_m=0.0,
        margin_m=0.0,
        geofence=spec,
        geofence_inflate_m=0.0,
    )
    # Cell at (0.1, 2.0) is outside keep-in.
    assert cm.blocked[10, 0]
    # Interior of the keep-in stays free.
    assert not cm.blocked[10, 10]


def test_costmap_blocks_keepout_polygon() -> None:
    hazard = np.zeros((20, 20), dtype=np.float32)
    slope = np.zeros_like(hazard)
    spec = GeofenceSpec(
        keep_in=[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
        keep_out=[[(1.6, 1.6), (2.4, 1.6), (2.4, 2.4), (1.6, 2.4)]],
    )
    cm = build_costmap(
        hazard,
        slope,
        resolution_m=0.2,
        width_m=4.0,
        height_m=4.0,
        max_climb_slope_rad=0.32,
        drain_clearance_m=0.0,
        margin_m=0.0,
        geofence=spec,
        geofence_inflate_m=0.0,
    )
    assert cm.blocked[10, 10]
    assert not cm.blocked[2, 2]


def test_raster_empty_spec_is_free() -> None:
    mask = rasterize_geofence((8, 8), resolution_m=0.25, spec=GeofenceSpec())
    assert not bool(mask.any())


def test_structured_geofence_yaml() -> None:
    scn = parse_scenario(
        {
            "kind": "scenario",
            "name": "fence",
            "geofence": {
                "keep_in": [[0.5, 0.5], [7.5, 0.5], [7.5, 7.5], [0.5, 7.5]],
                "keep_out": [[[3.0, 3.0], [4.0, 3.0], [4.0, 4.0], [3.0, 4.0]]],
            },
            "world": {
                "width_m": 8.0,
                "height_m": 8.0,
                "n_people": 0,
                "n_dogs": 0,
                "n_cats": 0,
                "n_birds": 0,
                "n_trees": 0,
                "n_furniture": 0,
                "n_toys": 0,
                "terrain": {"enabled": False},
            },
            "sensors": {"width": 16, "height": 12, "camera_count": 4},
        }
    )
    assert len(scn.geofence) == 4
    assert len(scn.keepout) == 1
    env = MowerEnv(config=scn)
    obs, info = env.reset(seed=1)
    assert info["geofence_spec"]["keep_out"]
    assert info["geofence_advice"] in {"ok", "slow", "stop"}
    assert "occupancy" in obs
    env.close()


def test_bundled_geofence_movers_scenario() -> None:
    scn = load_dsl_scenario("geofence_movers")
    assert scn.name == "geofence_movers"
    assert len(scn.geofence) >= 3
    assert len(scn.keepout) >= 1
    people = [o for o in scn.obstacles if o.kind == "person"]
    assert people and people[0].trajectory is not None
    env = MowerEnv(config=scn)
    env.cfg.sensors.width = 16
    env.cfg.sensors.height = 12
    env.cfg.sensors.camera_count = 4
    env.cfg.sensors.cameras = []
    _, info = env.reset(seed=2)
    assert info["scenario"] == "geofence_movers"
    living = [o for o in env._yard.obstacles if o.kind in {"person", "dog"}]
    assert len(living) >= 2
    env.close()


def test_demo_geofence_scenario(tmp_path: Path) -> None:
    from jims_mower.demo import run_demo

    summary = run_demo(
        tmp_path,
        steps=3,
        seed=3,
        cameras=4,
        policy="terrain",
        config="configs/scenarios/geofence_movers.yaml",
    )
    assert summary["steps_run"] >= 1
    assert (tmp_path / "bev_final.png").is_file()
    assert summary.get("geofence_advice") in {None, "ok", "slow", "stop"}

"""Bird's-eye visual debugger: costmap, plan, hazard, optional camera insets."""

from __future__ import annotations

from typing import Optional

import numpy as np

from jims_mower.kinematics import trimmer_xy
from jims_mower.planning.costmap import Costmap, build_costmap
from jims_mower.renderer import render_costmap_rgb, render_scalar_map, render_topdown

_INSET_H = 56


def _resize_nn(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbor resize (no extra deps)."""
    if image.shape[0] == height and image.shape[1] == width:
        return image
    ys = (np.linspace(0, image.shape[0] - 1, height)).astype(int)
    xs = (np.linspace(0, image.shape[1] - 1, width)).astype(int)
    return image[ys[:, None], xs[None, :]]


def _letterbox_row(tiles: list[np.ndarray], height: int, gap: int = 2) -> np.ndarray:
    if not tiles:
        return np.zeros((height, height, 3), dtype=np.uint8)
    resized: list[np.ndarray] = []
    for tile in tiles:
        h, w = tile.shape[:2]
        new_w = max(8, int(round(w * height / max(h, 1))))
        resized.append(_resize_nn(tile, height, new_w))
    width = sum(t.shape[1] for t in resized) + gap * max(0, len(resized) - 1)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = 0
    for tile in resized:
        canvas[:, x : x + tile.shape[1]] = tile
        x += tile.shape[1] + gap
    return canvas


def costmap_from_obs(obs: dict, cfg, geofence=None) -> Costmap:
    from jims_mower.geofence import GeofenceSpec

    hazard = np.asarray(obs["hazard"], dtype=np.float32)
    slope = np.asarray(obs["slope"], dtype=np.float32)
    occupancy = np.asarray(obs["occupancy"], dtype=np.float32) if "occupancy" in obs else None
    confidence = np.asarray(obs["confidence"], dtype=np.float32) if "confidence" in obs else None
    structure = np.asarray(obs["structure"]) if obs.get("structure") is not None else None
    elevation = np.asarray(obs["elevation"], dtype=np.float32) if "elevation" in obs else None
    elevation_prior = (
        np.asarray(obs["elevation_prior"], dtype=np.float32) if obs.get("elevation_prior") is not None else None
    )
    rows, cols = hazard.shape
    res = float(cfg.world.resolution_m)
    unc = getattr(cfg.planner, "uncertainty", None)
    blend = (
        float(getattr(cfg.planner, "elevation_prior_weight", 0.0))
        if bool(getattr(cfg.planner, "blend_elevation_prior", False))
        else 0.0
    )
    fence = geofence
    if fence is None:
        raw = obs.get("geofence_spec") if isinstance(obs, dict) else None
        if isinstance(raw, dict):
            fence = GeofenceSpec(
                keep_in=[tuple(p) for p in raw.get("keep_in") or []],
                keep_out=[[tuple(p) for p in poly] for poly in raw.get("keep_out") or []],
                inflate_m=cfg.planner.geofence_inflate_m,
            )
    return build_costmap(
        hazard,
        slope,
        resolution_m=res,
        width_m=cols * res,
        height_m=rows * res,
        max_climb_slope_rad=cfg.planner.max_climb_slope_rad,
        tip_lethal_slope_rad=float(cfg.planner.tip_lethal_frac)
        * min(float(cfg.robot.tip_roll_rad), float(cfg.robot.tip_pitch_rad)),
        contour_cost=cfg.planner.contour_cost,
        drain_clearance_m=cfg.planner.drain_clearance_m,
        occupancy=occupancy,
        occupancy_inflate_m=cfg.planner.occupancy_inflate_m,
        margin_m=cfg.robot.collision_radius_m,
        confidence=confidence,
        uncertainty_inflate=getattr(unc, "inflate", 0.0) if unc is not None else 0.0,
        uncertain_hazard_boost=getattr(unc, "hazard_boost", 0.0) if unc is not None else 0.0,
        uncertain_confidence_floor=getattr(unc, "confidence_floor", 0.25) if unc is not None else 0.25,
        geofence=fence,
        geofence_inflate_m=cfg.planner.geofence_inflate_m,
        structure=structure,
        path_cost=getattr(cfg.planner, "path_cost", 8.0),
        bunker_cost=getattr(cfg.planner, "bunker_cost", 12.0),
        elevation=elevation,
        elevation_prior=elevation_prior,
        prior_blend=blend,
    )


def render_bev(
    env,
    obs: Optional[dict] = None,
    *,
    waypoints: Optional[list[tuple[float, float]]] = None,
    waypoint_index: int = 0,
    costmap: Optional[Costmap] = None,
    cameras: Optional[dict[str, np.ndarray]] = None,
    include_insets: bool = True,
    panel_size: int = 240,
) -> np.ndarray:
    """Composite top-down: costmap | plan/coverage | hazard, plus camera strip."""
    if obs is None:
        obs = {
            "hazard": env.oracle_labels()["hazard"],
            "slope": env.oracle_labels()["slope"],
            "occupancy": np.zeros_like(env._coverage.as_float()),
            "cameras": env._last_images,
        }
    if costmap is None:
        spec = env.geofence_spec() if hasattr(env, "geofence_spec") else None
        costmap = costmap_from_obs(obs, env.cfg, geofence=spec)
    topdown = render_topdown(
        env._pose,
        env._coverage,
        env._yard.obstacles,
        trimmer_xy=trimmer_xy(env._pose, env.cfg.robot.trimmer.offset_m),
        trimmer_on=env._trimmer_on,
        terrain=env._terrain,
        waypoints=waypoints or [],
        waypoint_index=waypoint_index,
        image_size=panel_size,
    )
    hazard = np.asarray(obs["hazard"], dtype=np.float32)
    hazard_img = render_scalar_map(hazard, vmin=0.0, vmax=3.0, cmap="hazard", image_size=panel_size)
    cost_img = render_costmap_rgb(costmap.cost, costmap.blocked, image_size=panel_size)

    def _fit(img: np.ndarray) -> np.ndarray:
        return _resize_nn(img, panel_size, panel_size)

    row = np.concatenate([_fit(cost_img), _fit(topdown), _fit(hazard_img)], axis=1)
    if include_insets:
        cams = cameras if cameras is not None else obs.get("cameras") or env._last_images
        if cams:
            order = [c.name for c in env.cameras if c.name in cams]
            insets = [cams[name] for name in order]
            strip = _letterbox_row(insets, _INSET_H)
            if strip.shape[1] < row.shape[1]:
                pad = np.zeros((_INSET_H, row.shape[1] - strip.shape[1], 3), dtype=np.uint8)
                strip = np.concatenate([strip, pad], axis=1)
            elif strip.shape[1] > row.shape[1]:
                strip = strip[:, : row.shape[1]]
            row = np.concatenate([row, strip], axis=0)
    return row

"""Hazard-map action mask for BC / RL stubs.

Looks at observer ``hazard`` (not the god-view height field) in a short
forward cone. Drain lip / channel cells zero *forward* wheel commands and
the trimmer. Reverse and in-place pivots stay allowed so a policy can
back off. This is a safety clamp, not a claimed collision-free guarantee.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from jims_mower.constants import HAZARD_DRAIN_EDGE, HAZARD_STEEP


def pose_xy_theta(obs: dict[str, Any], info: Optional[dict[str, Any]] = None) -> tuple[float, float, float]:
    info = info or {}
    raw = info.get("pose")
    if isinstance(raw, dict) and "x" in raw:
        return float(raw["x"]), float(raw["y"]), float(raw.get("theta", 0.0))
    arr = np.asarray(obs.get("pose", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
    return (
        float(arr[0]) if arr.size > 0 else 0.0,
        float(arr[1]) if arr.size > 1 else 0.0,
        float(arr[2]) if arr.size > 2 else 0.0,
    )


def sample_grid(
    grid: Optional[np.ndarray],
    x: float,
    y: float,
    resolution_m: float,
) -> float:
    if grid is None:
        return 0.0
    arr = np.asarray(grid)
    if arr.ndim != 2 or arr.size == 0:
        return 0.0
    res = max(float(resolution_m), 1e-6)
    col = int(x / res)
    row = int(y / res)
    if row < 0 or col < 0 or row >= arr.shape[0] or col >= arr.shape[1]:
        return 0.0
    return float(arr[row, col])


def forward_cone_values(
    grid: Optional[np.ndarray],
    pose_xy: tuple[float, float, float],
    *,
    resolution_m: float,
    reach_m: float = 0.95,
    half_width_m: float = 0.38,
    n_along: int = 8,
    n_across: int = 5,
) -> np.ndarray:
    """Sample a grid in a short body-forward rectangle."""
    if grid is None:
        return np.zeros(0, dtype=np.float32)
    x, y, theta = pose_xy
    c, s = math.cos(theta), math.sin(theta)
    vals: list[float] = []
    for t in np.linspace(0.12, reach_m, n_along):
        for w in np.linspace(-half_width_m, half_width_m, n_across):
            px = x + float(t) * c - float(w) * s
            py = y + float(t) * s + float(w) * c
            vals.append(sample_grid(grid, px, py, resolution_m))
    return np.asarray(vals, dtype=np.float32)


def forward_hazard(
    obs: dict[str, Any],
    *,
    resolution_m: float,
    info: Optional[dict[str, Any]] = None,
    reach_m: float = 0.95,
) -> float:
    pose = pose_xy_theta(obs, info)
    vals = forward_cone_values(
        obs.get("hazard"),
        pose,
        resolution_m=resolution_m,
        reach_m=reach_m,
    )
    if vals.size == 0:
        return 0.0
    return float(np.max(vals))


def mask_action(
    action: np.ndarray,
    obs: dict[str, Any],
    *,
    resolution_m: float,
    info: Optional[dict[str, Any]] = None,
    blocked_threshold: float = float(HAZARD_DRAIN_EDGE),
    steep_threshold: float = float(HAZARD_STEEP),
    steep_scale: float = 0.45,
) -> np.ndarray:
    """Clamp a 3-vector action using the observer hazard map.

    Drain lip / channel ahead → no forward wheels, trimmer off.
    Steep-only ahead → scale forward wheels (still allow reverse).
    """
    out = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    if out.size < 3:
        padded = np.zeros(3, dtype=np.float32)
        padded[: out.size] = out
        out = padded
    hz = forward_hazard(obs, resolution_m=resolution_m, info=info)
    left, right = float(out[0]), float(out[1])
    if hz >= blocked_threshold:
        if left > 0.0:
            out[0] = 0.0
        if right > 0.0:
            out[1] = 0.0
        out[2] = 0.0
        return out[:3]
    if hz >= steep_threshold and left > 0.0 and right > 0.0:
        out[0] = left * steep_scale
        out[1] = right * steep_scale
    return out[:3]

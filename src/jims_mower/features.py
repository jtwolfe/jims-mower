"""Compact numpy feature vector for BC / RL stubs (no camera pixels)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from jims_mower.action_mask import forward_cone_values, pose_xy_theta, sample_grid
from jims_mower.constants import (
    BC_FEATURE_DIM,
    HAZARD_DRAIN,
    HAZARD_DRAIN_EDGE,
    HAZARD_STEEP,
    NEAR_MISS_LIVING_M,
    TERRAIN_ADVICE,
)

_ADVICE_RANK = {name: float(i) / 3.0 for i, name in enumerate(("ok", "slow", "reroute", "stop"))}


def _vec(obs: dict[str, Any], key: str, size: int, default: Optional[np.ndarray] = None) -> np.ndarray:
    raw = obs.get(key)
    if raw is None:
        return np.zeros(size, dtype=np.float32) if default is None else np.asarray(default, dtype=np.float32)
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    out = np.zeros(size, dtype=np.float32)
    n = min(size, int(arr.size))
    out[:n] = arr[:n]
    return out


def _advice_rank(info: dict[str, Any], key: str) -> float:
    raw = str(info.get(key) or "ok")
    return _ADVICE_RANK.get(raw if raw in _ADVICE_RANK else "ok", 0.0)


def _coverage_frac(obs: dict[str, Any], info: dict[str, Any]) -> float:
    if info.get("coverage_fraction") is not None:
        return float(info["coverage_fraction"])
    cov = obs.get("coverage")
    if cov is None:
        return 0.0
    arr = np.asarray(cov, dtype=np.float32)
    grass = arr >= 0.0
    if not np.any(grass):
        return 0.0
    return float(np.mean(arr[grass] > 0.5))


def extract_features(
    obs: dict[str, Any],
    info: Optional[dict[str, Any]] = None,
    *,
    resolution_m: float = 0.10,
) -> np.ndarray:
    """Fixed-length float32 vector. No claimed representation quality."""
    info = info or {}
    pose = _vec(obs, "pose", 6)
    imu = _vec(obs, "imu", 6, np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0], dtype=np.float32))
    gps = _vec(obs, "gps", 4)
    tof = _vec(obs, "tof", 4)
    trim = _vec(obs, "trimmer_enabled", 1)
    xyth = pose_xy_theta(obs, info)
    hazard = obs.get("hazard")
    occupancy = obs.get("occupancy")
    coverage = obs.get("coverage")
    cone = forward_cone_values(hazard, xyth, resolution_m=resolution_m)
    if cone.size == 0:
        max_hz = mean_hz = drain_f = lip_f = steep_f = 0.0
    else:
        max_hz = float(np.max(cone))
        mean_hz = float(np.mean(cone))
        drain_f = float(np.mean(cone >= HAZARD_DRAIN))
        lip_f = float(np.mean((cone >= HAZARD_DRAIN_EDGE) & (cone < HAZARD_DRAIN + 0.5)))
        steep_f = float(np.mean((cone >= HAZARD_STEEP) & (cone < HAZARD_DRAIN_EDGE)))
    x, y, theta = xyth
    c, s = float(np.cos(theta)), float(np.sin(theta))
    left_xy = (x - 0.45 * s, y + 0.45 * c)
    right_xy = (x + 0.45 * s, y - 0.45 * c)
    ahead_xy = (x + 0.70 * c, y + 0.70 * s)
    hz_l = sample_grid(hazard, left_xy[0], left_xy[1], resolution_m)
    hz_r = sample_grid(hazard, right_xy[0], right_xy[1], resolution_m)
    occ_a = sample_grid(occupancy, ahead_xy[0], ahead_xy[1], resolution_m)
    cov_a = sample_grid(coverage, ahead_xy[0], ahead_xy[1], resolution_m)
    uncut = 1.0 if 0.0 <= cov_a < 0.5 else 0.0
    person_m = info.get("nearest_person_m")
    if person_m is None or not np.isfinite(float(person_m)):
        person_n = 1.0
    else:
        person_n = float(min(1.0, float(person_m) / max(NEAR_MISS_LIVING_M, 1e-6)))
    hand = int(np.asarray(obs.get("hand_signal", 0)).reshape(-1)[0]) if "hand_signal" in obs else 0
    hand_oh = np.zeros(5, dtype=np.float32)
    if 0 <= hand < 5:
        hand_oh[hand] = 1.0
    feats = np.concatenate(
        [
            pose,
            imu,
            gps,
            tof,
            trim,
            np.array(
                [
                    max_hz,
                    mean_hz,
                    drain_f,
                    lip_f,
                    steep_f,
                    hz_l,
                    hz_r,
                    occ_a,
                    uncut,
                    person_n,
                    _advice_rank(info, "terrain_advice"),
                    _advice_rank(info, "living_advice"),
                    _advice_rank(info, "geofence_advice"),
                    _coverage_frac(obs, info),
                ],
                dtype=np.float32,
            ),
            hand_oh,
        ]
    )
    if feats.size < BC_FEATURE_DIM:
        padded = np.zeros(BC_FEATURE_DIM, dtype=np.float32)
        padded[: feats.size] = feats
        return padded
    return feats[:BC_FEATURE_DIM].astype(np.float32, copy=False)

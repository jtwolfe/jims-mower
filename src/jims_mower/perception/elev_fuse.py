"""Gym MAP-2 elevation fuse: stereo + ToF corners + local IMU grade.

Not a field stereo matcher. Gym ideal disparity is the metric stereo
source when a 6–12 cm pair exists. Locked / ``elevation_set`` cells stay
frozen. A mono-depth prior (MAP-2b) may fill gaps only — it cannot
override valid stereo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np

from jims_mower.kinematics import wheel_positions
from jims_mower.perception.grade import (
    attitude_from_accel,
    gradients_from_attitude,
    local_disk_mask,
)
from jims_mower.types import Pose

FUSE_NOTE = (
    "gym stereo (ideal disparity) + ToF corners + local IMU — "
    "not a field matcher, fps_claim null, map_claim null"
)


@runtime_checkable
class MonoDepthPrior(Protocol):
    """MAP-2b slot. Orin-class mono depth as a *prior* only."""

    def sample(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """Return (elevation, valid_mask). Valid cells may fill *gaps* only."""
        ...


class NullMonoDepthPrior:
    """Empty MAP-2b prior. Never writes."""

    def sample(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = int(shape[0]), int(shape[1])
        return (
            np.zeros((rows, cols), dtype=np.float32),
            np.zeros((rows, cols), dtype=bool),
        )


def apply_mono_prior(
    elev: np.ndarray,
    written: np.ndarray,
    prior: np.ndarray,
    prior_valid: np.ndarray,
) -> np.ndarray:
    """Fill gaps only. Stereo / ToF / IMU cells already in ``written`` stay."""
    out = np.asarray(elev, dtype=np.float32).copy()
    take = np.asarray(prior_valid, dtype=bool) & ~np.asarray(written, dtype=bool)
    plane = np.asarray(prior, dtype=np.float32)
    if plane.shape != out.shape or take.shape != out.shape:
        return out
    if np.any(take):
        out[take] = plane[take]
    return out


@dataclass
class ElevFuseResult:
    """One fused elev raster plus hit masks. Not a published score."""

    elevation: np.ndarray
    stereo_hits: np.ndarray
    tof_hits: np.ndarray
    imu_hits: np.ndarray
    prior_hits: np.ndarray
    source: str = "gym_stereo_tof_imu"
    note: str = FUSE_NOTE
    not_matcher: bool = True
    fps_claim: None = None
    map_claim: None = None
    n_stereo: int = 0
    n_tof: int = 0
    n_imu: int = 0
    n_prior: int = 0

    def as_info(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "note": self.note,
            "not_matcher": True,
            "fps_claim": None,
            "map_claim": None,
            "n_stereo": int(self.n_stereo),
            "n_tof": int(self.n_tof),
            "n_imu": int(self.n_imu),
            "n_prior": int(self.n_prior),
        }


def _locked_mask(locked: Optional[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """MAP READY lock. Stereo / ToF may still overwrite ``elevation_set``."""
    freeze = np.zeros(shape, dtype=bool)
    if locked is not None:
        lk = np.asarray(locked, dtype=bool)
        if lk.shape == shape:
            freeze |= lk
    return freeze


def _set_mask(elevation_set: Optional[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    have = np.zeros(shape, dtype=bool)
    if elevation_set is not None:
        st = np.asarray(elevation_set, dtype=bool)
        if st.shape == shape:
            have |= st
    return have


def _stamp_tof_metric(
    elev: np.ndarray,
    hits: np.ndarray,
    tof: np.ndarray,
    pose: Pose,
    *,
    resolution_m: float,
    length_m: float,
    track_m: float,
    hover_m: float,
    writable: np.ndarray,
) -> int:
    arr = np.asarray(tof, dtype=np.float32).reshape(-1)
    if arr.size < 4:
        return 0
    rows, cols = elev.shape
    res = max(float(resolution_m), 1e-6)
    wheels = wheel_positions(pose, length_m, track_m)
    rad = max(1, int(round(0.16 / res)))
    n = 0
    for rng, (wx, wy) in zip(arr[:4], wheels):
        if not np.isfinite(rng) or float(rng) <= 0.0:
            continue
        col = int(wx / res)
        row = int(wy / res)
        if not (0 <= row < rows and 0 <= col < cols):
            continue
        z = float(pose.z + hover_m - float(rng))
        r0, r1 = max(0, row - rad), min(rows, row + rad + 1)
        c0, c1 = max(0, col - rad), min(cols, col + rad + 1)
        if r0 >= r1 or c0 >= c1:
            continue
        patch_w = writable[r0:r1, c0:c1]
        if not np.any(patch_w):
            continue
        elev[r0:r1, c0:c1][patch_w] = z
        hits[r0:r1, c0:c1] |= patch_w
        n += int(patch_w.sum())
    return n


def _stamp_imu_local(
    elev: np.ndarray,
    hits: np.ndarray,
    pose: Pose,
    imu: np.ndarray,
    *,
    resolution_m: float,
    radius_m: float,
    writable: np.ndarray,
) -> int:
    """Local chassis-grade disk. Does not hinge the whole yard."""
    roll, pitch = attitude_from_accel(np.asarray(imu, dtype=np.float32))
    gx, gy = gradients_from_attitude(roll, pitch, pose.theta)
    disk = local_disk_mask(elev.shape, (pose.x, pose.y), radius_m, resolution_m)
    take = disk & writable
    if not np.any(take):
        return 0
    rows, cols = elev.shape
    res = max(float(resolution_m), 1e-6)
    yy = (np.arange(rows, dtype=np.float32) + 0.5) * res
    xx = (np.arange(cols, dtype=np.float32) + 0.5) * res
    grid_x, grid_y = np.meshgrid(xx, yy)
    plane = (
        np.float32(pose.z)
        + np.float32(gx) * (grid_x - np.float32(pose.x))
        + np.float32(gy) * (grid_y - np.float32(pose.y))
    )
    elev[take] = plane[take]
    hits[take] = True
    return int(take.sum())


def fuse_elev_stereo_tof_imu(
    *,
    shape: tuple[int, int],
    resolution_m: float,
    world_size: tuple[float, float],
    pose: Pose,
    stereo_elev: Optional[np.ndarray] = None,
    stereo_hits: Optional[np.ndarray] = None,
    tof: Optional[np.ndarray] = None,
    imu: Optional[np.ndarray] = None,
    length_m: float = 0.50,
    track_m: float = 0.40,
    hover_m: float = 0.06,
    imu_radius_m: float = 1.2,
    elevation: Optional[np.ndarray] = None,
    locked: Optional[np.ndarray] = None,
    elevation_set: Optional[np.ndarray] = None,
    rgb_prior: Optional[np.ndarray] = None,
    mono_prior: Optional[MonoDepthPrior] = None,
    prior_weight: float = 0.0,
) -> ElevFuseResult:
    """Unify elev updates. Priority: freeze > stereo > ToF > IMU > prior.

    ``prior_weight`` only mixes the RGB/grade prior into *unwritten* cells.
    ``mono_prior`` is the MAP-2b slot and cannot override valid stereo.
    ``world_size`` is accepted for call-site symmetry (unused here).
    """
    _ = world_size
    rows, cols = int(shape[0]), int(shape[1])
    elev = (
        np.zeros((rows, cols), dtype=np.float32)
        if elevation is None
        else np.asarray(elevation, dtype=np.float32).copy()
    )
    if elev.shape != (rows, cols):
        raise ValueError("elevation shape must match fuse shape")
    locked_m = _locked_mask(locked, (rows, cols))
    already = _set_mask(elevation_set, (rows, cols))
    # Metric stereo / ToF may overwrite unlocked first-stamp cells.
    metric_writable = ~locked_m
    stereo_m = np.zeros((rows, cols), dtype=bool)
    tof_m = np.zeros((rows, cols), dtype=bool)
    imu_m = np.zeros((rows, cols), dtype=bool)
    prior_m = np.zeros((rows, cols), dtype=bool)
    written = np.zeros((rows, cols), dtype=bool)

    if stereo_elev is not None and stereo_hits is not None:
        se = np.asarray(stereo_elev, dtype=np.float32)
        sh = np.asarray(stereo_hits, dtype=bool)
        if se.shape == elev.shape and sh.shape == elev.shape:
            take = sh & metric_writable
            if np.any(take):
                elev[take] = se[take]
                stereo_m[take] = True
                written[take] = True

    if tof is not None:
        tof_writable = metric_writable & ~stereo_m
        n_tof = _stamp_tof_metric(
            elev,
            tof_m,
            np.asarray(tof, dtype=np.float32),
            pose,
            resolution_m=resolution_m,
            length_m=length_m,
            track_m=track_m,
            hover_m=hover_m,
            writable=tof_writable,
        )
        if n_tof:
            written |= tof_m

    if imu is not None:
        # Local IMU grade: never flop locked, stereo, ToF, or first-stamp cells.
        imu_writable = ~locked_m & ~already & ~written
        n_imu = _stamp_imu_local(
            elev,
            imu_m,
            pose,
            np.asarray(imu, dtype=np.float32),
            resolution_m=resolution_m,
            radius_m=max(float(imu_radius_m), resolution_m),
            writable=imu_writable,
        )
        if n_imu:
            written |= imu_m

    if rgb_prior is not None:
        prior = np.asarray(rgb_prior, dtype=np.float32)
        if prior.shape == elev.shape:
            take = ~locked_m & ~already & ~written
            if np.any(take):
                w = float(np.clip(prior_weight, 0.0, 1.0))
                if w > 0.0:
                    elev[take] = (1.0 - w) * elev[take] + w * prior[take]
                else:
                    elev[take] = prior[take]
                prior_m[take] = True
                written[take] = True

    if mono_prior is not None:
        # Gaps only. Stereo hits must not be overwritten (MAP-2b contract).
        gap = ~locked_m & ~stereo_m
        plane, valid = mono_prior.sample((rows, cols))
        take = np.asarray(valid, dtype=bool) & gap & ~written
        plane_a = np.asarray(plane, dtype=np.float32)
        if plane_a.shape == elev.shape and take.shape == elev.shape and np.any(take):
            elev[take] = plane_a[take]
            prior_m[take] = True
            written[take] = True

    return ElevFuseResult(
        elevation=elev,
        stereo_hits=stereo_m,
        tof_hits=tof_m,
        imu_hits=imu_m,
        prior_hits=prior_m,
        n_stereo=int(stereo_m.sum()),
        n_tof=int(tof_m.sum()),
        n_imu=int(imu_m.sum()),
        n_prior=int(prior_m.sum()),
    )

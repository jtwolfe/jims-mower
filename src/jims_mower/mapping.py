"""Persistent BEV occupancy, gym elev fuse, sparse pose-assist stub.

None of this is a SLAM / COLMAP stack. Occupancy is detections + downward
ToF, not the god-view obstacle list. ``fuse_height_rgb_tof`` is the gym
MAP-2 path: ideal stereo (when a pair exists) + ToF corners + local IMU
grade, with RGB labels as a gap prior. Locked cells stay frozen. Loop
closure compares a coarse occupancy fingerprint; optional landmark
revisit is still ``not_slam: true``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np

from jims_mower.cameras import camera_world_pose, ground_hits
from jims_mower.constants import DEFAULT_RADII
from jims_mower.maps import OccupancyMap
from jims_mower.perception.cv_terrain import classify_terrain_rgb
from jims_mower.perception.elev_fuse import (
    MonoDepthPrior,
    fuse_elev_stereo_tof_imu,
)
from jims_mower.perception.stereo import (
    find_stereo_pair,
    rasterize_points,
    synthetic_stereo_points,
)
from jims_mower.types import CameraSpec, Detection, Pose


class PersistentOccupancy:
    """Decaying BEV occupancy from detections and short ToF hits."""

    def __init__(
        self,
        rows: int,
        cols: int,
        *,
        decay: float = 0.55,
        tof_hit_m: float = 0.18,
    ) -> None:
        self.decay = float(np.clip(decay, 0.0, 1.0))
        self.tof_hit_m = float(tof_hit_m)
        self.grid = np.zeros((int(rows), int(cols)), dtype=np.float32)

    def reset(self) -> None:
        self.grid.fill(0.0)

    def update(
        self,
        detections: Iterable[Detection],
        *,
        resolution_m: float,
        tof: Optional[np.ndarray] = None,
        pose: Optional[Pose] = None,
        length_m: float = 0.50,
        track_m: float = 0.40,
        inflate_m: float = 0.15,
    ) -> np.ndarray:
        self.grid *= self.decay
        occ = OccupancyMap(self.grid.shape[0], self.grid.shape[1])
        occ.grid = self.grid
        for det in detections:
            if det.world_xy is None:
                continue
            radius = DEFAULT_RADII.get(det.label, 0.2) + inflate_m
            occ.paint_circle(det.world_xy[0], det.world_xy[1], radius, resolution_m)
        if tof is not None and pose is not None:
            _stamp_tof_hits(
                occ,
                np.asarray(tof, dtype=np.float32).reshape(-1),
                pose,
                resolution_m=resolution_m,
                length_m=length_m,
                track_m=track_m,
                hit_m=self.tof_hit_m,
            )
        np.clip(self.grid, 0.0, 1.0, out=self.grid)
        return self.grid


def _stamp_tof_hits(
    occ: OccupancyMap,
    tof: np.ndarray,
    pose: Pose,
    *,
    resolution_m: float,
    length_m: float,
    track_m: float,
    hit_m: float,
) -> None:
    """Short downward ranges paint a local blob at that wheel — not god-view."""
    if tof.size < 4:
        return
    half_l = 0.5 * float(length_m)
    half_t = 0.5 * float(track_m)
    c = float(np.cos(pose.theta))
    s = float(np.sin(pose.theta))
    # FL, FR, RL, RR in body frame (x forward, y left).
    corners = (
        (half_l, half_t),
        (half_l, -half_t),
        (-half_l, half_t),
        (-half_l, -half_t),
    )
    for rng, (bx, by) in zip(tof[:4], corners):
        if not np.isfinite(rng) or float(rng) <= 0.0 or float(rng) > hit_m:
            continue
        wx = pose.x + bx * c - by * s
        wy = pose.y + bx * s + by * c
        occ.paint_circle(wx, wy, 0.18, resolution_m, value=0.65)


def _rgb_height_hint(
    images: dict[str, np.ndarray],
    cameras: list[CameraSpec],
    pose: Pose,
    *,
    shape: tuple[int, int],
    resolution_m: float,
    world_size: tuple[float, float],
    base: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Low-priority RGB drain/bank hint. Gaps only — stereo wins later."""
    rows, cols = shape
    elev = (
        np.zeros((rows, cols), dtype=np.float32)
        if base is None
        else np.asarray(base, dtype=np.float32).copy()
    )
    for cam in cameras:
        frame = images.get(cam.name)
        if frame is None:
            continue
        labels = classify_terrain_rgb(frame)
        world_cam = camera_world_pose(pose, cam)
        h, w = labels.shape
        hx, hy, valid = ground_hits(world_cam, w, h, ground_z=0.0)
        inside = (
            valid
            & (labels > 0)
            & np.isfinite(hx)
            & np.isfinite(hy)
            & (hx >= 0.0)
            & (hy >= 0.0)
            & (hx < world_size[0])
            & (hy < world_size[1])
        )
        if not np.any(inside):
            continue
        rr = np.floor(hy[inside] / max(resolution_m, 1e-6)).astype(np.int32)
        cc = np.floor(hx[inside] / max(resolution_m, 1e-6)).astype(np.int32)
        ok = (rr >= 0) & (cc >= 0) & (rr < rows) & (cc < cols)
        if not np.any(ok):
            continue
        lab = labels[inside][ok].astype(np.int32)
        rr, cc = rr[ok], cc[ok]
        hint = np.where(lab >= 3, -0.10, np.where(lab == 2, -0.04, 0.06))
        elev[rr, cc] = 0.7 * elev[rr, cc] + 0.3 * hint.astype(np.float32)
    return elev


def _gym_stereo_rasters(
    cameras: list[CameraSpec],
    pose: Pose,
    images: dict[str, np.ndarray],
    *,
    shape: tuple[int, int],
    resolution_m: float,
    world_size: tuple[float, float],
    height_at: Optional[Any] = None,
    width: int = 80,
    height: int = 60,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Ideal gym stereo when a 6–12 cm pair exists. Look-arounds → None."""
    pair = find_stereo_pair(cameras)
    if pair is None:
        return None, None
    if images:
        sample = next(iter(images.values()), None)
        if sample is not None and getattr(sample, "ndim", 0) >= 2:
            height = int(sample.shape[0])
            width = int(sample.shape[1])
    if height_at is None:
        height_at = lambda _x, _y: float(pose.z)
    xs, ys, zs = synthetic_stereo_points(
        pose,
        pair,
        width=int(width),
        height=int(height),
        height_at=height_at,
        pixel_stride=3 if height > 40 else 2,
    )
    raster, hits = rasterize_points(
        xs,
        ys,
        zs,
        shape=shape,
        resolution_m=resolution_m,
        width_m=float(world_size[0]),
        height_m=float(world_size[1]),
    )
    return raster, hits


def fuse_height_rgb_tof(
    images: dict[str, np.ndarray],
    cameras: list[CameraSpec],
    pose: Pose,
    tof: np.ndarray,
    *,
    shape: tuple[int, int],
    resolution_m: float,
    world_size: tuple[float, float],
    length_m: float = 0.50,
    track_m: float = 0.40,
    hover_m: float = 0.06,
    elevation: Optional[np.ndarray] = None,
    prior: Optional[np.ndarray] = None,
    prior_weight: float = 0.65,
    imu: Optional[np.ndarray] = None,
    locked: Optional[np.ndarray] = None,
    elevation_set: Optional[np.ndarray] = None,
    stereo_elev: Optional[np.ndarray] = None,
    stereo_hits: Optional[np.ndarray] = None,
    height_at: Optional[Any] = None,
    mono_prior: Optional[MonoDepthPrior] = None,
    image_size: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """Gym MAP-2 fuse: ideal stereo + ToF + local IMU. RGB is a gap prior.

    A true 6–12 cm pair reconstructs near-field height from gym ideal
    disparity (not a matcher). Look-around rigs skip stereo. Locked cells
    stay frozen. ``mono_prior`` (MAP-2b) cannot override valid stereo.
    """
    rows, cols = shape
    elev0 = (
        np.zeros((rows, cols), dtype=np.float32)
        if elevation is None
        else np.asarray(elevation, dtype=np.float32).copy()
    )
    if elev0.shape != (rows, cols):
        raise ValueError("elevation shape must match the grass grid")
    rgb_hint = _rgb_height_hint(
        images,
        cameras,
        pose,
        shape=shape,
        resolution_m=resolution_m,
        world_size=world_size,
        base=elev0,
    )
    se, sh = stereo_elev, stereo_hits
    if se is None or sh is None:
        w, h = (80, 60) if image_size is None else (int(image_size[0]), int(image_size[1]))
        sampler = height_at
        if sampler is None and prior is not None:
            plane = np.asarray(prior, dtype=np.float32)
            if plane.shape == (rows, cols):
                res = max(float(resolution_m), 1e-6)

                def sampler(x: float, y: float, _p=plane, _res=res) -> float:
                    if x < 0.0 or y < 0.0:
                        return float(pose.z)
                    rr = int(y / _res)
                    cc = int(x / _res)
                    if 0 <= rr < _p.shape[0] and 0 <= cc < _p.shape[1]:
                        return float(_p[rr, cc])
                    return float(pose.z)

        se, sh = _gym_stereo_rasters(
            cameras,
            pose,
            images,
            shape=shape,
            resolution_m=resolution_m,
            world_size=world_size,
            height_at=sampler,
            width=w,
            height=h,
        )
    result = fuse_elev_stereo_tof_imu(
        shape=shape,
        resolution_m=resolution_m,
        world_size=world_size,
        pose=pose,
        stereo_elev=se,
        stereo_hits=sh,
        tof=tof,
        imu=imu,
        length_m=length_m,
        track_m=track_m,
        hover_m=hover_m,
        elevation=elev0,
        locked=locked,
        elevation_set=elevation_set,
        rgb_prior=rgb_hint if prior is None else np.asarray(prior, dtype=np.float32),
        mono_prior=mono_prior,
        prior_weight=float(prior_weight) if prior is not None else 0.0,
    )
    return result.elevation


def _fuse_tof_elevation(
    elev: np.ndarray,
    tof: np.ndarray,
    pose: Pose,
    *,
    resolution_m: float,
    length_m: float,
    track_m: float,
    hover_m: float,
) -> None:
    if tof.size < 4:
        return
    rows, cols = elev.shape
    half_l = 0.5 * float(length_m)
    half_t = 0.5 * float(track_m)
    c = float(np.cos(pose.theta))
    s = float(np.sin(pose.theta))
    corners = (
        (half_l, half_t),
        (half_l, -half_t),
        (-half_l, half_t),
        (-half_l, -half_t),
    )
    rad = max(1, int(round(0.16 / max(resolution_m, 1e-6))))
    for rng, (bx, by) in zip(tof[:4], corners):
        if not np.isfinite(rng) or float(rng) <= 0.0:
            continue
        wx = pose.x + bx * c - by * s
        wy = pose.y + bx * s + by * c
        col = int(wx / max(resolution_m, 1e-6))
        row = int(wy / max(resolution_m, 1e-6))
        z = float(pose.z + hover_m - rng)
        r0, r1 = max(0, row - rad), min(rows, row + rad + 1)
        c0, c1 = max(0, col - rad), min(cols, col + rad + 1)
        if r0 >= r1 or c0 >= c1:
            continue
        elev[r0:r1, c0:c1] = 0.5 * elev[r0:r1, c0:c1] + 0.5 * z


@dataclass
class LoopHit:
    """A revisit fingerprint. No pose-graph update."""

    cell: tuple[int, int]
    score: float
    first_step: int
    step: int


@dataclass
class LoopClosureStub:
    """Detects 'I have been near here' from a coarse occupancy hash.

    Optional landmark revisit stores taught fence vertices and a 2-D
    translation on a matching revisit. Still not SLAM: no scan-match,
    no pose-graph optimizer, ``not_slam: true``.
    """

    cell_m: float = 2.0
    min_step_gap: int = 12
    match: float = 0.72
    max_correct_m: float = 1.50
    _seen: dict[tuple[int, int], tuple[np.ndarray, int]] = field(default_factory=dict)
    _step: int = 0
    last_hit: Optional[LoopHit] = None
    landmarks: list[tuple[float, float]] = field(default_factory=list)
    _landmark_xy: dict[int, tuple[float, float]] = field(default_factory=dict)
    _landmark_fp: dict[int, np.ndarray] = field(default_factory=dict)
    last_dx: float = 0.0
    last_dy: float = 0.0
    last_fence_err_m: Optional[float] = None

    def reset(self) -> None:
        self._seen.clear()
        self._step = 0
        self.last_hit = None
        self._landmark_xy.clear()
        self._landmark_fp.clear()
        self.last_dx = 0.0
        self.last_dy = 0.0
        self.last_fence_err_m = None

    def teach_vertices(self, vertices: Iterable[tuple[float, float]]) -> None:
        """Taught fence vertices for sparse revisit. Not a geofence Earth frame."""
        self.landmarks = [(float(x), float(y)) for x, y in vertices]

    def fingerprint(self, occupancy: np.ndarray, pose: Pose, resolution_m: float) -> np.ndarray:
        """3×3 coarse occupancy around the robot, 0/1."""
        grid = np.asarray(occupancy, dtype=np.float32)
        res = max(float(resolution_m), 1e-6)
        col = int(pose.x / res)
        row = int(pose.y / res)
        patch = np.zeros((3, 3), dtype=np.float32)
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                rr, cc = row + dr * 4, col + dc * 4
                if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
                    patch[dr + 1, dc + 1] = 1.0 if grid[rr, cc] > 0.35 else 0.0
        return patch.reshape(-1)

    def update(
        self,
        occupancy: np.ndarray,
        pose: Pose,
        resolution_m: float,
    ) -> Optional[LoopHit]:
        self._step += 1
        cell = (
            int(pose.x / max(self.cell_m, 1e-6)),
            int(pose.y / max(self.cell_m, 1e-6)),
        )
        fp = self.fingerprint(occupancy, pose, resolution_m)
        hit: Optional[LoopHit] = None
        best = 0.0
        best_key: Optional[tuple[int, int]] = None
        best_step = 0
        for key, (prev, first) in self._seen.items():
            if key == cell:
                continue
            if self._step - first < self.min_step_gap:
                continue
            # Hamming-ish overlap on the 9-bit patch.
            denom = float(np.maximum(fp + prev, 1e-6).clip(0.0, 2.0).sum())
            score = float(np.minimum(fp, prev).sum() / max(denom, 1.0) * 2.0)
            # Also accept an exact match on a sparse patch.
            if float(np.mean(fp == prev)) > score:
                score = float(np.mean(fp == prev))
            if score > best:
                best, best_key, best_step = score, key, first
        if best_key is not None and best >= self.match:
            hit = LoopHit(cell=best_key, score=best, first_step=best_step, step=self._step)
            self.last_hit = hit
        self._seen[cell] = (fp, self._seen[cell][1] if cell in self._seen else self._step)
        self._update_landmarks(fp, pose)
        return hit

    def _nearest_landmark(self, pose: Pose) -> Optional[int]:
        if not self.landmarks:
            return None
        best_i = None
        best_d = float("inf")
        for i, (x, y) in enumerate(self.landmarks):
            d = float(np.hypot(pose.x - x, pose.y - y))
            if d < best_d:
                best_d, best_i = d, i
        if best_i is None or best_d > max(self.cell_m * 1.5, 1.2):
            return None
        return best_i

    def _update_landmarks(self, fp: np.ndarray, pose: Pose) -> None:
        idx = self._nearest_landmark(pose)
        if idx is None:
            return
        if idx not in self._landmark_fp:
            self._landmark_fp[idx] = fp.copy()
            self._landmark_xy[idx] = (float(pose.x), float(pose.y))
            return
        prev = self._landmark_fp[idx]
        if float(np.mean(fp == prev)) < max(0.45, self.match * 0.7):
            return
        taught = self.landmarks[idx]
        # Revisit: pull the live pose back toward the taught vertex (capped).
        raw_dx = float(taught[0] - pose.x)
        raw_dy = float(taught[1] - pose.y)
        cap = max(float(self.max_correct_m), 0.0)
        nrm = float(np.hypot(raw_dx, raw_dy))
        if nrm > cap > 0.0:
            raw_dx *= cap / nrm
            raw_dy *= cap / nrm
        self.last_dx = raw_dx
        self.last_dy = raw_dy
        corrected = self.corrected_xy(pose)
        self.last_fence_err_m = float(np.hypot(corrected[0] - taught[0], corrected[1] - taught[1]))

    def corrected_xy(self, pose: Pose) -> tuple[float, float]:
        return float(pose.x + self.last_dx), float(pose.y + self.last_dy)

    def apply(self, pose: Pose) -> Pose:
        x, y = self.corrected_xy(pose)
        return Pose(x, y, pose.theta, z=pose.z, pitch=pose.pitch, roll=pose.roll)

    def fence_error_m(
        self,
        estimated: Iterable[tuple[float, float]],
        taught: Optional[Iterable[tuple[float, float]]] = None,
    ) -> float:
        """Mean vertex error vs taught fence. Stated gym metre band, not SLAM."""
        verts = list(self.landmarks if taught is None else taught)
        est = list(estimated)
        if not verts or not est:
            return 0.0
        n = min(len(verts), len(est))
        err = [
            float(np.hypot(est[i][0] - verts[i][0], est[i][1] - verts[i][1]))
            for i in range(n)
        ]
        return float(np.mean(err)) if err else 0.0

    def as_info(self) -> dict[str, Any]:
        hit = self.last_hit
        blob: dict[str, Any] = {
            "loop_closure": hit is not None,
            "loop_score": 0.0 if hit is None else float(hit.score),
            "not_slam": True,
            "pose_assist": bool(self.landmarks),
            "max_correct_m": float(self.max_correct_m),
            "assist_dx_m": float(self.last_dx),
            "assist_dy_m": float(self.last_dy),
        }
        if hit is not None:
            blob["loop_cell"] = list(hit.cell)
            blob["loop_first_step"] = int(hit.first_step)
        if self.last_fence_err_m is not None:
            blob["fence_err_m"] = float(self.last_fence_err_m)
        return blob

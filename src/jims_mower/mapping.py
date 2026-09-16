"""Persistent BEV occupancy, height-map fusion stub, loop-closure stub.

None of this is a SLAM / COLMAP stack. Occupancy is detections + downward
ToF, not the god-view obstacle list. ``fuse_height_rgb_tof`` back-projects
RGB terrain labels onto the ground plane and stamps ToF wheel corners —
not the live metric path. Near-field stereo lives in
``perception.stereo`` and stamps ``ObservedMap`` directly. Loop closure
only compares a coarse occupancy fingerprint to earlier cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np

from jims_mower.cameras import camera_world_pose, ground_hits
from jims_mower.constants import DEFAULT_RADII
from jims_mower.maps import OccupancyMap
from jims_mower.perception.cv_terrain import classify_terrain_rgb
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
) -> np.ndarray:
    """RGB ground-plane back-projection + downward ToF. Persistent-friendly.

    Drain / bank pixels from ``classify_terrain_rgb`` seed a height hint
    (negative for channels). ToF corners overwrite a disk around each wheel.
    This is a fusion *stub*, not a dense stereo / lidar map.
    """
    rows, cols = shape
    elev = (
        np.zeros((rows, cols), dtype=np.float32)
        if elevation is None
        else np.asarray(elevation, dtype=np.float32).copy()
    )
    if elev.shape != (rows, cols):
        raise ValueError("elevation shape must match the grass grid")
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
        # Channel → negative hint; lip/bank → small positive bump.
        hint = np.where(lab >= 3, -0.10, np.where(lab == 2, -0.04, 0.06))
        elev[rr, cc] = 0.7 * elev[rr, cc] + 0.3 * hint.astype(np.float32)
    _fuse_tof_elevation(
        elev,
        np.asarray(tof, dtype=np.float32).reshape(-1),
        pose,
        resolution_m=resolution_m,
        length_m=length_m,
        track_m=track_m,
        hover_m=hover_m,
    )
    if prior is not None:
        plane = np.asarray(prior, dtype=np.float32)
        if plane.shape == elev.shape:
            w = float(np.clip(prior_weight, 0.0, 1.0))
            elev = ((1.0 - w) * elev + w * plane).astype(np.float32)
    return elev


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

    Not SLAM: no scan-match, no pose-graph, no map correction.
    """

    cell_m: float = 2.0
    min_step_gap: int = 12
    match: float = 0.72
    _seen: dict[tuple[int, int], tuple[np.ndarray, int]] = field(default_factory=dict)
    _step: int = 0
    last_hit: Optional[LoopHit] = None

    def reset(self) -> None:
        self._seen.clear()
        self._step = 0
        self.last_hit = None

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
        return hit

    def as_info(self) -> dict[str, Any]:
        hit = self.last_hit
        if hit is None:
            return {"loop_closure": False, "loop_score": 0.0, "not_slam": True}
        return {
            "loop_closure": True,
            "loop_score": float(hit.score),
            "loop_cell": list(hit.cell),
            "loop_first_step": int(hit.first_step),
            "not_slam": True,
        }

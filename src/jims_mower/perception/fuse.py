"""Multi-camera ground-plane BEV fuse for heuristic / learned hazard stamps.

Each camera contributes class + confidence on the flat-yard plane. Cells
take the highest-confidence class (ties keep the stronger label). No
claimed detector scores — confidence is a relative merge weight only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from jims_mower.cameras import camera_world_pose, ground_hits
from jims_mower.constants import HAZARD_DRAIN, HAZARD_DRAIN_EDGE, HAZARD_STEEP
from jims_mower.perception.cv_terrain import DEFAULT_MAX_RANGE_M
from jims_mower.types import CameraSpec, Pose


@dataclass
class BevStamp:
    """Sparse ground-plane votes from one camera."""

    rows: np.ndarray
    cols: np.ndarray
    labels: np.ndarray
    confidence: np.ndarray
    radii: np.ndarray
    camera: str = ""


def collect_stamps(
    labels: np.ndarray,
    cam: CameraSpec,
    pose: Pose,
    *,
    resolution_m: float,
    world_size: tuple[float, float],
    confidence: Optional[np.ndarray] = None,
    ground_z: float = 0.0,
    max_range_m: float = DEFAULT_MAX_RANGE_M,
    map_shape: tuple[int, int],
) -> BevStamp:
    """Back-project labelled pixels onto map cells with a range-based weight."""
    height, width = labels.shape
    world_cam = camera_world_pose(pose, cam)
    hx, hy, valid = ground_hits(world_cam, width, height, ground_z=ground_z)
    rng = np.hypot(hx - world_cam.x, hy - world_cam.y)
    if confidence is None:
        conf = np.ones(labels.shape, dtype=np.float32)
    else:
        conf = np.asarray(confidence, dtype=np.float32)
        if conf.shape != labels.shape:
            raise ValueError("confidence must match labels H×W")
    inside = (
        valid
        & (labels > 0)
        & (conf > 0.0)
        & (hx >= 0.0)
        & (hy >= 0.0)
        & (hx < world_size[0])
        & (hy < world_size[1])
        & (rng < max_range_m)
        & np.isfinite(hx)
        & np.isfinite(hy)
    )
    empty = BevStamp(
        np.zeros(0, dtype=np.int32),
        np.zeros(0, dtype=np.int32),
        np.zeros(0, dtype=np.float32),
        np.zeros(0, dtype=np.float32),
        np.zeros(0, dtype=np.int32),
        camera=cam.name,
    )
    if not np.any(inside):
        return empty
    rows_f = hy[inside] / max(resolution_m, 1e-6)
    cols_f = hx[inside] / max(resolution_m, 1e-6)
    lab = labels[inside].astype(np.float32)
    rng_v = rng[inside]
    conf_v = conf[inside] * np.clip(1.0 - 0.55 * (rng_v / max(max_range_m, 1e-6)), 0.15, 1.0)
    map_rows, map_cols = map_shape
    rr = np.floor(rows_f).astype(np.int32)
    cc = np.floor(cols_f).astype(np.int32)
    inb = (rr >= 0) & (cc >= 0) & (rr < map_rows) & (cc < map_cols)
    if not np.any(inb):
        return empty
    rr, cc, lab, rng_v, conf_v = rr[inb], cc[inb], lab[inb], rng_v[inb], conf_v[inb]
    radii = np.clip(np.ceil((0.10 + 0.035 * rng_v) / max(resolution_m, 1e-6)), 1, 6).astype(np.int32)
    keys = rr.astype(np.int64) * map_cols + cc.astype(np.int64)
    order = np.argsort(keys)
    keys, rr, cc, lab, radii, conf_v = (
        keys[order],
        rr[order],
        cc[order],
        lab[order],
        radii[order],
        conf_v[order],
    )
    uniq = np.flatnonzero(np.concatenate(([True], keys[1:] != keys[:-1])))
    next_break = np.append(uniq[1:], keys.size)
    keep_r, keep_c, keep_lab, keep_rad, keep_conf = [], [], [], [], []
    for a, b in zip(uniq.tolist(), next_break.tolist()):
        chunk_lab = lab[a:b]
        chunk_conf = conf_v[a:b]
        best = int(np.argmax(chunk_conf))
        keep_r.append(int(rr[a]))
        keep_c.append(int(cc[a]))
        keep_lab.append(float(chunk_lab[best]))
        keep_rad.append(int(radii[a:b].max()))
        keep_conf.append(float(chunk_conf[best]))
    return BevStamp(
        np.asarray(keep_r, dtype=np.int32),
        np.asarray(keep_c, dtype=np.int32),
        np.asarray(keep_lab, dtype=np.float32),
        np.asarray(keep_conf, dtype=np.float32),
        np.asarray(keep_rad, dtype=np.int32),
        camera=cam.name,
    )


def fuse_stamps(
    stamps: list[BevStamp],
    shape: tuple[int, int],
    *,
    min_confidence: float = 0.12,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge camera stamps. Returns (hazard, confidence) rasters."""
    rows, cols = shape
    hazard = np.zeros((rows, cols), dtype=np.float32)
    conf = np.zeros((rows, cols), dtype=np.float32)
    for stamp in stamps:
        if stamp.rows.size == 0:
            continue
        for row, col, lab, weight, rad in zip(
            stamp.rows.tolist(),
            stamp.cols.tolist(),
            stamp.labels.tolist(),
            stamp.confidence.tolist(),
            stamp.radii.tolist(),
        ):
            if weight < min_confidence:
                continue
            r0 = max(0, int(row) - int(rad))
            r1 = min(rows, int(row) + int(rad) + 1)
            c0 = max(0, int(col) - int(rad))
            c1 = min(cols, int(col) + int(rad) + 1)
            if r0 >= r1 or c0 >= c1:
                continue
            yy = np.arange(r0, r1)
            xx = np.arange(c0, c1)
            dy = yy[:, None] - int(row)
            dx = xx[None, :] - int(col)
            mask = dy * dy + dx * dx <= int(rad) * int(rad)
            patch_c = conf[r0:r1, c0:c1]
            patch_h = hazard[r0:r1, c0:c1]
            better = mask & (weight >= patch_c)
            # Equal confidence: keep the stronger (larger) hazard class.
            tie = mask & (np.abs(weight - patch_c) < 1e-6) & (lab > patch_h)
            take = better | tie
            patch_h[take] = lab
            patch_c[take] = np.maximum(patch_c[take], weight)
    return hazard, conf


def paint_geometry_from_hazard(
    hazard: np.ndarray,
    *,
    elevation: np.ndarray,
    slope: np.ndarray,
    pose_z: float,
    steep_rad: float,
) -> None:
    """Fill elevation / slope from a fused hazard raster (in-place)."""
    drain = hazard >= HAZARD_DRAIN_EDGE
    if np.any(drain):
        drop = np.where(hazard >= HAZARD_DRAIN, 0.14, 0.05).astype(np.float32)
        elevation[drain] = np.minimum(elevation[drain], np.float32(pose_z) - drop[drain])
        slope[drain] = np.maximum(slope[drain], np.float32(steep_rad))
    bank = hazard == HAZARD_STEEP
    if np.any(bank):
        elevation[bank] = np.maximum(elevation[bank], np.float32(pose_z + 0.08))
        slope[bank] = np.maximum(slope[bank], np.float32(steep_rad))


def fuse_camera_labels(
    images: dict[str, np.ndarray],
    label_fn,
    cameras: list[CameraSpec],
    pose: Pose,
    *,
    resolution_m: float,
    world_size: tuple[float, float],
    map_shape: tuple[int, int],
    max_range_m: float = DEFAULT_MAX_RANGE_M,
    min_confidence: float = 0.12,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify each camera (label_fn → labels or (labels, conf)) and fuse."""
    cams = {c.name: c for c in cameras}
    stamps: list[BevStamp] = []
    for name, frame in images.items():
        cam = cams.get(name)
        if cam is None or frame.ndim != 3:
            continue
        out = label_fn(frame, cam)
        if isinstance(out, tuple):
            labels, confidence = out
        else:
            labels, confidence = out, None
        stamps.append(
            collect_stamps(
                np.asarray(labels),
                cam,
                pose,
                resolution_m=resolution_m,
                world_size=world_size,
                confidence=None if confidence is None else np.asarray(confidence),
                max_range_m=max_range_m,
                map_shape=map_shape,
            )
        )
    return fuse_stamps(stamps, map_shape, min_confidence=min_confidence)

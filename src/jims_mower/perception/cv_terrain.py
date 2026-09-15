"""RGB / ToF terrain cues for the sim heuristic. Replace on the Orin.

The gym renderer paints drain channels dark brown, lips lighter brown, and
banks olive-green, then shades them by slope. These functions recover that
palette and back-project hits onto the robot's seated tangent plane. They
do **not** read the god-view height field.
"""

from __future__ import annotations

import math

import numpy as np

from jims_mower.cameras import attitude_plane_hits, camera_world_pose
from jims_mower.constants import HAZARD_DRAIN, HAZARD_DRAIN_EDGE, HAZARD_STEEP
from jims_mower.types import CameraSpec, Pose

# Plane-projection horizon. Beyond this, one pixel covers too much yard.
DEFAULT_MAX_RANGE_M = 9.0


def classify_terrain_rgb(image: np.ndarray) -> np.ndarray:
    """Per-pixel hazard labels from renderer-style colour.

    ``0`` none / grass / sky, ``1`` steep/bank, ``2`` drain lip, ``3`` channel.
    Conservative on brown (false drain cells block coverage) and looser on
    olive banks (extra steep cost is a slow corridor).
    """
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError("image must be HxWx3")
    r = image[:, :, 0].astype(np.int16)
    g = image[:, :, 1].astype(np.int16)
    b = image[:, :, 2].astype(np.int16)
    value = r.astype(np.int32) + g.astype(np.int32) + b.astype(np.int32)
    sky = (b > 140) & (b > r + 15) & (b > g)
    # Obstacle blobs (people / toys) are saturated; skip them.
    hot = (r > 180) | ((r > 150) & (g < 120) & (b < 90))

    # Drain / dirt family: R ≥ G ≥ B, not bright. Channel is the dark core
    # (DRAIN_RGB ≈ 58,42,28 plus extra darkening by depth). Lip is the
    # lighter brown (DRAIN_EDGE_RGB ≈ 86,62,40).
    brown = (
        (r > g - 2)
        & (g >= b - 6)
        & (r > 28)
        & (r < 130)
        & (g < 100)
        & (b < 85)
        & (value < 270)
        & (r - b > 8)
        & ~sky
        & ~hot
    )
    channel = brown & (value < 180) & (r < 95) & (g < 72)
    lip = brown & ~channel

    # Banks are olive: green-dominant but a higher R/G than uncut grass
    # (BANK_RGB 72/118 vs UNCUT 46/140). Ratio is shading-invariant.
    g_safe = np.maximum(g.astype(np.float32), 1.0)
    rg = r.astype(np.float32) / g_safe
    bank = (
        (g > r + 6)
        & (g > b + 4)
        & (g > 55)
        & (g < 145)
        & (rg > 0.46)
        & (rg < 0.88)
        & (r > 38)
        & ~sky
        & ~hot
        & ~brown
    )

    labels = np.zeros(image.shape[:2], dtype=np.uint8)
    labels[bank] = HAZARD_STEEP
    labels[lip] = HAZARD_DRAIN_EDGE
    labels[channel] = HAZARD_DRAIN
    return labels


def drain_pixel_fraction(image: np.ndarray) -> float:
    """Share of pixels labelled lip or channel. Useful as a unit-test hook."""
    labels = classify_terrain_rgb(image)
    return float((labels >= HAZARD_DRAIN_EDGE).mean()) if labels.size else 0.0


def _stamp_disks(
    grid: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    radii: np.ndarray,
    values: np.ndarray,
    *,
    mode: str = "max",
) -> None:
    """Stamp disks onto ``grid`` (in-place). ``mode`` is max, min, or set."""
    h, w = grid.shape
    for row, col, rad, val in zip(rows.tolist(), cols.tolist(), radii.tolist(), values.tolist()):
        r0 = max(0, int(row) - int(rad))
        r1 = min(h, int(row) + int(rad) + 1)
        c0 = max(0, int(col) - int(rad))
        c1 = min(w, int(col) + int(rad) + 1)
        if r0 >= r1 or c0 >= c1:
            continue
        yy = np.arange(r0, r1)
        xx = np.arange(c0, c1)
        dy = yy[:, None] - int(row)
        dx = xx[None, :] - int(col)
        mask = dy * dy + dx * dx <= int(rad) * int(rad)
        patch = grid[r0:r1, c0:c1]
        if mode == "min":
            patch[mask] = np.minimum(patch[mask], val)
        elif mode == "set":
            patch[mask] = val
        else:
            patch[mask] = np.maximum(patch[mask], val)


def project_labels_to_maps(
    image: np.ndarray,
    labels: np.ndarray,
    cam: CameraSpec,
    pose: Pose,
    *,
    elevation: np.ndarray,
    slope: np.ndarray,
    hazard: np.ndarray,
    resolution_m: float,
    world_size: tuple[float, float],
    ground_z: float = 0.0,
    max_range_m: float = DEFAULT_MAX_RANGE_M,
    steep_rad: float = 0.30,
) -> int:
    """Back-project classified pixels onto persistent rasters. Returns hits."""
    if image.shape[:2] != labels.shape:
        raise ValueError("image and labels must share H×W")
    height, width = labels.shape
    world_cam = camera_world_pose(pose, cam)
    hx, hy, valid = attitude_plane_hits(world_cam, width, height, pose)
    _ = ground_z
    rng = np.hypot(hx - world_cam.x, hy - world_cam.y)
    inside = (
        valid
        & (labels > 0)
        & (hx >= 0.0)
        & (hy >= 0.0)
        & (hx < world_size[0])
        & (hy < world_size[1])
        & (rng < max_range_m)
        & np.isfinite(hx)
        & np.isfinite(hy)
    )
    if not np.any(inside):
        return 0
    rows_f = hy[inside] / max(resolution_m, 1e-6)
    cols_f = hx[inside] / max(resolution_m, 1e-6)
    lab = labels[inside].astype(np.float32)
    rng_v = rng[inside]
    map_rows, map_cols = hazard.shape
    rr = np.floor(rows_f).astype(np.int32)
    cc = np.floor(cols_f).astype(np.int32)
    inb = (rr >= 0) & (cc >= 0) & (rr < map_rows) & (cc < map_cols)
    if not np.any(inb):
        return 0
    rr, cc, lab, rng_v = rr[inb], cc[inb], lab[inb], rng_v[inb]
    # Far pixels cover more yard; stamp a disk that grows with range.
    radii = np.clip(np.ceil((0.10 + 0.035 * rng_v) / max(resolution_m, 1e-6)), 1, 6).astype(np.int32)
    # Collapse duplicate cells so a full frame is not a Python loop per pixel.
    keys = rr.astype(np.int64) * map_cols + cc.astype(np.int64)
    order = np.argsort(keys)
    keys, rr, cc, lab, radii = keys[order], rr[order], cc[order], lab[order], radii[order]
    uniq = np.flatnonzero(np.concatenate(([True], keys[1:] != keys[:-1])))
    # Per cell: strongest label, largest stamp.
    next_break = np.append(uniq[1:], keys.size)
    keep_r, keep_c, keep_lab, keep_rad = [], [], [], []
    for a, b in zip(uniq.tolist(), next_break.tolist()):
        keep_r.append(int(rr[a]))
        keep_c.append(int(cc[a]))
        keep_lab.append(float(lab[a:b].max()))
        keep_rad.append(int(radii[a:b].max()))
    rr = np.asarray(keep_r, dtype=np.int32)
    cc = np.asarray(keep_c, dtype=np.int32)
    lab = np.asarray(keep_lab, dtype=np.float32)
    radii = np.asarray(keep_rad, dtype=np.int32)
    _stamp_disks(hazard, rr, cc, radii, lab)
    drain = lab >= HAZARD_DRAIN_EDGE
    if np.any(drain):
        drop = np.where(lab[drain] >= HAZARD_DRAIN, 0.14, 0.05).astype(np.float32)
        _stamp_disks(
            elevation,
            rr[drain],
            cc[drain],
            radii[drain],
            pose.z - drop,
            mode="min",
        )
        _stamp_disks(
            slope,
            rr[drain],
            cc[drain],
            radii[drain],
            np.full(int(drain.sum()), steep_rad, dtype=np.float32),
        )
    bank = lab == HAZARD_STEEP
    if np.any(bank):
        _stamp_disks(
            elevation,
            rr[bank],
            cc[bank],
            radii[bank],
            np.full(int(bank.sum()), pose.z + 0.08, dtype=np.float32),
        )
        _stamp_disks(
            slope,
            rr[bank],
            cc[bank],
            radii[bank],
            np.full(int(bank.sum()), steep_rad, dtype=np.float32),
        )
    return int(rr.size)


def stamp_tof_corners(
    tof: np.ndarray,
    pose: Pose,
    *,
    hazard: np.ndarray,
    elevation: np.ndarray,
    slope: np.ndarray,
    resolution_m: float,
    length_m: float,
    track_m: float,
    hover_m: float = 0.06,
    drop_extra_m: float = 0.09,
    steep_rad: float = 0.30,
) -> int:
    """Mark a wheel-corner cell when downward ToF sees extra drop."""
    from jims_mower.kinematics import wheel_positions

    arr = np.asarray(tof, dtype=np.float32).reshape(-1)
    if arr.size < 4:
        return 0
    wheels = wheel_positions(pose, length_m, track_m)
    rows, cols = hazard.shape
    res = max(resolution_m, 1e-6)
    marked = 0
    for i, (wx, wy) in enumerate(wheels):
        range_m = float(arr[i])
        if range_m < hover_m + drop_extra_m:
            continue
        col = int(wx / res)
        row = int(wy / res)
        if not (0 <= row < rows and 0 <= col < cols):
            continue
        rad = max(1, int(math.ceil(0.16 / res)))
        r0, r1 = max(0, row - rad), min(rows, row + rad + 1)
        c0, c1 = max(0, col - rad), min(cols, col + rad + 1)
        hazard[r0:r1, c0:c1] = np.maximum(hazard[r0:r1, c0:c1], float(HAZARD_DRAIN))
        elevation[r0:r1, c0:c1] = np.minimum(elevation[r0:r1, c0:c1], pose.z - (range_m - hover_m))
        slope[r0:r1, c0:c1] = np.maximum(slope[r0:r1, c0:c1], steep_rad)
        marked += 1
    return marked

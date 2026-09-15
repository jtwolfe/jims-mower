"""Lightweight geometric multi-camera renderer (no OpenGL)."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from jims_mower.cameras import camera_world_pose, ground_hits, project_point
from jims_mower.constants import (
    CUT_GRASS_RGB,
    DIRT_RGB,
    KIND_RGB,
    SKY_RGB,
    UNCUT_GRASS_RGB,
)
from jims_mower.maps import GrassCoverageMap
from jims_mower.types import CameraSpec, Obstacle, Pose


def _rgb(color: tuple[int, int, int]) -> np.ndarray:
    return np.asarray(color, dtype=np.uint8)


def render_camera(
    pose: Pose,
    cam: CameraSpec,
    coverage: GrassCoverageMap,
    obstacles: list[Obstacle],
    width: int,
    height: int,
    yard_size: tuple[float, float],
) -> np.ndarray:
    """Synthesize one RGB view: ray-traced ground plus projected blobs."""
    world_cam = camera_world_pose(pose, cam)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = SKY_RGB

    hx, hy, valid = ground_hits(world_cam, width, height)
    inside = (
        valid
        & (hx >= 0.0)
        & (hy >= 0.0)
        & (hx < yard_size[0])
        & (hy < yard_size[1])
    )
    dirt = valid & ~inside
    image[dirt] = DIRT_RGB

    # Sample coverage at hit cells. Looping 80x60 is fine; keep it explicit.
    uncut = _rgb(UNCUT_GRASS_RGB)
    cut = _rgb(CUT_GRASS_RGB)
    rows, cols = np.where(inside)
    for r, c in zip(rows.tolist(), cols.tolist()):
        sample = coverage.sample_world(float(hx[r, c]), float(hy[r, c]))
        image[r, c] = cut if sample > 0.5 else uncut

    # Painter's algorithm: far objects first.
    drawn: list[tuple[float, Obstacle]] = []
    for obst in obstacles:
        proj = project_point(obst.x, obst.y, obst.visual_z, world_cam, width, height)
        if proj is None:
            continue
        _, _, depth = proj
        drawn.append((depth, obst))
    drawn.sort(key=lambda item: item[0], reverse=True)

    fx = 0.5 * width / math.tan(math.radians(world_cam.fov_deg) * 0.5)
    for depth, obst in drawn:
        proj = project_point(obst.x, obst.y, obst.visual_z, world_cam, width, height)
        if proj is None:
            continue
        u, v, _ = proj
        radius_px = max(1.5, fx * (obst.radius / max(depth, 1e-3)))
        _stamp_disk(image, u, v, radius_px, KIND_RGB.get(obst.kind, (20, 20, 20)))
    return image


def _stamp_disk(
    image: np.ndarray,
    u: float,
    v: float,
    radius_px: float,
    color: tuple[int, int, int],
) -> None:
    h, w = image.shape[:2]
    r = int(math.ceil(radius_px))
    x0 = max(0, int(u) - r)
    x1 = min(w, int(u) + r + 1)
    y0 = max(0, int(v) - r)
    y1 = min(h, int(v) + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    ys = np.arange(y0, y1, dtype=np.float32)
    xs = np.arange(x0, x1, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    mask = (xx - u) ** 2 + (yy - v) ** 2 <= radius_px**2
    image[y0:y1, x0:x1][mask] = color


def render_topdown(
    pose: Pose,
    coverage: GrassCoverageMap,
    obstacles: list[Obstacle],
    *,
    trimmer_xy: Optional[tuple[float, float]] = None,
    trimmer_on: bool = False,
    image_size: int = 240,
) -> np.ndarray:
    """Orthographic yard map for ``render_mode='rgb_array'``."""
    w_m = coverage.width_m
    h_m = coverage.height_m
    # Preserve aspect.
    if w_m >= h_m:
        width = image_size
        height = max(8, int(round(image_size * h_m / w_m)))
    else:
        height = image_size
        width = max(8, int(round(image_size * w_m / h_m)))
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = DIRT_RGB

    scale_x = width / w_m
    scale_y = height / h_m

    # Coverage as a nearest-neighbor upsample of the grass grid.
    grid = coverage.as_float()
    rows, cols = grid.shape
    yy = (np.arange(height) + 0.5) / scale_y
    xx = (np.arange(width) + 0.5) / scale_x
    # world y → image row from the top.
    src_r = np.clip((yy / coverage.resolution_m).astype(int), 0, rows - 1)
    src_c = np.clip((xx / coverage.resolution_m).astype(int), 0, cols - 1)
    # Flip Y so world +y is up on the image.
    src_r = src_r[::-1]
    sampled = grid[src_r[:, None], src_c[None, :]]
    image[sampled >= 0.0] = UNCUT_GRASS_RGB
    image[sampled > 0.5] = CUT_GRASS_RGB

    def to_px(x: float, y: float) -> tuple[float, float]:
        return x * scale_x, (h_m - y) * scale_y

    for obst in obstacles:
        u, v = to_px(obst.x, obst.y)
        rpx = max(2.0, obst.radius * scale_x)
        _stamp_disk(image, u, v, rpx, KIND_RGB.get(obst.kind, (20, 20, 20)))

    ru, rv = to_px(pose.x, pose.y)
    body_r = max(3.0, 0.25 * scale_x)
    _stamp_disk(image, ru, rv, body_r, (40, 40, 40))
    hx = pose.x + 0.28 * math.cos(pose.theta)
    hy = pose.y + 0.28 * math.sin(pose.theta)
    hu, hv = to_px(hx, hy)
    _stamp_disk(image, hu, hv, max(2.0, 0.08 * scale_x), (230, 230, 230))

    if trimmer_xy is not None:
        tu, tv = to_px(*trimmer_xy)
        color = (80, 220, 80) if trimmer_on else (200, 200, 200)
        _stamp_disk(image, tu, tv, max(2.0, 0.12 * scale_x), color)
    return image

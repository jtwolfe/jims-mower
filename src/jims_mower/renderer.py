"""Lightweight geometric multi-camera renderer (no OpenGL)."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from jims_mower.appearance import Appearance
from jims_mower.cameras import camera_world_pose, ground_hits, heightfield_hits, project_point
from jims_mower.constants import (
    BANK_RGB,
    CUT_GRASS_RGB,
    DIRT_RGB,
    DRAIN_EDGE_RGB,
    DRAIN_RGB,
    KIND_RGB,
    PUDDLE_RGB,
    SKY_RGB,
    TERRAIN_BANK,
    TERRAIN_DRAIN,
    TERRAIN_DRAIN_EDGE,
    TERRAIN_PUDDLE,
    UNCUT_GRASS_RGB,
)
from jims_mower.maps import GrassCoverageMap
from jims_mower.terrain import HeightField
from jims_mower.types import CameraSpec, Obstacle, Pose


def _rgb(color: tuple[int, int, int]) -> np.ndarray:
    return np.asarray(color, dtype=np.uint8)


def _shade_rgb(color: tuple[int, int, int], shade: float) -> np.ndarray:
    return np.clip(np.asarray(color, dtype=np.float32) * shade, 0, 255).astype(np.uint8)


def _terrain_base_color(
    coverage: GrassCoverageMap,
    terrain: Optional[HeightField],
    x: float,
    y: float,
) -> tuple[int, int, int]:
    if terrain is not None:
        label = terrain.sample_label(x, y)
        if label == TERRAIN_DRAIN:
            return DRAIN_RGB
        if label == TERRAIN_DRAIN_EDGE:
            return DRAIN_EDGE_RGB
        if label == TERRAIN_PUDDLE:
            return PUDDLE_RGB
    sample = coverage.sample_world(x, y)
    if sample < 0.0:
        return DIRT_RGB
    if sample > 0.5:
        return CUT_GRASS_RGB
    if terrain is not None and terrain.sample_label(x, y) == TERRAIN_BANK:
        return BANK_RGB
    return UNCUT_GRASS_RGB


def render_camera(
    pose: Pose,
    cam: CameraSpec,
    coverage: GrassCoverageMap,
    obstacles: list[Obstacle],
    width: int,
    height: int,
    yard_size: tuple[float, float],
    terrain: Optional[HeightField] = None,
    appearance: Optional[Appearance] = None,
) -> np.ndarray:
    """Synthesize one RGB view: ray-traced ground plus projected blobs.

    When a height field is present, rays iterate against elevation and
    ground pixels are shaded by slope (Lambert) and drain/bank labels so
    a CV hook can tell a ditch from flat grass.
    """
    app = appearance or Appearance.neutral()
    world_cam = camera_world_pose(pose, cam)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = app.sky_rgb

    if terrain is not None:
        hx, hy, _hz, valid = heightfield_hits(world_cam, width, height, terrain.sample_many)
    else:
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

    light = np.asarray(app.light_dir, dtype=np.float64)
    light = light / max(np.linalg.norm(light), 1e-9)
    rows, cols = np.where(inside)
    for r, c in zip(rows.tolist(), cols.tolist()):
        x = float(hx[r, c])
        y = float(hy[r, c])
        color = _terrain_base_color(coverage, terrain, x, y)
        shade = float(app.ambient)
        if terrain is not None:
            nx, ny, nz = terrain.normal_at(x, y)
            ndotl = nx * light[0] + ny * light[1] + nz * light[2]
            shade = float(np.clip(0.50 + 0.50 * ndotl, 0.28, 1.15)) * float(app.ambient)
            label = terrain.sample_label(x, y)
            if label == TERRAIN_DRAIN:
                depth = max(0.0, -terrain.sample(x, y))
                shade *= float(np.clip(1.0 - 1.6 * depth, 0.35, 1.0))
            if app.wet_specular and label in (0, TERRAIN_BANK, TERRAIN_PUDDLE):
                spec = max(0.0, float(ndotl)) ** 12 * (0.55 if label == TERRAIN_PUDDLE else 0.28)
                shade += spec
        for cx, cy, rx, ry, dark in app.shadow_blobs:
            if ((x - cx) / max(rx, 1e-6)) ** 2 + ((y - cy) / max(ry, 1e-6)) ** 2 <= 1.0:
                shade *= dark
                break
        if app.porch_lights:
            extra = 0.0
            for lx, ly, intensity, _rgb in app.porch_lights:
                dist2 = (x - lx) ** 2 + (y - ly) ** 2
                extra += intensity / (1.0 + 2.8 * dist2)
            shade += 0.22 * extra
        image[r, c] = _shade_rgb(color, shade)

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
        color = KIND_RGB.get(obst.kind, (20, 20, 20))
        if obst.length_m > 0.2:
            _stamp_segment(
                image,
                obst,
                world_cam,
                width,
                height,
                fx,
                color,
            )
            continue
        proj = project_point(obst.x, obst.y, obst.visual_z, world_cam, width, height)
        if proj is None:
            continue
        u, v, _ = proj
        radius_px = max(1.5, fx * (obst.radius / max(depth, 1e-3)))
        _stamp_disk(image, u, v, radius_px, color)
    if app.porch_lights:
        _stamp_porch_blooms(image, world_cam, width, height, app)
    return _apply_camera_effects(image, app)


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


def _stamp_segment(
    image: np.ndarray,
    obst: Obstacle,
    world_cam,
    width: int,
    height: int,
    fx: float,
    color: tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = obst.segment_ends()
    steps = max(3, int(math.ceil(max(obst.length_m, 0.3) / max(obst.radius * 2.0, 0.08))))
    for i in range(steps + 1):
        t = i / steps
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        proj = project_point(x, y, obst.visual_z, world_cam, width, height)
        if proj is None:
            continue
        u, v, depth = proj
        radius_px = max(1.2, fx * (obst.radius / max(depth, 1e-3)))
        _stamp_disk(image, u, v, radius_px, color)


def _stamp_porch_blooms(
    image: np.ndarray,
    world_cam,
    width: int,
    height: int,
    app: Appearance,
) -> None:
    for lx, ly, intensity, rgb in app.porch_lights:
        proj = project_point(lx, ly, 1.6, world_cam, width, height)
        if proj is None:
            continue
        u, v, depth = proj
        radius = max(2.5, 18.0 * intensity / max(depth, 0.4))
        _stamp_glow(image, u, v, radius, rgb, weight=0.55)


def _stamp_glow(
    image: np.ndarray,
    u: float,
    v: float,
    radius_px: float,
    color: tuple[int, int, int],
    weight: float,
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
    dist = np.sqrt((xx - u) ** 2 + (yy - v) ** 2) / max(radius_px, 1e-6)
    falloff = np.clip(1.0 - dist, 0.0, 1.0) ** 2
    patch = image[y0:y1, x0:x1].astype(np.float32)
    glow = np.asarray(color, dtype=np.float32)
    alpha = (weight * falloff)[..., None]
    image[y0:y1, x0:x1] = np.clip(patch * (1.0 - alpha) + glow * alpha, 0, 255).astype(np.uint8)


def _apply_camera_effects(image: np.ndarray, app: Appearance) -> np.ndarray:
    out = image.astype(np.float32)
    scale = np.asarray(app.colour_scale, dtype=np.float32)
    if np.any(np.abs(scale - 1.0) > 1e-4):
        out *= scale
    if app.night:
        out *= 0.72
    if app.vignette:
        h, w = out.shape[:2]
        yy = (np.arange(h, dtype=np.float32) + 0.5) / h - 0.5
        xx = (np.arange(w, dtype=np.float32) + 0.5) / w - 0.5
        rr = np.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)
        vig = np.clip(1.0 - 1.15 * (rr**2), 0.35, 1.0)
        out *= vig[..., None]
    if app.camera_dirt and app.dirt_specks:
        h, w = out.shape[:2]
        for nu, nv, nr, dark in app.dirt_specks:
            _stamp_dirt(out, nu * w, nv * h, nr * max(h, w), dark)
    if app.motion_blur:
        k = 5
        kernel = np.ones(k, dtype=np.float32) / k
        pad = k // 2
        padded = np.pad(out, ((0, 0), (pad, pad), (0, 0)), mode="edge")
        blurred = np.zeros_like(out)
        for i, coeff in enumerate(kernel):
            blurred += coeff * padded[:, i : i + out.shape[1], :]
        out = 0.55 * out + 0.45 * blurred
    return np.clip(out, 0, 255).astype(np.uint8)


def _stamp_dirt(image: np.ndarray, u: float, v: float, radius_px: float, dark: float) -> None:
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
    image[y0:y1, x0:x1][mask] *= float(dark)


def render_topdown(
    pose: Pose,
    coverage: GrassCoverageMap,
    obstacles: list[Obstacle],
    *,
    trimmer_xy: Optional[tuple[float, float]] = None,
    trimmer_on: bool = False,
    image_size: int = 240,
    terrain: Optional[HeightField] = None,
    waypoints: Optional[list[tuple[float, float]]] = None,
    waypoint_index: int = 0,
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

    if terrain is not None:
        trows, tcols = terrain.labels.shape
        src_tr = np.clip((yy / terrain.resolution_m).astype(int), 0, trows - 1)
        src_tc = np.clip((xx / terrain.resolution_m).astype(int), 0, tcols - 1)
        src_tr = src_tr[::-1]
        labels = terrain.labels[src_tr[:, None], src_tc[None, :]]
        image[labels == TERRAIN_BANK] = BANK_RGB
        image[labels == TERRAIN_DRAIN_EDGE] = DRAIN_EDGE_RGB
        image[labels == TERRAIN_DRAIN] = DRAIN_RGB
        image[labels == TERRAIN_PUDDLE] = PUDDLE_RGB
        # Recolor cut grass on banks so coverage is still visible.
        bank_cut = (labels == TERRAIN_BANK) & (sampled > 0.5)
        image[bank_cut] = CUT_GRASS_RGB

    def to_px(x: float, y: float) -> tuple[float, float]:
        return x * scale_x, (h_m - y) * scale_y

    for obst in obstacles:
        color = KIND_RGB.get(obst.kind, (20, 20, 20))
        if obst.length_m > 0.2:
            x0, y0, x1, y1 = obst.segment_ends()
            steps = max(3, int(math.ceil(obst.length_m / max(obst.radius * 2.0, 0.08))))
            for i in range(steps + 1):
                t = i / steps
                u, v = to_px(x0 + t * (x1 - x0), y0 + t * (y1 - y0))
                _stamp_disk(image, u, v, max(1.5, obst.radius * scale_x), color)
            continue
        u, v = to_px(obst.x, obst.y)
        rpx = max(2.0, obst.radius * scale_x)
        _stamp_disk(image, u, v, rpx, color)

    ru, rv = to_px(pose.x, pose.y)
    body_r = max(3.0, 0.25 * scale_x)
    _stamp_disk(image, ru, rv, body_r, (40, 40, 40))
    hx = pose.x + 0.28 * math.cos(pose.theta)
    hy = pose.y + 0.28 * math.sin(pose.theta)
    hu, hv = to_px(hx, hy)
    _stamp_disk(image, hu, hv, max(2.0, 0.08 * scale_x), (230, 230, 230))

    if waypoints:
        path_r = max(1.2, 0.045 * scale_x)
        for i, (wx, wy) in enumerate(waypoints):
            u, v = to_px(wx, wy)
            if i < waypoint_index:
                color = (30, 80, 100)
            elif i == waypoint_index:
                color = (250, 240, 80)
            else:
                color = (40, 210, 230)
            _stamp_disk(image, u, v, path_r, color)

    if trimmer_xy is not None:
        tu, tv = to_px(*trimmer_xy)
        color = (80, 220, 80) if trimmer_on else (200, 200, 200)
        _stamp_disk(image, tu, tv, max(2.0, 0.12 * scale_x), color)
    return image


def apply_weather_rgb(
    image: np.ndarray,
    *,
    night: bool = False,
    dawn: bool = False,
    wet: bool = False,
) -> np.ndarray:
    """Dim / tint a camera frame for night, dawn, or wet-surface flags.

    Visual only — no claimed sensor model. Night and dawn are exclusive.
    """
    out = np.asarray(image, dtype=np.float32)
    if night:
        out *= 0.28
        out[:, :, 2] = np.minimum(255.0, out[:, :, 2] * 1.15)
    elif dawn:
        out *= 0.55
        out[:, :, 0] = np.minimum(255.0, out[:, :, 0] * 1.25)
        out[:, :, 2] *= 0.85
    if wet:
        out *= 0.90
        out[:, :, 2] = np.minimum(255.0, out[:, :, 2] * 1.06)
    return np.clip(out, 0, 255).astype(np.uint8)


def render_costmap_rgb(
    cost: np.ndarray,
    blocked: np.ndarray,
    *,
    image_size: int = 240,
) -> np.ndarray:
    """False-color costmap: green free, yellow slow, red blocked."""
    rows, cols = cost.shape
    if cols >= rows:
        width = image_size
        height = max(8, int(round(image_size * rows / cols)))
    else:
        height = image_size
        width = max(8, int(round(image_size * cols / rows)))
    yy = (np.linspace(0, rows - 1, height)).astype(int)
    xx = (np.linspace(0, cols - 1, width)).astype(int)
    sampled = np.asarray(cost, dtype=np.float32)[yy[:, None], xx[None, :]]
    blk = np.asarray(blocked, dtype=bool)[yy[:, None], xx[None, :]]
    sampled = sampled[::-1]
    blk = blk[::-1]
    finite = np.isfinite(sampled)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (40, 90, 50)
    steep = finite & (sampled >= 4.0)
    image[steep] = (210, 160, 40)
    image[~finite | blk] = (140, 30, 28)
    return image


def render_scalar_map(
    grid: np.ndarray,
    *,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    image_size: int = 240,
    cmap: str = "elev",
) -> np.ndarray:
    """False-color raster for demo / tests (elevation, slope, or hazard)."""
    rows, cols = grid.shape
    if cols >= rows:
        width = image_size
        height = max(8, int(round(image_size * rows / cols)))
    else:
        height = image_size
        width = max(8, int(round(image_size * cols / rows)))
    yy = (np.linspace(0, rows - 1, height)).astype(int)
    xx = (np.linspace(0, cols - 1, width)).astype(int)
    sampled = grid[yy[:, None], xx[None, :]]
    # Flip Y so +world-y is up, matching render_topdown.
    sampled = sampled[::-1]
    lo = float(np.min(sampled)) if vmin is None else float(vmin)
    hi = float(np.max(sampled)) if vmax is None else float(vmax)
    span = max(hi - lo, 1e-6)
    t = np.clip((sampled.astype(np.float32) - lo) / span, 0.0, 1.0)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if cmap == "hazard":
        image[sampled == 0] = (40, 90, 50)
        image[sampled == 1] = (210, 160, 40)
        image[sampled == 2] = (200, 90, 30)
        image[sampled >= 3] = (90, 40, 20)
        return image
    if cmap == "slope":
        image[:, :, 0] = (40 + 200 * t).astype(np.uint8)
        image[:, :, 1] = (180 - 120 * t).astype(np.uint8)
        image[:, :, 2] = (50 + 20 * t).astype(np.uint8)
        return image
    # elevation: blue (low / drains) → green → yellow (banks)
    image[:, :, 0] = (30 + 200 * t).astype(np.uint8)
    image[:, :, 1] = (80 + 140 * t).astype(np.uint8)
    image[:, :, 2] = (160 - 120 * t).astype(np.uint8)
    return image

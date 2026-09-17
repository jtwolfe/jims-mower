"""Near-field metric stereo for a ~0.5 m mower — not COLMAP, not mAP.

Live control uses a **fixed forward stereo pair** (or gym synthetic
stereo from known extrinsics) in the 0.8–4 m band. Grass is
low-texture; this module does **not** run full-yard SfM. Semantic
labels still come from a terrain seg head.

Gym path: ideal disparity from known ray range (synthetic stereo).
That proves the metric stamp + lock contract. It is not a published
matcher / FPS / mAP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np

from jims_mower.cameras import camera_world_pose, focal_length_px, pixel_rays_world
from jims_mower.types import CameraSpec, Pose

# Near-field band for tip / lip / obstacle. Depth resolution worsens with z².
STEREO_Z_MIN_M = 0.80
STEREO_Z_MAX_M = 4.00
# 6–12 cm on a 70 cm body. Wider looks-around cams are not this pair.
STEREO_BASELINE_MIN_M = 0.06
STEREO_BASELINE_MAX_M = 0.12
STEREO_NOTE = (
    "synthetic / ideal stereo in gym — not a matcher, not COLMAP, "
    "fps_claim null, map_claim null"
)


@dataclass(frozen=True)
class StereoPair:
    """Two cameras that share yaw/pitch and sit on a short baseline."""

    left: CameraSpec
    right: CameraSpec
    baseline_m: float

    def as_info(self) -> dict[str, object]:
        return {
            "left": self.left.name,
            "right": self.right.name,
            "baseline_m": float(self.baseline_m),
            "z_min_m": STEREO_Z_MIN_M,
            "z_max_m": STEREO_Z_MAX_M,
            "note": STEREO_NOTE,
            "not_colmap": True,
            "fps_claim": None,
            "map_claim": None,
        }


def disparity_px(*, range_m: float, baseline_m: float, focal_px: float) -> float:
    """Ideal horizontal disparity. ``d = f B / Z`` (metres, pixels)."""
    z = max(float(range_m), 1e-6)
    return float(focal_px) * float(baseline_m) / z


def range_from_disparity(*, disparity_px: float, baseline_m: float, focal_px: float) -> float:
    d = max(float(disparity_px), 1e-6)
    return float(focal_px) * float(baseline_m) / d


def depth_resolution_m(
    *,
    range_m: float,
    baseline_m: float,
    focal_px: float,
    disparity_error_px: float = 1.0,
) -> float:
    """``δZ ≈ (Z² / (f B)) δd`` — plan cell size from this, not a published score."""
    z = float(range_m)
    denom = max(float(focal_px) * float(baseline_m), 1e-9)
    return (z * z / denom) * float(disparity_error_px)


def baseline_cm(pair: StereoPair) -> float:
    """Baseline in centimetres. Field band is 6–12 cm."""
    return float(pair.baseline_m) * 100.0


class StereoPairError(ValueError):
    """Cameras are not a 6–12 cm stereo pair (look-around, yaw mismatch, …)."""


def require_stereo_pair(cameras: Iterable[CameraSpec]) -> StereoPair:
    """Like :func:`find_stereo_pair` but reject non-pairs with a clear error."""
    pair = find_stereo_pair(cameras)
    if pair is None:
        raise StereoPairError(
            "no stereo pair: need stereo_left/stereo_right (or a 6–12 cm "
            "same-yaw pair). Default gym look-arounds are not a pair."
        )
    return pair


def find_stereo_pair(cameras: Iterable[CameraSpec]) -> Optional[StereoPair]:
    """Prefer named ``stereo_left`` / ``stereo_right`` if they are a true pair.

    A pair must share yaw/pitch (≤2°) and sit on a 6–12 cm baseline.
    The default gym look-around cams (40° yaw, 40 cm apart) are **not**
    a stereo pair. Named ``front_left`` / ``front_right`` are accepted
    only when they also sit in that band (the gym defaults do not).
    """
    cams = {c.name: c for c in cameras}
    named = None
    if "front_left" in cams and "front_right" in cams:
        named = (cams["front_left"], cams["front_right"])
    elif "stereo_left" in cams and "stereo_right" in cams:
        named = (cams["stereo_left"], cams["stereo_right"])
    candidates: list[tuple[CameraSpec, CameraSpec]] = []
    if named is not None:
        candidates.append(named)
    listed = list(cameras)
    for i, a in enumerate(listed):
        for b in listed[i + 1 :]:
            if named is not None and {a.name, b.name} == {named[0].name, named[1].name}:
                continue
            candidates.append((a, b))
    for a, b in candidates:
        pair = _as_pair(a, b)
        if pair is not None:
            return pair
    return None


def _as_pair(a: CameraSpec, b: CameraSpec) -> Optional[StereoPair]:
    if abs(float(a.yaw_deg) - float(b.yaw_deg)) > 2.0:
        return None
    if abs(float(a.pitch_deg) - float(b.pitch_deg)) > 2.0:
        return None
    dy = float(a.y) - float(b.y)
    dx = float(a.x) - float(b.x)
    dz = float(a.z) - float(b.z)
    baseline = math.hypot(dx, dy, dz)
    if baseline < STEREO_BASELINE_MIN_M - 1e-6 or baseline > STEREO_BASELINE_MAX_M + 1e-6:
        return None
    # Body +y is left. The more-positive y camera is the left eye.
    left, right = (a, b) if float(a.y) >= float(b.y) else (b, a)
    return StereoPair(left=left, right=right, baseline_m=float(baseline))


HeightSampler = Callable[[float, float], float]


def height_sampler_from_raster(
    elev: np.ndarray,
    *,
    resolution_m: float,
) -> HeightSampler:
    """Sample a world-XY height raster. Out of bounds is 0 (flat)."""
    ev = np.asarray(elev, dtype=np.float32)
    rows, cols = int(ev.shape[0]), int(ev.shape[1])
    res = max(float(resolution_m), 1e-6)

    def height_at(x: float, y: float) -> float:
        if x < 0.0 or y < 0.0:
            return 0.0
        rr = int(y / res)
        cc = int(x / res)
        if 0 <= rr < rows and 0 <= cc < cols:
            return float(ev[rr, cc])
        return 0.0

    return height_at


def synthetic_stereo_points(
    pose: Pose,
    pair: StereoPair,
    *,
    width: int,
    height: int,
    height_at: HeightSampler,
    pixel_stride: int = 2,
    z_min_m: float = STEREO_Z_MIN_M,
    z_max_m: float = STEREO_Z_MAX_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ideal stereo: ray → known height → disparity → reconstruct.

    Returns world ``(x, y, z)`` arrays in the near-field band. Gym only —
    correspondence is the true ray hit, not an RGB matcher.
    """
    left = camera_world_pose(pose, pair.left)
    fx = focal_length_px(width, left.fov_deg)
    ox, oy, oz, dirs = pixel_rays_world(left, width, height)
    stride = max(1, int(pixel_stride))
    dirs = dirs[::stride, ::stride]
    origin = np.array([ox, oy, oz], dtype=np.float64)
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for ray in dirs.reshape(-1, 3):
        hit = _ray_ground_hit(origin, ray, height_at, z_min_m=z_min_m, z_max_m=z_max_m)
        if hit is None:
            continue
        wx, wy, wz, range_m = hit
        disp = disparity_px(range_m=range_m, baseline_m=pair.baseline_m, focal_px=fx)
        recon = range_from_disparity(disparity_px=disp, baseline_m=pair.baseline_m, focal_px=fx)
        if recon < z_min_m or recon > z_max_m:
            continue
        # Scale the hit along the ray to the reconstructed range (metric path).
        scale = recon / max(range_m, 1e-6)
        xs.append(origin[0] + scale * (wx - origin[0]))
        ys.append(origin[1] + scale * (wy - origin[1]))
        zs.append(origin[2] + scale * (wz - origin[2]))
    if not xs:
        return (
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(zs, dtype=np.float32),
    )


def _ray_ground_hit(
    origin: np.ndarray,
    direction: np.ndarray,
    height_at: HeightSampler,
    *,
    z_min_m: float,
    z_max_m: float,
    steps: int = 24,
) -> Optional[tuple[float, float, float, float]]:
    d = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(d))
    if norm < 1e-9:
        return None
    d = d / norm
    o = np.asarray(origin, dtype=np.float64)
    # March the near-field band only.
    for t in np.linspace(z_min_m, z_max_m, steps):
        p = o + t * d
        ground = float(height_at(float(p[0]), float(p[1])))
        if p[2] <= ground + 0.01:
            range_m = float(np.linalg.norm(p - o))
            if z_min_m <= range_m <= z_max_m:
                return float(p[0]), float(p[1]), float(ground), range_m
    return None


def rasterize_points(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    *,
    shape: tuple[int, int],
    resolution_m: float,
    width_m: float,
    height_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean-z per cell plus a hit mask. Empty cells stay 0 / False."""
    rows, cols = int(shape[0]), int(shape[1])
    elev = np.zeros((rows, cols), dtype=np.float32)
    hits = np.zeros((rows, cols), dtype=np.float32)
    res = max(float(resolution_m), 1e-6)
    for x, y, z in zip(xs.tolist(), ys.tolist(), zs.tolist()):
        if x < 0.0 or y < 0.0 or x >= width_m or y >= height_m:
            continue
        rr = int(y / res)
        cc = int(x / res)
        if 0 <= rr < rows and 0 <= cc < cols:
            elev[rr, cc] += float(z)
            hits[rr, cc] += 1.0
    ok = hits > 0.0
    elev[ok] = elev[ok] / hits[ok]
    return elev, ok

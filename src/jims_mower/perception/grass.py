"""Class-aware grass coverage observer (CV-2) with colour fallback.

When terrain-seg classes exist (drain / lip / bank vs grass), coverage
uses those labels. Otherwise the gym palette heuristic runs. Phone cut %
stays the *gym grass grid* unless ``perception.coverage_source: observer``.

Gym strip fixture reports coverage drift vs a painted mask. That is **not**
field mAP / IoU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from jims_mower.constants import (
    COV_CLASS_NAMES,
    COV_CUT,
    COV_NON_GRASS,
    COV_UNCUT,
    CUT_GRASS_RGB,
    PATH_RGB,
    SKY_RGB,
    UNCUT_GRASS_RGB,
)
from jims_mower.perception.cv_terrain import classify_terrain_rgb


def uncut_grass_mask(image: np.ndarray) -> np.ndarray:
    """Pixels that look like the synthetic uncut-grass green."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be HxWx3")
    r = image[:, :, 0].astype(np.int16)
    g = image[:, :, 1].astype(np.int16)
    b = image[:, :, 2].astype(np.int16)
    # Tight on the rendered uncut color; still a real color test, not a no-op.
    target = UNCUT_GRASS_RGB
    close = (
        (np.abs(r - target[0]) < 30)
        & (np.abs(g - target[1]) < 30)
        & (np.abs(b - target[2]) < 30)
    )
    green_dominant = (g > r + 15) & (g > b + 10) & (g > 70)
    return close | green_dominant


def cut_grass_mask(image: np.ndarray) -> np.ndarray:
    """Pixels that look like the synthetic cut-grass tan."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be HxWx3")
    r = image[:, :, 0].astype(np.int16)
    g = image[:, :, 1].astype(np.int16)
    b = image[:, :, 2].astype(np.int16)
    target = CUT_GRASS_RGB
    close = (
        (np.abs(r - target[0]) < 36)
        & (np.abs(g - target[1]) < 36)
        & (np.abs(b - target[2]) < 36)
    )
    tan = (
        (r > 120)
        & (g > 100)
        & (b > 40)
        & (r > b + 20)
        & (g > b + 10)
        & (np.abs(r.astype(np.int32) - g.astype(np.int32)) < 40)
    )
    return close | tan


def grass_fraction(image: np.ndarray) -> float:
    mask = uncut_grass_mask(image)
    return float(mask.mean()) if mask.size else 0.0


def classify_coverage_rgb(
    image: np.ndarray,
    terrain_labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-pixel coverage class: 0 non-grass, 1 uncut, 2 cut.

    Terrain-seg classes (non-zero hazard: bank / lip / drain) override
    colour so a brown channel is never counted as grass. Sky / path stay
    non-grass. Not a published IoU head.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be HxWx3")
    uncut = uncut_grass_mask(image)
    cut = cut_grass_mask(image) & ~uncut
    labels = np.zeros(image.shape[:2], dtype=np.uint8)
    labels[uncut] = COV_UNCUT
    labels[cut] = COV_CUT
    if terrain_labels is None:
        terrain = classify_terrain_rgb(image)
    else:
        terrain = np.asarray(terrain_labels)
        if terrain.shape != labels.shape:
            raise ValueError("terrain_labels must match image H×W")
    labels[terrain > 0] = COV_NON_GRASS
    return labels


def uncut_fraction_from_labels(labels: np.ndarray) -> float:
    """Share of *image* pixels labelled uncut (same contract as grass_fraction)."""
    arr = np.asarray(labels)
    return float((arr == COV_UNCUT).mean()) if arr.size else 0.0


def cut_fraction_on_grass(labels: np.ndarray) -> float:
    """Cut / (cut + uncut) on grass pixels. Empty grass → 0."""
    arr = np.asarray(labels)
    grass = (arr == COV_UNCUT) | (arr == COV_CUT)
    n = int(grass.sum())
    if n <= 0:
        return 0.0
    return float((arr == COV_CUT).sum() / n)


class ColorGrassObserver:
    """Palette uncut fraction. Heuristic fallback when no class head is selected."""

    def estimate(self, images: dict[str, np.ndarray]) -> dict[str, float]:
        return {name: grass_fraction(frame) for name, frame in images.items()}


def grass_feature_vector(image: np.ndarray) -> np.ndarray:
    """Six HSV-ish stats. Not a claimed IoU / mAP head."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be HxWx3")
    rgb = image.astype(np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    g_safe = np.maximum(g, 1.0)
    green_dom = ((g > r + 10) & (g > b + 8) & (g > 60)).mean()
    rg = float(np.mean(r / g_safe))
    value = float((r + g + b).mean() / (3.0 * 255.0))
    sat = float(np.mean((np.max(rgb, axis=2) - np.min(rgb, axis=2)) / 255.0))
    tight = float(uncut_grass_mask(image).mean())
    chroma = float(np.mean(g - 0.5 * (r + b)) / 255.0)
    return np.array([green_dom, rg, value, sat, tight, chroma], dtype=np.float32)


def coverage_class_fractions(labels: np.ndarray) -> np.ndarray:
    """(non_grass, uncut, cut, terrain_nonzero_proxy) histogram. Length 4."""
    arr = np.asarray(labels)
    n = max(int(arr.size), 1)
    non = float((arr == COV_NON_GRASS).sum() / n)
    uncut = float((arr == COV_UNCUT).sum() / n)
    cut = float((arr == COV_CUT).sum() / n)
    return np.array([non, uncut, cut, 1.0 - uncut - cut], dtype=np.float32)


class FeatureGrassObserver:
    """Colour-heuristic plus optional linear weights. Replace on the Orin.

    Default weights recover the tight uncut-grass fraction. A ``.npz`` with
    ``w`` (6,) and ``b`` (scalar) can tilt the mix. No claimed accuracy.
    """

    def __init__(self, weights_path: str = "") -> None:
        self.w = np.array([0.35, 0.0, 0.0, 0.0, 0.65, 0.0], dtype=np.float32)
        self.b = 0.0
        self.weights_path = weights_path
        if weights_path:
            self._try_load(weights_path)

    def _try_load(self, path: str) -> None:
        dest = Path(path)
        if not dest.is_file():
            return
        data = np.load(dest)
        if "w" in data.files:
            w = np.asarray(data["w"], dtype=np.float32).reshape(-1)
            if w.size == 6:
                self.w = w
        if "b" in data.files:
            self.b = float(np.asarray(data["b"]).reshape(-1)[0])

    def estimate(self, images: dict[str, np.ndarray]) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, frame in images.items():
            feat = grass_feature_vector(frame)
            score = float(np.clip(float(feat @ self.w) + self.b, 0.0, 1.0))
            out[name] = score
        return out


@dataclass
class GrassCoverageEstimate:
    """Per-camera uncut fractions plus optional class rasters. Not field mAP."""

    per_camera: dict[str, float]
    labels: dict[str, np.ndarray] = field(default_factory=dict)
    source: str = "class_aware"
    used_terrain_classes: bool = False
    map_claim: None = None
    iou_claim: None = None


def paint_lawn_strip_fixture(
    height: int = 48,
    width: int = 64,
    *,
    sky_rows: int = 10,
    uncut_frac: float = 0.45,
    cut_frac: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic lawn: sky band + uncut / cut / path strips.

    Returns ``(image, truth)`` where truth is ``COV_*`` classes. Known
    geometry for gym coverage-error tests — not a field strip.
    """
    h = max(8, int(height))
    w = max(8, int(width))
    sky_n = int(np.clip(sky_rows, 0, h - 2))
    image = np.zeros((h, w, 3), dtype=np.uint8)
    truth = np.zeros((h, w), dtype=np.uint8)
    if sky_n > 0:
        image[:sky_n, :] = SKY_RGB
        truth[:sky_n, :] = COV_NON_GRASS
    lawn_w = w
    u = int(round(lawn_w * float(np.clip(uncut_frac, 0.0, 1.0))))
    c = int(round(lawn_w * float(np.clip(cut_frac, 0.0, 1.0))))
    u = max(1, min(u, lawn_w - 2))
    c = max(1, min(c, lawn_w - u - 1))
    p0 = u + c
    image[sky_n:, :u] = UNCUT_GRASS_RGB
    truth[sky_n:, :u] = COV_UNCUT
    image[sky_n:, u:p0] = CUT_GRASS_RGB
    truth[sky_n:, u:p0] = COV_CUT
    image[sky_n:, p0:] = PATH_RGB
    truth[sky_n:, p0:] = COV_NON_GRASS
    return image, truth


def coverage_error(
    pred: np.ndarray,
    truth: np.ndarray,
) -> dict[str, Any]:
    """Gym coverage drift vs a painted mask. Not field mAP / IoU.

    Reports pixel accuracy and uncut-fraction absolute error so a test
    can fail honestly. ``map_claim`` / ``iou_claim`` stay null.
    """
    p = np.asarray(pred)
    t = np.asarray(truth)
    if p.shape != t.shape:
        raise ValueError("pred and truth must share shape")
    n = max(int(p.size), 1)
    acc = float((p == t).sum() / n)
    grass_t = (t == COV_UNCUT) | (t == COV_CUT)
    grass_p = (p == COV_UNCUT) | (p == COV_CUT)
    grass_acc = float((grass_p == grass_t).sum() / n)
    uncut_err = abs(uncut_fraction_from_labels(p) - uncut_fraction_from_labels(t))
    cut_err = abs(cut_fraction_on_grass(p) - cut_fraction_on_grass(t))
    return {
        "pixel_accuracy": acc,
        "grass_vs_nongrass_accuracy": grass_acc,
        "uncut_fraction_abs_error": float(uncut_err),
        "cut_fraction_abs_error": float(cut_err),
        "n_pixels": int(p.size),
        "class_names": list(COV_CLASS_NAMES),
        "map_claim": None,
        "iou_claim": None,
        "field_strip": False,
        "note": "gym painted-strip error — not field mAP",
    }


class ClassAwareGrassObserver:
    """Terrain-class coverage observer with colour heuristic fallback.

    ``estimate`` keeps the GrassObserver contract (per-camera uncut
    fraction). Optional ``terrain_labels`` (per-camera HxW) mark
    non-grass when a seg head is available. Optional linear weights
    (``.npz`` ``w`` length 6 or 10, ``b``) tilt the mix — gym-trainable,
    not a field net.
    """

    def __init__(self, weights_path: str = "") -> None:
        self.w = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        self.b = 0.0
        self.use_linear = False
        self.weights_path = weights_path
        self.source_name = "class_aware"
        if weights_path:
            self._try_load(weights_path)

    def _try_load(self, path: str) -> None:
        dest = Path(path)
        if not dest.is_file():
            return
        data = np.load(dest)
        if "w" in data.files:
            w = np.asarray(data["w"], dtype=np.float32).reshape(-1)
            if w.size in {6, 10}:
                self.w = w
                self.use_linear = True
        if "b" in data.files:
            self.b = float(np.asarray(data["b"]).reshape(-1)[0])
            self.use_linear = True

    def classify(
        self,
        image: np.ndarray,
        terrain_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return classify_coverage_rgb(image, terrain_labels)

    def estimate(
        self,
        images: dict[str, np.ndarray],
        terrain_labels: Optional[dict[str, np.ndarray]] = None,
    ) -> dict[str, float]:
        return self.estimate_detail(images, terrain_labels).per_camera

    def estimate_detail(
        self,
        images: dict[str, np.ndarray],
        terrain_labels: Optional[dict[str, np.ndarray]] = None,
    ) -> GrassCoverageEstimate:
        used_classes = False
        per: dict[str, float] = {}
        labs: dict[str, np.ndarray] = {}
        for name, frame in images.items():
            tl = None
            if terrain_labels is not None and name in terrain_labels:
                tl = terrain_labels[name]
                used_classes = True
            labels = self.classify(frame, tl)
            labs[name] = labels
            if self.use_linear:
                feat = grass_feature_vector(frame)
                if self.w.size == 10:
                    feat = np.concatenate([feat, coverage_class_fractions(labels)])
                score = float(np.clip(float(feat @ self.w) + self.b, 0.0, 1.0))
                per[name] = score
            else:
                per[name] = uncut_fraction_from_labels(labels)
        return GrassCoverageEstimate(
            per_camera=per,
            labels=labs,
            source=self.source_name,
            used_terrain_classes=used_classes,
        )

    def stamp_bev(
        self,
        images: dict[str, np.ndarray],
        cameras: list[Any],
        pose: Any,
        *,
        shape: tuple[int, int],
        resolution_m: float,
        world_size: tuple[float, float],
        terrain_labels: Optional[dict[str, np.ndarray]] = None,
        max_range_m: float = 6.0,
    ) -> np.ndarray:
        """Back-project coverage classes onto a BEV raster (1 cut, 0 uncut, -1 other)."""
        from jims_mower.cameras import attitude_plane_hits, camera_world_pose

        rows, cols = int(shape[0]), int(shape[1])
        out = np.full((rows, cols), -1.0, dtype=np.float32)
        cams = {c.name: c for c in cameras}
        detail = self.estimate_detail(images, terrain_labels)
        res = max(float(resolution_m), 1e-6)
        for name, labels in detail.labels.items():
            cam = cams.get(name)
            if cam is None:
                continue
            h, w = labels.shape
            world_cam = camera_world_pose(pose, cam)
            hx, hy, valid = attitude_plane_hits(world_cam, w, h, pose)
            rng = np.hypot(hx - world_cam.x, hy - world_cam.y)
            grass = labels > COV_NON_GRASS
            inside = (
                valid
                & grass
                & np.isfinite(hx)
                & np.isfinite(hy)
                & (hx >= 0.0)
                & (hy >= 0.0)
                & (hx < world_size[0])
                & (hy < world_size[1])
                & (rng < float(max_range_m))
            )
            if not np.any(inside):
                continue
            rr = np.floor(hy[inside] / res).astype(np.int32)
            cc = np.floor(hx[inside] / res).astype(np.int32)
            lab = labels[inside]
            ok = (rr >= 0) & (cc >= 0) & (rr < rows) & (cc < cols)
            if not np.any(ok):
                continue
            rr, cc, lab = rr[ok], cc[ok], lab[ok]
            # Cut wins over uncut when both stamp the same cell.
            uncut = lab == COV_UNCUT
            cut = lab == COV_CUT
            if np.any(uncut):
                out[rr[uncut], cc[uncut]] = np.maximum(out[rr[uncut], cc[uncut]], 0.0)
            if np.any(cut):
                out[rr[cut], cc[cut]] = 1.0
        return out


def observer_coverage_fraction(raster: np.ndarray) -> float:
    """Cut / grass on a BEV observer raster (1 cut, 0 uncut, -1 non-grass)."""
    grid = np.asarray(raster, dtype=np.float32)
    grass = grid >= 0.0
    n = int(grass.sum())
    if n <= 0:
        return 0.0
    return float((grid[grass] >= 0.5).sum() / n)


def fit_linear_grass(
    images: list[np.ndarray],
    truths: list[np.ndarray],
    *,
    out_path: Optional[str] = None,
) -> tuple[np.ndarray, float]:
    """Least-squares uncut-fraction fit on gym strips. Not a field trainer."""
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for image, truth in zip(images, truths):
        labels = classify_coverage_rgb(image)
        feat = np.concatenate([grass_feature_vector(image), coverage_class_fractions(labels)])
        xs.append(feat)
        ys.append(uncut_fraction_from_labels(truth))
    if not xs:
        raise ValueError("fit_linear_grass needs at least one strip")
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    a = np.column_stack((x, np.ones(len(x))))
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    w = coef[:-1].astype(np.float32)
    b = float(coef[-1])
    if out_path:
        np.savez(out_path, w=w, b=np.float32(b), sim_only=True, map_claim=None)
    return w, b


def grass_observer_from_mode(mode: str, weights_path: str = ""):
    key = (mode or "color").strip().lower()
    if key in {"class", "terrain", "class_aware"}:
        return ClassAwareGrassObserver(weights_path)
    if key in {"feature", "learned", "net"}:
        return FeatureGrassObserver(weights_path)
    return ColorGrassObserver()

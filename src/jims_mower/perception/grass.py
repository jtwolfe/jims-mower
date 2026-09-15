"""Color-heuristic grass hook. Replace with a learned head on the Orin."""

from __future__ import annotations

import numpy as np

from jims_mower.constants import UNCUT_GRASS_RGB


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


def grass_fraction(image: np.ndarray) -> float:
    mask = uncut_grass_mask(image)
    return float(mask.mean()) if mask.size else 0.0


class ColorGrassObserver:
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
        from pathlib import Path

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


def grass_observer_from_mode(mode: str, weights_path: str = ""):
    key = (mode or "color").strip().lower()
    if key in {"feature", "learned", "net"}:
        return FeatureGrassObserver(weights_path)
    return ColorGrassObserver()

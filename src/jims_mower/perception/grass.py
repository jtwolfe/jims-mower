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

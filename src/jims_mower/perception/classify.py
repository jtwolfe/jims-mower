"""Simple appearance hooks for person / animal / toy crops.

Numpy colour + geometry features only. Not a claimed classifier, mAP, or
MOT score. Used by ``MockDetector`` to refine categories from the rendered
blob palette and by the hand-signal stub behind the curriculum.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from jims_mower.constants import ANIMAL_KINDS, HAND_SIGNALS, KIND_RGB

# Feature layout: mean RGB, std RGB, aspect, fill, brightness, rg, gb.
APPEAR_N_FEATURES = 10


def crop_bbox(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Safe uint8 crop; empty if the box is off-frame."""
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("image must be HxWx3")
    x, y, w, h = (int(v) for v in bbox)
    height, width = image.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width, x + max(w, 0))
    y1 = min(height, y + max(h, 0))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    return np.asarray(image[y0:y1, x0:x1, :3], dtype=np.uint8)


def appearance_features(crop: np.ndarray) -> np.ndarray:
    """Ten floats describing a detection crop. Zeros when the crop is empty."""
    out = np.zeros(APPEAR_N_FEATURES, dtype=np.float32)
    if crop.size == 0:
        return out
    rgb = crop.astype(np.float32)
    mean = rgb.reshape(-1, 3).mean(axis=0)
    std = rgb.reshape(-1, 3).std(axis=0)
    h, w = crop.shape[:2]
    out[0:3] = mean / 255.0
    out[3:6] = std / 255.0
    out[6] = float(h) / float(max(w, 1))
    sat = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    out[7] = float(sat.mean() / 255.0)
    out[8] = float(rgb.mean() / 255.0)
    out[9] = float(mean[1] - mean[0]) / 255.0
    return out


def _palette_distance(mean_rgb: np.ndarray, label: str) -> float:
    target = np.asarray(KIND_RGB.get(label, (128, 128, 128)), dtype=np.float32)
    return float(np.linalg.norm(mean_rgb - target))


def refine_category(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    prior_label: str,
) -> tuple[str, str, float]:
    """Return ``(label, category, appearance_score)``.

    Score is a relative palette match in ``[0, 1]``, not a published
    confidence. Living / toy labels stay in their family unless the crop
    is clearly closer to another family centroid.
    """
    from jims_mower.perception.mock import category_for

    crop = crop_bbox(image, bbox)
    prior_cat = category_for(prior_label)
    if crop.size == 0:
        return prior_label, prior_cat, 0.0
    mean = crop.reshape(-1, 3).astype(np.float32).mean(axis=0)
    families = {
        "person": ("person",),
        "animal": tuple(sorted(ANIMAL_KINDS)),
        "toy": ("toy",),
        "static": ("tree", "furniture"),
    }
    best_label = prior_label
    best_dist = _palette_distance(mean, prior_label)
    for labels in families.values():
        for lab in labels:
            dist = _palette_distance(mean, lab)
            if dist + 8.0 < best_dist:
                best_dist = dist
                best_label = lab
    # Stay inside the prior family unless the gap is large (renderer blobs
    # are saturated; a person crop should not flip to "tree").
    new_cat = category_for(best_label)
    if new_cat != prior_cat and best_dist > 70.0:
        best_label = prior_label
        new_cat = prior_cat
        best_dist = _palette_distance(mean, prior_label)
    score = float(np.clip(1.0 - best_dist / 220.0, 0.0, 1.0))
    return best_label, new_cat, score


def classify_hand_signal_features(features: np.ndarray) -> str:
    """Map numpy crop features to stop / go / follow / back.

    Rules use brightness, aspect, and red/green bias — a curriculum stub,
    not a gesture model. Deterministic so tests can pin a crop.
    """
    feat = np.asarray(features, dtype=np.float32).reshape(-1)
    if feat.size < APPEAR_N_FEATURES:
        feat = np.pad(feat, (0, APPEAR_N_FEATURES - feat.size))
    r, g, b = float(feat[0]), float(feat[1]), float(feat[2])
    aspect = float(feat[6])
    bright = float(feat[8])
    rg = float(feat[9])
    if bright < 0.28:
        return "stop"
    if aspect > 1.55:
        return "back"
    if r > g + 0.08 and r > b:
        return "stop"
    if rg > 0.04:
        return "go"
    if aspect < 0.85:
        return "follow"
    return "go"


class HandSignalClassifier:
    """Optional crop → signal head used when the curriculum asks for it."""

    def classify(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> Optional[str]:
        crop = crop_bbox(image, bbox)
        if crop.size == 0:
            return None
        name = classify_hand_signal_features(appearance_features(crop))
        return name if name in HAND_SIGNALS else None

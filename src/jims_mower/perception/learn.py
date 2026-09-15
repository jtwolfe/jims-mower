"""Tiny colour+position terrain classifier. Numpy baseline; torch optional.

Sim-only stub. Trains from the WAVE 1A exporter (oracle hazard on the
ground plane). No claimed mAP / IoU / FPS.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from jims_mower.cameras import camera_world_pose, ground_hits
from jims_mower.constants import HAZARD_DRAIN
from jims_mower.types import CameraSpec, Pose

WEIGHTS_SCHEMA = "jims_mower.terrain_weights.v1"
N_CLASSES = int(HAZARD_DRAIN) + 1  # 0..3
# r, g, b, u_norm, v_norm, x_norm, y_norm
N_FEATURES = 7
FEATURE_NAMES = ("r", "g", "b", "u", "v", "x", "y")


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=-1, keepdims=True)
    exp = np.exp(z)
    return (exp / np.clip(exp.sum(axis=-1, keepdims=True), 1e-12, None)).astype(np.float32)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


@dataclass
class TerrainMLP:
    """Softmax logistic (hidden=0) or one-hidden-layer ReLU MLP."""

    W1: np.ndarray
    b1: np.ndarray
    W2: Optional[np.ndarray] = None
    b2: Optional[np.ndarray] = None
    n_features: int = N_FEATURES
    n_classes: int = N_CLASSES
    hidden: int = 0
    kind: str = "logistic"

    @property
    def is_mlp(self) -> bool:
        return self.W2 is not None and int(self.hidden) > 0

    def logits(self, x: np.ndarray) -> np.ndarray:
        h = np.asarray(x, dtype=np.float32)
        if h.ndim == 1:
            h = h.reshape(1, -1)
        z = h @ self.W1 + self.b1
        if self.is_mlp:
            assert self.W2 is not None and self.b2 is not None
            z = relu(z) @ self.W2 + self.b2
        return z

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return _softmax(self.logits(x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=-1).astype(np.uint8)


def init_weights(
    *,
    hidden: int = 8,
    n_features: int = N_FEATURES,
    n_classes: int = N_CLASSES,
    rng: Optional[np.random.Generator] = None,
) -> TerrainMLP:
    rng = rng or np.random.default_rng(0)
    if hidden <= 0:
        scale = np.sqrt(2.0 / max(n_features, 1))
        return TerrainMLP(
            W1=(rng.normal(0.0, scale, (n_features, n_classes))).astype(np.float32),
            b1=np.zeros(n_classes, dtype=np.float32),
            n_features=n_features,
            n_classes=n_classes,
            hidden=0,
            kind="logistic",
        )
    s1 = np.sqrt(2.0 / max(n_features, 1))
    s2 = np.sqrt(2.0 / max(hidden, 1))
    return TerrainMLP(
        W1=(rng.normal(0.0, s1, (n_features, hidden))).astype(np.float32),
        b1=np.zeros(hidden, dtype=np.float32),
        W2=(rng.normal(0.0, s2, (hidden, n_classes))).astype(np.float32),
        b2=np.zeros(n_classes, dtype=np.float32),
        n_features=n_features,
        n_classes=n_classes,
        hidden=hidden,
        kind="mlp",
    )


def save_weights(model: TerrainMLP, path: Union[str, Path], extra: Optional[dict[str, Any]] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": np.asarray(WEIGHTS_SCHEMA),
        "kind": np.asarray(model.kind),
        "n_features": np.int32(model.n_features),
        "n_classes": np.int32(model.n_classes),
        "hidden": np.int32(model.hidden),
        "W1": np.asarray(model.W1, dtype=np.float32),
        "b1": np.asarray(model.b1, dtype=np.float32),
        "feature_names": np.asarray(FEATURE_NAMES[: model.n_features]),
    }
    if model.W2 is not None:
        payload["W2"] = np.asarray(model.W2, dtype=np.float32)
    if model.b2 is not None:
        payload["b2"] = np.asarray(model.b2, dtype=np.float32)
    if extra:
        for key, value in extra.items():
            payload[f"meta_{key}"] = np.asarray(value)
    np.savez_compressed(path, **payload)
    return path


def load_weights(path: Union[str, Path]) -> TerrainMLP:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"terrain weights not found: {path}")
    data = np.load(path, allow_pickle=True)
    hidden = int(data["hidden"]) if "hidden" in data.files else 0
    w2 = data["W2"] if "W2" in data.files else None
    b2 = data["b2"] if "b2" in data.files else None
    kind = str(data["kind"]) if "kind" in data.files else ("mlp" if hidden else "logistic")
    return TerrainMLP(
        W1=np.asarray(data["W1"], dtype=np.float32),
        b1=np.asarray(data["b1"], dtype=np.float32),
        W2=None if w2 is None else np.asarray(w2, dtype=np.float32),
        b2=None if b2 is None else np.asarray(b2, dtype=np.float32),
        n_features=int(data["n_features"]) if "n_features" in data.files else N_FEATURES,
        n_classes=int(data["n_classes"]) if "n_classes" in data.files else N_CLASSES,
        hidden=hidden,
        kind=kind,
    )


def pixel_features(
    image: np.ndarray,
    *,
    world_x: Optional[np.ndarray] = None,
    world_y: Optional[np.ndarray] = None,
    world_size: tuple[float, float] = (12.0, 12.0),
) -> np.ndarray:
    """(H, W, 7) float32: RGB/255 + pixel UV + ground-plane XY (0.5 if missing)."""
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError("image must be HxWx3")
    height, width = image.shape[:2]
    rgb = image[:, :, :3].astype(np.float32) / 255.0
    us = (np.arange(width, dtype=np.float32) + 0.5) / max(width, 1)
    vs = (np.arange(height, dtype=np.float32) + 0.5) / max(height, 1)
    uu, vv = np.meshgrid(us, vs)
    if world_x is None or world_y is None:
        xx = np.full((height, width), 0.5, dtype=np.float32)
        yy = np.full((height, width), 0.5, dtype=np.float32)
    else:
        xx = (np.asarray(world_x, dtype=np.float32) / max(world_size[0], 1e-6)).clip(0.0, 1.0)
        yy = (np.asarray(world_y, dtype=np.float32) / max(world_size[1], 1e-6)).clip(0.0, 1.0)
        bad = ~np.isfinite(xx) | ~np.isfinite(yy)
        xx = np.where(bad, 0.5, xx)
        yy = np.where(bad, 0.5, yy)
    return np.concatenate(
        [rgb, uu[..., None], vv[..., None], xx[..., None], yy[..., None]],
        axis=-1,
    ).astype(np.float32)


def classify_image(
    image: np.ndarray,
    model: TerrainMLP,
    *,
    cam: Optional[CameraSpec] = None,
    pose: Optional[Pose] = None,
    world_size: tuple[float, float] = (12.0, 12.0),
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel hazard class and softmax confidence. Invalid / skipped → 0."""
    height, width = image.shape[:2]
    labels = np.zeros((height, width), dtype=np.uint8)
    conf = np.zeros((height, width), dtype=np.float32)
    hx = hy = valid = None
    if cam is not None and pose is not None:
        world_cam = camera_world_pose(pose, cam)
        hx, hy, valid = ground_hits(world_cam, width, height, ground_z=0.0)
    feats = pixel_features(image, world_x=hx, world_y=hy, world_size=world_size)
    step = max(1, int(stride))
    sl = (slice(0, height, step), slice(0, width, step))
    grid = feats[sl]
    rows = np.arange(0, height, step)
    cols = np.arange(0, width, step)
    x = grid.reshape(-1, feats.shape[-1])
    proba = model.predict_proba(x)
    pred = np.argmax(proba, axis=-1).astype(np.uint8)
    pmax = proba.max(axis=-1).astype(np.float32)
    pred = pred.reshape(grid.shape[:2])
    pmax = pmax.reshape(grid.shape[:2])
    if valid is not None:
        vis = valid[sl]
        pred = np.where(vis, pred, 0)
        pmax = np.where(vis, pmax, 0.0)
    labels[sl] = pred
    conf[sl] = pmax
    del rows, cols
    return labels, conf


def fit_numpy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    hidden: int = 8,
    epochs: int = 40,
    lr: float = 0.15,
    batch_size: int = 256,
    l2: float = 1e-4,
    seed: int = 0,
) -> tuple[TerrainMLP, dict[str, Any]]:
    """SGD on softmax CE. Class-balanced sampling, no claimed accuracy."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("x must be (N, F) aligned with y")
    n_features = int(x.shape[1])
    n_classes = N_CLASSES
    rng = np.random.default_rng(int(seed))
    model = init_weights(hidden=hidden, n_features=n_features, n_classes=n_classes, rng=rng)

    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    weights = np.zeros(n_classes, dtype=np.float64)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    class_w = weights.astype(np.float32)

    n = int(x.shape[0])
    bs = max(8, min(int(batch_size), n))
    history: list[float] = []
    for _ in range(max(1, int(epochs))):
        order = rng.permutation(n)
        epoch_loss = 0.0
        seen = 0
        for start in range(0, n, bs):
            idx = order[start : start + bs]
            xb = x[idx]
            yb = y[idx]
            proba = model.predict_proba(xb)
            one = np.zeros_like(proba)
            one[np.arange(len(yb)), yb] = 1.0
            cw = class_w[yb][:, None]
            loss = float((-np.log(np.clip(proba, 1e-8, 1.0)) * one * cw).sum() / max(cw.sum(), 1e-6))
            grad = (proba - one) * cw / max(float(len(yb)), 1.0)

            if model.is_mlp:
                assert model.W2 is not None and model.b2 is not None
                hidden_pre = xb @ model.W1 + model.b1
                hidden_act = relu(hidden_pre)
                dW2 = hidden_act.T @ grad + l2 * model.W2
                db2 = grad.sum(axis=0)
                dhid = (grad @ model.W2.T) * (hidden_pre > 0)
                dW1 = xb.T @ dhid + l2 * model.W1
                db1 = dhid.sum(axis=0)
                model.W2 = model.W2 - np.float32(lr) * dW2.astype(np.float32)
                model.b2 = model.b2 - np.float32(lr) * db2.astype(np.float32)
                model.W1 = model.W1 - np.float32(lr) * dW1.astype(np.float32)
                model.b1 = model.b1 - np.float32(lr) * db1.astype(np.float32)
            else:
                dW = xb.T @ grad + l2 * model.W1
                db = grad.sum(axis=0)
                model.W1 = model.W1 - np.float32(lr) * dW.astype(np.float32)
                model.b1 = model.b1 - np.float32(lr) * db.astype(np.float32)
            epoch_loss += loss * len(yb)
            seen += len(yb)
        history.append(epoch_loss / max(seen, 1))

    stats = {
        "backend": "numpy",
        "epochs": int(epochs),
        "n_samples": n,
        "class_counts": counts.astype(int).tolist(),
        "final_loss": float(history[-1]) if history else None,
        "hidden": int(hidden),
        "kind": model.kind,
        "note": "train loss only — not mAP / IoU",
    }
    return model, stats


def try_import_torch():
    try:
        import torch  # type: ignore

        return torch
    except ImportError:
        return None

"""Behaviour cloning hook: log terrain-policy (obs→action), train a tiny numpy stub.

Weights are a 1-hidden-layer tanh MLP. No claimed imitation score / SOTA.
``jims-mower-demo --policy bc`` loads ``bc_weights.npz`` when present and
falls back to ``TerrainPolicy`` otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from jims_mower.action_mask import mask_action
from jims_mower.constants import BC_FEATURE_DIM, BC_SCHEMA, DEFAULT_BC_WEIGHTS
from jims_mower.env import MowerEnv
from jims_mower.features import extract_features
from jims_mower.planning import TerrainPolicy
from jims_mower.scenarios import load_source

HIDDEN = 16


def default_weights_path() -> Path:
    return Path(DEFAULT_BC_WEIGHTS)


class NumpyMlp:
    """Tiny tanh MLP: features → 3 wheel/trimmer commands."""

    def __init__(
        self,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: np.ndarray,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ) -> None:
        self.w1 = np.asarray(w1, dtype=np.float32)
        self.b1 = np.asarray(b1, dtype=np.float32)
        self.w2 = np.asarray(w2, dtype=np.float32)
        self.b2 = np.asarray(b2, dtype=np.float32)
        self.mean = np.zeros(self.w1.shape[0], dtype=np.float32) if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = np.ones(self.w1.shape[0], dtype=np.float32) if std is None else np.asarray(std, dtype=np.float32)
        self.std = np.maximum(self.std, 1e-3)

    @classmethod
    def random(cls, rng: np.random.Generator, feat_dim: int = BC_FEATURE_DIM, hidden: int = HIDDEN) -> "NumpyMlp":
        scale1 = 1.0 / np.sqrt(feat_dim)
        scale2 = 1.0 / np.sqrt(hidden)
        return cls(
            rng.normal(0.0, scale1, size=(feat_dim, hidden)).astype(np.float32),
            np.zeros(hidden, dtype=np.float32),
            rng.normal(0.0, scale2, size=(hidden, 3)).astype(np.float32),
            np.zeros(3, dtype=np.float32),
        )

    def normalize(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        return (x - self.mean) / self.std

    def forward(self, features: np.ndarray) -> np.ndarray:
        x = self.normalize(features)
        if x.ndim == 1:
            h = np.tanh(x @ self.w1 + self.b1)
            y = h @ self.w2 + self.b2
            return y.astype(np.float32)
        h = np.tanh(x @ self.w1 + self.b1)
        return (h @ self.w2 + self.b2).astype(np.float32)

    def act_raw(self, features: np.ndarray) -> np.ndarray:
        y = self.forward(features)
        if y.ndim == 1:
            out = y.copy()
        else:
            out = y[0].copy()
        out[0] = float(np.clip(out[0], -1.0, 1.0))
        out[1] = float(np.clip(out[1], -1.0, 1.0))
        out[2] = float(np.clip(out[2], 0.0, 1.0))
        return out.astype(np.float32)


def save_weights(path: Union[str, Path], net: NumpyMlp, extra: Optional[dict[str, Any]] = None) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": np.asarray(BC_SCHEMA),
        "w1": net.w1,
        "b1": net.b1,
        "w2": net.w2,
        "b2": net.b2,
        "mean": net.mean,
        "std": net.std,
    }
    if extra:
        payload["meta_json"] = np.asarray(json.dumps(extra))
    np.savez_compressed(dest, **payload)
    return dest


def load_weights(path: Union[str, Path]) -> NumpyMlp:
    data = np.load(Path(path), allow_pickle=False)
    return NumpyMlp(data["w1"], data["b1"], data["w2"], data["b2"], data["mean"], data["std"])


def weights_available(path: Optional[Union[str, Path]] = None) -> bool:
    dest = Path(path) if path else default_weights_path()
    return dest.is_file()


class BcPolicy:
    """Tiny cloned policy. Applies the hazard action mask after the MLP."""

    def __init__(self, net: NumpyMlp, resolution_m: float = 0.10) -> None:
        self.net = net
        self.resolution_m = float(resolution_m)
        self.last_advice = "ok"
        self.loaded = True

    @classmethod
    def load(cls, path: Union[str, Path], resolution_m: float = 0.10) -> "BcPolicy":
        return cls(load_weights(path), resolution_m=resolution_m)

    def reset(self, obs: dict[str, Any], info: Optional[dict[str, Any]] = None) -> None:
        return None

    def act(self, obs: dict[str, Any], info: Optional[dict[str, Any]] = None) -> np.ndarray:
        info = info or {}
        feats = extract_features(obs, info, resolution_m=self.resolution_m)
        raw = self.net.act_raw(feats)
        masked = mask_action(raw, obs, resolution_m=self.resolution_m, info=info)
        self.last_advice = str(info.get("terrain_advice") or "ok")
        return masked


def collect_demos(
    out_dir: Union[str, Path],
    *,
    steps: int = 40,
    seed: int = 7,
    config: Optional[Any] = None,
    cameras: Optional[int] = None,
    terrain_observer: Optional[str] = None,
) -> dict[str, Any]:
    """Run ``TerrainPolicy`` and write ``features.npy`` / ``actions.npy``."""
    cfg, scenario = load_source(config)
    if cameras is not None:
        cfg.sensors.camera_count = cameras
        cfg.sensors.cameras = []
    if terrain_observer:
        key = terrain_observer.strip().lower()
        if key not in {"oracle", "heuristic", "blind"}:
            raise ValueError(f"terrain_observer must be oracle|heuristic|blind; got {key!r}")
        cfg.perception.terrain_mode = key
    env = MowerEnv(config=cfg, scenario=scenario, render_mode=None)
    obs, info = env.reset(seed=seed)
    policy = TerrainPolicy(env.cfg)
    policy.reset(obs, info)
    feats: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    try:
        for _ in range(int(steps)):
            action = policy.act(obs, info)
            feats.append(extract_features(obs, info, resolution_m=env.cfg.world.resolution_m))
            actions.append(np.asarray(action, dtype=np.float32).reshape(-1)[:3])
            obs, _reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    finally:
        env.close()
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    x = np.stack(feats, axis=0) if feats else np.zeros((0, BC_FEATURE_DIM), dtype=np.float32)
    y = np.stack(actions, axis=0) if actions else np.zeros((0, 3), dtype=np.float32)
    np.save(dest / "features.npy", x)
    np.save(dest / "actions.npy", y)
    meta = {
        "schema": BC_SCHEMA,
        "n_samples": int(x.shape[0]),
        "feature_dim": int(BC_FEATURE_DIM),
        "seed": int(seed),
        "policy": "terrain",
        "not_a_benchmark": True,
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.savez_compressed(dest / "dataset.npz", features=x, actions=y)
    return meta


def _load_xy(source: Union[str, Path]) -> tuple[np.ndarray, np.ndarray]:
    path = Path(source)
    if path.is_dir():
        feat_p = path / "features.npy"
        act_p = path / "actions.npy"
        if feat_p.is_file() and act_p.is_file():
            return np.load(feat_p), np.load(act_p)
        npz = path / "dataset.npz"
        if npz.is_file():
            data = np.load(npz)
            return data["features"], data["actions"]
        from jims_mower.episode import EpisodeReader

        reader = EpisodeReader(path)
        feats: list[np.ndarray] = []
        acts: list[np.ndarray] = []
        res = float(reader.config().world.resolution_m)
        for rec in reader.steps:
            feats.append(extract_features(rec["obs"], rec.get("info") or {}, resolution_m=res))
            acts.append(np.asarray(rec["action_arr"], dtype=np.float32).reshape(-1)[:3])
        if not feats:
            raise ValueError(f"no BC samples in episode {path}")
        return np.stack(feats), np.stack(acts)
    data = np.load(path)
    if "features" in data.files and "actions" in data.files:
        return data["features"], data["actions"]
    raise FileNotFoundError(f"no BC dataset at {path}")


def train_bc(
    source: Union[str, Path],
    out_path: Union[str, Path],
    *,
    hidden: int = HIDDEN,
    epochs: int = 80,
    lr: float = 0.05,
    seed: int = 0,
    batch_size: int = 32,
) -> dict[str, Any]:
    """SGD on a tiny tanh MLP. Reports train MSE only — not a published score."""
    x, y = _load_xy(source)
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or y.shape[1] != 3:
        raise ValueError("BC dataset must be features (N, D) and actions (N, 3)")
    if x.shape[0] < 1:
        raise ValueError("BC dataset is empty")
    if x.shape[1] != BC_FEATURE_DIM:
        raise ValueError(f"feature dim {x.shape[1]} != {BC_FEATURE_DIM}")
    rng = np.random.default_rng(seed)
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 1e-3)
    xn = (x - mean) / std
    net = NumpyMlp.random(rng, feat_dim=x.shape[1], hidden=hidden)
    net.mean = mean.astype(np.float32)
    net.std = std.astype(np.float32)
    n = xn.shape[0]
    last_mse = 0.0
    for _ in range(int(epochs)):
        idx = rng.permutation(n)
        total = 0.0
        seen = 0
        for start in range(0, n, max(1, int(batch_size))):
            batch = idx[start : start + max(1, int(batch_size))]
            xb = xn[batch]
            yb = y[batch]
            h = np.tanh(xb @ net.w1 + net.b1)
            pred = h @ net.w2 + net.b2
            err = pred - yb
            total += float(np.mean(err * err))
            seen += 1
            grad_y = (2.0 / max(len(batch), 1)) * err
            dw2 = h.T @ grad_y
            db2 = grad_y.sum(axis=0)
            dh = (grad_y @ net.w2.T) * (1.0 - h * h)
            dw1 = xb.T @ dh
            db1 = dh.sum(axis=0)
            net.w2 -= np.float32(lr) * dw2.astype(np.float32)
            net.b2 -= np.float32(lr) * db2.astype(np.float32)
            net.w1 -= np.float32(lr) * dw1.astype(np.float32)
            net.b1 -= np.float32(lr) * db1.astype(np.float32)
        last_mse = total / max(seen, 1)
    extra = {
        "schema": BC_SCHEMA,
        "n_samples": int(n),
        "hidden": int(hidden),
        "epochs": int(epochs),
        "train_mse": last_mse,
        "not_a_benchmark": True,
    }
    save_weights(out_path, net, extra)
    extra["out"] = str(Path(out_path))
    return extra


def maybe_load_bc(
    path: Optional[Union[str, Path]],
    resolution_m: float,
) -> Optional[BcPolicy]:
    dest = Path(path) if path else default_weights_path()
    if dest.is_file():
        return BcPolicy.load(dest, resolution_m=resolution_m)
    return None

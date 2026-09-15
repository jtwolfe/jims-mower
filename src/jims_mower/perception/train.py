"""Train a tiny terrain stub from a WAVE 1A export folder.

Default backend is the numpy colour+position logistic / MLP. Torch is an
optional extra (``pip install -e ".[torch]"``) and is never required for CI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from PIL import Image

from jims_mower.cameras import camera_world_pose, ground_hits
from jims_mower.perception.learn import (
    N_CLASSES,
    N_FEATURES,
    TerrainMLP,
    fit_numpy,
    pixel_features,
    save_weights,
    try_import_torch,
)
from jims_mower.types import CameraSpec, Pose


def _pose_from_sidecar(raw: Any) -> Pose:
    if not isinstance(raw, dict):
        raise ValueError("frame sidecar pose must be a mapping")
    return Pose(
        float(raw.get("x", 0.0)),
        float(raw.get("y", 0.0)),
        float(raw.get("theta", 0.0)),
        float(raw.get("z", 0.0)),
        float(raw.get("pitch", 0.0)),
        float(raw.get("roll", 0.0)),
    )


def _cameras_from_meta(meta: dict[str, Any]) -> list[CameraSpec]:
    specs = meta.get("camera_specs") or []
    out: list[CameraSpec] = []
    for item in specs:
        if not isinstance(item, dict) or "name" not in item:
            continue
        out.append(
            CameraSpec(
                name=str(item["name"]),
                x=float(item.get("x", 0.0)),
                y=float(item.get("y", 0.0)),
                z=float(item.get("z", 0.38)),
                yaw_deg=float(item.get("yaw_deg", 0.0)),
                pitch_deg=float(item.get("pitch_deg", -12.0)),
                fov_deg=float(item.get("fov_deg", 70.0)),
            )
        )
    return out


def samples_from_export(
    dataset_dir: Union[str, Path],
    *,
    stride: int = 2,
    max_per_class: int = 400,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Pixel colour+position features labelled by the oracle hazard raster."""
    root = Path(dataset_dir)
    meta_path = root / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"export meta.json missing: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cameras = {c.name: c for c in _cameras_from_meta(meta)}
    if not cameras:
        raise ValueError(
            "export meta.json has no camera_specs; re-export with this WAVE 2B exporter"
        )
    resolution = float(meta.get("resolution_m") or 0.10)
    world = meta.get("world_size") or [12.0, 12.0]
    world_size = (float(world[0]), float(world[1]))
    map_shape = meta.get("map_shape") or [1, 1]
    map_rows, map_cols = int(map_shape[0]), int(map_shape[1])

    frames_dir = root / "frames"
    sidecars = sorted(frames_dir.glob("*.json"))
    if not sidecars:
        raise FileNotFoundError(f"no frame sidecars in {frames_dir}")

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for sidecar_path in sidecars:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        pose = _pose_from_sidecar(sidecar.get("pose"))
        haz_rel = (sidecar.get("labels") or {}).get("hazard")
        if not haz_rel:
            continue
        hazard = np.asarray(Image.open(root / haz_rel).convert("L"), dtype=np.uint8)
        if hazard.shape[0] != map_rows or hazard.shape[1] != map_cols:
            # Still usable if we index with world XY / resolution.
            map_rows, map_cols = hazard.shape
        cam_paths = sidecar.get("cameras") or {}
        for cam_name, rel in cam_paths.items():
            cam = cameras.get(cam_name)
            if cam is None:
                continue
            image = np.asarray(Image.open(root / rel).convert("RGB"), dtype=np.uint8)
            height, width = image.shape[:2]
            world_cam = camera_world_pose(pose, cam)
            hx, hy, valid = ground_hits(world_cam, width, height, ground_z=0.0)
            feats = pixel_features(image, world_x=hx, world_y=hy, world_size=world_size)
            rows = np.floor(hy / max(resolution, 1e-6)).astype(np.int32)
            cols = np.floor(hx / max(resolution, 1e-6)).astype(np.int32)
            inb = (
                valid
                & (rows >= 0)
                & (cols >= 0)
                & (rows < hazard.shape[0])
                & (cols < hazard.shape[1])
            )
            if stride > 1:
                grid = np.zeros((height, width), dtype=bool)
                grid[0::stride, 0::stride] = True
                inb = inb & grid
            if not np.any(inb):
                continue
            lab = hazard[rows[inb], cols[inb]].astype(np.int64)
            lab = np.clip(lab, 0, N_CLASSES - 1)
            xs.append(feats[inb])
            ys.append(lab)

    if not xs:
        raise ValueError("no labelled ground-plane pixels in export (empty cameras?)")
    x_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    rng = np.random.default_rng(int(seed))
    keep_idx: list[np.ndarray] = []
    for cls in range(N_CLASSES):
        idx = np.flatnonzero(y_all == cls)
        if idx.size == 0:
            continue
        if idx.size > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep_idx.append(idx)
    keep = np.concatenate(keep_idx)
    keep = rng.permutation(keep)
    x_all = x_all[keep]
    y_all = y_all[keep]
    info = {
        "n_pixels": int(x_all.shape[0]),
        "class_counts": np.bincount(y_all, minlength=N_CLASSES).astype(int).tolist(),
        "n_frames": len(sidecars),
        "schema": meta.get("schema"),
        "domain_randomization": meta.get("domain_randomization"),
        "n_features": N_FEATURES,
    }
    return x_all, y_all, info


def _fit_torch(
    x: np.ndarray,
    y: np.ndarray,
    *,
    hidden: int,
    epochs: int,
    lr: float,
    seed: int,
) -> tuple[TerrainMLP, dict[str, Any]]:
    torch = try_import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed; use --backend numpy")
    torch.manual_seed(int(seed))
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    n_features = int(x.shape[1])
    if hidden <= 0:
        net = torch.nn.Linear(n_features, N_CLASSES)
    else:
        net = torch.nn.Sequential(
            torch.nn.Linear(n_features, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, N_CLASSES),
        )
    opt = torch.optim.SGD(net.parameters(), lr=float(lr), weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    last = 0.0
    for _ in range(max(1, int(epochs))):
        opt.zero_grad()
        logits = net(xt)
        loss = loss_fn(logits, yt)
        loss.backward()
        opt.step()
        last = float(loss.detach())
    # Copy tensors back into TerrainMLP so inference stays numpy-only.
    if hidden <= 0:
        W1 = net.weight.detach().cpu().numpy().T.astype(np.float32)
        b1 = net.bias.detach().cpu().numpy().astype(np.float32)
        model = TerrainMLP(W1=W1, b1=b1, n_features=n_features, hidden=0, kind="logistic")
    else:
        linear1, _, linear2 = net[0], net[1], net[2]
        model = TerrainMLP(
            W1=linear1.weight.detach().cpu().numpy().T.astype(np.float32),
            b1=linear1.bias.detach().cpu().numpy().astype(np.float32),
            W2=linear2.weight.detach().cpu().numpy().T.astype(np.float32),
            b2=linear2.bias.detach().cpu().numpy().astype(np.float32),
            n_features=n_features,
            hidden=hidden,
            kind="mlp",
        )
    stats = {
        "backend": "torch",
        "epochs": int(epochs),
        "n_samples": int(x.shape[0]),
        "final_loss": last,
        "hidden": int(hidden),
        "kind": model.kind,
        "note": "train loss only — not mAP / IoU",
    }
    return model, stats


def train_from_export(
    dataset_dir: Union[str, Path],
    out_path: Union[str, Path],
    *,
    hidden: int = 8,
    epochs: int = 35,
    lr: float = 0.15,
    stride: int = 2,
    max_per_class: int = 400,
    seed: int = 0,
    backend: str = "numpy",
) -> dict[str, Any]:
    x, y, info = samples_from_export(
        dataset_dir, stride=stride, max_per_class=max_per_class, seed=seed
    )
    key = (backend or "numpy").strip().lower()
    if key == "torch":
        if try_import_torch() is None:
            key = "numpy"
            info["torch_fallback"] = "torch unavailable; trained with numpy"
        else:
            model, stats = _fit_torch(x, y, hidden=hidden, epochs=epochs, lr=lr, seed=seed)
            path = save_weights(model, out_path, extra={"dataset": str(dataset_dir)})
            out = {**info, **stats, "weights": str(path)}
            return out
    model, stats = fit_numpy(x, y, hidden=hidden, epochs=epochs, lr=lr, seed=seed)
    path = save_weights(model, out_path, extra={"dataset": str(dataset_dir)})
    return {**info, **stats, "weights": str(path)}


def export_and_train(
    out_dir: Union[str, Path],
    *,
    steps: int = 6,
    seed: int = 7,
    cameras: int = 4,
    domain_rand: bool = False,
    hidden: int = 8,
    epochs: int = 25,
    config: Optional[str] = None,
) -> dict[str, Any]:
    """Minimal labelled dump + train. Used by tests and the CLI."""
    from jims_mower.export import export_dataset

    out_dir = Path(out_dir)
    dataset = out_dir / "dataset"
    weights = out_dir / "terrain_mlp.npz"
    meta = export_dataset(
        dataset,
        steps=steps,
        seed=seed,
        cameras=cameras,
        policy="scripted",
        config=config,
        domain_rand=domain_rand,
    )
    stats = train_from_export(
        dataset,
        weights,
        hidden=hidden,
        epochs=epochs,
        seed=seed,
        backend="numpy",
    )
    stats["export"] = meta
    stats["dataset"] = str(dataset)
    return stats


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a tiny numpy (or optional torch) terrain stub from an export"
    )
    p.add_argument("--dataset", type=Path, default=None, help="Existing export folder")
    p.add_argument("--out", type=Path, default=Path("terrain_mlp.npz"))
    p.add_argument("--export-out", type=Path, default=None, help="If no --dataset, write a tiny export here")
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cameras", type=int, default=4)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--domain-rand", action="store_true", help="Export with renderer DR on")
    p.add_argument("--hidden", type=int, default=8, help="0 = logistic; >0 = one hidden layer")
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--lr", type=float, default=0.15)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--max-per-class", type=int, default=400)
    p.add_argument(
        "--backend",
        choices=("numpy", "torch"),
        default="numpy",
        help="torch is optional and unused in CI",
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    dataset = args.dataset
    if dataset is None:
        from jims_mower.export import export_dataset

        dataset = args.export_out or (args.out.parent / "dataset")
        dataset.mkdir(parents=True, exist_ok=True)
        export_dataset(
            dataset,
            steps=args.steps,
            seed=args.seed,
            cameras=args.cameras,
            policy="scripted",
            config=args.config,
            domain_rand=args.domain_rand,
        )
    stats = train_from_export(
        dataset,
        args.out,
        hidden=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        stride=args.stride,
        max_per_class=args.max_per_class,
        seed=args.seed,
        backend=args.backend,
    )
    stats["dataset"] = str(dataset)
    print(
        f"trained {stats.get('kind')} backend={stats.get('backend')} "
        f"n={stats.get('n_pixels') or stats.get('n_samples')} "
        f"loss={stats.get('final_loss')} → {stats.get('weights')}"
    )
    print("note:", stats.get("note"))


if __name__ == "__main__":
    main()

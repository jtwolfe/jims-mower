"""Deterministic dataset exporter: synced RGB + sensors + oracle labels.

Folder layout (PNG + JSON sidecars, COCO-seg-like index)::

    <out>/
      LAYOUT.md
      meta.json
      coco.json
      images/{frame:06d}_{camera}.png
      labels/{frame:06d}_hazard.png      # uint8 0–3 (oracle)
      labels/{frame:06d}_grass.png       # uint8 0 uncut, 1 cut, 255 non-grass
      labels/{frame:06d}_elevation.npy   # float32 metres (oracle)
      labels/{frame:06d}_slope.npy       # float32 radians (oracle)
      frames/{frame:06d}.json            # imu / gps / tof / pose / paths

Oracle rasters come from the true height field and grass map, not the
heuristic observer. Seeds are passed to ``env.reset(seed=...)``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from jims_mower.constants import DATASET_SCHEMA, LABEL_TO_ID, TERRAIN_MODE_CLI
from jims_mower.dataset import assign_frame_split, validate_dataset_meta
from jims_mower.env import MowerEnv
from jims_mower.runtime.capture import adapter_kind
from jims_mower.metrics import POLICIES, _random_action, _scripted_action
from jims_mower.planning import TerrainPolicy
from jims_mower.scenarios import load_source

LAYOUT_MD = """# jims-mower dataset layout (WAVE 1A)

Synced multi-camera RGB plus oracle yard labels. No claimed mAP / FPS.

```
<out>/
  LAYOUT.md                 this file
  meta.json                 seed, cameras, scenario, schema version
  coco.json                 COCO-like images / annotations / categories
  images/{frame:06d}_{cam}.png
  labels/{frame:06d}_hazard.png     uint8: 0 free, 1 steep, 2 drain lip, 3 channel
  labels/{frame:06d}_grass.png      uint8: 0 uncut, 1 cut, 255 non-grass
  labels/{frame:06d}_elevation.npy  float32 metres, same grid as coverage
  labels/{frame:06d}_slope.npy      float32 radians
  frames/{frame:06d}.json           imu, gps, optional tof, pose, file paths
  split.json                train / val frame indices (last-frac val)
```

`coco.json` lists every camera frame as an image and mock/oracle boxes as
annotations (`bbox` is `[u, v, w, h]` in pixels). Semantic rasters are
referenced from each frame sidecar, not as COCO RLE (keep it numpy/PNG).

Train/val: last ``val_frac`` of frames are val (see ``meta.split`` /
``split.json``). Fake CSI / FakeGst frames are valid for the *pipeline*
in CI — they are not real photos. Label protocol: ``docs/DATASET.md``.
No published mAP / IoU.
"""


def _save_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(path)


def _save_gray(path: Path, image: np.ndarray) -> None:
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="L").save(path)


def _grass_png(coverage: np.ndarray) -> np.ndarray:
    """0 uncut, 1 cut, 255 non-grass."""
    cov = np.asarray(coverage, dtype=np.float32)
    out = np.zeros(cov.shape, dtype=np.uint8)
    out[cov > 0.5] = 1
    out[cov < 0.0] = 255
    return out


def export_dataset(
    out_dir: Path,
    *,
    steps: int = 20,
    seed: int = 7,
    config: Optional[str] = None,
    cameras: Optional[int] = None,
    policy: str = "terrain",
    include_tof: bool = True,
    terrain_observer: Optional[str] = None,
    domain_rand: bool = False,
    adapter: Optional[str] = None,
    val_frac: float = 0.20,
) -> dict[str, Any]:
    name = (policy or "terrain").strip().lower()
    if name not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}; got {policy!r}")
    cfg, scenario = load_source(config)
    if cameras is not None:
        cfg.sensors.camera_count = cameras
        cfg.sensors.cameras = []
    if terrain_observer:
        from jims_mower.perception.terrain import normalize_terrain_mode

        key = normalize_terrain_mode(terrain_observer)
        cfg.perception.terrain_mode = key
    if domain_rand:
        cfg.domain_randomization.enabled = True
        cfg.domain_randomization.lighting = True
        cfg.domain_randomization.colour_jitter = True
        cfg.domain_randomization.shadow_blobs = True
        cfg.domain_randomization.camera_dirt = True
        cfg.domain_randomization.vignette = True
    if adapter:
        kind = adapter_kind(adapter)
        if kind not in {"renderer", "fake_gst", "fake_csi", "gst"}:
            raise ValueError(
                "adapter must be renderer|fake_gst|fake_csi|gst; "
                f"got {adapter!r}"
            )
        cfg.runtime.cameras.adapter = "" if kind == "renderer" else kind
    source_kind = adapter_kind(getattr(cfg.runtime.cameras, "adapter", ""))
    env = MowerEnv(config=cfg, scenario=scenario, render_mode=None)
    obs, info = env.reset(seed=seed)
    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    frames_dir = out_dir / "frames"
    for folder in (images_dir, labels_dir, frames_dir):
        folder.mkdir(parents=True, exist_ok=True)

    terrain_policy: Optional[TerrainPolicy] = None
    rng = np.random.default_rng(seed)
    if name == "terrain":
        terrain_policy = TerrainPolicy(env.cfg)
        terrain_policy.reset(obs, info)

    cam_names = list(obs["cameras"].keys())
    coco_images: list[dict[str, Any]] = []
    coco_anns: list[dict[str, Any]] = []
    image_id = 1
    ann_id = 1
    dumped = 0

    def _dump(frame_i: int, obs_i: dict[str, Any], info_i: dict[str, Any]) -> None:
        nonlocal image_id, ann_id
        token = f"{frame_i:06d}"
        labels = env.oracle_labels()
        cam_paths = {}
        h = int(env.cfg.sensors.height)
        w = int(env.cfg.sensors.width)
        for cam_name, frame in obs_i["cameras"].items():
            rel = f"images/{token}_{cam_name}.png"
            _save_rgb(out_dir / rel, frame)
            cam_paths[cam_name] = rel
            coco_images.append(
                {
                    "id": image_id,
                    "file_name": rel,
                    "width": w,
                    "height": h,
                    "camera": cam_name,
                    "frame_index": frame_i,
                }
            )
            for det in info_i.get("detections") or []:
                if det.get("camera") != cam_name:
                    continue
                bbox = det.get("bbox") or [0, 0, 0, 0]
                coco_anns.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": int(LABEL_TO_ID.get(det.get("label"), 0)),
                        "bbox": [int(x) for x in bbox],
                        "score": float(det.get("confidence") or 0.0),
                        "iscrowd": 0,
                        "world_xy": det.get("world_xy"),
                    }
                )
                ann_id += 1
            image_id += 1
        haz_rel = f"labels/{token}_hazard.png"
        grass_rel = f"labels/{token}_grass.png"
        elev_rel = f"labels/{token}_elevation.npy"
        slope_rel = f"labels/{token}_slope.npy"
        _save_gray(out_dir / haz_rel, np.clip(labels["hazard"], 0, 3).astype(np.uint8))
        _save_gray(out_dir / grass_rel, _grass_png(labels["grass"]))
        np.save(out_dir / elev_rel, labels["elevation"])
        np.save(out_dir / slope_rel, labels["slope"])
        sidecar = {
            "frame": frame_i,
            "pose": info_i.get("pose"),
            "imu": info_i.get("imu"),
            "gps": info_i.get("gps"),
            "tof": info_i.get("tof") if include_tof else None,
            "trimmer_enabled": info_i.get("trimmer_enabled"),
            "terrain_advice": info_i.get("terrain_advice"),
            "scenario": info_i.get("scenario"),
            "weather": info_i.get("weather"),
            "cameras": cam_paths,
            "labels": {
                "hazard": haz_rel,
                "grass": grass_rel,
                "elevation": elev_rel,
                "slope": slope_rel,
            },
        }
        (frames_dir / f"{token}.json").write_text(
            json.dumps(sidecar, indent=2), encoding="utf-8"
        )

    _dump(0, obs, info)
    dumped = 1
    for t in range(max(0, int(steps))):
        if name == "scripted":
            action = _scripted_action()
        elif name == "random":
            action = _random_action(env, rng)
        else:
            assert terrain_policy is not None
            action = terrain_policy.act(obs, info)
        obs, _reward, terminated, truncated, info = env.step(action)
        _dump(t + 1, obs, info)
        dumped += 1
        if terminated or truncated:
            break

    categories = [{"id": i, "name": lab, "supercategory": "object"} for lab, i in LABEL_TO_ID.items()]
    meta = {
        "schema": DATASET_SCHEMA,
        "seed": int(seed),
        "steps_requested": int(steps),
        "frames": dumped,
        "cameras": cam_names,
        "policy": name,
        "scenario": env.scenario.name if env.scenario else "",
        "include_tof": bool(include_tof),
        "terrain_mode_obs": env.cfg.perception.terrain_mode,
        "labels": "oracle height-field + grass map",
        "image_size": [int(env.cfg.sensors.width), int(env.cfg.sensors.height)],
        "map_shape": list(env._coverage.cut.shape),
        "resolution_m": env.cfg.world.resolution_m,
        "world_size": [env.cfg.world.width_m, env.cfg.world.height_m],
        "camera_specs": [
            {
                "name": c.name,
                "x": c.x,
                "y": c.y,
                "z": c.z,
                "yaw_deg": c.yaw_deg,
                "pitch_deg": c.pitch_deg,
                "fov_deg": c.fov_deg,
            }
            for c in env.cameras
        ],
        "domain_randomization": {
            "enabled": bool(env.cfg.domain_randomization.enabled),
            "lighting": bool(env.cfg.domain_randomization.lighting),
            "colour_jitter": bool(env.cfg.domain_randomization.colour_jitter),
            "shadow_blobs": bool(env.cfg.domain_randomization.shadow_blobs),
            "camera_dirt": bool(env.cfg.domain_randomization.camera_dirt),
            "vignette": bool(env.cfg.domain_randomization.vignette),
        },
        "source": source_kind,
        "adapter": source_kind,
        "split": assign_frame_split(dumped, val_frac=val_frac),
        "map_claim": None,
        "fps_claim": None,
    }
    coco = {
        "info": {
            "description": "jims-mower WAVE 1A export",
            "seed": int(seed),
            "version": "1.0",
        },
        "images": coco_images,
        "annotations": coco_anns,
        "categories": categories,
        "semantic": {
            "hazard": {"0": "free", "1": "steep", "2": "drain_lip", "3": "channel"},
            "grass": {"0": "uncut", "1": "cut", "255": "non_grass"},
        },
    }
    validate_dataset_meta(meta)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "coco.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")
    (out_dir / "split.json").write_text(
        json.dumps(meta["split"], indent=2), encoding="utf-8"
    )
    (out_dir / "LAYOUT.md").write_text(LAYOUT_MD, encoding="utf-8")
    env.close()
    return meta


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export a synced jims-mower dataset folder")
    p.add_argument("--out", type=Path, default=Path("dataset_out"))
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--config", type=str, default=None, help="Env YAML or scenario name/path")
    p.add_argument("--cameras", type=int, default=None)
    p.add_argument("--policy", choices=POLICIES, default="terrain")
    p.add_argument("--no-tof", action="store_true", help="Omit ToF from sidecars")
    p.add_argument(
        "--terrain-observer",
        choices=TERRAIN_MODE_CLI,
        default=None,
        help="Observer used for env obs (labels stay oracle)",
    )
    p.add_argument(
        "--domain-rand",
        action="store_true",
        help="Enable renderer domain randomisation for training exports (see docs/WAVE2B.md)",
    )
    p.add_argument(
        "--adapter",
        choices=("renderer", "fake_gst", "fake_csi", "gst"),
        default=None,
        help="Camera source. fake_csi / fake_gst are OK in CI (not real photos).",
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.20,
        help="Last-fraction val split written to meta.split / split.json",
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    meta = export_dataset(
        args.out,
        steps=args.steps,
        seed=args.seed,
        config=args.config,
        cameras=args.cameras,
        policy=args.policy,
        include_tof=not args.no_tof,
        terrain_observer=args.terrain_observer,
        domain_rand=args.domain_rand,
        adapter=args.adapter,
        val_frac=args.val_frac,
    )
    print(
        f"Wrote {meta['frames']} frames, cameras={meta['cameras']}, "
        f"scenario={meta['scenario']!r} → {args.out}"
    )


if __name__ == "__main__":
    main()

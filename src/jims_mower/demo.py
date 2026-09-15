"""Headless demo: dump multi-camera frames, detections, and a coverage map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from jims_mower.config import load_config
from jims_mower.env import MowerEnv


def _save_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image, mode="RGB").save(path)


def _montage(images: dict[str, np.ndarray], order: list[str]) -> np.ndarray:
    frames = [images[name] for name in order if name in images]
    if not frames:
        raise ValueError("no camera frames to montage")
    h, w = frames[0].shape[:2]
    cols = min(3, len(frames))
    rows = int(np.ceil(len(frames) / cols))
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        r, c = divmod(i, cols)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = frame
    return canvas


def run_demo(
    out_dir: Path,
    *,
    steps: int = 40,
    seed: int = 7,
    cameras: Optional[int] = None,
    hand_signals: bool = False,
    config: Optional[str] = None,
) -> dict:
    cfg = load_config(config)
    if cameras is not None:
        cfg.sensors.camera_count = cameras
        cfg.sensors.cameras = []
    env = MowerEnv(config=cfg, render_mode="rgb_array", hand_signals=hand_signals)
    obs, info = env.reset(seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = list(obs["cameras"].keys())
    records: list[dict] = []
    dump_steps = {0, max(0, steps // 2), max(0, steps - 1)}

    def _dump_step(step_dir: Path, obs: dict, info: dict) -> None:
        step_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in obs["cameras"].items():
            _save_rgb(step_dir / f"cam_{name}.png", frame)
        _save_rgb(step_dir / "montage.png", _montage(obs["cameras"], names))
        topdown = env.render()
        if topdown is not None:
            _save_rgb(step_dir / "topdown.png", topdown)
        for layer, img in env.terrain_layer_images().items():
            _save_rgb(step_dir / f"{layer}.png", img)
        (step_dir / "detections.json").write_text(
            json.dumps(info["detections"], indent=2),
            encoding="utf-8",
        )
        (step_dir / "sensors.json").write_text(
            json.dumps(
                {
                    "imu": info.get("imu"),
                    "gps": info.get("gps"),
                    "tof": info.get("tof"),
                    "pose": info.get("pose"),
                    "n_drains": info.get("n_drains"),
                    "n_banks": info.get("n_banks"),
                    "terrain_source": info.get("terrain_source"),
                    "terrain_advice": info.get("terrain_advice"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    for t in range(steps):
        if t in dump_steps:
            _dump_step(out_dir / f"step_{t:03d}", obs, info)

        # Slow forward creep with the trimmer requested.
        action = np.array([0.55, 0.50, 1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        records.append(
            {
                "step": t,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "coverage_fraction": info["coverage_fraction"],
                "trimmer_enabled": info["trimmer_enabled"],
                "n_detections": len(info["detections"]),
                "collision": info.get("collision"),
                "tipover": info.get("tipover"),
                "drain_drop": info.get("drain_drop"),
                "steep": info.get("steep"),
                "terrain_advice": info.get("terrain_advice"),
                "imu": info.get("imu"),
                "gps": info.get("gps"),
            }
        )
        if terminated or truncated:
            break

    topdown = env.render()
    if topdown is not None:
        _save_rgb(out_dir / "coverage_final.png", topdown)
    layers = env.terrain_layer_images()
    for layer, img in layers.items():
        _save_rgb(out_dir / f"{layer}_final.png", img)
    summary = {
        "steps_run": len(records),
        "cameras": names,
        "final_coverage_fraction": records[-1]["coverage_fraction"] if records else 0.0,
        "final_detections": info["detections"],
        "hand_signals_enabled": hand_signals,
        "n_drains": info.get("n_drains"),
        "n_banks": info.get("n_banks"),
        "final_pose": info.get("pose"),
        "final_imu": info.get("imu"),
        "final_gps": info.get("gps"),
        "log": records,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    env.close()
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dump Jim's Mower camera frames + detections")
    p.add_argument("--out", type=Path, default=Path("demo_out"))
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cameras", type=int, default=None, help="4, 5, or 6")
    p.add_argument("--hand-signals", action="store_true")
    p.add_argument("--config", type=str, default=None)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    summary = run_demo(
        args.out,
        steps=args.steps,
        seed=args.seed,
        cameras=args.cameras,
        hand_signals=args.hand_signals,
        config=args.config,
    )
    print(
        f"Wrote {summary['steps_run']} steps, cameras={summary['cameras']} → {args.out}"
    )
    print(f"Final coverage: {summary['final_coverage_fraction']:.4f}")
    print(f"Detections on last step: {len(summary['final_detections'])}")


if __name__ == "__main__":
    main()

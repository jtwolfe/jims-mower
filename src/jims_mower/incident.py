"""Incident replay viewer: scrub a recorded episode (cameras + hazard + advice)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from PIL import Image

from jims_mower.episode import EpisodeReader
from jims_mower.renderer import render_scalar_map

HAZARD_CMAP = np.array(
    [
        [46, 140, 58],
        [200, 160, 40],
        [180, 90, 40],
        [90, 40, 30],
    ],
    dtype=np.uint8,
)


def _save_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(path)


def _hazard_rgb(hazard: np.ndarray, size: int = 160) -> np.ndarray:
    arr = np.asarray(hazard, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return render_scalar_map(arr, image_size=size, cmap="hazard")


def _nn_square(image: np.ndarray, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    ys = (np.linspace(0, h - 1, size)).astype(int)
    xs = (np.linspace(0, w - 1, size)).astype(int)
    return image[ys[:, None], xs[None, :]]


def _montage(cameras: dict[str, np.ndarray]) -> np.ndarray:
    frames = [np.asarray(v, dtype=np.uint8) for v in cameras.values()]
    if not frames:
        return np.zeros((60, 80, 3), dtype=np.uint8)
    h, w = frames[0].shape[:2]
    cols = min(3, len(frames))
    rows = int(np.ceil(len(frames) / cols))
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        r, c = divmod(i, cols)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = frame
    return canvas


def _banner(width: int, text: str, *, height: int = 22) -> np.ndarray:
    bar = np.full((height, width, 3), 24, dtype=np.uint8)
    # Tiny 5x7 hex-ish marker so the advice is visible without a font rasterizer.
    color = (240, 220, 80)
    if "stop" in text:
        color = (230, 60, 60)
    elif "reroute" in text:
        color = (230, 140, 40)
    elif "slow" in text:
        color = (230, 200, 60)
    bar[:, :8] = color
    return bar


def _stack_view(cameras: dict[str, np.ndarray], hazard: np.ndarray, advice: str) -> np.ndarray:
    mont = _montage(cameras)
    haz = _hazard_rgb(hazard, size=max(80, mont.shape[0]))
    if haz.shape[0] != mont.shape[0]:
        haz = _nn_square(haz, mont.shape[0])
        # restore width after square resize
        if haz.shape[1] != mont.shape[0]:
            pass
    gap = np.zeros((mont.shape[0], 4, 3), dtype=np.uint8)
    # Match hazard height to montage.
    if haz.shape[0] != mont.shape[0]:
        ys = (np.linspace(0, haz.shape[0] - 1, mont.shape[0])).astype(int)
        xs = (np.linspace(0, haz.shape[1] - 1, mont.shape[0])).astype(int)
        haz = haz[ys[:, None], xs[None, :]]
    row = np.concatenate([mont, gap, haz], axis=1)
    banner = _banner(row.shape[1], advice)
    return np.concatenate([banner, row], axis=0)


def write_incident_viewer(
    episode_dir: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    stride: int = 1,
) -> dict[str, Any]:
    """Dump per-step PNGs + an HTML scrubber. Offline — no re-sim."""
    reader = EpisodeReader(episode_dir)
    dest = Path(out_dir)
    frames_dir = dest / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    stride = max(1, int(stride))
    for rec in reader.steps[::stride]:
        step = int(rec.get("step", len(records)))
        obs = rec["obs"]
        info = rec.get("info") or {}
        extra = rec.get("extra") or {}
        advice = str(
            extra.get("policy_advice")
            or info.get("terrain_advice")
            or info.get("living_advice")
            or "ok"
        )
        view = _stack_view(
            obs.get("cameras") or {},
            np.asarray(obs.get("hazard", np.zeros((8, 8), dtype=np.float32))),
            advice,
        )
        name = f"{step:06d}.png"
        _save_rgb(frames_dir / name, view)
        records.append(
            {
                "step": step,
                "file": f"frames/{name}",
                "advice": advice,
                "living_advice": info.get("living_advice"),
                "geofence_advice": info.get("geofence_advice"),
                "coverage_fraction": info.get("coverage_fraction"),
                "nearest_person_m": info.get("nearest_person_m"),
                "tipover": info.get("tipover"),
                "drain_drop": info.get("drain_drop"),
                "action": rec.get("action"),
            }
        )
    index = {
        "source": str(Path(episode_dir)),
        "n_frames": len(records),
        "stride": stride,
        "frames": records,
        "not_a_benchmark": True,
    }
    (dest / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    (dest / "index.html").write_text(_html_scrubber(index), encoding="utf-8")
    return index


def _html_scrubber(index: dict[str, Any]) -> str:
    n = int(index.get("n_frames") or 0)
    last = max(0, n - 1)
    disabled = "disabled" if n == 0 else ""
    payload = json.dumps(index).replace("</", "<\\/")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>jims-mower incident replay</title>
  <style>
    body {{ font-family: sans-serif; background: #111; color: #eee; margin: 16px; }}
    input[type=range] {{ width: min(720px, 100%); }}
    img {{ max-width: 100%; image-rendering: pixelated; background: #000; }}
    pre {{ background: #1c1c1c; padding: 8px; overflow: auto; }}
  </style>
</head>
<body>
  <h1>Incident replay</h1>
  <p>Scrub cameras + hazard + advice. Offline dump — not a benchmark.</p>
  <input id="scrub" type="range" min="0" max="{last}" value="0" {disabled}/>
  <div id="meta">step —</div>
  <img id="view" alt="episode frame"/>
  <pre id="json"></pre>
  <script type="application/json" id="index-data">{payload}</script>
  <script>
    const INDEX = JSON.parse(document.getElementById("index-data").textContent);
    const scrub = document.getElementById("scrub");
    const view = document.getElementById("view");
    const meta = document.getElementById("meta");
    const box = document.getElementById("json");
    function show(i) {{
      const rec = (INDEX.frames || [])[i];
      if (!rec) return;
      view.src = rec.file;
      meta.textContent = "step " + rec.step + "  advice=" + rec.advice;
      box.textContent = JSON.stringify(rec, null, 2);
    }}
    scrub.addEventListener("input", (e) => show(Number(e.target.value)));
    show(0);
  </script>
</body>
</html>
"""

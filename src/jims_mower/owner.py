"""Owner UX stub: phone-sized HTML overlay of yard + geofence + plan."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

from jims_mower.constants import OWNER_OVERLAY_SCHEMA
from jims_mower.env import MowerEnv
from jims_mower.geofence import GeofenceSpec
from jims_mower.planning import TerrainPolicy
from jims_mower.scenarios import load_source


def _poly_svg(points: list[tuple[float, float]], **attrs: Any) -> str:
    if len(points) < 2:
        return ""
    pts = " ".join(f"{x:.3f},{y:.3f}" for x, y in points)
    extra = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f'<polygon points="{pts}" {extra}/>'


def _path_svg(points: list[tuple[float, float]], **attrs: Any) -> str:
    if len(points) < 2:
        return ""
    d = "M " + " L ".join(f"{x:.3f},{y:.3f}" for x, y in points)
    extra = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f'<path d="{d}" {extra}/>'


def overlay_payload(
    *,
    width_m: float,
    height_m: float,
    geofence: Optional[GeofenceSpec],
    waypoints: list[tuple[float, float]],
    pose: dict[str, float],
    coverage_pct: float,
    scenario: str = "",
) -> dict[str, Any]:
    spec = geofence or GeofenceSpec()
    return {
        "schema": OWNER_OVERLAY_SCHEMA,
        "scenario": scenario,
        "width_m": float(width_m),
        "height_m": float(height_m),
        "geofence": spec.as_info(),
        "waypoints": [{"x": x, "y": y} for x, y in waypoints],
        "pose": pose,
        "coverage_pct": float(coverage_pct),
        "not_a_benchmark": True,
        "note": "Phone overlay mock — not a shipping owner app.",
    }


def render_overlay_html(payload: dict[str, Any]) -> str:
    w = float(payload["width_m"])
    h = float(payload["height_m"])
    fence = payload.get("geofence") or {}
    keep_in = [tuple(p) for p in (fence.get("keep_in") or [])]
    keep_out = [[tuple(p) for p in poly] for poly in (fence.get("keep_out") or [])]
    wps = [(float(p["x"]), float(p["y"])) for p in payload.get("waypoints") or []]
    pose = payload.get("pose") or {"x": 1.0, "y": 1.0, "theta": 0.0}
    px, py, th = float(pose.get("x", 0.0)), float(pose.get("y", 0.0)), float(pose.get("theta", 0.0))
    # Tiny heading triangle in world metres.
    import math

    nose = (px + 0.35 * math.cos(th), py + 0.35 * math.sin(th))
    left = (px - 0.18 * math.sin(th), py + 0.18 * math.cos(th))
    right = (px + 0.18 * math.sin(th), py - 0.18 * math.cos(th))
    parts = [
        f'<rect x="0" y="0" width="{w:.3f}" height="{h:.3f}" fill="#2e8c3a"/>',
        _poly_svg(
            keep_in,
            fill="none",
            stroke="#ffcc33",
            stroke_width="0.12",
        ),
    ]
    for poly in keep_out:
        parts.append(
            _poly_svg(poly, fill="rgba(200,40,40,0.35)", stroke="#c82828", stroke_width="0.08")
        )
    parts.append(
        _path_svg(wps, fill="none", stroke="#2ad4e6", stroke_width="0.08")
    )
    parts.append(
        _poly_svg(
            [nose, left, right],
            fill="#222",
            stroke="#fff",
            stroke_width="0.04",
        )
    )
    svg_inner = "\n".join(p for p in parts if p)
    cov = float(payload.get("coverage_pct") or 0.0)
    name = payload.get("scenario") or "yard"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>jims-mower owner overlay</title>
  <style>
    body {{ margin: 0; background: #0b0b0b; color: #eee; font-family: system-ui, sans-serif; }}
    .phone {{
      width: 390px; min-height: 720px; margin: 16px auto; background: #161616;
      border-radius: 28px; border: 8px solid #333; box-sizing: border-box;
      padding: 18px 14px 24px; box-shadow: 0 12px 40px rgba(0,0,0,0.45);
    }}
    h1 {{ font-size: 18px; margin: 0 0 8px; }}
    .sub {{ color: #aaa; font-size: 12px; margin-bottom: 12px; }}
    svg {{ width: 100%; height: auto; background: #1a1a1a; border-radius: 12px; }}
    .stats {{ display: flex; gap: 12px; margin-top: 12px; font-size: 13px; }}
    .chip {{ background: #222; padding: 8px 10px; border-radius: 10px; flex: 1; }}
  </style>
</head>
<body>
  <div class="phone">
    <h1>Yard overlay</h1>
    <div class="sub">{name} · geofence + plan mock · not a shipping app</div>
    <svg viewBox="0 0 {w:.3f} {h:.3f}" xmlns="http://www.w3.org/2000/svg">
      <g transform="translate(0 {h:.3f}) scale(1 -1)">
        {svg_inner}
      </g>
    </svg>
    <div class="stats">
      <div class="chip">Coverage<br><strong>{cov:.1f}%</strong></div>
      <div class="chip">Waypoints<br><strong>{len(wps)}</strong></div>
      <div class="chip">Pose<br><strong>{px:.1f}, {py:.1f}</strong></div>
    </div>
  </div>
</body>
</html>
"""


def write_owner_overlay(
    out_path: Union[str, Path],
    payload: dict[str, Any],
) -> Path:
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(render_overlay_html(payload), encoding="utf-8")
    sidecar = dest.with_suffix(".json")
    sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dest


def export_owner_overlay(
    out_path: Union[str, Path],
    *,
    config: Optional[str] = None,
    seed: int = 7,
    cameras: Optional[int] = 4,
) -> dict[str, Any]:
    """Reset the env, plan once, write the phone HTML mock."""
    cfg, scenario = load_source(config)
    if cameras is not None:
        cfg.sensors.camera_count = cameras
        cfg.sensors.cameras = []
    env = MowerEnv(config=cfg, scenario=scenario, render_mode=None)
    obs, info = env.reset(seed=seed)
    policy = TerrainPolicy(env.cfg)
    policy.reset(obs, info)
    spec = env.geofence_spec()
    pose = info.get("pose") or {"x": env._pose.x, "y": env._pose.y, "theta": env._pose.theta}
    payload = overlay_payload(
        width_m=env.cfg.world.width_m,
        height_m=env.cfg.world.height_m,
        geofence=spec,
        waypoints=policy.waypoints,
        pose=pose,
        coverage_pct=100.0 * float(info.get("coverage_fraction") or 0.0),
        scenario=str(info.get("scenario") or (scenario.name if scenario else "")),
    )
    env.close()
    write_owner_overlay(out_path, payload)
    payload["out"] = str(Path(out_path))
    return payload

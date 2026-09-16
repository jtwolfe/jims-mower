"""World Viewer: static HTML+JS + stdlib HTTP server over a demo/record dir."""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from PIL import Image

from jims_mower.constants import LIVE_SCHEMA, VIEWER_SCHEMA
from jims_mower.mesh import (
    coverage_to_rgb,
    export_mesh,
    mesh_from_elevation,
    mesh_from_env,
    occupancy_to_rgb,
    write_map_npy,
    write_map_png,
)
from jims_mower.renderer import render_scalar_map

STATIC_NAMES = ("index.html", "app.js", "style.css")


def static_dir() -> Path:
    return Path(__file__).resolve().parent / "viewer_static"


def viewer_assets_present(folder: Optional[Path] = None) -> bool:
    root = folder or static_dir()
    return all((root / name).is_file() for name in STATIC_NAMES)


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(path)


def dump_step_frames(
    step_dir: Path,
    obs: dict[str, Any],
    *,
    names: Optional[list[str]] = None,
) -> list[str]:
    """Write camera PNGs for the viewer PiP (demo / teach / live)."""
    step_dir.mkdir(parents=True, exist_ok=True)
    cameras = obs.get("cameras") or {}
    order = names or list(cameras.keys())
    written: list[str] = []
    for name in order:
        frame = cameras.get(name)
        if frame is None:
            continue
        _save_rgb(step_dir / f"cam_{name}.png", frame)
        written.append(name)
    return written


def _pose_dict(raw: Any) -> dict[str, float]:
    if isinstance(raw, dict):
        return {
            "x": float(raw.get("x", 0.0)),
            "y": float(raw.get("y", 0.0)),
            "theta": float(raw.get("theta", 0.0)),
            "z": float(raw.get("z", 0.0)),
            "pitch": float(raw.get("pitch", 0.0)),
            "roll": float(raw.get("roll", 0.0)),
        }
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    keys = ("x", "y", "theta", "z", "pitch", "roll")
    return {k: float(arr[i]) if arr.size > i else 0.0 for i, k in enumerate(keys)}


def _scan_camera_steps(out_dir: Path) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for child in sorted(out_dir.glob("step_*")):
        if not child.is_dir():
            continue
        cams = sorted(p.name[4:-4] for p in child.glob("cam_*.png"))
        if not cams:
            continue
        try:
            step = int(child.name.split("_")[-1])
        except ValueError:
            step = len(found)
        found.append({"step": step, "dir": child.name, "cameras": cams})
    return found


def write_viewer_bundle(
    out_dir: Union[str, Path],
    *,
    env: Any = None,
    poses: Optional[list[Any]] = None,
    plan: Optional[dict[str, Any]] = None,
    profile: Optional[dict[str, Any]] = None,
    cameras: Optional[list[str]] = None,
    policy: str = "",
    stride: int = 2,
    mission: Optional[dict[str, Any]] = None,
    live: bool = False,
) -> dict[str, Any]:
    """Write mesh + rasters + viewer.json so the HTTP UI can open this run."""
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    maps_dir = dest / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    width_m = height_m = 12.0
    resolution_m = 0.20
    mesh = None
    if env is not None:
        width_m = float(env.cfg.world.width_m)
        height_m = float(env.cfg.world.height_m)
        resolution_m = float(env.cfg.world.resolution_m)
        mesh = mesh_from_env(env, stride=stride)
        elev = np.asarray(env._terrain.elevation, dtype=np.float32)
        hazard = env._terrain.hazard_map(env.cfg.robot.steep_slope_rad)
        coverage = env._coverage.as_float()
        occupancy = None
        if hasattr(env, "_occ_persist") and env._occ_persist is not None:
            occupancy = np.asarray(env._occ_persist.grid, dtype=np.float32)
        write_map_npy(maps_dir / "elevation.npy", elev)
        write_map_npy(maps_dir / "hazard.npy", hazard)
        write_map_npy(maps_dir / "coverage.npy", coverage)
        write_map_png(maps_dir / "coverage.png", coverage_to_rgb(coverage))
        write_map_png(
            maps_dir / "hazard.png",
            render_scalar_map(hazard, vmin=0.0, vmax=3.0, cmap="hazard"),
        )
        obs_elev = None
        if getattr(env, "_last_terrain_est", None) is not None:
            obs_elev = np.asarray(env._last_terrain_est.elevation, dtype=np.float32)
            write_map_npy(maps_dir / "elevation_observer.npy", obs_elev)
            err = np.abs(obs_elev - elev)
            write_map_npy(maps_dir / "elevation_error.npy", err)
            write_map_png(
                maps_dir / "elevation_error.png",
                render_scalar_map(err, vmin=0.0, vmax=max(0.25, float(err.max()) or 0.25), cmap="slope"),
            )
        (maps_dir / "elevation.json").write_text(
            json.dumps(
                {
                    "rows": int(elev.shape[0]),
                    "cols": int(elev.shape[1]),
                    "resolution_m": resolution_m,
                    "values": elev.astype(float).reshape(-1).tolist(),
                }
            ),
            encoding="utf-8",
        )
        if occupancy is not None:
            write_map_npy(maps_dir / "occupancy.npy", occupancy)
            write_map_png(maps_dir / "occupancy.png", occupancy_to_rgb(occupancy))
        else:
            write_map_png(
                maps_dir / "occupancy.png",
                occupancy_to_rgb(np.zeros_like(coverage, dtype=np.float32)),
            )
    elif (dest / "maps" / "elevation.npy").is_file():
        elev = np.load(dest / "maps" / "elevation.npy")
        hazard = (
            np.load(dest / "maps" / "hazard.npy")
            if (dest / "maps" / "hazard.npy").is_file()
            else None
        )
        coverage = (
            np.load(dest / "maps" / "coverage.npy")
            if (dest / "maps" / "coverage.npy").is_file()
            else None
        )
        meta_guess = dest / "viewer.json"
        if meta_guess.is_file():
            prev = json.loads(meta_guess.read_text(encoding="utf-8"))
            width_m = float(prev.get("width_m") or width_m)
            height_m = float(prev.get("height_m") or height_m)
            resolution_m = float(prev.get("resolution_m") or resolution_m)
        mesh = mesh_from_elevation(
            elev,
            width_m=width_m,
            height_m=height_m,
            resolution_m=resolution_m,
            hazard=hazard,
            coverage=coverage,
            stride=stride,
        )

    written_mesh: dict[str, str] = {}
    if mesh is not None and not mesh.is_empty():
        written_mesh = export_mesh(mesh, dest / "yard.glb")

    pose_rows = [_pose_dict(p) for p in (poses or []) if p]
    if pose_rows:
        (dest / "poses.json").write_text(
            json.dumps({"schema": VIEWER_SCHEMA, "poses": pose_rows}, indent=2),
            encoding="utf-8",
        )
    if plan is not None:
        (dest / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if profile is not None:
        (dest / "profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    if mission is not None:
        (dest / "mission.json").write_text(json.dumps(mission, indent=2), encoding="utf-8")

    cam_steps = _scan_camera_steps(dest)
    manifest = {
        "schema": VIEWER_SCHEMA,
        "policy": policy,
        "width_m": width_m,
        "height_m": height_m,
        "resolution_m": resolution_m,
        "mesh": "yard.glb" if written_mesh else None,
        "mesh_json": "yard.json" if written_mesh else None,
        "maps": {
            "coverage": "maps/coverage.png" if (maps_dir / "coverage.png").is_file() else None,
            "hazard": "maps/hazard.png" if (maps_dir / "hazard.png").is_file() else None,
            "occupancy": "maps/occupancy.png" if (maps_dir / "occupancy.png").is_file() else None,
            "elevation": "maps/elevation.json" if (maps_dir / "elevation.json").is_file() else None,
            "elevation_error": "maps/elevation_error.png"
            if (maps_dir / "elevation_error.png").is_file()
            else None,
            "observed": "maps/observed.png" if (maps_dir / "observed.png").is_file() else None,
            "fog": "maps/fog.png" if (maps_dir / "fog.png").is_file() else None,
            "observed_mesh": "maps/observed_mesh.json"
            if (maps_dir / "observed_mesh.json").is_file()
            else None,
        },
        "live": bool(live),
        "owner_mode": "observed_fog" if live else "",
        "fog": bool(live) or (maps_dir / "fog.png").is_file(),
        "relief_scale": 4.0,
        "poses": "poses.json" if pose_rows else None,
        "plan": "plan.json" if (dest / "plan.json").is_file() else None,
        "profile": "profile.json" if (dest / "profile.json").is_file() else None,
        "mission": "mission.json" if (dest / "mission.json").is_file() else None,
        "observed": "maps/observed.png" if (maps_dir / "observed.png").is_file() else None,
        "cameras": cameras or (cam_steps[0]["cameras"] if cam_steps else []),
        "camera_steps": cam_steps,
        "n_poses": len(pose_rows),
        "vertex_count": int(mesh.vertex_count) if mesh is not None else 0,
        "triangle_count": int(mesh.triangle_count) if mesh is not None else 0,
        "health": {"placeholder": True, "label": "health", "value": None},
        "radio": {"placeholder": True, "label": "radio", "value": None},
        "not_a_benchmark": True,
        "note": (
            "Live owner session — fog-of-war over unknown cells, no mAP/FPS."
            if live
            else "World Viewer bundle — mesh + cameras from sim, no mAP/FPS."
        ),
    }
    (dest / "viewer.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _extract_episode_cameras(episode_dir: Path, reader: Any, *, every: Optional[int] = None) -> None:
    from jims_mower.demo import camera_dump_indices

    if reader.reset_obs and reader.reset_obs.get("cameras"):
        dump_step_frames(episode_dir / "step_000", reader.reset_obs)
    if not reader.steps:
        return
    n = len(reader.steps)
    picks = camera_dump_indices(n, every)
    for i, rec in enumerate(reader.steps):
        if i in picks:
            obs = rec.get("obs") or {}
            if obs.get("cameras"):
                dump_step_frames(episode_dir / f"step_{i:03d}", obs)


def prepare_viewer_dir(source: Union[str, Path], *, stride: int = 2) -> dict[str, Any]:
    """Make a demo or record directory openable by the World Viewer."""
    root = Path(source)
    if not root.is_dir():
        raise FileNotFoundError(f"episode/demo directory not found: {root}")

    poses: list[Any] = []
    plan = None
    profile = None
    mission = None
    env = None
    policy = ""

    if (root / "poses.json").is_file():
        raw = json.loads((root / "poses.json").read_text(encoding="utf-8"))
        poses = raw.get("poses") or raw if isinstance(raw, list) else raw.get("poses") or []
    if (root / "plan.json").is_file():
        plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    if (root / "profile.json").is_file():
        profile = json.loads((root / "profile.json").read_text(encoding="utf-8"))
    if (root / "mission.json").is_file():
        mission = json.loads((root / "mission.json").read_text(encoding="utf-8"))
    if (root / "summary.json").is_file():
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        policy = str(summary.get("policy") or "")
        if not poses:
            poses = [row.get("pose") for row in (summary.get("log") or []) if row.get("pose")]
            if summary.get("final_pose"):
                poses.append(summary["final_pose"])

    if (root / "manifest.json").is_file() and (root / "steps.jsonl").is_file():
        from jims_mower.episode import EpisodeReader

        reader = EpisodeReader(root)
        policy = policy or str(reader.manifest.get("policy") or "")
        if not poses:
            if reader.reset_info.get("pose"):
                poses.append(reader.reset_info["pose"])
            for rec in reader.steps:
                pose = (rec.get("info") or {}).get("pose")
                if pose:
                    poses.append(pose)
        cfg = reader.config()
        if not (root / "maps" / "elevation.npy").is_file() and reader.reset_obs:
            maps_dir = root / "maps"
            maps_dir.mkdir(parents=True, exist_ok=True)
            obs = reader.reset_obs
            if "elevation" in obs:
                write_map_npy(maps_dir / "elevation.npy", obs["elevation"])
            if "hazard" in obs:
                write_map_npy(maps_dir / "hazard.npy", obs["hazard"])
                write_map_png(
                    maps_dir / "hazard.png",
                    render_scalar_map(obs["hazard"], vmin=0.0, vmax=3.0, cmap="hazard"),
                )
            if "coverage" in obs:
                write_map_npy(maps_dir / "coverage.npy", obs["coverage"])
                write_map_png(maps_dir / "coverage.png", coverage_to_rgb(obs["coverage"]))
            if "occupancy" in obs:
                write_map_npy(maps_dir / "occupancy.npy", obs["occupancy"])
                write_map_png(maps_dir / "occupancy.png", occupancy_to_rgb(obs["occupancy"]))
            # stash world size for mesh_from_elevation
            (root / "viewer.json").write_text(
                json.dumps(
                    {
                        "schema": VIEWER_SCHEMA,
                        "width_m": cfg.world.width_m,
                        "height_m": cfg.world.height_m,
                        "resolution_m": cfg.world.resolution_m,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        _extract_episode_cameras(root, reader)

    manifest = write_viewer_bundle(
        root,
        env=env,
        poses=poses,
        plan=plan,
        profile=profile,
        policy=policy,
        stride=stride,
        mission=mission,
    )
    return manifest


class ViewerHandler(SimpleHTTPRequestHandler):
    """Serve package static files plus an episode/demo data directory."""

    data_dir: Path
    static_root: Path
    session: Any = None

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self.send_error(404, f"missing {path.name}")
            return
        data = path.read_bytes()
        self._send_bytes(data, content_type)

    def _safe_under(self, root: Path, rel: str) -> Optional[Path]:
        if not rel or rel.startswith("/"):
            return None
        candidate = (root / rel).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            return None
        return candidate

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path in {"", "/"}:
            self._send_file(self.static_root / "index.html", "text/html; charset=utf-8")
            return
        if path == "/api/manifest":
            if self.session is not None and hasattr(self.session, "manifest"):
                body = json.dumps(self.session.manifest()).encode("utf-8")
                self._send_bytes(body, "application/json")
                return
            manifest = self.data_dir / "viewer.json"
            self._send_file(manifest, "application/json")
            return
        if path in {"/api/live", "/api/live/"}:
            self._sse_live()
            return
        if path == "/api/live/snapshot":
            self._send_live_snapshot()
            return
        if path == "/api/live/observed.png":
            self._send_live_bytes(self._live_observed(), "image/png")
            return
        if path == "/api/live/fog.png":
            self._send_live_bytes(self._live_fog(), "image/png")
            return
        if path == "/api/live/observed_mesh.json":
            self._send_live_bytes(self._live_observed_mesh(), "application/json")
            return
        if path.startswith("/api/live/cam/"):
            name = path[len("/api/live/cam/") :].split("?")[0]
            self._send_live_bytes(self._live_camera(name), "image/jpeg")
            return
        if path == "/api/profile":
            dest = self.data_dir / "profile.json"
            if not dest.is_file():
                self._send_bytes(b"{}", "application/json")
                return
            self._send_file(dest, "application/json")
            return
        if path.startswith("/data/"):
            rel = path[len("/data/") :]
            target = self._safe_under(self.data_dir, rel)
            if target is None:
                self.send_error(400, "bad path")
                return
            ctype = _guess_type(target)
            self._send_file(target, ctype)
            return
        rel = path.lstrip("/")
        target = self._safe_under(self.static_root, rel)
        if target is not None and target.is_file():
            self._send_file(target, _guess_type(target))
            return
        self.send_error(404, "not found")

    def _send_live_bytes(self, payload: bytes, content_type: str) -> None:
        if not payload:
            self.send_error(404, "live asset not ready")
            return
        self._send_bytes(payload, content_type)

    def _live_observed(self) -> bytes:
        session = self.session
        if session is None:
            path = self.data_dir / "maps" / "observed.png"
            return path.read_bytes() if path.is_file() else b""
        return session.observed_png_bytes()

    def _live_fog(self) -> bytes:
        session = self.session
        if session is None:
            path = self.data_dir / "maps" / "fog.png"
            return path.read_bytes() if path.is_file() else b""
        return session.fog_png_bytes()

    def _live_observed_mesh(self) -> bytes:
        session = self.session
        if session is None:
            path = self.data_dir / "maps" / "observed_mesh.json"
            return path.read_bytes() if path.is_file() else b""
        if hasattr(session, "observed_mesh_bytes"):
            return session.observed_mesh_bytes()
        return b""

    def _live_camera(self, name: str) -> bytes:
        session = self.session
        if session is None:
            return b""
        return session.camera_bytes(name)

    def _send_live_snapshot(self) -> None:
        session = self.session
        if session is None:
            self.send_error(404, "no live session")
            return
        body = json.dumps(session.snapshot()).encode("utf-8")
        self._send_bytes(body, "application/json")

    def _sse_live(self) -> None:
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        try:
            limit = int((qs.get("n") or ["0"])[0])
        except ValueError:
            limit = 0
        session = self.session
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close" if limit > 0 else "keep-alive")
        self.end_headers()
        import time

        n = 0
        try:
            while limit <= 0 or n < limit:
                if session is None:
                    payload = json.dumps({"schema": LIVE_SCHEMA, "live": False})
                else:
                    payload = json.dumps(session.snapshot())
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
                n += 1
                time.sleep(0.04 if limit > 0 else 0.12)
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/profile":
            self.send_error(404, "not found")
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(400, "invalid json")
            return
        dest = self.data_dir / "profile.json"
        dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        body = json.dumps({"ok": True, "path": "profile.json"}).encode("utf-8")
        self._send_bytes(body, "application/json")


def _guess_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".html": "text/html; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".json": "application/json",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".glb": "model/gltf-binary",
        ".gltf": "model/gltf+json",
        ".obj": "text/plain; charset=utf-8",
        ".npy": "application/octet-stream",
    }.get(suffix, "application/octet-stream")


def serve_viewer(
    data_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    session: Any = None,
) -> ThreadingHTTPServer:
    handler = partial(ViewerHandler)
    ViewerHandler.data_dir = data_dir.resolve()
    ViewerHandler.static_root = static_dir().resolve()
    ViewerHandler.session = session
    server = ThreadingHTTPServer((host, port), handler)
    return server


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Jim's Mower World Viewer (low-poly yard + cameras)")
    p.add_argument(
        "--episode",
        type=Path,
        default=None,
        help="demo_out / record dir / teach_out (writes viewer.json if missing)",
    )
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--prepare-only",
        action="store_true",
        help="write mesh/viewer.json and exit (CI / headless)",
    )
    p.add_argument("--stride", type=int, default=2)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if not viewer_assets_present():
        raise SystemExit(f"viewer static assets missing under {static_dir()}")
    if args.episode is None:
        data = Path("demo_out")
        if not data.is_dir():
            # Headless default: export a tiny suburban mesh so the UI still opens.
            from jims_mower.env import MowerEnv
            from jims_mower.scenarios import load_source

            data = Path(".jims_mower_viewer")
            data.mkdir(parents=True, exist_ok=True)
            cfg, scenario = load_source(None)
            cfg.sensors.camera_count = 4
            cfg.sensors.cameras = []
            env = MowerEnv(config=cfg, scenario=scenario, render_mode=None)
            obs, info = env.reset(seed=7)
            dump_step_frames(data / "step_000", obs)
            write_viewer_bundle(
                data,
                env=env,
                poses=[info.get("pose")],
                policy="idle",
                stride=args.stride,
            )
            env.close()
        else:
            prepare_viewer_dir(data, stride=args.stride)
    else:
        data = Path(args.episode)
        prepare_viewer_dir(data, stride=args.stride)
    if args.prepare_only:
        print(f"viewer ready → {data / 'viewer.json'}")
        return
    server = serve_viewer(data, host=args.host, port=args.port)
    url = f"http://{args.host}:{args.port}/"
    print(f"World Viewer {data} → {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nviewer stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

"""Live mission session: wall-clock MissionPolicy + SSE owner view.

This is the owner loop that ``jims-mower-mission-demo`` + the World Viewer
scrubber were missing. The sim still uses the true height field for
physics. Control stays on ``ObservedMap``. The default owner view is a
fog veil over unknown cells — not a finished god-view mesh from step 0.

No claimed mAP / FPS. Not a coverage benchmark.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from PIL import Image

from jims_mower.constants import LIVE_SCHEMA, VIEWER_SCHEMA
from jims_mower.env import MowerEnv
from jims_mower.mission_demo import (
    FAST_CAM_HEIGHT,
    FAST_CAM_WIDTH,
    FAST_STEPS,
    resolve_mission_config,
)
from jims_mower.mesh import mesh_from_observed, mesh_to_payload
from jims_mower.mission_flow import (
    PHASE_LABELS,
    MissionPolicy,
    mission_timeline,
    scale_mission_budget,
)
from jims_mower.planning.observed import fog_rgba
from jims_mower.profile import write_yard_profile
from jims_mower.scenarios import load_source
from jims_mower.teach import keepouts_from_env
from jims_mower.viewer import (
    dump_step_frames,
    serve_viewer,
    static_dir,
    viewer_assets_present,
    write_viewer_bundle,
)

DEFAULT_CONFIG = "acre_yard"
DEFAULT_STEPS = 8000
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAP_MAX_SIDE = 96
PLAN_MAX_POINTS = 160
CAM_JPEG_QUALITY = 62
MAP_STRIDE_DEFAULT = 4
MESH_STRIDE_DEFAULT = 8
MESH_MAX_SIDE = 48
CAM_WALL_S = 0.40
POSE_FLUSH_STRIDE = 20
ACRE_LIVE_CAM_WIDTH = 48
ACRE_LIVE_CAM_HEIGHT = 36

SPEED_ALIASES = {
    "1": 1.0,
    "1x": 1.0,
    "2": 2.0,
    "2x": 2.0,
    "5": 5.0,
    "5x": 5.0,
    "max": 0.0,
    "inf": 0.0,
}


def parse_speed(raw: Any) -> float:
    """Return a realtime multiplier. ``0`` means unpaced (CI / ``--speed max``)."""
    if raw is None:
        return 1.0
    if isinstance(raw, (int, float)):
        val = float(raw)
        return 0.0 if val <= 0.0 else val
    key = str(raw).strip().lower()
    if key in SPEED_ALIASES:
        return SPEED_ALIASES[key]
    val = float(key)
    return 0.0 if val <= 0.0 else val


def resolve_live_config(config: Optional[Any], *, fast: bool) -> Any:
    """``--fast`` uses ``mission_tiny``. Default live yard is ``acre_yard``."""
    if fast:
        if config is None or config in {DEFAULT_CONFIG, "golf_rough"}:
            return resolve_mission_config(None, fast=True)
        return resolve_mission_config(config, fast=fast)
    return config if config is not None else DEFAULT_CONFIG


def coarsen2d(arr: np.ndarray, max_side: int = MAP_MAX_SIDE) -> np.ndarray:
    """Stride-downsample a raster so acre maps stay browser-cheap."""
    grid = np.asarray(arr)
    if grid.ndim < 2:
        return grid
    rows, cols = int(grid.shape[0]), int(grid.shape[1])
    stride = max(1, int(math.ceil(max(rows, cols) / max(1, int(max_side)))))
    if stride <= 1:
        return grid
    return grid[::stride, ::stride]


def _png_bytes(image: np.ndarray) -> bytes:
    buf = io.BytesIO()
    arr = np.asarray(image)
    mode = "RGBA" if arr.ndim == 3 and arr.shape[2] == 4 else "RGB"
    Image.fromarray(arr.astype(np.uint8), mode=mode).save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(image: np.ndarray, *, quality: int = CAM_JPEG_QUALITY) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(
        buf, format="JPEG", quality=int(quality)
    )
    return buf.getvalue()


def _compact_mesh_payload(mesh: Any) -> dict[str, Any]:
    payload = mesh_to_payload(mesh)
    for key in ("positions", "normals", "colors", "uvs"):
        raw = payload.get(key) or []
        payload[key] = [round(float(v), 4) for v in raw]
    payload["kind"] = "observed"
    payload["honesty"] = "physics uses true height; this mesh is ObservedMap only"
    return payload


def _downsample_xy(points: list[tuple[float, float]], limit: int = PLAN_MAX_POINTS) -> list[dict[str, float]]:
    if not points:
        return []
    if len(points) <= int(limit):
        return [{"x": float(x), "y": float(y)} for x, y in points]
    step = max(1, int(math.ceil(len(points) / float(limit))))
    picked = points[::step]
    if picked[-1] != points[-1]:
        picked.append(points[-1])
    return [{"x": float(x), "y": float(y)} for x, y in picked]


def _pose_row(raw: Any) -> dict[str, float]:
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


class LiveSession:
    """One running ``MissionPolicy`` plus the latest owner-view assets."""

    def __init__(
        self,
        *,
        config: Optional[Any] = None,
        fast: bool = False,
        speed: Any = 1.0,
        steps: Optional[int] = None,
        seed: int = 7,
        cameras: int = 4,
        out_dir: Union[str, Path] = "live_out",
        cam_stride: int = 20,
        map_stride: int = MAP_STRIDE_DEFAULT,
        mesh_stride: Optional[int] = None,
        observed_mesh_stride: int = MESH_STRIDE_DEFAULT,
        phase_budget: float = 1.0,
        calibrate_stride: Optional[float] = None,
        calibrate_confirm: Optional[float] = None,
    ) -> None:
        self.config_name = resolve_live_config(config, fast=fast)
        self.fast = bool(fast)
        self.speed = parse_speed(speed)
        self.seed = int(seed)
        self.cameras = int(cameras)
        self.out_dir = Path(out_dir)
        self.cam_stride = max(1, int(cam_stride))
        self.map_stride = max(1, int(map_stride))
        self.mesh_stride = mesh_stride
        self.observed_mesh_stride = max(1, int(observed_mesh_stride))
        self.phase_budget = min(1.0, max(0.05, float(phase_budget)))
        self.calibrate_stride = calibrate_stride
        self.calibrate_confirm = calibrate_confirm
        self.max_steps = int(steps if steps is not None else (FAST_STEPS if fast else DEFAULT_STEPS))
        self.env: Optional[MowerEnv] = None
        self.policy: Optional[MissionPolicy] = None
        self.obs: dict[str, Any] = {}
        self.info: dict[str, Any] = {}
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._seq = 0
        self._map_seq = 0
        self._mesh_seq = 0
        self._cam_seq = 0
        self._last_cam_wall = 0.0
        self._n_observed = 0
        self._last_phase = ""
        self.poses: list[dict[str, Any]] = []
        self.camera_names: list[str] = []
        self.observed_png = b""
        self.fog_png = b""
        self.observed_mesh_json = b""
        self.cam_jpeg: dict[str, bytes] = {}
        self.done = False
        self.started = False

    @property
    def dt(self) -> float:
        if self.env is None:
            return 0.10
        return float(self.env.cfg.dt)

    def reset(self) -> dict[str, Any]:
        cfg, scenario = load_source(self.config_name)
        cfg.sensors.camera_count = self.cameras
        cfg.sensors.cameras = []
        if self.fast:
            cfg.sensors.width = min(int(cfg.sensors.width), FAST_CAM_WIDTH)
            cfg.sensors.height = min(int(cfg.sensors.height), FAST_CAM_HEIGHT)
        cfg.max_steps = max(int(cfg.max_steps), self.max_steps + 2)
        if not self.fast:
            cells = (float(cfg.world.width_m) / max(float(cfg.world.resolution_m), 1e-6)) * (
                float(cfg.world.height_m) / max(float(cfg.world.resolution_m), 1e-6)
            )
            if cells >= 8000:
                cfg.sensors.width = min(int(cfg.sensors.width), ACRE_LIVE_CAM_WIDTH)
                cfg.sensors.height = min(int(cfg.sensors.height), ACRE_LIVE_CAM_HEIGHT)
        if self.calibrate_stride is not None:
            cfg.mission.calibrate_stride_m = float(self.calibrate_stride)
        if self.calibrate_confirm is not None:
            cfg.mission.calibrate_confirm_m = float(self.calibrate_confirm)
        scale_mission_budget(cfg.mission, self.phase_budget)
        if self.env is not None:
            self.env.close()
        self.env = MowerEnv(config=cfg, scenario=scenario, render_mode="rgb_array")
        self.obs, self.info = self.env.reset(seed=self.seed)
        self.policy = MissionPolicy(self.env.cfg, fast=self.fast)
        self.policy.reset(self.obs, self.info)
        self.camera_names = list(self.obs.get("cameras") or {})
        self.poses = []
        self.done = False
        self.started = True
        self._seq = 0
        self._map_seq = 0
        self._mesh_seq = 0
        self._cam_seq = 0
        self._last_cam_wall = 0.0
        self._n_observed = 0
        self._last_phase = self.policy.phase.value
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._refresh_maps(force=True)
        self._refresh_cameras(force=True)
        self._record_pose()
        self._write_bundle()
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return self._frame_unlocked()

    def manifest(self) -> dict[str, Any]:
        path = self.out_dir / "viewer.json"
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                data["live"] = True
                data["owner_mode"] = "observed_terrain"
                data["fog"] = True
                data["observed_mesh"] = True
                return data
            except json.JSONDecodeError:
                pass
        width = height = 12.0
        res = 0.20
        if self.env is not None:
            width = float(self.env.cfg.world.width_m)
            height = float(self.env.cfg.world.height_m)
            res = float(self.env.cfg.world.resolution_m)
        return {
            "schema": VIEWER_SCHEMA,
            "live": True,
            "owner_mode": "observed_terrain",
            "fog": True,
            "observed_mesh": True,
            "policy": "mission",
            "width_m": width,
            "height_m": height,
            "resolution_m": res,
            "cameras": list(self.camera_names),
            "relief_scale": 4.0,
            "not_a_benchmark": True,
            "note": (
                "Live owner session — observed elevation grows with the map. "
                "Unknown stays fog. True elev is a debug toggle. No mAP/FPS."
            ),
        }

    def observed_png_bytes(self) -> bytes:
        with self.lock:
            return self.observed_png

    def fog_png_bytes(self) -> bytes:
        with self.lock:
            return self.fog_png

    def observed_mesh_bytes(self) -> bytes:
        with self.lock:
            return self.observed_mesh_json

    def camera_bytes(self, name: str) -> bytes:
        with self.lock:
            return self.cam_jpeg.get(name, b"")

    def step_once(self) -> dict[str, Any]:
        if self.env is None or self.policy is None:
            raise RuntimeError("LiveSession.reset() before stepping")
        if self.done:
            return self.snapshot()
        action = self.policy.act(self.obs, self.info)
        self.obs, _reward, terminated, truncated, self.info = self.env.step(action)
        if terminated or truncated or self.policy.done:
            self.done = True
        force_map = self.policy.phase.value != self._last_phase
        self._last_phase = self.policy.phase.value
        self._refresh_maps(force=force_map)
        self._refresh_cameras(force=False)
        self._record_pose()
        with self.lock:
            self._seq += 1
        if self.policy.step % POSE_FLUSH_STRIDE == 0 or self.done:
            self._flush_incremental()
        return self.snapshot()

    def run_n(self, n: int) -> dict[str, Any]:
        last = self.snapshot() if self.started else self.reset()
        for _ in range(max(0, int(n))):
            if self.done or self._stop.is_set():
                break
            last = self.step_once()
        return last

    def run_blocking(self) -> dict[str, Any]:
        if not self.started:
            self.reset()
        t0 = time.perf_counter()
        step_i = 0
        last = self.snapshot()
        while not self._stop.is_set() and not self.done and step_i < self.max_steps:
            last = self.step_once()
            step_i += 1
            if self.speed > 0.0:
                target = t0 + step_i * (self.dt / self.speed)
                now = time.perf_counter()
                delay = target - now
                if delay > 0.0:
                    time.sleep(delay)
        self._flush_incremental(final=True)
        return last

    def start_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if not self.started:
            self.reset()
        self._stop.clear()
        self._thread = threading.Thread(target=self.run_blocking, name="jims-mower-live", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._flush_incremental(final=True)

    def close(self) -> None:
        self.stop()
        if self.env is not None:
            self.env.close()
            self.env = None

    def _record_pose(self) -> None:
        assert self.policy is not None
        pose = _pose_row(self.info.get("pose") if self.info else {})
        status = self.policy.status(self.info)
        pose["phase"] = status["phase"]
        pose["phase_label"] = status["phase_label"]
        pose["map_completion"] = status["map_completion"]
        pose["actual_coverage_fraction"] = status["actual_coverage_fraction"]
        pose["step"] = status["step"]
        with self.lock:
            self.poses.append(pose)

    def _refresh_maps(self, *, force: bool) -> None:
        assert self.policy is not None
        omap = self.policy.observed
        if omap is None:
            return
        n_obs = int(omap.observed.sum())
        step = int(self.policy.step)
        changed = n_obs != self._n_observed
        if not force and not changed and step % self.map_stride != 0:
            return
        self._n_observed = n_obs
        frontier_cells = [
            omap.world_to_cell(x, y)
            for x, y in self.policy._frontier_xy
            if omap.world_to_cell(x, y) is not None
        ]
        rgb = coarsen2d(omap.as_rgb(frontiers=frontier_cells), MAP_MAX_SIDE)
        fog = coarsen2d(fog_rgba(omap.observed), MAP_MAX_SIDE)
        observed_png = _png_bytes(rgb)
        fog_png = _png_bytes(fog)
        mesh_json = b""
        if force or step % self.observed_mesh_stride == 0 or changed:
            mesh = mesh_from_observed(omap, stride=1, max_side=MESH_MAX_SIDE)
            mesh_json = json.dumps(_compact_mesh_payload(mesh), separators=(",", ":")).encode("utf-8")
        with self.lock:
            self.observed_png = observed_png
            self.fog_png = fog_png
            self._map_seq += 1
            if mesh_json:
                self.observed_mesh_json = mesh_json
                self._mesh_seq += 1

    def _refresh_cameras(self, *, force: bool) -> None:
        now = time.perf_counter()
        if not force and (now - self._last_cam_wall) < CAM_WALL_S:
            return
        cameras = (self.obs or {}).get("cameras") or {}
        if not cameras:
            return
        encoded = {name: _jpeg_bytes(frame) for name, frame in cameras.items() if frame is not None}
        if not encoded:
            return
        self._last_cam_wall = now
        with self.lock:
            self.cam_jpeg = encoded
            self._cam_seq += 1

    def _frame_unlocked(self) -> dict[str, Any]:
        policy = self.policy
        if policy is None:
            return {
                "schema": LIVE_SCHEMA,
                "live": True,
                "seq": 0,
                "step": 0,
                "phase": "calibrate_boundary",
                "done": False,
                "not_a_benchmark": True,
            }
        status = policy.status(self.info)
        pose = dict(self.poses[-1]) if self.poses else _pose_row((self.info or {}).get("pose"))
        frontiers = [{"x": float(x), "y": float(y)} for x, y in policy._frontier_xy]
        explore = _downsample_xy(
            list(policy.explore_plan.waypoints) if policy.explore_plan else []
        )
        plan: list[dict[str, float]] = []
        if policy.phase.value in {"review", "mow", "return_home", "complete"}:
            wps = policy.global_plan.waypoints if policy.global_plan is not None else policy.waypoints
            plan = _downsample_xy(list(wps))
        keep_in: list[list[float]] = []
        keep_out: list[list[list[float]]] = []
        if policy.profile is not None:
            keep_in = [list(pt) for pt in policy.profile.keep_in]
            keep_out = [[list(pt) for pt in hole] for hole in (policy.profile.keep_out or [])]
        cam_name = self.camera_names[0] if self.camera_names else "front"
        n_obs = self._n_observed
        if policy.observed is not None:
            n_obs = int(policy.observed.observed.sum())
        return {
            "schema": LIVE_SCHEMA,
            "live": True,
            "seq": self._seq,
            "map_seq": self._map_seq,
            "cam_seq": self._cam_seq,
            "step": status["step"],
            "phase": status["phase"],
            "phase_label": status["phase_label"],
            "pose": pose,
            "map_pct": float(status["map_completion"]),
            "cut_pct": float(status["actual_coverage_fraction"]),
            "coverage_pct": float(status["actual_coverage_fraction"]),
            "planned_pct": float(status.get("planned_coverage_fraction") or 0.0),
            "reachable": int(status.get("reachable_mowable_cells") or 0),
            "unreachable": int(status.get("unreachable_mowable_cells") or 0),
            "n_frontiers": int(status["n_frontiers"]),
            "n_observed": n_obs,
            "mesh_seq": self._mesh_seq,
            "n_waypoints": int(status["n_waypoints"]),
            "frontiers": frontiers,
            "explore": explore,
            "plan": plan,
            "keep_in": keep_in,
            "keep_out": keep_out,
            "trimmer_on": bool((self.info or {}).get("trimmer_enabled")),
            "trimmer_allowed": bool(status["trimmer_allowed"]),
            "observed_url": f"/api/live/observed.png?v={self._map_seq}",
            "fog_url": f"/api/live/fog.png?v={self._map_seq}",
            "observed_mesh_url": f"/api/live/observed_mesh.json?v={self._mesh_seq}",
            "cam_url": f"/api/live/cam/{cam_name}?v={self._cam_seq}",
            "cameras": list(self.camera_names),
            "done": bool(self.done),
            "owner_mode": "observed_terrain",
            "speed": self.speed,
            "dt": self.dt,
            "not_a_benchmark": True,
        }

    def _mesh_stride_for_world(self) -> int:
        if self.mesh_stride is not None:
            return max(1, int(self.mesh_stride))
        if self.env is None:
            return 2
        cells = (self.env.cfg.world.width_m / max(self.env.cfg.world.resolution_m, 1e-6)) * (
            self.env.cfg.world.height_m / max(self.env.cfg.world.resolution_m, 1e-6)
        )
        if cells >= 8000:
            return 4
        if cells >= 2500:
            return 3
        return 2

    def _write_bundle(self) -> None:
        assert self.env is not None and self.policy is not None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        maps_dir = self.out_dir / "maps"
        maps_dir.mkdir(parents=True, exist_ok=True)
        if self.observed_png:
            (maps_dir / "observed.png").write_bytes(self.observed_png)
        if self.fog_png:
            (maps_dir / "fog.png").write_bytes(self.fog_png)
        if self.observed_mesh_json:
            (maps_dir / "observed_mesh.json").write_bytes(self.observed_mesh_json)
        timeline = mission_timeline(
            self.policy,
            actual_coverage=float((self.info or {}).get("coverage_fraction") or 0.0),
        )
        write_viewer_bundle(
            self.out_dir,
            env=self.env,
            poses=self.poses,
            plan={"policy": "mission", "waypoints": [], "explore": []},
            profile=self.policy.profile.as_dict() if self.policy.profile is not None else None,
            cameras=self.camera_names,
            policy="mission",
            stride=self._mesh_stride_for_world(),
            mission=timeline,
            live=True,
        )
        if self.obs.get("cameras"):
            dump_step_frames(self.out_dir / "step_000", self.obs, names=self.camera_names)

    def _flush_incremental(self, *, final: bool = False) -> None:
        if self.policy is None:
            return
        dest = self.out_dir
        dest.mkdir(parents=True, exist_ok=True)
        maps_dir = dest / "maps"
        maps_dir.mkdir(parents=True, exist_ok=True)
        with self.lock:
            poses = list(self.poses)
            observed_png = self.observed_png
            fog_png = self.fog_png
            observed_mesh_json = self.observed_mesh_json
        (dest / "poses.json").write_text(
            json.dumps({"schema": VIEWER_SCHEMA, "poses": poses}, indent=2),
            encoding="utf-8",
        )
        if observed_png:
            (maps_dir / "observed.png").write_bytes(observed_png)
        if fog_png:
            (maps_dir / "fog.png").write_bytes(fog_png)
        if observed_mesh_json:
            (maps_dir / "observed_mesh.json").write_bytes(observed_mesh_json)
        timeline = mission_timeline(
            self.policy,
            actual_coverage=float((self.info or {}).get("coverage_fraction") or 0.0),
        )
        (dest / "mission.json").write_text(json.dumps(timeline, indent=2), encoding="utf-8")
        if self.policy.profile is not None:
            if not self.policy.profile.keep_out and self.env is not None:
                self.policy.profile.keep_out = keepouts_from_env(self.env)
            write_yard_profile(dest / "profile.json", self.policy.profile)
        if final or (self.policy.step % self.cam_stride == 0 and self.obs.get("cameras")):
            dump_step_frames(
                dest / f"step_{self.policy.step:03d}",
                self.obs,
                names=self.camera_names,
            )
        if (dest / "viewer.json").is_file():
            try:
                manifest = json.loads((dest / "viewer.json").read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                manifest = {}
            manifest["live"] = True
            manifest["owner_mode"] = "observed_terrain"
            manifest["fog"] = True
            manifest["observed_mesh"] = True
            manifest["n_poses"] = len(poses)
            manifest["maps"] = manifest.get("maps") or {}
            manifest["maps"]["observed"] = "maps/observed.png"
            manifest["maps"]["fog"] = "maps/fog.png"
            if observed_mesh_json:
                manifest["maps"]["observed_mesh"] = "maps/observed_mesh.json"
            (dest / "viewer.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_live(
    *,
    config: Optional[Any] = None,
    fast: bool = False,
    speed: Any = 1.0,
    steps: Optional[int] = None,
    seed: int = 7,
    cameras: int = 4,
    out_dir: Union[str, Path] = "live_out",
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    prepare_only: bool = False,
    cam_stride: int = 20,
    map_stride: int = MAP_STRIDE_DEFAULT,
    mesh_stride: Optional[int] = None,
    observed_mesh_stride: int = MESH_STRIDE_DEFAULT,
    phase_budget: float = 1.0,
    calibrate_stride: Optional[float] = None,
    calibrate_confirm: Optional[float] = None,
) -> dict[str, Any]:
    """Start a live mission. ``prepare_only`` runs unattended (CI) then exits."""
    session = LiveSession(
        config=config,
        fast=fast,
        speed=0.0 if prepare_only else speed,
        steps=steps,
        seed=seed,
        cameras=cameras,
        out_dir=out_dir,
        cam_stride=cam_stride,
        map_stride=map_stride,
        mesh_stride=mesh_stride,
        observed_mesh_stride=observed_mesh_stride,
        phase_budget=phase_budget,
        calibrate_stride=calibrate_stride,
        calibrate_confirm=calibrate_confirm,
    )
    summary = session.reset()
    if prepare_only:
        session.speed = 0.0
        last = session.run_blocking()
        session.close()
        last["out"] = str(Path(out_dir))
        return last
    if not viewer_assets_present():
        session.close()
        raise SystemExit(f"viewer static assets missing under {static_dir()}")
    session.start_thread()
    server = serve_viewer(session.out_dir, host=host, port=port, session=session)
    url = f"http://{host}:{port}/"
    print(f"Live mission {session.config_name} @ {session.speed or 'max'}× → {url}")
    print("Owner view: growing observed terrain + fog. Toggle true elev for god-view.")
    print("Physics uses true height; owner/control use ObservedMap.")
    print(f"phase {summary.get('phase')}  map {float(summary.get('map_pct') or 0.0):.1%}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nlive session stopped")
    finally:
        server.server_close()
        session.close()
    return session.snapshot()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Jim's Mower live mission: wall-clock calibrate → explore → mow + observed terrain"
    )
    p.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="scenario (default acre_yard; live demo: acre_yard_demo; --fast → mission_tiny)",
    )
    p.add_argument("--out", type=Path, default=Path("live_out"))
    p.add_argument("--steps", type=int, default=None, help="episode budget (default 8000, or 420 with --fast)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cameras", type=int, default=4)
    p.add_argument(
        "--fast",
        action="store_true",
        help="tiny-yard CI smoke: mission_tiny + short phase budgets",
    )
    p.add_argument(
        "--speed",
        type=str,
        default="1",
        help="wall-clock multiplier: 1, 2, 5, or max (unpaced)",
    )
    p.add_argument("--host", type=str, default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument(
        "--prepare-only",
        action="store_true",
        help="run N steps headless (no HTTP) and write the live bundle",
    )
    p.add_argument("--cam-stride", type=int, default=20)
    p.add_argument("--map-stride", type=int, default=MAP_STRIDE_DEFAULT)
    p.add_argument("--stride", type=int, default=None, help="true-mesh decimation stride (acre default 4)")
    p.add_argument(
        "--observed-mesh-stride",
        type=int,
        default=MESH_STRIDE_DEFAULT,
        help="rebuild the observed elevation mesh every N steps (default 8)",
    )
    p.add_argument(
        "--phase-budget",
        type=float,
        default=1.0,
        help="scale mission phase caps (0.05–1). Live demos can use 0.4; does not change physics.",
    )
    p.add_argument(
        "--calibrate-stride",
        type=float,
        default=None,
        help="calibrate waypoint spacing in metres (0/omit = auto from yard size)",
    )
    p.add_argument(
        "--calibrate-confirm",
        type=float,
        default=None,
        help="close calibrate after this many metres of trail (demo). 0 = full lap.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    summary = run_live(
        config=args.config,
        fast=args.fast,
        speed=args.speed,
        steps=args.steps,
        seed=args.seed,
        cameras=args.cameras,
        out_dir=args.out,
        host=args.host,
        port=args.port,
        prepare_only=args.prepare_only,
        cam_stride=args.cam_stride,
        map_stride=args.map_stride,
        mesh_stride=args.stride,
        observed_mesh_stride=args.observed_mesh_stride,
        phase_budget=args.phase_budget,
        calibrate_stride=args.calibrate_stride,
        calibrate_confirm=args.calibrate_confirm,
    )
    if args.prepare_only:
        print(
            f"Live prepare {summary.get('phase')} after {summary.get('step')} steps "
            f"→ {args.out}  map {float(summary.get('map_pct') or 0.0):.3f}"
        )


if __name__ == "__main__":
    main()

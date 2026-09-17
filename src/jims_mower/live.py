"""Live mission session: wall-clock MissionPolicy + SSE owner view.

This is the owner loop that ``jims-mower-mission-demo`` + the World Viewer
scrubber were missing. The sim still uses the true height field for
physics. Control stays on ``ObservedMap``. The default owner view is a
fog veil over unknown cells plus a growing observed elevation mesh —
already-mapped heights stay put when the IMU tips. Not a finished
god-view mesh from step 0, and not a sheet hinged to chassis tilt.

No claimed mAP / FPS. Not a coverage benchmark.

At ``--speed max`` (and 5×) owner-view PNG / observed-mesh / disk flush
is coarsened and no longer rebuilds on every newly observed cell — that
used to starve acre physics down to ~2 Hz.
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

from jims_mower.constants import GYM_PAIR_PIN, LIVE_CONTROL_CMDS, LIVE_SCHEMA, VIEWER_SCHEMA
from jims_mower.pairing import PairingMachine
from jims_mower.env import MowerEnv
from jims_mower.mission_demo import (
    FAST_CAM_HEIGHT,
    FAST_CAM_WIDTH,
    FAST_STEPS,
    resolve_mission_config,
)
from jims_mower.mesh import coverage_to_rgb, mesh_from_observed, mesh_to_payload
from jims_mower.mission_flow import (
    PHASE_LABELS,
    MissionPolicy,
    mission_timeline,
    scale_mission_budget,
    session_summary,
)
from jims_mower.path_overlay import build_path_overlay, mission_from_phase
from jims_mower.planning.observed import fog_rgba
from jims_mower.profile import (
    YardProfile,
    apply_profile_to_scenario,
    keep_in_usable,
    load_yard_profile,
    repair_keep_in,
    write_yard_profile,
)
from jims_mower.scenarios import load_source
from jims_mower.teach import TeachPolicy, keepouts_from_env
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
# High-speed floors. Per-cell map/mesh rebuilds starve acre physics.
MAP_STRIDE_FAST = 8
MESH_STRIDE_FAST = 16
FLUSH_STRIDE_FAST = 40
CAM_WALL_FAST_S = 0.80
MAP_MAX_SIDE_FAST = 64
# Do not swap this in for MESH_MAX_SIDE mid-session — a 32↔48 grid
# change rebuilds acre verts (696→1920) and looks like a local flop.
MESH_MAX_SIDE_FAST = 32
MAP_STRIDE_MAX = 16
MESH_STRIDE_MAX = 32
FLUSH_STRIDE_MAX = 80
CAM_WALL_MAX_S = 1.25
ACRE_LIVE_CAM_WIDTH = 48
ACRE_LIVE_CAM_HEIGHT = 36
REVIEW_HOLD_WALL_S = 2.0
LIVE_YARDS = ("acre_yard_demo", "acre_yard", "mission_tiny", "golf_rough")
OWNER_COPY = {
    "idle": "Yard unknown — start a job when ready.",
    "taught_idle": "Yard taught — start a job when ready.",
    "teach": "Teaching the keep-in — drive the perimeter or edit vertices, then save.",
    "paused": "Paused.",
    "estop": "E-STOP — hold.",
    "hw_estop": "Hardware E-STOP — rails dead. Reset the paddle.",
    "hold": "Hold.",
    "calibrate_boundary": "Calibrating boundary…",
    "explore": "Exploring unknown yard…",
    "review": "Map ready — start mow?",
    "reteach": "Fence too small — re-teach the keep-in.",
    "mow": "Mowing…",
    "resuming_mow": "Resuming mow",
    "resuming_explore": "Resuming explore",
    "return_home": "Heading home…",
    "low_battery": "Low battery — returning to charge",
    "charging": "Charging…",
    "complete": "Done.",
    "fault": "Fault — retrieve.",
    "safe": "Hold — safe.",
    "immobilised": "SOS — immobilised. Retrieve the mower.",
    "stuck": "Stuck — recovering (reverse / pivot).",
    "blocked_remap": "Blocked — remapping around obstacle",
    "unpaired": "Pair Bluetooth before Start.",
    "pairing": "Pairing…",
    "pair_failed": "Pairing failed — check the gym PIN.",
    "radio_lost": "Radio lost — hold safe. Pair again to Start.",
}
RADIO_PATH_CHIPS = (
    {"id": "bt", "label": "BT teach", "role": "teach"},
    {"id": "wifi", "label": "Wi-Fi map", "role": "map"},
    {"id": "lora", "label": "LoRa sparse", "role": "sparse"},
)
INJECT_ALIASES = {
    "sos": "motor_left",
    "dead_motor": "motor_left",
    "immobilised": "motor_left",
    "motor": "motor_left",
    "stuck": "stuck",
    "paddle": "hw_estop",
    "hw_estop": "hw_estop",
    "hardware_estop": "hw_estop",
    "estop_paddle": "hw_estop",
    "radio_lost": "radio_lost",
    "lost": "radio_lost",
    "radio-lost": "radio_lost",
    "bt_lost": "bt_lost",
    "low_soc": "low_soc",
    "soc": "low_soc",
    "battery": "low_soc",
    "soc_low": "low_soc",
}

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


def speed_label(speed: float) -> str:
    if speed <= 0.0:
        return "max"
    if abs(speed - 1.0) < 1e-6:
        return "1"
    if abs(speed - 2.0) < 1e-6:
        return "2"
    if abs(speed - 5.0) < 1e-6:
        return "5"
    return str(speed)


def owner_copy_for(
    job_state: str,
    phase: str,
    fault: Optional[dict[str, Any]] = None,
    *,
    taught: bool = False,
    fence_unusable: bool = False,
    done: bool = False,
    hw_estop: bool = False,
    pairing_state: Optional[str] = None,
    require_pair: bool = False,
    explore_reason: Optional[dict[str, Any]] = None,
    charge_state: str = "",
    return_kind: str = "",
) -> str:
    blob = fault if isinstance(fault, dict) else None
    if hw_estop or (blob and str(blob.get("code") or "") == "HW_ESTOP"):
        return OWNER_COPY["hw_estop"]
    if job_state == "estop":
        return OWNER_COPY["estop"]
    if blob:
        code = str(blob.get("code") or "")
        if blob.get("retrieve") or code == "FAULT_IMMOBILISED":
            return OWNER_COPY["immobilised"]
        if code == "STUCK":
            return OWNER_COPY["stuck"]
    if require_pair and pairing_state in {"unpaired", "pairing", "failed", "lost"}:
        if pairing_state == "lost":
            return OWNER_COPY["radio_lost"]
        if pairing_state == "failed":
            return OWNER_COPY["pair_failed"]
        if pairing_state == "pairing":
            return OWNER_COPY["pairing"]
        if job_state in {"idle", "paused", "hold"}:
            return OWNER_COPY["unpaired"]
    if done and job_state == "idle":
        return OWNER_COPY["complete"]
    if job_state == "teach" or phase == "teach":
        return OWNER_COPY["teach"]
    if fence_unusable:
        return OWNER_COPY["reteach"]
    if phase == "charging" or charge_state == "charging":
        return OWNER_COPY["charging"]
    if phase == "return_home" and (return_kind == "battery" or charge_state == "returning"):
        return OWNER_COPY["low_battery"]
    if charge_state == "resuming":
        return OWNER_COPY["resuming_explore"] if phase == "explore" else OWNER_COPY["resuming_mow"]
    # MAP READY is a review beat, not SafeState / ESTOP.
    if phase == "review":
        return OWNER_COPY["review"]
    if phase == "explore" and isinstance(explore_reason, dict) and explore_reason.get("label"):
        return str(explore_reason["label"])
    if phase == "safe":
        return OWNER_COPY["safe"]
    if job_state in OWNER_COPY and job_state in {"idle", "paused", "hold"}:
        if job_state == "idle" and taught:
            return OWNER_COPY["taught_idle"]
        return OWNER_COPY[job_state]
    return OWNER_COPY.get(phase, PHASE_LABELS.get(phase, phase))


def radio_path_for(phase: str, job_state: str = "running") -> dict[str, Any]:
    """Simulated bearer chips: BT teach / Wi-Fi map / LoRa sparse. No RF hardware."""
    if job_state in {"idle", "teach"} or phase in {"", "idle", "teach", "calibrate_boundary"}:
        active = "bt"
    elif phase in {"explore", "review"}:
        active = "wifi"
    else:
        active = "lora"
    chips = [{**chip, "active": chip["id"] == active} for chip in RADIO_PATH_CHIPS]
    return {
        "active": active,
        "chips": chips,
        "rf_claim": None,
        "simulated": True,
        "not_rf_hardware": True,
    }


def robot_status_for(
    *,
    paired: bool,
    job_state: str,
    faults: Optional[list[dict[str, Any]]] = None,
    done: bool = False,
) -> str:
    """Owner pill: idle / pairing / live / fault."""
    for item in faults or []:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "")
        if item.get("retrieve") or code in {"FAULT_IMMOBILISED", "HW_ESTOP"}:
            return "fault"
    if job_state == "estop":
        return "fault"
    if not paired:
        return "pairing"
    if job_state in {"running", "paused", "hold", "teach"} and not done:
        return "live"
    return "idle"


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


def _compact_mesh_payload(mesh: Any, *, cheap: bool = False) -> dict[str, Any]:
    payload = mesh_to_payload(mesh)
    if not cheap:
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
        yard_profile: Optional[YardProfile] = None,
        yard_path: Optional[Union[str, Path]] = None,
        first_run: bool = False,
        require_pair: Optional[bool] = None,
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
        self.yard_profile = yard_profile
        self.yard_path = Path(yard_path) if yard_path else None
        self.first_run = bool(first_run)
        self.owner_taught = bool(yard_profile is not None and len(yard_profile.keep_in) >= 3)
        self._taught_env_dirty = False
        self.teach_policy: Optional[TeachPolicy] = None
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
        self._n_map_builds = 0
        self._n_mesh_builds = 0
        self._last_phase = ""
        self.poses: list[dict[str, Any]] = []
        self.camera_names: list[str] = []
        self.observed_png = b""
        self.fog_png = b""
        self.observed_mesh_json = b""
        self.coverage_png = b""
        self.areas_png = b""
        self.cam_jpeg: dict[str, bytes] = {}
        self.done = False
        self.started = False
        self.job_state = "idle"
        self.unattended = False
        self.estop = False
        pin = GYM_PAIR_PIN
        self._require_pair_explicit = require_pair is not None
        require = bool(require_pair) if require_pair is not None else False
        persisted = None
        if yard_profile is not None:
            persisted = getattr(yard_profile, "pairing", None)
        self.pairing = PairingMachine.from_profile(persisted, require_pair=require, pin=pin)
        self.require_pair = require
        self._t0_wall = 0.0
        self._review_wall0: Optional[float] = None
        self.session_card: dict[str, Any] = {}
        self._fault_overlay: dict[str, Any] = {}

    @property
    def paired(self) -> bool:
        return self.pairing.is_paired

    @paired.setter
    def paired(self, value: bool) -> None:
        if value:
            if not self.pairing.is_paired:
                self.pairing.force_paired()
        elif self.pairing.is_paired:
            self.pairing.unpair()

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
        taught = self.yard_profile if self.owner_taught and self.yard_profile is not None else None
        if taught is not None and len(taught.keep_in) >= 3:
            apply_profile_to_scenario(scenario, taught, resize_world=False)
        if self.env is not None:
            self.env.close()
        self.env = MowerEnv(config=cfg, scenario=scenario, render_mode="rgb_array")
        owner = getattr(cfg, "owner", None)
        if owner is not None:
            self.pairing.pin = str(getattr(owner, "pair_pin", GYM_PAIR_PIN) or GYM_PAIR_PIN)
            if not self._require_pair_explicit:
                self.require_pair = bool(getattr(owner, "require_pair", False))
                self.pairing.require_pair = self.require_pair
        reset_opts: dict[str, Any] = {
            "blackbox": str(self.out_dir / "blackbox.jsonl"),
            "save_mission": str(self.out_dir / "session.npz"),
        }
        if taught is not None:
            reset_opts["yard_profile"] = taught
            reset_opts["resize_world"] = False
        self.obs, self.info = self.env.reset(seed=self.seed, options=reset_opts)
        self.policy = MissionPolicy(self.env.cfg, fast=self.fast)
        self.policy.reset(self.obs, self.info, profile=taught)
        self.policy.attach_to_env(self.env)
        self.teach_policy = None
        self._taught_env_dirty = False
        self.camera_names = list(self.obs.get("cameras") or {})
        self.poses = []
        self.done = False
        self.started = True
        self.job_state = "idle"
        self.estop = False
        self._fault_overlay = {}
        self._t0_wall = 0.0
        self._review_wall0 = None
        self.session_card = {}
        self._seq = 0
        self._map_seq = 0
        self._mesh_seq = 0
        self._cam_seq = 0
        self._last_cam_wall = 0.0
        self._n_observed = 0
        self._n_map_builds = 0
        self._n_mesh_builds = 0
        self._last_phase = self.policy.phase.value
        if self.unattended:
            self.job_state = "running"
            self._t0_wall = time.perf_counter()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._refresh_maps(force=True)
        self._refresh_cameras(force=True)
        self._record_pose()
        self._write_bundle()
        self._capture_summary()
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

    def coverage_png_bytes(self) -> bytes:
        with self.lock:
            return self.coverage_png

    def areas_png_bytes(self) -> bytes:
        with self.lock:
            return self.areas_png

    def camera_bytes(self, name: str) -> bytes:
        with self.lock:
            return self.cam_jpeg.get(name, b"")

    def step_once(self) -> dict[str, Any]:
        self._advance()
        return self.snapshot()

    def _advance(self) -> None:
        """One physics step. Owner-view encode is throttled — do not snapshot."""
        if self.env is None or self.policy is None:
            raise RuntimeError("LiveSession.reset() before stepping")
        if self.done:
            return
        if self.job_state == "teach":
            self._step_teach()
            return
        if self.estop:
            self.info = dict(self.info or {})
            self.info["estop"] = True
        self._maybe_auto_mow()
        action = self.policy.act(self.obs, self.info)
        self.obs, _reward, terminated, truncated, self.info = self.env.step(action)
        if self.estop:
            self.info = dict(self.info or {})
            self.info["estop"] = True
        if terminated or truncated or self.policy.done:
            self.done = True
            self._capture_summary()
            if self.policy.phase.value == "complete":
                self.job_state = "idle"
        force_map = self.policy.phase.value != self._last_phase
        if self.policy.phase.value == "review" and self._last_phase != "review":
            self._review_wall0 = time.perf_counter()
        self._last_phase = self.policy.phase.value
        self._refresh_maps(force=force_map)
        self._refresh_cameras(force=False)
        self._record_pose()
        with self.lock:
            self._seq += 1
        budget = self._stream_budget()
        if self.policy.step % int(budget["flush_stride"]) == 0 or self.done:
            self._flush_incremental()

    def run_n(self, n: int) -> dict[str, Any]:
        self.unattended = True
        self._stop.clear()
        if self.job_state == "idle":
            self.job_state = "running"
            self._t0_wall = time.perf_counter()
        if not self.started:
            self.reset()
        for _ in range(max(0, int(n))):
            if self.done or self._stop.is_set():
                break
            self._advance()
        self._flush_incremental(final=True)
        return self.snapshot()

    def run_blocking(self) -> dict[str, Any]:
        if not self.started:
            self.reset()
        if self.unattended and self.job_state == "idle":
            self.job_state = "running"
            self._t0_wall = time.perf_counter()
        t0 = time.perf_counter()
        step_i = 0
        last_speed = self.speed
        while not self._stop.is_set() and not self.done and step_i < self.max_steps:
            if self.job_state not in {"running", "teach"}:
                time.sleep(0.05)
                t0 = time.perf_counter()
                step_i = 0
                continue
            self._advance()
            step_i += 1
            if self.speed != last_speed:
                t0 = time.perf_counter()
                step_i = 1
                last_speed = self.speed
            if self.speed > 0.0:
                target = t0 + step_i * (self.dt / self.speed)
                now = time.perf_counter()
                delay = target - now
                if delay > 0.0:
                    time.sleep(delay)
        self._flush_incremental(final=True)
        return self.snapshot()

    def start_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if not self.started:
            self.reset()
        if self.policy is not None and not self.unattended:
            self.policy.settings.review_hold_steps = max(
                10_000, int(self.policy.settings.review_hold_steps)
            )
        self._stop.clear()
        self._thread = threading.Thread(target=self.run_blocking, name="jims-mower-live", daemon=True)
        self._thread.start()

    def control(self, cmd: str, **kwargs: Any) -> dict[str, Any]:
        """Owner bar: start / pause / resume / speed / start_mow / ESTOP / teach."""
        key = str(cmd or "").strip().lower()
        if key not in LIVE_CONTROL_CMDS:
            return {"ok": False, "error": f"unknown command {cmd}", **self.snapshot()}
        if not self.started:
            self.reset()
        if key in {"start", "resume", "start_mow", "explore", "mow", "return"}:
            refused = self.pairing.refuse_start()
            if refused:
                return {
                    "ok": False,
                    "error": "not paired — Pair Bluetooth before Start",
                    "reason": refused,
                    "cmd": key,
                    **self.snapshot(),
                }
        radio_block = self._command_via_radio(key)
        if radio_block is not None:
            return radio_block
        if key == "start":
            yard = kwargs.get("yard")
            if yard:
                self._apply_yard(str(yard))
            if kwargs.get("speed") is not None:
                self.speed = parse_speed(kwargs.get("speed"))
            self.estop = False
            if self._needs_job_rebuild():
                self._reset_for_next_job()
            if self.policy is not None:
                self.policy.clear_owner_hold()
                if self.policy.safe.mode == "estop":
                    self.policy.safe.clear()
            self.job_state = "running"
            self._t0_wall = self._t0_wall or time.perf_counter()
            self._stop.clear()
            self.start_thread()
        elif key == "pause":
            if self.job_state == "running":
                self.job_state = "paused"
                if self.policy is not None:
                    self.policy.request_hold("owner pause")
                self.persist_session()
        elif key == "resume":
            if self.job_state in {"paused", "hold"} or self.done:
                self.estop = False
                if self._needs_job_rebuild():
                    self._reset_for_next_job()
                if self.policy is not None:
                    self.policy.clear_owner_hold()
                self.job_state = "running"
                self._stop.clear()
                self.start_thread()
        elif key == "speed":
            self.speed = parse_speed(kwargs.get("speed", self.speed))
        elif key == "start_mow" or key == "mow":
            if self.policy is not None:
                self.policy.request_start_mow()
            if self.job_state == "idle":
                self.job_state = "running"
                self._t0_wall = time.perf_counter()
                self._stop.clear()
                self.start_thread()
        elif key == "reexplore" or key == "explore":
            full = bool(kwargs.get("full") or kwargs.get("full_explore"))
            if self.policy is not None:
                self.policy.request_explore(full=full)
            if self.job_state == "idle":
                self.job_state = "running"
                self._t0_wall = self._t0_wall or time.perf_counter()
                self._stop.clear()
                self.start_thread()
        elif key == "return":
            if self.policy is not None:
                self.policy.request_return_home(reason=str(kwargs.get("reason") or "owner"))
            if self.job_state == "idle":
                self.job_state = "running"
                self._t0_wall = self._t0_wall or time.perf_counter()
                self._stop.clear()
                self.start_thread()
        elif key == "full_explore":
            enabled = kwargs.get("enabled", True)
            if isinstance(enabled, str):
                enabled = enabled.strip().lower() not in {"0", "false", "off", "no"}
            if self.policy is not None:
                self.policy.apply_full_explore_mode(bool(enabled))
        elif key == "estop":
            self.estop = True
            self.job_state = "estop"
            if self.policy is not None:
                self.policy.request_estop("owner estop")
        elif key == "hw_estop":
            if self.env is not None:
                self.env.hit_hw_estop(str(kwargs.get("reason") or "owner paddle"))
            self.info = dict(self.info or {})
            self.info.update(self.env.hw_estop.as_info() if self.env is not None else {})
        elif key == "hw_reset":
            if self.env is not None:
                self.env.reset_hw_estop()
            self.info = dict(self.info or {})
            self.info.update(self.env.hw_estop.as_info() if self.env is not None else {})
        elif key == "hold":
            if self.job_state == "running":
                self.job_state = "hold"
            if self.policy is not None:
                self.policy.request_hold("owner hold")
        elif key == "clear":
            self.estop = False
            if self.policy is not None:
                self.policy.safe.clear()
                self.policy.clear_owner_hold()
                if self.policy.phase.value == "safe":
                    self.policy.phase = self.policy.phase
            self.job_state = "paused"
        elif key == "yard":
            self._apply_yard(str(kwargs.get("yard") or self.config_name))
            self.job_state = "idle"
        elif key == "pair":
            pin = kwargs.get("pin") if kwargs.get("pin") is not None else kwargs.get("code")
            result = self.pairing.request_pair(None if pin is None else str(pin))
            self.require_pair = self.pairing.require_pair
            self._persist_pairing()
            snap = self.snapshot()
            return {"ok": bool(result.ok), "cmd": key, "reason": result.reason, **snap}
        elif key == "unpair":
            self.pairing.unpair()
            self._hold_after_link_loss("unpaired")
            self._persist_pairing()
        elif key == "teach":
            return self._begin_teach(**kwargs)
        elif key == "save_yard":
            return self._save_yard(**kwargs)
        elif key == "load_yard":
            return self._load_yard(**kwargs)
        elif key == "inject":
            soc_raw = kwargs.get("soc")
            soc = None if soc_raw is None else float(soc_raw)
            self._inject_fault(
                str(kwargs.get("kind") or kwargs.get("fault") or "stuck"),
                mode=str(kwargs.get("mode") or "open_circuit"),
                soc=soc,
            )
        return {"ok": True, "cmd": key, **self.snapshot()}

    def _command_via_radio(self, key: str) -> Optional[dict[str, Any]]:
        """When radio sim is enabled, route owner cmds. BT lost → LoRa far-fence."""
        if self.env is None:
            return None
        radio = getattr(self.env, "radio", None)
        if radio is None or not (bool(getattr(radio, "enabled", False)) or getattr(radio, "any_lost", False)):
            return None
        if not getattr(radio, "any_lost", False) and not radio.enabled:
            return None
        if key not in {"start", "pause", "resume", "estop", "hold", "hw_estop", "start_mow", "explore", "mow", "return"}:
            return None
        if not getattr(radio, "any_lost", False) and not radio.enabled:
            return None
        if not getattr(radio, "any_lost", False):
            return None
        delivery = radio.route_command(key, paired=self.pairing.is_paired)
        if delivery.ok:
            return None
        return {
            "ok": False,
            "error": str(delivery.reason or "radio_refused"),
            "cmd": key,
            "radio_channel": delivery.channel,
            **self.snapshot(),
        }

    def _hold_after_link_loss(self, reason: str) -> None:
        """Unpair / radio-lost: hold safe, not ESTOP."""
        if self.job_state == "running":
            self.job_state = "hold"
            if self.policy is not None:
                self.policy.request_hold(f"{reason} — hold safe")

    def _persist_pairing(self) -> None:
        blob = self.pairing.persist()
        if self.yard_profile is not None:
            self.yard_profile.pairing = blob
            if self.yard_path is not None:
                write_yard_profile(self.yard_path, self.yard_profile)

    def _inject_fault(self, kind: str, *, mode: str = "open_circuit", soc: Optional[float] = None) -> None:
        raw = str(kind or "stuck").strip().lower()
        resolved = INJECT_ALIASES.get(raw, raw)
        if resolved == "low_soc":
            target = 0.12 if soc is None else float(soc)
            if self.env is not None:
                self.env.budget.set_soc(target)
                self.info = dict(self.info or {})
                self.info.update(self.env.budget.as_info())
            return
        if resolved == "radio_lost":
            self.pairing.lose(reason="radio_lost")
            if self.env is not None:
                self.env.radio.lose_link("wifi")
                self.env.radio.lose_link("bt")
            self._hold_after_link_loss("radio_lost")
            self._persist_pairing()
            return
        if resolved == "bt_lost":
            if self.env is not None:
                self.env.radio.lose_link("bt")
            return
        if self.env is None:
            return
        self.env.inject_fault(resolved, mode=mode)
        self.info = dict(self.info or {})
        self.info.update(self.env.hw_estop.as_info())
        blob = self.env.fault_bus.as_info((self.info or {}).get("pose") if self.info else None)
        self.info["fault"] = blob
        hw_fault = self.env.hw_estop.as_fault()
        self._fault_overlay = hw_fault or blob

    def _hw_estop_latched(self) -> bool:
        if self.env is not None and bool(getattr(self.env.hw_estop, "latched", False)):
            return True
        info = self.info if isinstance(self.info, dict) else {}
        return bool(info.get("hw_estop") or info.get("hw_estop_latched"))

    def _current_fault(self) -> dict[str, Any]:
        if self.env is not None:
            hw = self.env.hw_estop.as_fault()
            if hw is not None:
                return hw
        info = self.info if isinstance(self.info, dict) else {}
        if info.get("hw_estop"):
            return {
                "code": "HW_ESTOP",
                "detail": str(info.get("hw_estop_reason") or "paddle latched — rails dead"),
                "retrieve": False,
                "kind": "hardware",
            }
        blob = info.get("fault") if isinstance(info, dict) else None
        if isinstance(blob, dict) and str(blob.get("code") or "ok") not in {"", "ok"}:
            return blob
        if isinstance(self._fault_overlay, dict) and str(self._fault_overlay.get("code") or "ok") not in {"", "ok"}:
            return self._fault_overlay
        return {}

    def _faults_payload(self, fault: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        if isinstance(fault, dict):
            code = str(fault.get("code") or "ok")
            if code and code != "ok":
                items.append(
                    {
                        "code": code,
                        "detail": str(
                            fault.get("detail")
                            or fault.get("reason")
                            or fault.get("component")
                            or code
                        ),
                        "retrieve": bool(fault.get("retrieve")),
                        "kind": str(fault.get("kind") or ("hardware" if code == "HW_ESTOP" else "software")),
                    }
                )
        if self._hw_estop_latched() and not any(f.get("code") == "HW_ESTOP" for f in items):
            reason = None
            if self.env is not None:
                reason = self.env.hw_estop.reason
            if not reason and isinstance(self.info, dict):
                reason = self.info.get("hw_estop_reason")
            items.append(
                {
                    "code": "HW_ESTOP",
                    "detail": str(reason or "paddle latched — rails dead"),
                    "retrieve": False,
                    "kind": "hardware",
                }
            )
        return items

    def _apply_yard(self, yard: str) -> None:
        name = str(yard).strip()
        if name not in LIVE_YARDS:
            return
        if name == self.config_name and self.started:
            return
        was_running = self.job_state == "running"
        self.stop()
        self.config_name = name
        self.fast = name == "mission_tiny"
        self.reset()
        if was_running:
            self.job_state = "running"
            self._t0_wall = time.perf_counter()

    def _needs_job_rebuild(self) -> bool:
        """Start/resume must rebuild after a finished job or a taught fence."""
        if self.done:
            return True
        if self.policy is not None and self.policy.phase.value in {"complete", "fault", "safe"}:
            return True
        return self._needs_taught_job_reset()

    def _reset_for_next_job(self) -> None:
        """Rebuild env/policy for a new Start. Keeps the taught yard if any."""
        pairing = self.pairing
        require_pair = self.require_pair
        explicit = self._require_pair_explicit
        taught = self.owner_taught
        profile = self.yard_profile
        first = self.first_run
        dest = self.yard_path
        self.stop()
        self.yard_profile = profile
        self.owner_taught = taught
        self.first_run = first
        self.yard_path = dest
        if taught and profile is not None:
            self._apply_usable_keep_in(profile)
            self._snap_home_inside_keep_in(profile)
        self.reset()
        self.pairing = pairing
        self.require_pair = require_pair
        self._require_pair_explicit = explicit
        self.pairing.require_pair = require_pair
        self.done = False

    def _needs_taught_job_reset(self) -> bool:
        """True when Start must rebuild so a saved fence skips authored calibrate.

        Teach completion sets ``job_state='paused'``. Treating that like a
        mid-mow pause used to resume the idle calibrate session on a dirty
        env — UI stuck on “Calibrating…” / IDLE after 1–2 steps.
        """
        if not self.owner_taught or self.yard_profile is None:
            return False
        if len(self.yard_profile.keep_in) < 3:
            return False
        if self.done or self._taught_env_dirty or self.job_state == "teach":
            return True
        if self.policy is None:
            return True
        if self.policy.phase.value in {"calibrate_boundary", "complete", "fault", "safe"}:
            return True
        if self.policy.profile is None or len(self.policy.profile.keep_in) < 3:
            return True
        env = self.env
        if env is not None and int(getattr(env, "_steps", 0)) >= max(1, int(env.cfg.max_steps) - 2):
            return True
        return False

    def _reset_for_taught_job(self) -> None:
        """Rebuild the env/policy so Start uses the saved YardProfile as fence."""
        if not self.owner_taught or self.yard_profile is None:
            return
        self._reset_for_next_job()

    def _snap_home_inside_keep_in(self, profile: YardProfile) -> None:
        """Keep spawn inside the taught fence so Start does not OOB/drain-drop."""
        from jims_mower.geofence import inside_keep_in

        keep = list(profile.keep_in)
        if len(keep) < 3:
            return
        home = profile.home_pose()
        if inside_keep_in(home.x, home.y, keep):
            return
        cx = sum(p[0] for p in keep) / float(len(keep))
        cy = sum(p[1] for p in keep) / float(len(keep))
        if not inside_keep_in(cx, cy, keep) and keep:
            vx, vy = keep[0]
            cx = vx + 0.35 * (cx - vx)
            cy = vy + 0.35 * (cy - vy)
        profile.home = {"x": float(cx), "y": float(cy), "theta": float(home.theta)}

    def _begin_teach(self, **kwargs: Any) -> dict[str, Any]:
        if not self.started:
            self.reset()
        if kwargs.get("speed") is not None:
            self.speed = parse_speed(kwargs.get("speed"))
        self.estop = False
        self.done = False
        self.teach_policy = None
        self._taught_env_dirty = True
        if self.policy is not None:
            self.policy.clear_owner_hold()
        self.job_state = "teach"
        self._t0_wall = self._t0_wall or time.perf_counter()
        self._stop.clear()
        self.start_thread()
        return {"ok": True, "cmd": "teach", **self.snapshot()}

    def _profile_from_vertices(
        self,
        keep_in: Any,
        *,
        keep_out: Optional[Any] = None,
        home: Optional[Any] = None,
        name: str = "",
    ) -> YardProfile:
        from jims_mower.profile import parse_yard_profile

        env = self.env
        width = float(env.cfg.world.width_m) if env is not None else 16.0
        height = float(env.cfg.world.height_m) if env is not None else 12.0
        res = float(env.cfg.world.resolution_m) if env is not None else 0.20
        pose = (self.info or {}).get("pose") if isinstance(self.info, dict) else {}
        holes = keep_out
        if holes is None and env is not None:
            holes = [list(p) for p in env.geofence_spec().keep_out]
        if not holes and self.yard_profile is not None:
            holes = [list(p) for p in self.yard_profile.keep_out]
        if not holes and env is not None:
            holes = keepouts_from_env(env)
        home_pose = home
        if home_pose is None and isinstance(pose, dict) and pose:
            home_pose = {
                "x": float(pose.get("x", 1.0)),
                "y": float(pose.get("y", 1.0)),
                "theta": float(pose.get("theta", 0.0)),
            }
        payload = {
            "schema": "jims_mower.yard.v1",
            "name": name or (self.yard_profile.name if self.yard_profile else str(self.config_name)),
            "width_m": width,
            "height_m": height,
            "resolution_m": res,
            "keep_in": keep_in,
            "keep_out": holes or [],
            "home": home_pose or {"x": 1.0, "y": 1.0, "theta": 0.0},
            "mesh": "yard.glb",
        }
        return parse_yard_profile(payload)

    def _draft_profile_from_teach(self) -> YardProfile:
        env = self.env
        keep_out: list[list[tuple[float, float]]] = []
        if env is not None:
            keep_out = [list(p) for p in env.geofence_spec().keep_out]
        if not keep_out and env is not None:
            keep_out = keepouts_from_env(env)
        if self.teach_policy is not None:
            profile = self.teach_policy.to_profile(name=str(self.config_name), keep_out=keep_out)
        elif self.yard_profile is not None and len(self.yard_profile.keep_in) >= 3:
            profile = self.yard_profile
        elif env is not None:
            spec = env.geofence_spec()
            profile = YardProfile(
                name=str(self.config_name),
                width_m=float(env.cfg.world.width_m),
                height_m=float(env.cfg.world.height_m),
                resolution_m=float(env.cfg.world.resolution_m),
                keep_in=list(spec.keep_in),
                keep_out=keep_out or [list(p) for p in spec.keep_out],
                home={"x": 1.0, "y": 1.0, "theta": 0.0},
            )
        else:
            raise RuntimeError("no teach trail or yard to save")
        if env is not None:
            profile.width_m = float(env.cfg.world.width_m)
            profile.height_m = float(env.cfg.world.height_m)
            profile.resolution_m = float(env.cfg.world.resolution_m)
        pose = (self.info or {}).get("pose") if isinstance(self.info, dict) else {}
        if isinstance(pose, dict) and pose and not (self.teach_policy and self.teach_policy.trail):
            profile.home = {
                "x": float(pose.get("x", profile.home.get("x", 1.0))),
                "y": float(pose.get("y", profile.home.get("y", 1.0))),
                "theta": float(pose.get("theta", profile.home.get("theta", 0.0))),
            }
        return profile

    def _authored_keep_in(self) -> list[tuple[float, float]]:
        env = self.env
        if env is not None:
            spec = env.geofence_spec()
            ring = list(spec.keep_in or [])
            if len(ring) >= 3:
                return [(float(x), float(y)) for x, y in ring]
        if self.teach_policy is not None:
            planned = self.teach_policy.planned_ring()
            if len(planned) >= 3:
                return list(planned)
        return []

    def _apply_usable_keep_in(self, profile: YardProfile) -> str:
        """Reject scribble fences; inflate or fall back to the authored yard."""
        env = self.env
        width = float(env.cfg.world.width_m) if env is not None else float(profile.width_m or 16.0)
        height = float(env.cfg.world.height_m) if env is not None else float(profile.height_m or 12.0)
        profile.width_m = width
        profile.height_m = height
        ring, source = repair_keep_in(
            list(profile.keep_in),
            width_m=width,
            height_m=height,
            fallback=self._authored_keep_in(),
            inflate_m=2.5,
        )
        profile.keep_in = ring
        return source

    def _persist_yard(self, profile: YardProfile, dest: Optional[Union[str, Path]] = None) -> Path:
        path = Path(dest) if dest is not None else (self.yard_path or (self.out_dir / "profile.json"))
        write_yard_profile(path, profile)
        self.yard_path = path
        return path

    def _save_yard(self, **kwargs: Any) -> dict[str, Any]:
        keep_in = kwargs.get("keep_in")
        if keep_in:
            profile = self._profile_from_vertices(
                keep_in,
                keep_out=kwargs.get("keep_out"),
                home=kwargs.get("home"),
                name=str(kwargs.get("name") or ""),
            )
        else:
            profile = self._draft_profile_from_teach()
        if len(profile.keep_in) < 3:
            return {"ok": False, "error": "keep_in needs at least 3 vertices", **self.snapshot()}
        source = self._apply_usable_keep_in(profile)
        if source == "unusable" or not keep_in_usable(
            profile.keep_in,
            float(profile.width_m),
            float(profile.height_m),
        ):
            return {
                "ok": False,
                "error": "keep-in is a scribble — drive the perimeter or edit the starter rectangle",
                "needs_reteach": True,
                **self.snapshot(),
            }
        dest = kwargs.get("path")
        self._snap_home_inside_keep_in(profile)
        path = self._persist_yard(profile, dest)
        self.yard_profile = profile
        self.owner_taught = True
        self.done = False
        self.session_card = {}
        self.job_state = "idle"
        self.stop()
        return {
            "ok": True,
            "cmd": "save_yard",
            "path": str(path),
            "keep_in_source": source,
            "keep_in_repaired": source != "taught",
            **self.snapshot(),
        }

    def _load_yard(self, **kwargs: Any) -> dict[str, Any]:
        dest = Path(kwargs.get("path") or self.yard_path or (self.out_dir / "profile.json"))
        if not dest.is_file():
            return {"ok": False, "error": f"YardProfile not found: {dest}", **self.snapshot()}
        profile = load_yard_profile(dest)
        if len(profile.keep_in) < 3:
            return {"ok": False, "error": "saved yard missing keep_in", **self.snapshot()}
        source = self._apply_usable_keep_in(profile)
        if source == "unusable" or not keep_in_usable(
            profile.keep_in,
            float(profile.width_m),
            float(profile.height_m),
        ):
            return {
                "ok": False,
                "error": "saved keep-in is too small — re-teach the yard",
                "needs_reteach": True,
                **self.snapshot(),
            }
        self.yard_profile = profile
        self.owner_taught = True
        self.yard_path = dest
        self._snap_home_inside_keep_in(profile)
        self.done = False
        self.session_card = {}
        self.job_state = "idle"
        return {
            "ok": True,
            "cmd": "load_yard",
            "path": str(dest),
            "keep_in_source": source,
            "keep_in_repaired": source != "taught",
            **self.snapshot(),
        }

    def _step_teach(self) -> None:
        assert self.env is not None
        if self.teach_policy is None:
            self.teach_policy = TeachPolicy(self.env.cfg, spec=self.env.geofence_spec())
            self.teach_policy.reset(self.obs, self.info)
        action = self.teach_policy.act(self.obs, self.info)
        self.obs, _reward, terminated, truncated, self.info = self.env.step(action)
        if self.policy is not None:
            from jims_mower.types import Pose

            raw = (self.info or {}).get("pose") or {}
            pose = Pose(
                float(raw.get("x", 0.0)),
                float(raw.get("y", 0.0)),
                float(raw.get("theta", 0.0)),
            )
            self.policy._stamp(self.obs, self.info, pose, explored=True)
        if terminated or truncated or self.teach_policy.done:
            self.job_state = "paused"
            self.yard_profile = self._draft_profile_from_teach()
        self._refresh_maps(force=False)
        self._refresh_cameras(force=False)
        self._record_pose()
        with self.lock:
            self._seq += 1

    def _maybe_auto_mow(self) -> None:
        if self.policy is None or self.unattended:
            return
        if self.policy.phase.value != "review":
            return
        if getattr(self.policy, "fence_unusable", False):
            return
        if getattr(self.policy, "_empty_mow_plan", lambda: False)():
            return
        if self._review_wall0 is None:
            self._review_wall0 = time.perf_counter()
            return
        if (time.perf_counter() - self._review_wall0) >= REVIEW_HOLD_WALL_S:
            self.policy.request_start_mow()

    def _live_session_card(self) -> dict[str, Any]:
        if self.policy is None:
            return dict(self.session_card)
        wall = 0.0
        if self._t0_wall:
            wall = time.perf_counter() - self._t0_wall
        return session_summary(
            self.policy,
            actual_coverage=float((self.info or {}).get("coverage_fraction") or 0.0),
            yard=str(self.config_name),
            wall_s=wall,
            info=self.info if isinstance(self.info, dict) else None,
        )

    def _capture_summary(self) -> None:
        if self.policy is None:
            return
        self.session_card = self._live_session_card()
        dest = self.out_dir / "session_summary.json"
        dest.write_text(json.dumps(self.session_card, indent=2), encoding="utf-8")

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._flush_incremental(final=True)

    def persist_session(self) -> Optional[Path]:
        """Write ObservedMap + uncut + pose + yard for a cold day-2 load."""
        if self.env is None or self.policy is None:
            return None
        self.policy.attach_to_env(self.env)
        dest = self.out_dir / "session.npz"
        return self.policy.save_session(
            dest,
            self.env._coverage,
            self.env._pose,
            scenario=self.env.scenario.name if self.env.scenario else "",
            seed=getattr(self.env, "_episode_seed", None),
        )

    def close(self) -> None:
        self.stop()
        try:
            self.persist_session()
        except Exception:
            pass
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

    def _stream_budget(self) -> dict[str, Any]:
        """Owner-view cadence. High speed coarsens so physics steps run.

        Map / mesh used to rebuild on every newly observed cell. On acre
        that is almost every explore step and drops ``--speed max`` to ~2 Hz.
        """
        speed = float(self.speed)
        map_stride = max(1, int(self.map_stride))
        mesh_stride = max(1, int(self.observed_mesh_stride))
        flush_stride = POSE_FLUSH_STRIDE
        cam_wall = CAM_WALL_S
        map_side = MAP_MAX_SIDE
        # Keep the observed-mesh vertex grid fixed for the session.
        # Switching 32↔48 mid-run rebuilds the decimation (696→1920 verts
        # on acre) and looks like a local flop even when heights are stable.
        mesh_side = MESH_MAX_SIDE
        cheap = False
        skip_coverage = False
        if speed <= 0.0:
            map_stride = max(map_stride, MAP_STRIDE_MAX)
            mesh_stride = max(mesh_stride, MESH_STRIDE_MAX)
            flush_stride = FLUSH_STRIDE_MAX
            cam_wall = CAM_WALL_MAX_S
            map_side = MAP_MAX_SIDE_FAST
            cheap = True
            skip_coverage = True
        elif speed >= 4.5:
            map_stride = max(map_stride, MAP_STRIDE_FAST)
            mesh_stride = max(mesh_stride, MESH_STRIDE_FAST)
            flush_stride = FLUSH_STRIDE_FAST
            cam_wall = CAM_WALL_FAST_S
            map_side = MAP_MAX_SIDE_FAST
            cheap = True
            skip_coverage = True
        return {
            "map_stride": map_stride,
            "mesh_stride": mesh_stride,
            "flush_stride": flush_stride,
            "cam_wall": cam_wall,
            "map_side": int(map_side),
            "mesh_side": int(mesh_side),
            "cheap": cheap,
            "skip_coverage": skip_coverage,
        }

    def _refresh_maps(self, *, force: bool) -> None:
        assert self.policy is not None
        omap = self.policy.observed
        if omap is None:
            return
        budget = self._stream_budget()
        n_obs = int(omap.observed.sum())
        step = int(self.policy.step)
        self._n_observed = n_obs
        if not force and step % int(budget["map_stride"]) != 0:
            return
        frontier_cells = [
            omap.world_to_cell(x, y)
            for x, y in self.policy._frontier_xy
            if omap.world_to_cell(x, y) is not None
        ]
        rgb = coarsen2d(omap.as_rgb(frontiers=frontier_cells), int(budget["map_side"]))
        fog = coarsen2d(fog_rgba(omap.observed), int(budget["map_side"]))
        keep = self.policy.keep_in_mask
        mowable = None
        if self.policy.global_plan is not None or self.policy.phase.value in {"review", "mow", "return_home", "charging", "complete"}:
            mowable = omap.mowable_mask(keep)
        areas = coarsen2d(omap.areas_rgb(keep, mowable=mowable), int(budget["map_side"]))
        observed_png = _png_bytes(rgb)
        fog_png = _png_bytes(fog)
        areas_png = _png_bytes(areas)
        coverage_png = b""
        phase = self.policy.phase.value
        want_coverage = (
            force
            or not budget["skip_coverage"]
            or phase in {"mow", "return_home", "complete"}
        )
        if want_coverage and self.env is not None and hasattr(self.env, "_coverage"):
            try:
                cov = coarsen2d(coverage_to_rgb(self.env._coverage.as_float()), int(budget["map_side"]))
                coverage_png = _png_bytes(cov)
            except Exception:
                coverage_png = b""
        mesh_json = b""
        if force or step % int(budget["mesh_stride"]) == 0:
            mesh = mesh_from_observed(omap, stride=1, max_side=int(budget["mesh_side"]))
            mesh_json = json.dumps(
                _compact_mesh_payload(mesh, cheap=bool(budget["cheap"])),
                separators=(",", ":"),
            ).encode("utf-8")
            self._n_mesh_builds += 1
        self._n_map_builds += 1
        with self.lock:
            self.observed_png = observed_png
            self.fog_png = fog_png
            self.areas_png = areas_png
            if coverage_png:
                self.coverage_png = coverage_png
            self._map_seq += 1
            if mesh_json:
                self.observed_mesh_json = mesh_json
                self._mesh_seq += 1

    def _refresh_cameras(self, *, force: bool) -> None:
        now = time.perf_counter()
        wall = float(self._stream_budget()["cam_wall"])
        if not force and (now - self._last_cam_wall) < wall:
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
        fault = self._current_fault()
        radio_path = radio_path_for("idle", self.job_state)
        if policy is None:
            idle_overlay = build_path_overlay(
                phase="idle",
                job_state=self.job_state,
                pose=_pose_row((self.info or {}).get("pose")) if self.info else None,
                poses=self.poses,
            )
            return {
                "schema": LIVE_SCHEMA,
                "live": True,
                "seq": 0,
                "step": 0,
                "phase": "idle",
                "phase_label": "IDLE",
                "job_state": self.job_state,
                "owner_copy": owner_copy_for(
                    self.job_state,
                    "idle",
                    fault,
                    hw_estop=self._hw_estop_latched(),
                    pairing_state=self.pairing.state,
                    require_pair=self.require_pair,
                ),
                "radio_path": radio_path,
                "faults": self._faults_payload(fault),
                "hw_estop": self._hw_estop_latched(),
                "estop_kind": "hardware" if self._hw_estop_latched() else None,
                "paired": bool(self.paired),
                "pairing": self.pairing.as_info(),
                "require_pair": bool(self.require_pair),
                "path_overlay": idle_overlay,
                "mode_banner": idle_overlay["mode"],
                "mission": mission_from_phase("idle", self.job_state),
                "waypoint_index": 0,
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
        src = policy.profile if policy.profile is not None else self.yard_profile
        if src is not None and len(src.keep_in) >= 3:
            keep_in = [list(pt) for pt in src.keep_in]
            keep_out = [[list(pt) for pt in hole] for hole in (src.keep_out or [])]
        phase = "teach" if self.job_state == "teach" else status["phase"]
        phase_label = "TEACH" if phase == "teach" else status["phase_label"]
        cam_name = self.camera_names[0] if self.camera_names else "front"
        n_obs = self._n_observed
        if policy.observed is not None:
            n_obs = int(policy.observed.observed.sum())
        live_card = self._live_session_card()
        card = self.session_card if (self.done and self.session_card) else live_card
        fence_unusable = bool(getattr(policy, "fence_unusable", False))
        if status["phase"] == "review":
            if getattr(policy, "_keep_in_too_small", lambda: False)():
                fence_unusable = True
            if int(status.get("n_waypoints") or 0) < 2 and int(status.get("planned_mowable_cells") or 0) <= 0:
                fence_unusable = True
        if (
            self.job_state == "idle"
            and self.owner_taught
            and self.yard_profile is not None
            and keep_in_usable(
                self.yard_profile.keep_in,
                float(self.yard_profile.width_m),
                float(self.yard_profile.height_m),
            )
        ):
            fence_unusable = False
        teach_trail = []
        if self.teach_policy is not None and self.teach_policy.trail:
            teach_trail = list(self.teach_policy.trail)
        path_overlay = build_path_overlay(
            phase=phase,
            job_state=self.job_state,
            pose=pose,
            poses=self.poses,
            trail=teach_trail or None,
            plan=plan,
            explore=explore,
            frontiers=frontiers,
            n_waypoints=int(status["n_waypoints"]),
            waypoint_index=int(status.get("waypoint_index") or 0),
        )
        return {
            "schema": LIVE_SCHEMA,
            "live": True,
            "seq": self._seq,
            "map_seq": self._map_seq,
            "cam_seq": self._cam_seq,
            "step": status["step"],
            "phase": phase,
            "phase_label": phase_label,
            "job_state": self.job_state,
            "owner_copy": owner_copy_for(
                self.job_state,
                phase,
                fault,
                taught=self.owner_taught,
                fence_unusable=fence_unusable,
                done=self.done,
                hw_estop=self._hw_estop_latched(),
                pairing_state=self.pairing.state,
                require_pair=self.require_pair,
                explore_reason=status.get("explore_reason") if isinstance(status.get("explore_reason"), dict) else None,
                charge_state=str(status.get("charge_state") or ""),
                return_kind=str(status.get("return_kind") or ""),
            ),
            "radio_path": radio_path_for(phase, self.job_state),
            "taught": bool(self.owner_taught),
            "first_run": bool(self.first_run),
            "needs_teach": bool(self.first_run and not self.owner_taught),
            "yard_path": str(self.yard_path) if self.yard_path else "",
            "yard_saved": bool(self.yard_path is not None and Path(self.yard_path).is_file()),
            "faults": self._faults_payload(fault),
            "paired": bool(self.paired),
            "pairing": self.pairing.as_info(),
            "require_pair": bool(self.require_pair),
            "yards": list(LIVE_YARDS),
            "yard": str(self.config_name),
            "width_m": float(self.env.cfg.world.width_m) if self.env is not None else 16.0,
            "height_m": float(self.env.cfg.world.height_m) if self.env is not None else 12.0,
            "can_start_mow": (status["phase"] in {"review", "explore", "complete", "charging"} or bool(status.get("planned_mowable_cells"))) and not fence_unusable,
            "can_reexplore": status["phase"] in {"review", "mow", "return_home", "complete", "charging"},
            "can_explore": True,
            "can_mow": not fence_unusable,
            "can_return": self.job_state in {"running", "paused", "hold"} or status["phase"] not in {"idle", "complete", ""},
            "explore_reason": status.get("explore_reason") or {},
            "full_explore": bool(status.get("full_explore")),
            "charge_state": status.get("charge_state") or "",
            "return_kind": status.get("return_kind") or "",
            "area_legend": status.get("area_legend") or [],
            "needs_reteach": fence_unusable,
            "fence_unusable": fence_unusable,
            "pose": pose,
            "map_pct": float(status["map_completion"]),
            "cut_pct": float(status["actual_coverage_fraction"]),
            "coverage_pct": float(status["actual_coverage_fraction"]),
            "coverage_source": str(status.get("coverage_source") or (self.info or {}).get("coverage_source") or "gym_grid"),
            "world_cut_pct": float(status.get("world_coverage_fraction") or 0.0),
            "planned_pct": float(status.get("planned_coverage_fraction") or 0.0),
            "reachable": int(status.get("reachable_mowable_cells") or 0),
            "unreachable": int(status.get("unreachable_mowable_cells") or 0),
            "skips": int(status.get("skipped_global") or 0),
            "duration_s": float(card.get("duration_s") or 0.0),
            "wall_s": float(card.get("wall_s") or 0.0),
            "n_frontiers": int(status["n_frontiers"]),
            "blocked_cells": int(status.get("blocked_cells") or 0),
            "blocked_frontiers": int(status.get("blocked_frontiers") or 0),
            "n_blockages": int(status.get("n_blockages") or 0),
            "n_observed": n_obs,
            "mesh_seq": self._mesh_seq,
            "n_waypoints": int(status["n_waypoints"]),
            "waypoint_index": int(status.get("waypoint_index") or 0),
            "mission": mission_from_phase(phase, self.job_state),
            "frontiers": frontiers,
            "explore": explore,
            "plan": plan,
            "path_overlay": path_overlay,
            "mode_banner": path_overlay["mode"],
            "keep_in": keep_in,
            "keep_out": keep_out,
            "trimmer_on": bool((self.info or {}).get("trimmer_enabled")),
            "trimmer_allowed": bool(status["trimmer_allowed"]),
            "estop": bool(self.estop or status["phase"] == "safe"),
            "hw_estop": self._hw_estop_latched(),
            "estop_kind": (
                "both"
                if self._hw_estop_latched() and (self.estop or status["phase"] == "safe")
                else "hardware"
                if self._hw_estop_latched()
                else "software"
                if (self.estop or status["phase"] == "safe")
                else None
            ),
            "safe_mode": getattr(policy.safe, "mode", "run"),
            "observed_url": f"/api/live/observed.png?v={self._map_seq}",
            "fog_url": f"/api/live/fog.png?v={self._map_seq}",
            "observed_mesh_url": f"/api/live/observed_mesh.json?v={self._mesh_seq}",
            "coverage_url": f"/api/live/coverage.png?v={self._map_seq}",
            "areas_url": f"/api/live/areas.png?v={self._map_seq}",
            "cam_url": f"/api/live/cam/{cam_name}?v={self._cam_seq}",
            "cameras": list(self.camera_names),
            "done": bool(self.done),
            "session_summary": card,
            "owner_mode": "observed_terrain",
            "speed": self.speed,
            "speed_label": speed_label(self.speed),
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
        if self.coverage_png:
            (maps_dir / "coverage.png").write_bytes(self.coverage_png)
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
        budget = self._stream_budget()
        cheap = bool(budget["cheap"]) and not final
        with self.lock:
            poses = list(self.poses)
            observed_png = self.observed_png
            fog_png = self.fog_png
            observed_mesh_json = self.observed_mesh_json
            coverage_png = self.coverage_png
            card = dict(self.session_card)
        pose_kwargs: dict[str, Any] = {"separators": (",", ":")} if cheap else {"indent": 2}
        (dest / "poses.json").write_text(
            json.dumps({"schema": VIEWER_SCHEMA, "poses": poses}, **pose_kwargs),
            encoding="utf-8",
        )
        if observed_png:
            (maps_dir / "observed.png").write_bytes(observed_png)
        if fog_png:
            (maps_dir / "fog.png").write_bytes(fog_png)
        if observed_mesh_json:
            (maps_dir / "observed_mesh.json").write_bytes(observed_mesh_json)
        if coverage_png:
            (maps_dir / "coverage.png").write_bytes(coverage_png)
        live_card = self._live_session_card()
        if self.done:
            if not card:
                self.session_card = live_card
            card = dict(self.session_card or live_card)
        else:
            card = live_card
        timeline = mission_timeline(
            self.policy,
            actual_coverage=float((self.info or {}).get("coverage_fraction") or 0.0),
        )
        card.setdefault("yard", str(self.config_name))
        timeline["session_summary"] = card
        card_kwargs: dict[str, Any] = {"separators": (",", ":")} if cheap else {"indent": 2}
        (dest / "session_summary.json").write_text(json.dumps(card, **card_kwargs), encoding="utf-8")
        (dest / "mission.json").write_text(json.dumps(timeline, **card_kwargs), encoding="utf-8")
        if self.policy.profile is not None:
            if not self.policy.profile.keep_out and self.env is not None:
                self.policy.profile.keep_out = keepouts_from_env(self.env)
            write_yard_profile(dest / "profile.json", self.policy.profile)
        dump_cams = final or (
            not cheap
            and self.policy.step % self.cam_stride == 0
            and self.obs.get("cameras")
        )
        if dump_cams:
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
            if coverage_png:
                manifest["maps"]["coverage"] = "maps/coverage.png"
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
        session.unattended = True
        session.speed = 0.0
        session.job_state = "running"
        session._t0_wall = time.perf_counter()
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
    print(f"Live owner session {session.config_name} @ {session.speed or 'max'}× → {url}")
    print("Idle until Start. Speed 1× 2× 5× max. MAP READY waits 2s or Start mow.")
    print("Owner view: growing observed terrain + fog. Toggle true elev for god-view.")
    print("Physics uses true height; owner/control use ObservedMap.")
    print(f"{summary.get('owner_copy')}  map {float(summary.get('map_pct') or 0.0):.1%}")
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

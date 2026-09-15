"""Sim / episode / in-memory backends for the owner-app API."""

from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any, Optional, Protocol, Union

import numpy as np

from jims_mower.constants import (
    APP_COMMANDS,
    APP_STATUS_SCHEMA,
    COVERAGE_MAP_SCHEMA,
    MESH_MAP_SCHEMA,
    SAFE_MODES,
)
from jims_mower.episode import EpisodeReader
from jims_mower.geofence import allowed_xy
from jims_mower.safe_state import SafeStateMachine
from jims_mower.yard_profile import (
    HomePose,
    RadioPrefs,
    YardProfile,
    YardProfileError,
    default_yard_profile,
    load_yard_profile,
    save_yard_profile,
    yard_profile_from_geofence,
)


class AppBackend(Protocol):
    def status(self) -> dict[str, Any]: ...
    def get_yard(self) -> dict[str, Any]: ...
    def put_yard(self, profile: YardProfile) -> dict[str, Any]: ...
    def command(self, cmd: str, *, reason: str = "") -> dict[str, Any]: ...
    def mesh(self) -> dict[str, Any]: ...
    def coverage(self) -> dict[str, Any]: ...
    def tick(self) -> dict[str, Any]: ...
    def close(self) -> None: ...


def _pose_dict(x: float, y: float, theta: float) -> dict[str, float]:
    return {"x": float(x), "y": float(y), "theta": float(theta)}


def _radio_status(radio: RadioPrefs, *, rssi: int = -88) -> dict[str, Any]:
    primary = radio.primary
    if primary == "lora" and not radio.lora_enabled:
        primary = "bluetooth" if radio.bluetooth else "wifi"
    if primary == "wifi" and not radio.wifi_enabled:
        primary = "lora" if radio.lora_enabled else "bluetooth"
    ok = True
    if primary == "bluetooth":
        ok = bool(radio.bluetooth)
    elif primary == "wifi":
        ok = bool(radio.wifi_enabled)
    elif primary == "lora":
        ok = bool(radio.lora_enabled)
    return {
        "link": primary if ok else "none",
        "ok": bool(ok),
        "rssi": int(rssi),
        "bluetooth": {"paired": bool(radio.bluetooth), "ok": bool(radio.bluetooth)},
        "wifi": {"enabled": bool(radio.wifi_enabled), "ssid": radio.wifi_ssid, "ok": bool(radio.wifi_enabled)},
        "lora": {
            "enabled": bool(radio.lora_enabled),
            "channel": int(radio.lora_channel),
            "ok": bool(radio.lora_enabled),
        },
    }


def _downsample(grid: np.ndarray, max_side: int = 48) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros((1, 1), dtype=np.float32)
    rows, cols = arr.shape
    if max(rows, cols) <= max_side:
        return arr
    scale = max(rows, cols) / float(max_side)
    nr = max(1, int(round(rows / scale)))
    nc = max(1, int(round(cols / scale)))
    ys = (np.linspace(0, rows - 1, nr)).astype(int)
    xs = (np.linspace(0, cols - 1, nc)).astype(int)
    return arr[ys[:, None], xs[None, :]]


def _coverage_payload(grid: np.ndarray, yard: YardProfile) -> dict[str, Any]:
    small = _downsample(grid)
    return {
        "schema": COVERAGE_MAP_SCHEMA,
        "width_m": float(yard.width_m),
        "height_m": float(yard.height_m),
        "resolution_m": float(yard.resolution_m),
        "rows": int(small.shape[0]),
        "cols": int(small.shape[1]),
        "values": [float(v) for v in small.reshape(-1)],
        "legend": {"cut": 1.0, "uncut": 0.0, "non_grass": -1.0},
    }


def _mesh_from_grid(grid: np.ndarray, yard: YardProfile, *, path: str = "") -> dict[str, Any]:
    small = _downsample(np.asarray(grid, dtype=np.float32), max_side=32)
    rows, cols = small.shape
    cells: list[list[int]] = []
    for r in range(rows):
        for c in range(cols):
            if float(small[r, c]) >= 0.45:
                cells.append([r, c])
    return {
        "schema": MESH_MAP_SCHEMA,
        "path": path or yard.mesh_path,
        "width_m": float(yard.width_m),
        "height_m": float(yard.height_m),
        "resolution_m": float(yard.width_m) / max(cols, 1),
        "rows": int(rows),
        "cols": int(cols),
        "occupied": cells,
        "viewer": "/#/map",
        "ux_a_href": "/static/ux_a/index.html",
        "note": "2D occupancy mesh stub. UX-A three.js viewer deep-links via ux_a_href when present.",
        "not_slam": True,
    }


def _load_mesh_file(mesh_path: str, root: Optional[Path]) -> Optional[dict[str, Any]]:
    if not mesh_path:
        return None
    candidates = [Path(mesh_path)]
    if root is not None:
        candidates.append(Path(root) / mesh_path)
    for cand in candidates:
        if cand.is_file():
            import json

            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
            if isinstance(data, dict):
                return data
    return None


class MemoryBackend:
    """In-memory kinematic stub used by tests and `--backend memory`."""

    def __init__(
        self,
        yard: Optional[YardProfile] = None,
        *,
        yard_path: Optional[Path] = None,
    ) -> None:
        self._lock = threading.Lock()
        self.yard = yard or default_yard_profile()
        self.yard_path = Path(yard_path) if yard_path else None
        self.pose = dict(self.yard.home.as_dict())
        self.mission = "idle"
        self.safe = SafeStateMachine()
        self.soc = 0.92
        self.temp_c = 42.0
        self.faults: list[dict[str, str]] = []
        self.hours_mowed = 1.4
        rows = max(1, int(round(self.yard.height_m / self.yard.resolution_m)))
        cols = max(1, int(round(self.yard.width_m / self.yard.resolution_m)))
        self._coverage = np.zeros((rows, cols), dtype=np.float32)
        self._occ = np.zeros((rows, cols), dtype=np.float32)
        self._paint_keepout()

    def _paint_keepout(self) -> None:
        spec = self.yard.to_geofence_spec()
        rows, cols = self._coverage.shape
        res = self.yard.resolution_m
        for r in range(rows):
            y = (r + 0.5) * res
            for c in range(cols):
                x = (c + 0.5) * res
                if not allowed_xy(x, y, spec):
                    self._coverage[r, c] = -1.0
                    self._occ[r, c] = 1.0

    def _cell(self, x: float, y: float) -> Optional[tuple[int, int]]:
        col = int(x / self.yard.resolution_m)
        row = int(y / self.yard.resolution_m)
        if 0 <= row < self._coverage.shape[0] and 0 <= col < self._coverage.shape[1]:
            return row, col
        return None

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_unlocked()

    def _status_unlocked(self) -> dict[str, Any]:
        grass = self._coverage >= 0.0
        cut = grass & (self._coverage >= 0.5)
        total = int(grass.sum())
        coverage_pct = 100.0 * (float(cut.sum()) / total) if total else 0.0
        mode = self.safe.mode if self.safe.mode in SAFE_MODES else "run"
        mission = self.mission
        if mode == "estop":
            mission = "estop"
        return {
            "schema": APP_STATUS_SCHEMA,
            "pose": _pose_dict(self.pose["x"], self.pose["y"], self.pose["theta"]),
            "battery": {
                "soc": float(self.soc),
                "temp_c": float(self.temp_c),
                "not_a_power_trace": True,
            },
            "state": {
                "mission": mission,
                "machine": mode,
                "help_requested": bool(self.safe.command().help_requested),
                "reason": self.safe.last_reason,
            },
            "radio": _radio_status(self.yard.radio),
            "faults": list(self.faults),
            "coverage_pct": coverage_pct,
            "hours_mowed": float(self.hours_mowed),
            "yard": self.yard.name,
            "not_a_benchmark": True,
        }

    def get_yard(self) -> dict[str, Any]:
        with self._lock:
            return self.yard.as_dict()

    def put_yard(self, profile: YardProfile) -> dict[str, Any]:
        with self._lock:
            self.yard = profile
            rows = max(1, int(round(profile.height_m / profile.resolution_m)))
            cols = max(1, int(round(profile.width_m / profile.resolution_m)))
            if self._coverage.shape != (rows, cols):
                self._coverage = np.zeros((rows, cols), dtype=np.float32)
                self._occ = np.zeros((rows, cols), dtype=np.float32)
            self._paint_keepout()
            if self.yard_path is not None:
                save_yard_profile(profile, self.yard_path)
            return profile.as_dict()

    def command(self, cmd: str, *, reason: str = "") -> dict[str, Any]:
        key = str(cmd or "").strip().lower()
        if key not in APP_COMMANDS:
            raise YardProfileError(f"unknown command {cmd!r}; expected one of {APP_COMMANDS}")
        with self._lock:
            if key == "estop":
                self.safe.request_estop(reason or "owner estop")
                self.mission = "estop"
                self.faults = [{"code": "ESTOP", "detail": reason or "owner estop"}]
            elif key == "start":
                if self.safe.mode == "estop":
                    self.safe.clear()
                self.faults = []
                self.mission = "mowing"
            elif key == "stop":
                if self.safe.mode != "estop":
                    self.mission = "idle"
            elif key == "return":
                if self.safe.mode == "estop":
                    raise YardProfileError("cannot return while ESTOP is latched")
                self.mission = "returning"
            elif key == "teach":
                if self.safe.mode == "estop":
                    raise YardProfileError("cannot teach while ESTOP is latched")
                self.mission = "teach"
            return self._status_unlocked()

    def mesh(self) -> dict[str, Any]:
        with self._lock:
            payload = _mesh_from_grid(self._occ, self.yard)
            extra = _load_mesh_file(self.yard.mesh_path, self.yard_path.parent if self.yard_path else None)
            if extra:
                payload["file"] = extra
            return payload

    def coverage(self) -> dict[str, Any]:
        with self._lock:
            return _coverage_payload(self._coverage, self.yard)

    def tick(self) -> dict[str, Any]:
        with self._lock:
            if self.safe.mode == "estop":
                return self._status_unlocked()
            if self.mission == "mowing":
                self.pose["theta"] = float(self.pose["theta"])
                self.pose["x"] = float(self.pose["x"]) + 0.18 * math.cos(self.pose["theta"])
                self.pose["y"] = float(self.pose["y"]) + 0.18 * math.sin(self.pose["theta"])
                spec = self.yard.to_geofence_spec()
                if not allowed_xy(self.pose["x"], self.pose["y"], spec):
                    self.pose["theta"] += 0.65
                    self.pose["x"] = float(np.clip(self.pose["x"], 0.4, self.yard.width_m - 0.4))
                    self.pose["y"] = float(np.clip(self.pose["y"], 0.4, self.yard.height_m - 0.4))
                cell = self._cell(self.pose["x"], self.pose["y"])
                if cell is not None and self._coverage[cell] >= 0.0:
                    self._coverage[cell] = 1.0
                self.soc = max(0.05, self.soc - 0.0008)
                self.hours_mowed += 0.002
            elif self.mission == "returning":
                hx, hy = self.yard.home.x, self.yard.home.y
                dx, dy = hx - self.pose["x"], hy - self.pose["y"]
                dist = math.hypot(dx, dy)
                if dist < 0.25:
                    self.pose = dict(self.yard.home.as_dict())
                    self.mission = "idle"
                else:
                    self.pose["theta"] = math.atan2(dy, dx)
                    step = min(0.25, dist)
                    self.pose["x"] += step * math.cos(self.pose["theta"])
                    self.pose["y"] += step * math.sin(self.pose["theta"])
            return self._status_unlocked()

    def close(self) -> None:
        return None


class SimBackend:
    """Live ``MowerEnv`` (demo env) plus a YardProfile overlay."""

    def __init__(
        self,
        *,
        config: Optional[str] = "geofence_movers",
        seed: int = 7,
        cameras: int = 4,
        yard: Optional[YardProfile] = None,
        yard_path: Optional[Path] = None,
    ) -> None:
        from jims_mower.env import MowerEnv
        from jims_mower.planning import TerrainPolicy
        from jims_mower.scenarios import load_source

        self._lock = threading.Lock()
        cfg, scenario = load_source(config)
        if cameras is not None:
            cfg.sensors.camera_count = cameras
            cfg.sensors.cameras = []
        self.env = MowerEnv(config=cfg, scenario=scenario, render_mode=None)
        self._obs, self._info = self.env.reset(seed=seed)
        self._policy = TerrainPolicy(self.env.cfg)
        self._policy.reset(self._obs, self._info)
        spec = self.env.geofence_spec()
        self.yard = yard or yard_profile_from_geofence(
            spec,
            name=str(self._info.get("scenario") or (scenario.name if scenario else "sim")),
            width_m=self.env.cfg.world.width_m,
            height_m=self.env.cfg.world.height_m,
            resolution_m=self.env.cfg.world.resolution_m,
            home=HomePose(
                x=float(self._info["pose"]["x"]),
                y=float(self._info["pose"]["y"]),
                theta=float(self._info["pose"]["theta"]),
            ),
        )
        self.yard_path = Path(yard_path) if yard_path else None
        self.mission = "idle"
        self.safe = SafeStateMachine()
        self.faults: list[dict[str, str]] = []
        self.hours_mowed = 0.0

    def _pose(self) -> dict[str, float]:
        pose = self._info.get("pose") or {}
        return _pose_dict(float(pose.get("x", 0.0)), float(pose.get("y", 0.0)), float(pose.get("theta", 0.0)))

    def _status_unlocked(self) -> dict[str, Any]:
        mode = self.safe.mode
        mission = self.mission
        if mode == "estop":
            mission = "estop"
        soc = float(self._info.get("battery_soc", 0.9))
        temp = float(self._info.get("thermal_c", 42.0))
        faults = list(self.faults)
        if self._info.get("tipover"):
            faults.append({"code": "TIP", "detail": "tip-over advice"})
        if self._info.get("drain_drop"):
            faults.append({"code": "DRAIN", "detail": "wheel in channel"})
        return {
            "schema": APP_STATUS_SCHEMA,
            "pose": self._pose(),
            "battery": {"soc": soc, "temp_c": temp, "not_a_power_trace": True},
            "state": {
                "mission": mission,
                "machine": mode,
                "help_requested": bool(self.safe.command().help_requested),
                "reason": self.safe.last_reason or self._info.get("terrain_reason"),
                "terrain_advice": self._info.get("terrain_advice"),
                "living_advice": self._info.get("living_advice"),
            },
            "radio": _radio_status(self.yard.radio),
            "faults": faults,
            "coverage_pct": 100.0 * float(self._info.get("coverage_fraction") or 0.0),
            "hours_mowed": float(self.hours_mowed),
            "yard": self.yard.name,
            "not_a_benchmark": True,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_unlocked()

    def get_yard(self) -> dict[str, Any]:
        with self._lock:
            return self.yard.as_dict()

    def put_yard(self, profile: YardProfile) -> dict[str, Any]:
        with self._lock:
            self.yard = profile
            if self.yard_path is not None:
                save_yard_profile(profile, self.yard_path)
            return profile.as_dict()

    def command(self, cmd: str, *, reason: str = "") -> dict[str, Any]:
        key = str(cmd or "").strip().lower()
        if key not in APP_COMMANDS:
            raise YardProfileError(f"unknown command {cmd!r}; expected one of {APP_COMMANDS}")
        with self._lock:
            if key == "estop":
                self.safe.request_estop(reason or "owner estop")
                self.mission = "estop"
                self.faults = [{"code": "ESTOP", "detail": reason or "owner estop"}]
            elif key == "start":
                if self.safe.mode == "estop":
                    self.safe.clear()
                self.faults = []
                self.mission = "mowing"
            elif key == "stop":
                if self.safe.mode != "estop":
                    self.mission = "idle"
            elif key == "return":
                if self.safe.mode == "estop":
                    raise YardProfileError("cannot return while ESTOP is latched")
                self.mission = "returning"
            elif key == "teach":
                if self.safe.mode == "estop":
                    raise YardProfileError("cannot teach while ESTOP is latched")
                self.mission = "teach"
            return self._status_unlocked()

    def mesh(self) -> dict[str, Any]:
        with self._lock:
            occ = np.asarray(self._obs.get("occupancy"), dtype=np.float32)
            payload = _mesh_from_grid(occ, self.yard)
            extra = _load_mesh_file(self.yard.mesh_path, self.yard_path.parent if self.yard_path else None)
            if extra:
                payload["file"] = extra
            return payload

    def coverage(self) -> dict[str, Any]:
        with self._lock:
            grid = np.asarray(self._obs.get("coverage"), dtype=np.float32)
            return _coverage_payload(grid, self.yard)

    def tick(self) -> dict[str, Any]:
        with self._lock:
            if self.safe.mode == "estop" or self.mission in {"idle", "teach"}:
                return self._status_unlocked()
            if self.mission == "returning":
                pose = self._info.get("pose") or {}
                hx, hy = self.yard.home.x, self.yard.home.y
                heading = math.atan2(hy - float(pose.get("y", 0.0)), hx - float(pose.get("x", 0.0)))
                # Crude equal-speed creep toward home; planner still owns mowing.
                action = np.array([0.35, 0.35, 0.0], dtype=np.float32)
                if abs((heading - float(pose.get("theta", 0.0)) + math.pi) % (2 * math.pi) - math.pi) > 0.4:
                    action = np.array([0.25, -0.25, 0.0], dtype=np.float32)
            else:
                action = self._policy.act(self._obs, self._info)
            action = self.safe.apply(action)
            obs, _reward, terminated, truncated, info = self.env.step(action)
            self._obs, self._info = obs, info
            self.hours_mowed += float(self.env.cfg.dt) / 3600.0
            if self.mission == "returning":
                pose = info.get("pose") or {}
                if math.hypot(float(pose.get("x", 0.0)) - self.yard.home.x, float(pose.get("y", 0.0)) - self.yard.home.y) < 0.6:
                    self.mission = "idle"
            if terminated or truncated:
                self.mission = "idle"
                if info.get("tipover") or info.get("drain_drop"):
                    self.faults.append({"code": "HALT", "detail": str(info.get("terrain_reason") or "episode ended")})
            return self._status_unlocked()

    def close(self) -> None:
        with self._lock:
            self.env.close()


class EpisodeBackend:
    """Replay a ``jims-mower-record`` directory as the live pose / maps."""

    def __init__(
        self,
        episode_dir: Union[str, Path],
        *,
        yard: Optional[YardProfile] = None,
        yard_path: Optional[Path] = None,
    ) -> None:
        self._lock = threading.Lock()
        self.reader = EpisodeReader(episode_dir)
        self.index = 0
        self.mission = "idle"
        self.safe = SafeStateMachine()
        self.faults: list[dict[str, str]] = []
        self.yard_path = Path(yard_path) if yard_path else None
        self.yard = yard or self._yard_from_episode()
        self.hours_mowed = 0.0

    def _records(self) -> list[dict[str, Any]]:
        recs: list[dict[str, Any]] = []
        if self.reader.reset_obs is not None:
            recs.append({"obs": self.reader.reset_obs, "info": self.reader.reset_info, "kind": "reset"})
        recs.extend(self.reader.steps)
        return recs

    def _current(self) -> dict[str, Any]:
        recs = self._records()
        if not recs:
            return {"obs": {}, "info": {}}
        idx = min(max(self.index, 0), len(recs) - 1)
        return recs[idx]

    def _yard_from_episode(self) -> YardProfile:
        info = self.reader.reset_info or {}
        fence = info.get("geofence_spec") or {}
        keep_in = [tuple(p) for p in (fence.get("keep_in") or info.get("geofence") or [])]
        keep_out = [[tuple(p) for p in poly] for poly in (fence.get("keep_out") or [])]
        pose = info.get("pose") or {}
        manifest = self.reader.manifest or {}
        cfg = manifest.get("config") or {}
        world = cfg.get("world") or {}
        return YardProfile(
            name=str(info.get("scenario") or manifest.get("scenario") or "episode"),
            width_m=float(world.get("width_m") or 16.0),
            height_m=float(world.get("height_m") or 12.0),
            resolution_m=float(world.get("resolution_m") or 0.20),
            home=HomePose(
                x=float(pose.get("x", 1.0)),
                y=float(pose.get("y", 1.0)),
                theta=float(pose.get("theta", 0.0)),
            ),
            keep_in=keep_in,
            keep_out=keep_out,
        )

    def _status_unlocked(self) -> dict[str, Any]:
        rec = self._current()
        info = rec.get("info") if isinstance(rec.get("info"), dict) else {}
        obs = rec.get("obs") if isinstance(rec.get("obs"), dict) else {}
        pose = info.get("pose") or {}
        if not pose and "pose" in obs:
            arr = np.asarray(obs["pose"], dtype=np.float32).reshape(-1)
            pose = {"x": float(arr[0]), "y": float(arr[1]), "theta": float(arr[2]) if arr.size > 2 else 0.0}
        mode = self.safe.mode
        mission = self.mission
        if mode == "estop":
            mission = "estop"
        return {
            "schema": APP_STATUS_SCHEMA,
            "pose": _pose_dict(float(pose.get("x", 0.0)), float(pose.get("y", 0.0)), float(pose.get("theta", 0.0))),
            "battery": {
                "soc": float(info.get("battery_soc", 0.88)),
                "temp_c": float(info.get("thermal_c", 41.0)),
                "not_a_power_trace": True,
            },
            "state": {
                "mission": mission,
                "machine": mode,
                "help_requested": bool(self.safe.command().help_requested),
                "reason": self.safe.last_reason,
                "terrain_advice": info.get("terrain_advice"),
            },
            "radio": _radio_status(self.yard.radio),
            "faults": list(self.faults),
            "coverage_pct": 100.0 * float(info.get("coverage_fraction") or 0.0),
            "hours_mowed": float(self.hours_mowed),
            "yard": self.yard.name,
            "episode": str(self.reader.episode_dir),
            "not_a_benchmark": True,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_unlocked()

    def get_yard(self) -> dict[str, Any]:
        with self._lock:
            return self.yard.as_dict()

    def put_yard(self, profile: YardProfile) -> dict[str, Any]:
        with self._lock:
            self.yard = profile
            if self.yard_path is not None:
                save_yard_profile(profile, self.yard_path)
            return profile.as_dict()

    def command(self, cmd: str, *, reason: str = "") -> dict[str, Any]:
        key = str(cmd or "").strip().lower()
        if key not in APP_COMMANDS:
            raise YardProfileError(f"unknown command {cmd!r}; expected one of {APP_COMMANDS}")
        with self._lock:
            if key == "estop":
                self.safe.request_estop(reason or "owner estop")
                self.mission = "estop"
                self.faults = [{"code": "ESTOP", "detail": reason or "owner estop"}]
            elif key == "start":
                if self.safe.mode == "estop":
                    self.safe.clear()
                self.faults = []
                self.mission = "mowing"
            elif key == "stop":
                if self.safe.mode != "estop":
                    self.mission = "idle"
            elif key == "return":
                if self.safe.mode == "estop":
                    raise YardProfileError("cannot return while ESTOP is latched")
                self.mission = "returning"
                self.index = max(len(self._records()) - 1, 0)
                self.mission = "idle"
            elif key == "teach":
                if self.safe.mode == "estop":
                    raise YardProfileError("cannot teach while ESTOP is latched")
                self.mission = "teach"
            return self._status_unlocked()

    def mesh(self) -> dict[str, Any]:
        with self._lock:
            obs = self._current().get("obs") or {}
            occ = np.asarray(obs.get("occupancy") if obs.get("occupancy") is not None else np.zeros((8, 8)), dtype=np.float32)
            return _mesh_from_grid(occ, self.yard)

    def coverage(self) -> dict[str, Any]:
        with self._lock:
            obs = self._current().get("obs") or {}
            grid = np.asarray(obs.get("coverage") if obs.get("coverage") is not None else np.zeros((8, 8)), dtype=np.float32)
            return _coverage_payload(grid, self.yard)

    def tick(self) -> dict[str, Any]:
        with self._lock:
            if self.mission == "mowing" and self.safe.mode != "estop":
                recs = self._records()
                if self.index + 1 < len(recs):
                    self.index += 1
                    self.hours_mowed += 0.002
                else:
                    self.mission = "idle"
            return self._status_unlocked()

    def close(self) -> None:
        return None


def make_backend(
    *,
    kind: str = "sim",
    config: Optional[str] = "geofence_movers",
    episode: Optional[Union[str, Path]] = None,
    yard: Optional[Union[str, Path, YardProfile]] = None,
    yard_path: Optional[Path] = None,
    seed: int = 7,
    cameras: int = 4,
) -> AppBackend:
    profile: Optional[YardProfile] = None
    persist = Path(yard_path) if yard_path else None
    if isinstance(yard, YardProfile):
        profile = yard
    elif yard is not None:
        persist = persist or Path(yard)
        profile = load_yard_profile(yard)
    if episode or kind == "episode":
        if not episode:
            raise YardProfileError("--episode is required for the episode backend")
        return EpisodeBackend(episode, yard=profile, yard_path=persist)
    if kind == "memory":
        return MemoryBackend(profile, yard_path=persist)
    if kind not in {"sim", "demo"}:
        raise YardProfileError(f"unknown backend {kind!r}")
    return SimBackend(config=config, seed=seed, cameras=cameras, yard=profile, yard_path=persist)

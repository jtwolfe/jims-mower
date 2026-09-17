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
    MESH_SCHEMA,
    RADIO_SCHEMA,
    SAFE_MODES,
    VIEWER_SCHEMA,
)
from jims_mower.episode import EpisodeReader
from jims_mower.geofence import allowed_xy
from jims_mower.mesh import mesh_from_elevation, mesh_to_payload
from jims_mower.pairing import PairingMachine
from jims_mower.profile import RadioPrefs
from jims_mower.safe_state import SafeStateMachine
from jims_mower.pack import GYM_STUB_CAPACITY_WH, battery_status_block
from jims_mower.schedule import (
    Clock,
    ScheduleHook,
    gates_from_owner_state,
    rain_from_weather,
)
from jims_mower.notify import NotificationLog
from jims_mower.ota import ota_status
from jims_mower.yards import YardStore
from jims_mower.yard_profile import (
    YardProfile,
    YardProfileError,
    default_yard_profile,
    load_yard_profile,
    radio_prefs,
    save_yard_profile,
    yard_profile_from_geofence,
)


class AppBackend(Protocol):
    def status(self) -> dict[str, Any]: ...
    def get_yard(self) -> dict[str, Any]: ...
    def put_yard(self, profile: YardProfile) -> dict[str, Any]: ...
    def command(self, cmd: str, *, reason: str = "", **kwargs: Any) -> dict[str, Any]: ...
    def mesh(self) -> dict[str, Any]: ...
    def coverage(self) -> dict[str, Any]: ...
    def tick(self) -> dict[str, Any]: ...
    def close(self) -> None: ...


def _pose_dict(
    x: float,
    y: float,
    theta: float,
    *,
    pitch: float = 0.0,
    roll: float = 0.0,
) -> dict[str, float]:
    return {
        "x": float(x),
        "y": float(y),
        "theta": float(theta),
        "pitch": float(pitch),
        "roll": float(roll),
    }


def _bind_pairing(yard: Any, *, require_pair: bool = False) -> PairingMachine:
    return PairingMachine.from_profile(getattr(yard, "pairing", None), require_pair=require_pair)


def _persist_pair_on_yard(yard: Any, pairing: PairingMachine) -> None:
    if yard is not None and hasattr(yard, "pairing"):
        yard.pairing = pairing.persist()


def _pair_kwargs_pin(kwargs: dict[str, Any]) -> Optional[str]:
    if kwargs.get("pin") is not None:
        return str(kwargs["pin"])
    if kwargs.get("code") is not None:
        return str(kwargs["code"])
    return None


def _weather_status(*, rain: bool, extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    blob = {"rain": bool(rain), "wet": bool(rain)}
    if isinstance(extra, dict):
        blob.update({k: extra[k] for k in extra if k not in blob})
        blob["rain"] = bool(rain or extra.get("wet") or extra.get("rain"))
        blob["wet"] = bool(blob["rain"])
    return blob


def _radio_status(radio: Any, *, pairing: Any = None) -> dict[str, Any]:
    if not isinstance(radio, RadioPrefs):
        raw = radio or {}
        if isinstance(raw, YardProfile):
            radio = radio_prefs(raw)
        else:
            wifi = raw.get("wifi") if isinstance(raw.get("wifi"), dict) else {}
            lora = raw.get("lora") if isinstance(raw.get("lora"), dict) else {}
            radio = RadioPrefs(
                bluetooth=bool(raw.get("bluetooth", True)),
                wifi_enabled=bool(wifi.get("enabled", False)),
                wifi_ssid=str(wifi.get("ssid") or ""),
                lora_enabled=bool(lora.get("enabled", True)),
                lora_channel=int(lora.get("channel", 1)),
                primary=str(raw.get("primary") or "lora"),
            )
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
    pair_state = None
    if pairing is not None:
        pair_state = getattr(pairing, "state", None) or (pairing.get("state") if isinstance(pairing, dict) else None)
    paired = bool(getattr(pairing, "is_paired", False)) if pairing is not None else bool(radio.bluetooth)
    if pairing is not None and hasattr(pairing, "is_paired"):
        paired = bool(pairing.is_paired)
    elif pair_state == "paired":
        paired = True
    return {
        "link": primary if ok else "none",
        "ok": bool(ok),
        "rf_claim": None,
        "transport": {"bluetooth": "bt", "wifi": "wifi", "lora": "lora"}.get(primary if ok else "", None),
        "bluetooth": {
            "state": pair_state or ("paired" if paired else "unpaired"),
            "paired": bool(paired),
            "ok": bool(paired if pair_state else radio.bluetooth),
        },
        "wifi": {"enabled": bool(radio.wifi_enabled), "ssid": radio.wifi_ssid, "ok": bool(radio.wifi_enabled)},
        "lora": {
            "enabled": bool(radio.lora_enabled),
            "channel": int(radio.lora_channel),
            "ok": bool(radio.lora_enabled),
            "far_fence": True,
        },
        "not_rf_hardware": True,
        "simulated": True,
    }


_SIM_LINK = {"wifi": "wifi", "bt": "bluetooth", "lora": "lora"}


def _radio_sim_from_info(info: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """UX-B RadioSim fields are flattened onto env ``info``."""
    if not isinstance(info, dict):
        return None
    if "radio_enabled" not in info and "radio_channel" not in info and info.get("schema") != RADIO_SCHEMA:
        return None
    return {
        "schema": str(info.get("schema") or RADIO_SCHEMA),
        "radio_enabled": bool(info.get("radio_enabled")),
        "radio_channel": info.get("radio_channel"),
        "radio_lost": bool(info.get("radio_lost")),
        "radio_on_loss": info.get("radio_on_loss"),
        "rf_claim": None,
        "transports": info.get("transports"),
        "not_rf_hardware": True,
    }


def _overlay_radio_sim(
    status_radio: dict[str, Any],
    sim: Optional[dict[str, Any]],
    *,
    env_radio: Any = None,
) -> dict[str, Any]:
    if env_radio is not None and hasattr(env_radio, "owner_status"):
        owner = env_radio.owner_status()
        out = dict(status_radio)
        out.update({k: owner[k] for k in owner if k not in {"link"} or owner.get("link") != "none"})
        out["rf_claim"] = None
        out["sim"] = sim
        if sim and sim.get("radio_lost") and not getattr(env_radio, "any_lost", False):
            out["ok"] = False
        return out
    if not isinstance(sim, dict):
        status_radio = dict(status_radio)
        status_radio.setdefault("rf_claim", None)
        return status_radio
    out = dict(status_radio)
    out["sim"] = sim
    out["rf_claim"] = None
    if sim.get("radio_lost"):
        out["link"] = "none"
        out["ok"] = False
        out["transport"] = None
    elif sim.get("radio_enabled") and sim.get("radio_channel"):
        out["link"] = _SIM_LINK.get(str(sim["radio_channel"]), str(sim["radio_channel"]))
        out["transport"] = sim.get("radio_channel")
    return out


def _ux_b_faults(info: Optional[dict[str, Any]], extra: list[dict[str, Any]]) -> list[dict[str, Any]]:
    faults = [dict(item) for item in extra]
    blob = (info or {}).get("fault") if isinstance(info, dict) else None
    if isinstance(blob, dict):
        code = str(blob.get("code") or "ok")
        if code and code != "ok" and not any(f.get("code") == code for f in faults):
            faults.append(
                {
                    "code": code,
                    "detail": str(blob.get("reason") or blob.get("component") or code),
                    "retrieve": bool(blob.get("retrieve")),
                }
            )
    if isinstance(info, dict) and (info.get("hw_estop") or info.get("hw_estop_latched")):
        if not any(f.get("code") == "HW_ESTOP" for f in faults):
            faults.append(
                {
                    "code": "HW_ESTOP",
                    "detail": str(info.get("hw_estop_reason") or "paddle latched — rails dead"),
                    "retrieve": False,
                    "kind": "hardware",
                }
            )
    return faults


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


def _mesh_from_grid(
    elev: Optional[np.ndarray],
    yard: YardProfile,
    *,
    coverage: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    rows = max(8, min(48, int(round(yard.height_m / max(yard.resolution_m, 0.2)))))
    cols = max(8, min(48, int(round(yard.width_m / max(yard.resolution_m, 0.2)))))
    if elev is None:
        elev = np.zeros((rows, cols), dtype=np.float32)
    else:
        elev = np.asarray(elev, dtype=np.float32)
        if elev.ndim != 2 or elev.size == 0:
            elev = np.zeros((rows, cols), dtype=np.float32)
    cov = None if coverage is None else np.asarray(coverage, dtype=np.float32)
    mesh = mesh_from_elevation(
        elev,
        width_m=float(yard.width_m),
        height_m=float(yard.height_m),
        resolution_m=float(yard.resolution_m),
        coverage=cov if cov is not None and cov.shape == elev.shape else None,
        stride=max(1, int(max(elev.shape) / 24)),
    )
    payload = mesh_to_payload(mesh)
    payload.update(
        {
            "schema": MESH_SCHEMA,
            "path": yard.mesh,
            "viewer": "/viewer",
            "ux_a_href": "/viewer",
            "note": "UX-A TerrainMesh payload (three.js World Viewer). No second WebGL stack.",
            "not_slam": True,
        }
    )
    return payload


def viewer_manifest(backend: AppBackend) -> dict[str, Any]:
    session = getattr(backend, "session", None)
    if session is not None and hasattr(session, "manifest"):
        return session.manifest()
    status = backend.status()
    yard = backend.get_yard()
    mesh = backend.mesh()
    return {
        "schema": VIEWER_SCHEMA,
        "width_m": float(yard.get("width_m") or 16.0),
        "height_m": float(yard.get("height_m") or 12.0),
        "resolution_m": float(yard.get("resolution_m") or 0.20),
        "mesh": None,
        "mesh_json": "yard.json",
        "maps": {},
        "profile": "profile.json",
        "poses": None,
        "vertex_count": int(mesh.get("vertex_count") or 0),
        "triangle_count": int(mesh.get("triangle_count") or 0),
        "health": {
            "placeholder": False,
            "label": "battery",
            "value": (status.get("battery") or {}).get("soc"),
        },
        "radio": {
            "placeholder": False,
            "label": (status.get("radio") or {}).get("link"),
            "value": (status.get("radio") or {}).get("rf_claim"),
        },
        "not_a_benchmark": True,
        "note": "Owner-app live bundle — same UX-A viewer_static / mesh payload.",
    }


class MemoryBackend:
    """In-memory kinematic stub used by tests and `--backend memory`."""

    def __init__(
        self,
        yard: Optional[YardProfile] = None,
        *,
        yard_path: Optional[Path] = None,
        require_pair: bool = False,
    ) -> None:
        self._lock = threading.Lock()
        self.yard = yard or default_yard_profile()
        self.yard_path = Path(yard_path) if yard_path else None
        self.pose = dict(self.yard.home)
        self.mission = "idle"
        self.safe = SafeStateMachine()
        self.soc = 0.92
        self.temp_c = 42.0
        self.faults: list[dict[str, str]] = []
        self.hours_mowed = 1.4
        self.rain = False
        self.require_pair = bool(require_pair)
        self.pairing = _bind_pairing(self.yard, require_pair=self.require_pair)
        self.schedule_hook = ScheduleHook.from_profile_schedule(self.yard.schedule)
        rows = max(1, int(round(self.yard.height_m / self.yard.resolution_m)))
        cols = max(1, int(round(self.yard.width_m / self.yard.resolution_m)))
        self._coverage = np.zeros((rows, cols), dtype=np.float32)
        self._occ = np.zeros((rows, cols), dtype=np.float32)
        self._paint_keepout()
        self._yards: dict[str, YardProfile] = {self.yard.name: self.yard}
        self.notify = NotificationLog(
            (Path(yard_path).parent / "notifications.jsonl") if yard_path else None
        )
        self._yard_store = None
        if yard_path:
            self._yard_store = YardStore(Path(yard_path).parent / "yards")
            self._yard_store.put(self.yard)

    def set_clock(self, clock: Clock) -> None:
        self.schedule_hook.set_clock(clock)

    def _gates_unlocked(self) -> Any:
        mode = self.safe.mode if self.safe.mode in SAFE_MODES else "run"
        mission = self.mission
        if mode == "estop":
            mission = "estop"
        return gates_from_owner_state(
            soc=float(self.soc),
            rain=bool(self.rain),
            faults=self.faults,
            mission=mission,
            machine=mode,
            capacity_wh=GYM_STUB_CAPACITY_WH,
            pack_measured=False,
        )

    def _arm_from_schedule(self) -> None:
        if not self.pairing.can_start:
            return
        if self.safe.mode == "estop":
            self.safe.clear()
        self.faults = []
        self.mission = "mowing"

    def _stop_from_schedule(self) -> None:
        if self.safe.mode != "estop":
            self.mission = "idle"

    def _hold_safe_unlocked(self, reason: str) -> None:
        """Unpair / radio-lost: idle hold, not ESTOP."""
        if self.safe.mode == "estop":
            return
        if self.mission in {"mowing", "returning", "teach"}:
            self.mission = "idle"
            self.notify.emit("info", f"{reason} — hold safe", yard=self.yard.name)

    def _poll_schedule_unlocked(self) -> None:
        self.schedule_hook.poll(
            self._gates_unlocked(),
            start=self._arm_from_schedule,
            stop=self._stop_from_schedule,
            spec=self.yard.schedule,
        )

    def _paint_keepout(self) -> None:
        spec = self.yard.geofence_spec()
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
        self._poll_schedule_unlocked()
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
            "battery": battery_status_block(soc=float(self.soc), temp_c=float(self.temp_c)),
            "state": {
                "mission": mission,
                "machine": mode,
                "help_requested": bool(self.safe.command().help_requested),
                "reason": self.safe.last_reason,
            },
            "radio": _radio_status(self.yard.radio, pairing=self.pairing),
            "pairing": self.pairing.as_info(),
            "require_pair": bool(self.require_pair),
            "paired": bool(self.pairing.is_paired),
            "faults": list(self.faults),
            "coverage_pct": coverage_pct,
            "coverage_source": "gym_grid",
            "hours_mowed": float(self.hours_mowed),
            "yard": self.yard.name,
            "weather": _weather_status(rain=self.rain),
            "schedule": self.schedule_hook.status_dict(),
            "not_a_benchmark": True,
        }

    def get_yard(self) -> dict[str, Any]:
        with self._lock:
            return self.yard.as_dict()

    def put_yard(self, profile: YardProfile) -> dict[str, Any]:
        with self._lock:
            return self._put_yard_unlocked(profile)

    def _put_yard_unlocked(self, profile: YardProfile) -> dict[str, Any]:
        prev = getattr(self, "yard", None)
        self.yard = profile
        self._yards[profile.name] = profile
        rows = max(1, int(round(profile.height_m / profile.resolution_m)))
        cols = max(1, int(round(profile.width_m / profile.resolution_m)))
        if self._coverage.shape != (rows, cols):
            self._coverage = np.zeros((rows, cols), dtype=np.float32)
            self._occ = np.zeros((rows, cols), dtype=np.float32)
        self._coverage[:, :] = 0.0
        self._occ[:, :] = 0.0
        self._paint_keepout()
        self.pose = dict(profile.home)
        if self.yard_path is not None:
            save_yard_profile(profile, self.yard_path)
        if self._yard_store is not None:
            self._yard_store.put(profile)
            self._yard_store.select(profile.name)
        self.schedule_hook.sync(profile.schedule)
        self.pairing = _bind_pairing(profile, require_pair=self.require_pair)
        if prev is not None and prev.name != profile.name:
            self.notify.emit(
                "info",
                f"switched yard {prev.name} → {profile.name}",
                yard=profile.name,
            )
        return profile.as_dict()

    def list_yards(self) -> dict[str, Any]:
        with self._lock:
            if self._yard_store is not None:
                return self._yard_store.as_info()
            return {
                "schema": "jims_mower.yards.v1",
                "active": self.yard.name,
                "yards": [
                    {
                        "name": p.name,
                        "active": p.name == self.yard.name,
                        "keep_in_vertices": len(p.keep_in),
                        "home": dict(p.home),
                    }
                    for p in self._yards.values()
                ],
                "note": "switch replaces keep-in/home — no fence bleed",
            }

    def select_yard(self, name: str) -> dict[str, Any]:
        with self._lock:
            key = str(name or "").strip()
            profile = self._yards.get(key)
            if profile is None and self._yard_store is not None:
                profile = self._yard_store.select(key)
                self._yards[profile.name] = profile
            if profile is None:
                raise YardProfileError(f"yard not found: {name}")
            return self._put_yard_unlocked(profile)

    def notifications(self, *, last_n: int = 50) -> dict[str, Any]:
        with self._lock:
            return {
                "schema": "jims_mower.notify.v1",
                "items": self.notify.list(last_n=last_n),
                "sms": False,
                "channel": "in_app",
            }

    def notify_event(self, kind: str, reason: str) -> dict[str, Any]:
        with self._lock:
            return self.notify.emit(kind, reason, yard=self.yard.name)

    def ota(self) -> dict[str, Any]:
        return ota_status()

    def command(self, cmd: str, *, reason: str = "", **kwargs: Any) -> dict[str, Any]:
        key = str(cmd or "").strip().lower()
        if key not in APP_COMMANDS:
            raise YardProfileError(f"unknown command {cmd!r}; expected one of {APP_COMMANDS}")
        with self._lock:
            if key == "estop":
                self.safe.request_estop(reason or "owner estop")
                self.mission = "estop"
                self.faults = [{"code": "ESTOP", "detail": reason or "owner estop"}]
                self.notify.emit("fault", reason or "owner estop", yard=self.yard.name)
            elif key == "start":
                refused = self.pairing.refuse_start()
                if refused:
                    raise YardProfileError("not paired — Pair Bluetooth before Start")
                if self.safe.mode == "estop":
                    self.safe.clear()
                self.faults = []
                self.mission = "mowing"
                self.notify.emit("info", reason or "owner start", yard=self.yard.name)
            elif key == "stop":
                if self.safe.mode != "estop":
                    self.mission = "idle"
                    self.notify.emit("finish", reason or "owner stop", yard=self.yard.name)
            elif key == "return":
                if self.safe.mode == "estop":
                    raise YardProfileError("cannot return while ESTOP is latched")
                self.mission = "returning"
            elif key == "explore":
                if self.safe.mode == "estop":
                    raise YardProfileError("cannot explore while ESTOP is latched")
                self.mission = "explore"
            elif key == "mow":
                if self.safe.mode == "estop":
                    raise YardProfileError("cannot mow while ESTOP is latched")
                self.mission = "mowing"
            elif key == "teach":
                if self.safe.mode == "estop":
                    raise YardProfileError("cannot teach while ESTOP is latched")
                self.mission = "teach"
            elif key == "pair":
                result = self.pairing.request_pair(_pair_kwargs_pin(kwargs))
                _persist_pair_on_yard(self.yard, self.pairing)
                if not result.ok:
                    raise YardProfileError(result.reason or "pair_failed")
            elif key == "unpair":
                self.pairing.unpair()
                self._hold_safe_unlocked("unpaired")
                _persist_pair_on_yard(self.yard, self.pairing)
            return self._status_unlocked()

    def mesh(self) -> dict[str, Any]:
        with self._lock:
            return _mesh_from_grid(None, self.yard, coverage=self._coverage)

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
                spec = self.yard.geofence_spec()
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
                hx, hy = float(self.yard.home.get("x", 1.0)), float(self.yard.home.get("y", 1.0))
                dx, dy = hx - self.pose["x"], hy - self.pose["y"]
                dist = math.hypot(dx, dy)
                if dist < 0.25:
                    self.pose = dict(self.yard.home)
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
            home={
                "x": float(self._info["pose"]["x"]),
                "y": float(self._info["pose"]["y"]),
                "theta": float(self._info["pose"]["theta"]),
            },
        )
        self.yard_path = Path(yard_path) if yard_path else None
        self.mission = "idle"
        self.safe = SafeStateMachine()
        self.faults: list[dict[str, str]] = []
        self.hours_mowed = 0.0
        self.rain = rain_from_weather(self._info.get("weather"))
        self.require_pair = False
        self.pairing = _bind_pairing(self.yard, require_pair=False)
        self.schedule_hook = ScheduleHook.from_profile_schedule(self.yard.schedule)

    def set_clock(self, clock: Clock) -> None:
        self.schedule_hook.set_clock(clock)

    def _gates_unlocked(self) -> Any:
        mode = self.safe.mode
        mission = "estop" if mode == "estop" else self.mission
        return gates_from_owner_state(
            soc=float(self._info.get("battery_soc", 0.9)),
            rain=bool(self.rain or rain_from_weather(self._info.get("weather"))),
            faults=_ux_b_faults(self._info, self.faults),
            mission=mission,
            machine=mode,
            capacity_wh=self._info.get("capacity_wh", GYM_STUB_CAPACITY_WH),
            pack_measured=bool(self._info.get("pack_measured", False)),
        )

    def _arm_from_schedule(self) -> None:
        if self.safe.mode == "estop":
            self.safe.clear()
        self.faults = []
        self.mission = "mowing"

    def _stop_from_schedule(self) -> None:
        if self.safe.mode != "estop":
            self.mission = "idle"

    def _poll_schedule_unlocked(self) -> None:
        self.rain = bool(self.rain or rain_from_weather(self._info.get("weather")))
        self.schedule_hook.poll(
            self._gates_unlocked(),
            start=self._arm_from_schedule,
            stop=self._stop_from_schedule,
            spec=self.yard.schedule,
        )

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
        faults = _ux_b_faults(self._info, faults)
        self._poll_schedule_unlocked()
        mode = self.safe.mode
        mission = self.mission
        if mode == "estop":
            mission = "estop"
        weather = self._info.get("weather") if isinstance(self._info.get("weather"), dict) else {}
        rain = bool(self.rain or rain_from_weather(weather))
        return {
            "schema": APP_STATUS_SCHEMA,
            "pose": self._pose(),
            "battery": battery_status_block(soc=soc, temp_c=temp, info=self._info),
            "state": {
                "mission": mission,
                "machine": mode,
                "help_requested": bool(self.safe.command().help_requested),
                "reason": self.safe.last_reason or self._info.get("terrain_reason"),
                "terrain_advice": self._info.get("terrain_advice"),
                "living_advice": self._info.get("living_advice"),
            },
            "radio": _overlay_radio_sim(
                _radio_status(self.yard.radio, pairing=self.pairing),
                _radio_sim_from_info(self._info),
                env_radio=getattr(self.env, "radio", None),
            ),
            "pairing": self.pairing.as_info(),
            "require_pair": bool(self.require_pair),
            "paired": bool(self.pairing.is_paired),
            "faults": faults,
            "coverage_pct": 100.0 * float(self._info.get("coverage_fraction") or 0.0),
            "coverage_source": str(self._info.get("coverage_source") or "gym_grid"),
            "hours_mowed": float(self.hours_mowed),
            "yard": self.yard.name,
            "weather": _weather_status(rain=rain, extra=weather if isinstance(weather, dict) else None),
            "schedule": self.schedule_hook.status_dict(),
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
            self.schedule_hook.sync(profile.schedule)
            self.pairing = _bind_pairing(profile, require_pair=self.require_pair)
            return profile.as_dict()

    def command(self, cmd: str, *, reason: str = "", **kwargs: Any) -> dict[str, Any]:
        key = str(cmd or "").strip().lower()
        if key not in APP_COMMANDS:
            raise YardProfileError(f"unknown command {cmd!r}; expected one of {APP_COMMANDS}")
        with self._lock:
            if key == "estop":
                self.safe.request_estop(reason or "owner estop")
                self.mission = "estop"
                self.faults = [{"code": "ESTOP", "detail": reason or "owner estop"}]
            elif key == "start":
                refused = self.pairing.refuse_start()
                if refused:
                    raise YardProfileError("not paired — Pair Bluetooth before Start")
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
            elif key == "explore":
                if self.safe.mode == "estop":
                    raise YardProfileError("cannot explore while ESTOP is latched")
                self.mission = "explore"
            elif key == "mow":
                if self.safe.mode == "estop":
                    raise YardProfileError("cannot mow while ESTOP is latched")
                self.mission = "mowing"
            elif key == "teach":
                if self.safe.mode == "estop":
                    raise YardProfileError("cannot teach while ESTOP is latched")
                self.mission = "teach"
            elif key == "pair":
                result = self.pairing.request_pair(_pair_kwargs_pin(kwargs))
                _persist_pair_on_yard(self.yard, self.pairing)
                if not result.ok:
                    raise YardProfileError(result.reason or "pair_failed")
            elif key == "unpair":
                self.pairing.unpair()
                if self.safe.mode != "estop" and self.mission in {"mowing", "returning", "teach"}:
                    self.mission = "idle"
                _persist_pair_on_yard(self.yard, self.pairing)
            return self._status_unlocked()

    def mesh(self) -> dict[str, Any]:
        with self._lock:
            elev = getattr(getattr(self.env, "_terrain", None), "elevation", None)
            return _mesh_from_grid(elev, self.yard, coverage=self._obs.get("coverage"))

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
                hx, hy = float(self.yard.home.get("x", 1.0)), float(self.yard.home.get("y", 1.0))
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
                if math.hypot(float(pose.get("x", 0.0)) - float(self.yard.home.get("x", 1.0)), float(pose.get("y", 0.0)) - float(self.yard.home.get("y", 1.0))) < 0.6:
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
        self.rain = False
        self.require_pair = False
        self.pairing = _bind_pairing(self.yard, require_pair=False)
        self.schedule_hook = ScheduleHook.from_profile_schedule(self.yard.schedule)

    def set_clock(self, clock: Clock) -> None:
        self.schedule_hook.set_clock(clock)

    def _gates_unlocked(self, info: Optional[dict[str, Any]] = None) -> Any:
        blob = info if isinstance(info, dict) else {}
        mode = self.safe.mode
        mission = "estop" if mode == "estop" else self.mission
        return gates_from_owner_state(
            soc=float(blob.get("battery_soc", 0.88)),
            rain=bool(self.rain or rain_from_weather(blob.get("weather"))),
            faults=_ux_b_faults(blob, self.faults),
            mission=mission,
            machine=mode,
            capacity_wh=blob.get("capacity_wh", GYM_STUB_CAPACITY_WH),
            pack_measured=bool(blob.get("pack_measured", False)),
        )

    def _arm_from_schedule(self) -> None:
        if self.safe.mode == "estop":
            self.safe.clear()
        self.faults = []
        self.mission = "mowing"

    def _stop_from_schedule(self) -> None:
        if self.safe.mode != "estop":
            self.mission = "idle"

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
            home={
                "x": float(pose.get("x", 1.0)),
                "y": float(pose.get("y", 1.0)),
                "theta": float(pose.get("theta", 0.0)),
            },
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
        self.schedule_hook.poll(
            self._gates_unlocked(info),
            start=self._arm_from_schedule,
            stop=self._stop_from_schedule,
            spec=self.yard.schedule,
        )
        mode = self.safe.mode
        mission = self.mission
        if mode == "estop":
            mission = "estop"
        weather = info.get("weather") if isinstance(info.get("weather"), dict) else {}
        rain = bool(self.rain or rain_from_weather(weather) or rain_from_weather(info.get("weather")))
        return {
            "schema": APP_STATUS_SCHEMA,
            "pose": _pose_dict(float(pose.get("x", 0.0)), float(pose.get("y", 0.0)), float(pose.get("theta", 0.0))),
            "battery": battery_status_block(
                soc=float(info.get("battery_soc", 0.88)),
                temp_c=float(info.get("thermal_c", 41.0)),
                info=info,
            ),
            "state": {
                "mission": mission,
                "machine": mode,
                "help_requested": bool(self.safe.command().help_requested),
                "reason": self.safe.last_reason,
                "terrain_advice": info.get("terrain_advice"),
            },
            "radio": _overlay_radio_sim(_radio_status(self.yard.radio, pairing=self.pairing), _radio_sim_from_info(info)),
            "pairing": self.pairing.as_info(),
            "require_pair": bool(self.require_pair),
            "paired": bool(self.pairing.is_paired),
            "faults": _ux_b_faults(info, list(self.faults)),
            "coverage_pct": 100.0 * float(info.get("coverage_fraction") or 0.0),
            "coverage_source": str(info.get("coverage_source") or "gym_grid"),
            "hours_mowed": float(self.hours_mowed),
            "yard": self.yard.name,
            "episode": str(self.reader.episode_dir),
            "weather": _weather_status(rain=rain, extra=weather if isinstance(weather, dict) else None),
            "schedule": self.schedule_hook.status_dict(),
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
            self.schedule_hook.sync(profile.schedule)
            self.pairing = _bind_pairing(profile, require_pair=self.require_pair)
            return profile.as_dict()

    def command(self, cmd: str, *, reason: str = "", **kwargs: Any) -> dict[str, Any]:
        key = str(cmd or "").strip().lower()
        if key not in APP_COMMANDS:
            raise YardProfileError(f"unknown command {cmd!r}; expected one of {APP_COMMANDS}")
        with self._lock:
            if key == "estop":
                self.safe.request_estop(reason or "owner estop")
                self.mission = "estop"
                self.faults = [{"code": "ESTOP", "detail": reason or "owner estop"}]
            elif key == "start":
                refused = self.pairing.refuse_start()
                if refused:
                    raise YardProfileError("not paired — Pair Bluetooth before Start")
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
            elif key == "explore":
                if self.safe.mode == "estop":
                    raise YardProfileError("cannot explore while ESTOP is latched")
                self.mission = "explore"
            elif key == "mow":
                if self.safe.mode == "estop":
                    raise YardProfileError("cannot mow while ESTOP is latched")
                self.mission = "mowing"
            elif key == "teach":
                if self.safe.mode == "estop":
                    raise YardProfileError("cannot teach while ESTOP is latched")
                self.mission = "teach"
            elif key == "pair":
                result = self.pairing.request_pair(_pair_kwargs_pin(kwargs))
                _persist_pair_on_yard(self.yard, self.pairing)
                if not result.ok:
                    raise YardProfileError(result.reason or "pair_failed")
            elif key == "unpair":
                self.pairing.unpair()
                if self.safe.mode != "estop" and self.mission in {"mowing", "returning", "teach"}:
                    self.mission = "idle"
                _persist_pair_on_yard(self.yard, self.pairing)
            return self._status_unlocked()

    def mesh(self) -> dict[str, Any]:
        with self._lock:
            obs = self._current().get("obs") or {}
            elev = obs.get("elevation")
            return _mesh_from_grid(elev, self.yard, coverage=obs.get("coverage"))

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
    fast: bool = False,
    speed: Any = 5.0,
    steps: Optional[int] = None,
    out_dir: Optional[Union[str, Path]] = None,
    session: Any = None,
    reset: bool = True,
    first_run: bool = False,
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
    if kind == "live":
        from jims_mower.app.live_backend import LiveBackend

        return LiveBackend(
            session=session,
            config=config or "acre_yard_demo",
            fast=fast,
            speed=speed,
            steps=steps,
            seed=seed,
            cameras=cameras,
            yard=profile,
            yard_path=persist,
            out_dir=out_dir or "live_out",
            reset=reset,
            first_run=first_run,
        )
    if kind not in {"sim", "demo"}:
        raise YardProfileError(f"unknown backend {kind!r}")
    return SimBackend(config=config, seed=seed, cameras=cameras, yard=profile, yard_path=persist)

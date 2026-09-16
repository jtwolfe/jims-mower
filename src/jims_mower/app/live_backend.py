"""App backend that wraps ``LiveSession`` — no second mission loop."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from jims_mower.app.backend import (
    _coverage_payload,
    _mesh_from_grid,
    _overlay_radio_sim,
    _pose_dict,
    _radio_sim_from_info,
    _radio_status,
    _ux_b_faults,
)
from jims_mower.constants import APP_STATUS_SCHEMA, LIVE_CONTROL_CMDS
from jims_mower.live import (
    LiveSession,
    robot_status_for,
)
from jims_mower.yard_profile import (
    YardProfile,
    YardProfileError,
    default_yard_profile,
    save_yard_profile,
    yard_profile_from_geofence,
)


class LiveBackend:
    """Phone-app facade over one ``LiveSession``.

    Start / pause / ESTOP / MAP READY go to ``LiveSession.control``. The
    desktop ``jims-mower-live`` viewer keeps the same session type and
    ``POST /api/live/control`` contract.
    """

    def __init__(
        self,
        *,
        session: Optional[LiveSession] = None,
        config: Optional[str] = "acre_yard_demo",
        fast: bool = False,
        speed: Any = 5.0,
        steps: Optional[int] = None,
        seed: int = 7,
        cameras: int = 4,
        yard: Optional[YardProfile] = None,
        yard_path: Optional[Path] = None,
        out_dir: Union[str, Path] = "live_out",
        reset: bool = True,
    ) -> None:
        self._lock = threading.Lock()
        self.session = session or LiveSession(
            config=config,
            fast=fast,
            speed=speed,
            steps=steps,
            seed=seed,
            cameras=cameras,
            out_dir=out_dir,
        )
        self.yard_path = Path(yard_path) if yard_path else None
        self.yard = yard or default_yard_profile(name=str(self.session.config_name))
        self.paired = bool(getattr(self.session, "paired", False))
        if reset and not self.session.started:
            self.session.reset()
        self._sync_yard_from_session()

    def _sync_yard_from_session(self) -> None:
        policy = self.session.policy
        env = self.session.env
        if policy is not None and policy.profile is not None:
            self.yard = policy.profile
            return
        if env is None:
            return
        spec = env.geofence_spec()
        pose = (self.session.info or {}).get("pose") or {}
        self.yard = yard_profile_from_geofence(
            spec,
            name=str(self.session.config_name),
            width_m=env.cfg.world.width_m,
            height_m=env.cfg.world.height_m,
            resolution_m=env.cfg.world.resolution_m,
            home={
                "x": float(pose.get("x", 1.0)),
                "y": float(pose.get("y", 1.0)),
                "theta": float(pose.get("theta", 0.0)),
            },
        )

    def live_control(self, cmd: str, **kwargs: Any) -> dict[str, Any]:
        key = str(cmd or "").strip().lower()
        if key == "pair":
            self.paired = True
            self.session.paired = True
        result = self.session.control(cmd, **kwargs)
        if key == "start":
            self.paired = True
            self.session.paired = True
        self._sync_yard_from_session()
        return result

    def _status_unlocked(self) -> dict[str, Any]:
        snap = self.session.snapshot()
        info = self.session.info if isinstance(self.session.info, dict) else {}
        pose = snap.get("pose") or {}
        faults = list(snap.get("faults") or [])
        faults = _ux_b_faults(info, faults)
        job_state = str(snap.get("job_state") or "idle")
        mission = {
            "idle": "idle",
            "running": "mowing",
            "paused": "idle",
            "hold": "idle",
            "estop": "estop",
        }.get(job_state, job_state)
        if snap.get("phase") == "return_home":
            mission = "returning"
        if snap.get("can_start_mow"):
            mission = "review"
        radio = _overlay_radio_sim(_radio_status(self.yard.radio), _radio_sim_from_info(info))
        radio["path"] = snap.get("radio_path") or {}
        robot = robot_status_for(
            paired=self.paired,
            job_state=job_state,
            faults=faults,
            done=bool(snap.get("done")),
        )
        return {
            "schema": APP_STATUS_SCHEMA,
            "backend": "live",
            "live": True,
            "pose": _pose_dict(
                float(pose.get("x", 0.0)),
                float(pose.get("y", 0.0)),
                float(pose.get("theta", 0.0)),
            ),
            "battery": {
                "soc": float(info.get("battery_soc", 0.9)),
                "temp_c": float(info.get("thermal_c", 42.0)),
                "not_a_power_trace": True,
            },
            "state": {
                "mission": mission,
                "machine": snap.get("safe_mode") or ("estop" if job_state == "estop" else "run"),
                "job_state": job_state,
                "phase": snap.get("phase"),
                "phase_label": snap.get("phase_label"),
                "help_requested": any(f.get("retrieve") for f in faults),
                "reason": (faults[0].get("detail") if faults else "") or snap.get("owner_copy"),
            },
            "radio": radio,
            "radio_path": snap.get("radio_path") or {},
            "faults": faults,
            "coverage_pct": 100.0 * float(snap.get("cut_pct") or 0.0),
            "map_pct": 100.0 * float(snap.get("map_pct") or 0.0),
            "cut_pct": 100.0 * float(snap.get("cut_pct") or 0.0),
            "hours_mowed": float(snap.get("duration_s") or 0.0) / 3600.0,
            "yard": str(snap.get("yard") or self.yard.name),
            "owner_copy": snap.get("owner_copy"),
            "can_start_mow": bool(snap.get("can_start_mow")),
            "can_reexplore": bool(snap.get("can_reexplore")),
            "session_summary": snap.get("session_summary") or {},
            "speed": snap.get("speed"),
            "speed_label": snap.get("speed_label"),
            "paired": bool(self.paired),
            "robot": robot,
            "done": bool(snap.get("done")),
            "fog_url": snap.get("fog_url"),
            "observed_url": snap.get("observed_url"),
            "viewer": "/viewer",
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
        mapped = {"stop": "pause", "return": "hold", "teach": "pair"}.get(key, key)
        if mapped == "pair":
            with self._lock:
                self.paired = True
                self.session.paired = True
                return self._status_unlocked()
        if mapped not in LIVE_CONTROL_CMDS:
            raise YardProfileError(f"unknown command {cmd!r}; expected live control or pair")
        extra: dict[str, Any] = {}
        if reason:
            extra["reason"] = reason
        with self._lock:
            self.live_control(mapped, **extra)
            return self._status_unlocked()

    def mesh(self) -> dict[str, Any]:
        with self._lock:
            env = self.session.env
            elev = getattr(getattr(env, "_terrain", None), "elevation", None) if env is not None else None
            obs = self.session.obs if isinstance(self.session.obs, dict) else {}
            return _mesh_from_grid(elev, self.yard, coverage=obs.get("coverage"))

    def coverage(self) -> dict[str, Any]:
        with self._lock:
            obs = self.session.obs if isinstance(self.session.obs, dict) else {}
            grid = np.asarray(
                obs.get("coverage") if obs.get("coverage") is not None else np.zeros((8, 8)),
                dtype=np.float32,
            )
            return _coverage_payload(grid, self.yard)

    def tick(self) -> dict[str, Any]:
        return self.status()

    def close(self) -> None:
        self.session.close()

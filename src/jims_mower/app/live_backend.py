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
    _weather_status,
)
from jims_mower.constants import APP_STATUS_SCHEMA, LIVE_CONTROL_CMDS
from jims_mower.live import (
    LiveSession,
    robot_status_for,
)
from jims_mower.path_overlay import mission_from_phase
from jims_mower.pack import GYM_STUB_CAPACITY_WH, battery_status_block
from jims_mower.schedule import (
    Clock,
    ScheduleHook,
    gates_from_owner_state,
    rain_from_weather,
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
        first_run: bool = False,
        require_pair: bool = True,
    ) -> None:
        self._lock = threading.Lock()
        persist = Path(yard_path) if yard_path else Path(out_dir) / "profile.json"
        self.session = session or LiveSession(
            config=config,
            fast=fast,
            speed=speed,
            steps=steps,
            seed=seed,
            cameras=cameras,
            out_dir=out_dir,
            yard_profile=yard,
            yard_path=persist,
            first_run=first_run,
            require_pair=require_pair,
        )
        if session is not None:
            if yard is not None:
                self.session.yard_profile = yard
                if len(yard.keep_in) >= 3:
                    self.session.owner_taught = True
            if yard_path is not None:
                self.session.yard_path = persist
            self.session.first_run = bool(first_run or self.session.first_run)
        self.session.require_pair = True if require_pair else bool(self.session.require_pair)
        self.session._require_pair_explicit = True
        self.session.pairing.require_pair = bool(self.session.require_pair)
        if yard is not None and getattr(yard, "pairing", None):
            if str(yard.pairing.get("state") or "") == "paired":
                self.session.pairing.force_paired()
        self.yard_path = persist
        self.session.yard_path = persist
        self.yard = yard or default_yard_profile(name=str(self.session.config_name))
        self.paired = bool(getattr(self.session, "paired", False))
        self.rain = False
        self.schedule_hook = ScheduleHook.from_profile_schedule(self.yard.schedule)
        if reset and not self.session.started:
            self.session.reset()
        self._sync_yard_from_session()
        self.schedule_hook.sync(self.yard.schedule)

    def set_clock(self, clock: Clock) -> None:
        self.schedule_hook.set_clock(clock)

    def _sync_yard_from_session(self) -> None:
        if self.session.yard_profile is not None and len(self.session.yard_profile.keep_in) >= 3:
            self.yard = self.session.yard_profile
            return
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
        result = self.session.control(cmd, **kwargs)
        self.paired = bool(self.session.paired)
        if key in {"save_yard", "load_yard", "teach", "pair", "unpair"}:
            self._sync_yard_from_session()
        self._sync_yard_from_session()
        return result

    def _arm_from_schedule(self) -> None:
        if not self.session.pairing.can_start:
            return
        self.session.control("start")

    def _stop_from_schedule(self) -> None:
        self.session.control("pause")

    def _status_unlocked(self) -> dict[str, Any]:
        snap = self.session.snapshot()
        info = self.session.info if isinstance(self.session.info, dict) else {}
        pose = snap.get("pose") or {}
        faults = list(snap.get("faults") or [])
        faults = _ux_b_faults(info, faults)
        job_state = str(snap.get("job_state") or "idle")
        tipped = bool(snap.get("chassis_tipped"))
        mission = mission_from_phase(
            str(snap.get("phase") or "idle"),
            job_state,
            tipped=tipped,
            immobilised=tipped,
        )
        radio = _overlay_radio_sim(
            _radio_status(self.yard.radio, pairing=self.session.pairing),
            _radio_sim_from_info(info),
            env_radio=getattr(self.session.env, "radio", None) if self.session.env is not None else None,
        )
        radio["path"] = snap.get("radio_path") or {}
        self.paired = bool(self.session.paired)
        robot = robot_status_for(
            paired=self.paired,
            job_state=job_state,
            faults=faults,
            done=bool(snap.get("done")),
        )
        weather = info.get("weather") if isinstance(info.get("weather"), dict) else {}
        rain = bool(self.rain or rain_from_weather(weather))
        self.schedule_hook.poll(
            gates_from_owner_state(
                soc=float(info.get("battery_soc", 0.9)),
                rain=rain,
                faults=faults,
                mission=mission,
                machine=(
                    "estop"
                    if snap.get("hw_estop")
                    else str(snap.get("safe_mode") or ("estop" if job_state == "estop" else "run"))
                ),
                capacity_wh=info.get("capacity_wh", GYM_STUB_CAPACITY_WH),
                pack_measured=bool(info.get("pack_measured", False)),
            ),
            start=self._arm_from_schedule,
            stop=self._stop_from_schedule,
            spec=self.yard.schedule,
        )
        snap = self.session.snapshot()
        info = self.session.info if isinstance(self.session.info, dict) else info
        pose = snap.get("pose") or pose
        job_state = str(snap.get("job_state") or job_state)
        tipped = bool(snap.get("chassis_tipped"))
        faults = list(snap.get("faults") or [])
        faults = _ux_b_faults(info, faults)
        if tipped and not any(f.get("code") == "FAULT_IMMOBILISED" for f in faults):
            faults.append(
                {
                    "code": "FAULT_IMMOBILISED",
                    "detail": "tipped — immobilised, retrieve",
                    "retrieve": True,
                    "kind": "software",
                }
            )
        mission = mission_from_phase(
            str(snap.get("phase") or "idle"),
            job_state,
            tipped=tipped,
            immobilised=tipped,
        )
        robot = robot_status_for(
            paired=self.paired,
            job_state=job_state,
            faults=faults,
            done=bool(snap.get("done")),
            tipped=tipped,
        )
        return {
            "schema": APP_STATUS_SCHEMA,
            "backend": "live",
            "live": True,
            "pose": _pose_dict(
                float(pose.get("x", 0.0)),
                float(pose.get("y", 0.0)),
                float(pose.get("theta", 0.0)),
                pitch=float(pose.get("pitch", 0.0)),
                roll=float(pose.get("roll", 0.0)),
            ),
            "battery": battery_status_block(
                soc=float(info.get("battery_soc", 0.9)),
                temp_c=float(info.get("thermal_c", 42.0)),
                info=info,
            ),
            "state": {
                "mission": mission,
                "machine": (
                    "estop"
                    if snap.get("hw_estop")
                    else snap.get("safe_mode") or ("estop" if job_state == "estop" else "run")
                ),
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
            "coverage_source": str(snap.get("coverage_source") or "gym_grid"),
            "map_pct": 100.0 * float(snap.get("map_pct") or 0.0),
            "cut_pct": 100.0 * float(snap.get("cut_pct") or 0.0),
            "planned_pct": 100.0 * float(snap.get("planned_pct") or 0.0),
            "waypoint_index": int(
                snap.get("waypoint_index")
                if snap.get("waypoint_index") is not None
                else (snap.get("path_overlay") or {}).get("waypoint_index") or 0
            ),
            "frontiers": snap.get("frontiers") or (snap.get("path_overlay") or {}).get("frontiers") or [],
            "explore": snap.get("explore") or (snap.get("path_overlay") or {}).get("explore") or [],
            "plan": snap.get("plan") or (snap.get("path_overlay") or {}).get("plan") or [],
            "hours_mowed": float(snap.get("duration_s") or 0.0) / 3600.0,
            "yard": str(snap.get("yard") or self.yard.name),
            "owner_copy": snap.get("owner_copy"),
            "can_start_mow": bool(snap.get("can_start_mow")),
            "can_reexplore": bool(snap.get("can_reexplore")),
            "can_explore": bool(snap.get("can_explore", True)),
            "can_mow": bool(snap.get("can_mow", snap.get("can_start_mow"))),
            "can_return": bool(snap.get("can_return")),
            "explore_reason": snap.get("explore_reason") or {},
            "tilt_kind": snap.get("tilt_kind") or "",
            "chassis_tipped": bool(snap.get("chassis_tipped")),
            "full_explore": bool(snap.get("full_explore")),
            "charge_state": snap.get("charge_state") or "",
            "return_kind": snap.get("return_kind") or "",
            "area_legend": snap.get("area_legend") or [],
            "blocked_cells": int(snap.get("blocked_cells") or 0),
            "blocked_frontiers": int(snap.get("blocked_frontiers") or 0),
            "n_blockages": int(snap.get("n_blockages") or 0),
            "areas_url": snap.get("areas_url"),
            "needs_reteach": bool(snap.get("needs_reteach")),
            "fence_unusable": bool(snap.get("fence_unusable")),
            "session_summary": snap.get("session_summary") or {},
            "speed": snap.get("speed"),
            "speed_label": snap.get("speed_label"),
            "paired": bool(self.paired),
            "pairing": self.session.pairing.as_info(),
            "require_pair": bool(self.session.require_pair),
            "robot": robot,
            "done": bool(snap.get("done")),
            "fog_url": snap.get("fog_url"),
            "observed_url": snap.get("observed_url"),
            "coverage_url": snap.get("coverage_url"),
            "path_overlay": snap.get("path_overlay") or {},
            "mode_banner": snap.get("mode_banner") or (snap.get("path_overlay") or {}).get("mode") or {},
            "n_waypoints": snap.get("n_waypoints"),
            "path_remaining": (snap.get("path_overlay") or {}).get("path_remaining"),
            "keep_in": snap.get("keep_in") or [list(p) for p in self.yard.keep_in],
            "keep_out": snap.get("keep_out") or [[list(p) for p in hole] for hole in self.yard.keep_out],
            "taught": bool(snap.get("taught") or self.session.owner_taught),
            "first_run": bool(snap.get("first_run") or self.session.first_run),
            "needs_teach": bool(snap.get("needs_teach") or (self.session.first_run and not self.session.owner_taught)),
            "yard_path": snap.get("yard_path") or (str(self.yard_path) if self.yard_path else ""),
            "yard_saved": bool(snap.get("yard_saved") or (self.yard_path is not None and self.yard_path.is_file())),
            "viewer": "/viewer",
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
            self.session.yard_profile = profile
            dest = self.yard_path or (Path(self.session.out_dir) / "profile.json")
            save_yard_profile(profile, dest)
            self.yard_path = dest
            self.session.yard_path = dest
            self.schedule_hook.sync(profile.schedule)
            return profile.as_dict()

    def command(self, cmd: str, *, reason: str = "", **kwargs: Any) -> dict[str, Any]:
        key = str(cmd or "").strip().lower()
        mapped = {"stop": "pause"}.get(key, key)
        extra: dict[str, Any] = dict(kwargs)
        if reason:
            extra["reason"] = reason
        if mapped not in LIVE_CONTROL_CMDS:
            raise YardProfileError(f"unknown command {cmd!r}; expected live control or pair")
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

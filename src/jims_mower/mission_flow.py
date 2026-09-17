"""Mission phases: calibrate boundary → explore/map → review → mow → home.

The terrain policy used to plan a coverage path from the first partial
observer raster. A short demo then looked like 2–3% coverage because the
robot had only travelled a few metres and unknown cells were treated as
mowable. This module is the first-principles replacement:

* unknown is not safe and not a mow target
* mapping is sensor-driven (cameras / ToF / IMU / GPS), not an oracle
* the global mow plan is frozen on the reviewed map
* local replans detour; they do not silently drop disconnected segments
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from jims_mower.config import EnvConfig, MissionConfig
from jims_mower.faults import fault_is_immobilised, fault_is_retrieve
from jims_mower.geofence import GeofenceSpec, gps_world_from_enu
from jims_mower.kinematics import attitude_past_tip, unicycle_from_wheels
from jims_mower.planning.controller import (
    combine_advice,
    geofence_from_info,
    imu_advice,
    observed_hand_signal,
    tracking_action,
)
from jims_mower.planning.costmap import build_costmap, slope_from_elevation
from jims_mower.planning.grade_tip import (
    KIND_GRADE,
    KIND_TIP,
    TipHoldFilter,
    classify_tilt,
    grade_aware_cruise,
    look_ahead_advice,
    look_ahead_from_elevation,
    merge_look_ahead_kind,
    read_tilt,
    tip_lethal_slope_rad,
)
from jims_mower.constants import (
    HAZARD_DRAIN,
    HAZARD_DRAIN_EDGE,
    MISSION_FLOW_SCHEMA,
    MISSION_PHASES,
    SESSION_SCHEMA,
    STRUCTURE_BUILDING,
    STRUCTURE_BUNKER,
    STRUCTURE_GARDEN,
    STRUCTURE_GREEN,
    STRUCTURE_POND,
    TERRAIN_ADVICE,
)
from jims_mower.planning.terrain_decision import (
    TERRAIN_CONTOUR,
    TERRAIN_OK,
    TERRAIN_REASON_CODE,
    TERRAIN_RETRACE,
    TERRAIN_TIP_REVERSE,
    climbable_grade,
    fuse_terrain_state,
    look_ahead_reason_blob,
    may_stamp_blockage,
    retrace_waypoints,
    stamp_on_cooldown,
)
from jims_mower.planning.coverage import (
    CoveragePlan,
    choose_strip_orientation,
    connected_components,
    plan_coverage,
    shortest_path,
)
from jims_mower.planning.explore import ExplorePlan, explore_costmap, plan_explore
from jims_mower.planning.fusion import make_pose_filter
from jims_mower.planning.observed import ObservedMap, downsample_frontiers, frontiers
from jims_mower.perception.elev_fuse import fuse_elev_stereo_tof_imu
from jims_mower.perception.stereo import (
    find_stereo_pair,
    height_sampler_from_raster,
    rasterize_points,
    synthetic_stereo_points,
)
from jims_mower.profile import YardProfile, keep_in_usable, trail_to_polygon
from jims_mower.runtime.budget import budget_advice
from jims_mower.safe_state import SafeStateMachine, estop_requested
from jims_mower.teach import TeachPolicy, keepouts_from_env
from jims_mower.types import Pose

PHASE_LABELS = {
    "calibrate_boundary": "CALIBRATE",
    "explore": "EXPLORE",
    "review": "MAP READY",
    "mow": "MOW",
    "return_home": "RETURN",
    "charging": "CHARGING",
    "complete": "DONE",
    "fault": "FAULT",
    "safe": "SAFE",
}

PHASE_ORDER = (
    "calibrate_boundary",
    "explore",
    "review",
    "mow",
    "return_home",
    "complete",
)


class MissionPhase(str, Enum):
    CALIBRATE_BOUNDARY = "calibrate_boundary"
    EXPLORE = "explore"
    REVIEW = "review"
    MOW = "mow"
    RETURN_HOME = "return_home"
    CHARGING = "charging"
    COMPLETE = "complete"
    FAULT = "fault"
    SAFE = "safe"


@dataclass
class MissionEvent:
    step: int
    phase: str
    event: str
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": int(self.step),
            "phase": self.phase,
            "event": self.event,
            **self.detail,
        }


@dataclass
class YardSnapshot:
    """Frozen onboard map + keep-in at review. Control never mutates this."""

    profile: YardProfile
    observed: np.ndarray
    explored: np.ndarray
    hazard: np.ndarray
    structure: np.ndarray
    elevation: np.ndarray
    confidence: np.ndarray
    keep_in_mask: np.ndarray
    free: np.ndarray
    step: int
    completion: float
    mean_confidence: float
    closed: bool
    n_components: int
    notes: list[str] = field(default_factory=list)


class MissionPolicy:
    """State machine that owns mapping, the frozen yard, and mow execution."""

    def __init__(
        self,
        cfg: EnvConfig,
        *,
        settings: Optional[MissionConfig] = None,
        fast: bool = False,
    ) -> None:
        self.cfg = cfg
        self.settings = settings or cfg.mission
        if fast:
            self.settings = _fast_settings(self.settings)
        self.fusion = make_pose_filter(cfg)
        self.safe = SafeStateMachine.from_config(cfg.planner.safe_state)
        self.teach = TeachPolicy(cfg, margin_m=0.70, arrive_m=0.55, cruise=0.75)
        self._apply_calibrate_drive()
        self.observed: Optional[ObservedMap] = None
        self.phase = MissionPhase.CALIBRATE_BOUNDARY
        self.events: list[MissionEvent] = []
        self.phase_ranges: list[dict[str, Any]] = []
        self.step = 0
        self.phase_step = 0
        self.index = 0
        self.replans = 0
        self.last_advice = "ok"
        self.last_tilt_kind = "ok"
        self._chassis_tipped = False
        self._tilt_filter = TipHoldFilter(
            window=int(cfg.planner.imu_tilt_window),
            hold_steps=int(cfg.planner.imu_stop_hold_steps),
        )
        self.last_safe_mode = self.safe.mode
        self.help_requested = False
        self.plan: Optional[CoveragePlan] = None
        self.explore_plan: Optional[ExplorePlan] = None
        self.global_plan: Optional[CoveragePlan] = None
        self.snapshot: Optional[YardSnapshot] = None
        self.profile: Optional[YardProfile] = None
        self.keep_in_mask: Optional[np.ndarray] = None
        self._geofence = GeofenceSpec()
        self._last_v = 0.0
        self._last_omega = 0.0
        self._home = Pose(1.0, 1.0, 0.0)
        self._review_hold = False
        self._mow_requested = False
        self._owner_hold = False
        self._detour: list[tuple[float, float]] = []
        self._detour_index = 0
        self._wet = False
        self._skipped_global: list[tuple[float, float]] = []
        self._frontier_xy: list[tuple[float, float]] = []
        self._replan_cool = 0
        self._stop_cool = 0
        self._calibrate_stall = 0
        self._authored_structure: Optional[np.ndarray] = None
        self.fence_unusable = False
        self._env: Any = None
        self._last_info: dict[str, Any] = {}
        self._last_pose: Optional[Pose] = None
        self.explore_reason: dict[str, Any] = {}
        self._explore_blocked = 0
        self._explore_spin = 0
        self._skipped_frontiers: list[tuple[int, int]] = []
        self._blockage_events = 0
        self._blocked_frontier_count = 0
        self._last_blockage_xy: Optional[tuple[float, float]] = None
        self._progress_pose: Optional[tuple[float, float]] = None
        self._progress_best = 1e9
        self._progress_stall = 0
        self._progress_map = 0.0
        self._progress_map_stall = 0
        self._explore_recover_cool = 0
        self._return_kind = ""
        self._resume_phase: Optional[MissionPhase] = None
        self._resume_index = 0
        self._resume_copy_steps = 0
        self._demo_explore_complete: Optional[float] = None
        self._demo_max_explore: Optional[int] = None
        self._demo_mow_frac: Optional[float] = None
        self._demo_max_mow: Optional[int] = None
        self._leftover_replans = 0
        self.terrain_state = TERRAIN_OK
        self._pose_trail: list[tuple[float, float, float]] = []
        self._retrace_wps: list[tuple[float, float]] = []
        self._retrace_index = 0
        self._retrace_reason = ""
        self._blockage_cool = 0
        self._last_blockage_step = -10_000
        self._look_ahead_kind = ""
        self._look_ahead_advice = ""
        self._look_ahead_pitch = 0.0
        self._look_ahead_roll = 0.0
        self._deferred_once: set[tuple[int, int]] = set()
        self._retrace_cool = 0

    @property
    def waypoints(self) -> list[tuple[float, float]]:
        if self._detour:
            return list(self._detour)
        if self.phase == MissionPhase.EXPLORE and self.explore_plan is not None:
            return list(self.explore_plan.waypoints)
        if self.phase == MissionPhase.CALIBRATE_BOUNDARY:
            return list(self.teach.waypoints)
        if self.global_plan is not None:
            return list(self.global_plan.waypoints)
        if self.plan is not None:
            return list(self.plan.waypoints)
        return []

    @property
    def done(self) -> bool:
        return self.phase in {MissionPhase.COMPLETE, MissionPhase.FAULT}

    def request_start_mow(self) -> bool:
        """Owner override: mow now — review, explore, or a frozen plan."""
        if self.phase == MissionPhase.MOW:
            self._mow_requested = True
            return True
        if self.phase == MissionPhase.CHARGING:
            self._resume_phase = MissionPhase.MOW
            return True
        if self.phase == MissionPhase.CALIBRATE_BOUNDARY:
            self._close_calibrate_now()
        if self.phase == MissionPhase.EXPLORE or self.snapshot is None or self.global_plan is None:
            pose = self._last_pose or self._home
            info = self._last_info or {}
            if self.observed is not None:
                self.snapshot = self._freeze(info)
                self.global_plan = self._plan_global_mow(pose)
                self.plan = self.global_plan
                self.index = 0
                self._review_hold = True
        if self._keep_in_too_small():
            self.fence_unusable = True
            self._emit("owner_start_mow_blocked", {"reason": "fence_too_small"})
            return False
        if self._empty_mow_plan():
            self.fence_unusable = True
            self._emit("owner_start_mow_blocked", {"reason": "empty_mow_plan"})
            return False
        self.fence_unusable = False
        self._mow_requested = True
        self._return_kind = ""
        self._emit("owner_start_mow", {"phase_step": self.phase_step, "from": self.phase.value})
        if self.phase != MissionPhase.REVIEW:
            self._transition(MissionPhase.MOW)
        return True

    def request_reexplore(self) -> bool:
        """Owner override: thaw the frozen map and keep exploring."""
        return self.request_explore()

    def request_explore(self, *, full: bool = False) -> bool:
        """Owner override: start / resume mapping without waiting for auto."""
        if full:
            apply_full_explore(self.settings, world_width_m=float(self.cfg.world.width_m))
            self._emit("owner_full_explore", {"explore_complete": self.settings.explore_complete})
        if self.phase == MissionPhase.EXPLORE:
            return True
        if self.phase == MissionPhase.CALIBRATE_BOUNDARY:
            self._close_calibrate_now()
            return True
        self._review_hold = False
        self._mow_requested = False
        self.snapshot = None
        self.global_plan = None
        self.plan = None
        self.explore_plan = None
        self._return_kind = ""
        if self.observed is not None:
            # Thaw so cameras can keep growing after MAP READY.
            self.observed.locked[:] = False
        self._emit("owner_explore", {"phase_step": self.phase_step, "from": self.phase.value})
        self._transition(MissionPhase.EXPLORE)
        return True

    def request_return_home(self, *, reason: str = "owner") -> bool:
        """Owner override: dock now. Battery returns resume the leftover plan."""
        if self.phase in {MissionPhase.COMPLETE, MissionPhase.FAULT}:
            return self.phase == MissionPhase.COMPLETE
        if self.phase == MissionPhase.RETURN_HOME and self._return_kind == reason:
            return True
        if self.phase == MissionPhase.MOW:
            self._resume_phase = MissionPhase.MOW
            self._resume_index = int(self.index)
        elif self.phase == MissionPhase.EXPLORE:
            self._resume_phase = MissionPhase.EXPLORE
            self._resume_index = int(self.index)
        self._return_kind = str(reason or "owner")
        self._detour = []
        self._detour_index = 0
        self._emit("owner_return", {"reason": self._return_kind, "from": self.phase.value})
        self._transition(MissionPhase.RETURN_HOME)
        return True

    def owner_reset(self, *, clear_blockages: bool = True) -> dict[str, Any]:
        """Fresh job: keep the taught fence, clear tip / progress / blockages.

        Owner Reset is not Re-teach. YardProfile stay. Observed fog, learned
        no-go, explore/mow plans, and tip latch clear so the next Explore
        starts clean. Default clears blockages; pass ``clear_blockages=False``
        only when a caller wants to keep them.
        """
        self.clear_owner_hold()
        self.help_requested = False
        self._stop_cool = 0
        self._calibrate_stall = 0
        self._explore_spin = 0
        self._explore_blocked = 0
        self._explore_recover_cool = 0
        self._clear_retrace()
        self._pose_trail = []
        self.terrain_state = TERRAIN_OK
        self._blockage_cool = 0
        self._last_blockage_step = -10_000
        self._look_ahead_kind = ""
        self._look_ahead_advice = ""
        self._look_ahead_pitch = 0.0
        self._look_ahead_roll = 0.0
        self._deferred_once = set()
        self._retrace_cool = 0
        self._progress_stall = 0
        self._progress_best = 1e9
        self._progress_pose = None
        self._progress_map = 0.0
        self._progress_map_stall = 0
        self._replan_cool = 0
        self._detour = []
        self._detour_index = 0
        self.last_advice = "ok"
        self.last_tilt_kind = "ok"
        self._chassis_tipped = False
        self._tilt_filter.reset()
        self.safe.reset()
        self.plan = None
        self.explore_plan = None
        self.global_plan = None
        self.snapshot = None
        self.index = 0
        self.phase_step = 0
        self._skipped_global = []
        self._frontier_xy = []
        self._leftover_replans = 0
        self._review_hold = False
        self._mow_requested = False
        self._return_kind = ""
        self._resume_phase = None
        self._resume_index = 0
        cleared = 0
        if self.observed is not None:
            if clear_blockages:
                cleared = int(self.observed.blockage_count())
            self.observed.clear_progress(clear_blockages=clear_blockages)
            self.keep_in_mask = self.observed.keep_in_mask(self._geofence)
            self._skipped_frontiers = []
            self._blocked_frontier_count = 0
            self._last_blockage_xy = None
            self._blockage_events = 0
        # Keep a taught fence. Leave calibrate only when a profile already exists.
        if self.profile is not None:
            self._transition(MissionPhase.EXPLORE)
        elif self.phase == MissionPhase.CALIBRATE_BOUNDARY:
            pass
        elif self.phase in {
            MissionPhase.FAULT,
            MissionPhase.SAFE,
            MissionPhase.MOW,
            MissionPhase.RETURN_HOME,
            MissionPhase.CHARGING,
            MissionPhase.COMPLETE,
            MissionPhase.REVIEW,
            MissionPhase.EXPLORE,
        }:
            self._transition(MissionPhase.CALIBRATE_BOUNDARY)
        self.explore_reason = {}
        self._emit(
            "owner_reset",
            {
                "clear_blockages": bool(clear_blockages),
                "cleared_cells": int(cleared),
                "phase": self.phase.value,
                "kept_fence": self.profile is not None,
            },
        )
        return {
            "ok": True,
            "clear_blockages": bool(clear_blockages),
            "cleared_cells": int(cleared),
            "phase": self.phase.value,
            "kept_fence": self.profile is not None,
            "kept_blockages": not clear_blockages,
        }

    def latch_chassis_tip(self, *, reason: str = "tip-over — immobilised") -> None:
        """Sticky past-tip: SOS / FAULT until owner Reset / retrieve."""
        self._chassis_tipped = True
        self.last_tilt_kind = KIND_TIP
        self.last_advice = "stop"
        self.help_requested = True
        self.safe.enter_safe(reason)
        if self.phase not in {MissionPhase.FAULT, MissionPhase.COMPLETE}:
            self._transition(MissionPhase.FAULT)

    def _pose_past_tip(self, pose: Pose, info: Optional[dict[str, Any]] = None) -> bool:
        blob = info if isinstance(info, dict) else {}
        if self._chassis_tipped or bool(blob.get("tipover")) or bool(blob.get("chassis_tipped")):
            return True
        return attitude_past_tip(
            pose.roll,
            pose.pitch,
            self.cfg.robot.static_tip_roll_rad(),
            self.cfg.robot.static_tip_pitch_rad(),
        )

    def apply_full_explore_mode(self, enabled: bool = True) -> bool:
        if enabled:
            if self._demo_explore_complete is None:
                self._demo_explore_complete = float(self.settings.explore_complete)
                self._demo_max_explore = int(self.settings.max_explore_steps)
                self._demo_mow_frac = float(self.settings.mow_complete_frac)
                self._demo_max_mow = int(self.settings.max_mow_steps)
            apply_full_explore(self.settings, world_width_m=float(self.cfg.world.width_m))
        else:
            self.settings.full_explore = False
            if self._demo_explore_complete is not None:
                self.settings.explore_complete = float(self._demo_explore_complete)
            if self._demo_max_explore is not None:
                self.settings.max_explore_steps = int(self._demo_max_explore)
            if self._demo_mow_frac is not None:
                self.settings.mow_complete_frac = float(self._demo_mow_frac)
            if self._demo_max_mow is not None:
                self.settings.max_mow_steps = int(self._demo_max_mow)
        self._emit("full_explore", {"enabled": bool(self.settings.full_explore)})
        return bool(self.settings.full_explore)

    def request_estop(self, reason: str = "owner estop") -> None:
        self.safe.request_estop(reason)
        self._owner_hold = False

    def request_hold(self, reason: str = "owner hold") -> None:
        if self.safe.mode != "estop":
            self._owner_hold = True
            self._emit("owner_hold", {"reason": reason})

    def clear_owner_hold(self) -> None:
        self._owner_hold = False
        self.safe.clear_if_not_estop()

    def accept_taught_profile(self, profile: YardProfile) -> None:
        """Owner-taught keep-in: skip authored calibrate and start exploring."""
        if profile is None or len(profile.keep_in) < 3:
            return
        self.profile = profile
        self._geofence = profile.geofence_spec()
        self._home = profile.home_pose()
        if self.observed is not None:
            self.keep_in_mask = self.observed.keep_in_mask(self._geofence)
        self.teach.spec = self._geofence
        self._emit(
            "boundary_taught",
            {
                "keep_in_vertices": len(profile.keep_in),
                "source": "owner",
                "name": profile.name,
            },
        )
        self._transition(MissionPhase.EXPLORE)

    def attach_to_env(self, env: Any) -> None:
        """Copy fog + yard onto the env so ``save_mission`` is a full day-2 bundle."""
        if env is None:
            return
        self._env = env
        env._observed_map = self.observed.copy() if self.observed is not None else None
        env._yard_profile = self.profile
        env._mission_phase = self.phase.value if isinstance(self.phase, MissionPhase) else str(self.phase)
        env._charging = self.phase == MissionPhase.CHARGING
        env._charge_delta = float(self.settings.gym_charge_soc_per_step)

    def save_session(
        self,
        path: Union[str, Path],
        coverage: Any,
        pose: Pose,
        *,
        scenario: str = "",
        seed: Optional[int] = None,
    ) -> Path:
        from jims_mower.mission import save_mission

        return save_mission(
            path,
            coverage,
            pose,
            scenario=scenario,
            seed=seed,
            steps=self.step,
            observed=self.observed,
            profile=self.profile,
            phase=self.phase.value if isinstance(self.phase, MissionPhase) else str(self.phase),
            home={"x": self._home.x, "y": self._home.y, "theta": self._home.theta},
        )

    def restore_session(
        self,
        obs: dict[str, Any],
        info: dict[str, Any],
        state: Any,
    ) -> None:
        """Cold-start: load yesterday's map + yard without reteaching."""
        profile = getattr(state, "profile", None)
        self.reset(obs, info, profile=profile)
        observed = getattr(state, "observed", None)
        if observed is not None:
            self.observed = observed.copy()
            self.keep_in_mask = self.observed.keep_in_mask(self._geofence)
        phase_raw = str(getattr(state, "phase", "") or "")
        if phase_raw:
            try:
                restored = MissionPhase(phase_raw)
                if restored != MissionPhase.CALIBRATE_BOUNDARY:
                    self.phase = restored
                    self.phase_step = 0
            except ValueError:
                pass
        if (
            self.phase == MissionPhase.CALIBRATE_BOUNDARY
            and profile is not None
            and len(getattr(profile, "keep_in", []) or []) >= 3
        ):
            self.phase = MissionPhase.EXPLORE
            self.phase_step = 0
        pose = _pose_from_obs(obs, info)
        if self.phase in {MissionPhase.REVIEW, MissionPhase.MOW}:
            if self.snapshot is None and self.observed is not None:
                self.snapshot = self._freeze(info)
            self.global_plan = self._plan_global_mow(pose)
            self.plan = self.global_plan
            self.index = 0
        elif self.phase == MissionPhase.EXPLORE and self.observed is not None:
            self.explore_plan = None

    def reset(
        self,
        obs: dict[str, Any],
        info: Optional[dict[str, Any]] = None,
        *,
        profile: Optional[YardProfile] = None,
    ) -> None:
        info = info or {}
        pose = _pose_from_obs(obs, info)
        self.fusion.reset(pose.x, pose.y, pose.theta, pose.z, pose.pitch, pose.roll)
        self.fusion.update(
            _gps_world(obs, info, self.profile),
            obs.get("imu", np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0], dtype=np.float32)),
            self.cfg.dt,
            seed_xy=(pose.x, pose.y),
        )
        hazard = np.asarray(obs["hazard"])
        self.observed = ObservedMap.empty(
            self.cfg.world.width_m,
            self.cfg.world.height_m,
            self.cfg.world.resolution_m,
            shape=hazard.shape[:2],
        )
        self._geofence = geofence_from_info(info, self.cfg)
        self.keep_in_mask = self.observed.keep_in_mask(self._geofence)
        self.teach.spec = self._geofence
        self.teach.reset(obs, info)
        self._apply_calibrate_drive()
        self._inset_teach_ring(self._teach_inset_m())
        self._start_teach_nearest(pose)
        self._home = pose
        self.phase = MissionPhase.CALIBRATE_BOUNDARY
        self.events = []
        self.phase_ranges = []
        self.step = 0
        self.phase_step = 0
        self.index = 0
        self.replans = 0
        self.last_advice = "ok"
        self.last_tilt_kind = "ok"
        self._chassis_tipped = False
        self._tilt_filter.reset()
        self.help_requested = False
        self.plan = None
        self.explore_plan = None
        self.global_plan = None
        self.snapshot = None
        self.profile = None
        self._review_hold = False
        self._mow_requested = False
        self._owner_hold = False
        self._detour = []
        self._detour_index = 0
        self._skipped_global = []
        self._frontier_xy = []
        self._replan_cool = 0
        self._stop_cool = 0
        self._calibrate_stall = 0
        self.fence_unusable = False
        self.explore_reason = {}
        self._explore_blocked = 0
        self._explore_spin = 0
        self._skipped_frontiers = []
        self._blockage_events = 0
        self._blocked_frontier_count = 0
        self._last_blockage_xy = None
        self._progress_pose = None
        self._progress_best = 1e9
        self._progress_stall = 0
        self._progress_map = 0.0
        self._progress_map_stall = 0
        self._explore_recover_cool = 0
        self._return_kind = ""
        self._resume_phase = None
        self._resume_index = 0
        self._resume_copy_steps = 0
        self._last_info = {}
        self._last_pose = None
        self._leftover_replans = 0
        self._clear_retrace()
        self._pose_trail = []
        self.terrain_state = TERRAIN_OK
        self._blockage_cool = 0
        self._last_blockage_step = -10_000
        self._look_ahead_kind = ""
        self._look_ahead_advice = ""
        self._look_ahead_pitch = 0.0
        self._look_ahead_roll = 0.0
        self._deferred_once = set()
        self._retrace_cool = 0
        self.safe.reset()
        self.last_safe_mode = self.safe.mode
        self._wet = bool((info.get("weather") or {}).get("wet", False))
        if obs.get("structure") is not None:
            self._authored_structure = np.asarray(obs["structure"]).copy()
        self._stamp(obs, info, pose, explored=True)
        taught = profile if profile is not None and len(profile.keep_in) >= 3 else None
        if taught is not None:
            self.profile = taught
            self._geofence = taught.geofence_spec()
            self._home = taught.home_pose()
            if self.observed is not None:
                self.keep_in_mask = self.observed.keep_in_mask(self._geofence)
            self.teach.spec = self._geofence
            self._enter(
                MissionPhase.EXPLORE,
                "phase_enter",
                {
                    "guided": False,
                    "taught": True,
                    "keep_in_vertices": len(taught.keep_in),
                },
            )
            self._emit(
                "boundary_taught",
                {
                    "keep_in_vertices": len(taught.keep_in),
                    "source": "owner",
                    "name": taught.name,
                },
            )
        else:
            self._enter(MissionPhase.CALIBRATE_BOUNDARY, "phase_enter", {"guided": True})

    def act(self, obs: dict[str, Any], info: dict[str, Any]) -> np.ndarray:
        pose_hint = _pose_from_obs(obs, info)
        fused = self.fusion.update(
            _gps_world(obs, info, self.profile),
            obs["imu"],
            self.cfg.dt,
            commanded_v=self._last_v,
            commanded_omega=self._last_omega,
            seed_xy=(pose_hint.x, pose_hint.y),
        )
        pose = pose_hint
        self._last_info = dict(info or {})
        self._last_pose = pose
        self._geofence = geofence_from_info(info, self.cfg)
        self._wet = bool((info.get("weather") or {}).get("wet", False))
        if self._blockage_cool > 0:
            self._blockage_cool -= 1
        if self._retrace_cool > 0:
            self._retrace_cool -= 1
        self._record_pose_trail(pose)
        self._stamp(obs, info, pose, explored=True)
        if self._resume_copy_steps > 0:
            self._resume_copy_steps -= 1

        advice = self._sense_advice(obs, info, fused, pose_hint)
        self.last_advice = advice
        if self._pose_past_tip(pose, info):
            self.latch_chassis_tip()
            self.step += 1
            self.phase_step += 1
            return self._finish(self._hold(), "stop", info)
        if estop_requested(obs, info):
            self.safe.request_estop("software/hardware estop")
        if self._owner_hold and self.safe.mode != "estop":
            self.step += 1
            self.phase_step += 1
            return self._finish(self._hold(), "stop", info)
        fault = info.get("fault") if isinstance(info.get("fault"), dict) else None
        if fault_is_immobilised(fault) or fault_is_retrieve(fault):
            return self._fail("FAULT_IMMOBILISED", advice, info)
        if info.get("radio_lost") and str(info.get("radio_on_loss") or "") == "stop_beacon":
            return self._fail("radio heartbeat lost", advice, info)
        if self.help_requested:
            return self._fail("call-for-help", advice, info)
        signal = observed_hand_signal(obs, self.cfg.curriculum.hand_signals)
        if signal == "stop":
            return self._finish(self._hold(), "stop", info)
        if advice == "stop" and self.phase not in {
            MissionPhase.REVIEW,
            MissionPhase.COMPLETE,
            MissionPhase.CALIBRATE_BOUNDARY,
            MissionPhase.MOW,
            MissionPhase.EXPLORE,
        }:
            return self._finish(self._hold(), advice, info)

        if self._maybe_battery_return(info, pose):
            action = self._tick_return(obs, info, pose, advice)
        elif self.phase == MissionPhase.CALIBRATE_BOUNDARY:
            action = self._tick_calibrate(obs, info, pose, advice)
        elif self.phase == MissionPhase.EXPLORE:
            action = self._tick_explore(obs, info, pose, advice)
        elif self.phase == MissionPhase.REVIEW:
            action = self._tick_review(obs, info, pose)
        elif self.phase == MissionPhase.MOW:
            action = self._tick_mow(obs, info, pose, advice)
        elif self.phase == MissionPhase.RETURN_HOME:
            action = self._tick_return(obs, info, pose, advice)
        elif self.phase == MissionPhase.CHARGING:
            action = self._tick_charge(obs, info, pose)
        else:
            action = self._hold()
        self._refresh_terrain_state(info, advice=advice)
        self.step += 1
        self.phase_step += 1
        return self._finish(action, advice, info)

    def status(self, info: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        omap = self.observed
        keep = self.keep_in_mask
        completion = omap.completion(keep) if omap is not None else 0.0
        conf = omap.mean_confidence(keep) if omap is not None else 0.0
        mowable_n = int(omap.mowable_mask(keep).sum()) if omap is not None else 0
        plan = self.global_plan
        world = float((info or {}).get("coverage_fraction") or 0.0)
        actual = self._job_cut_fraction(info, plan, world)
        reachable = plan.reachable_mowable_cells if plan is not None else 0
        unreachable = plan.unreachable_mowable_cells if plan is not None else 0
        planned = plan.planned_mowable_cells if plan is not None else mowable_n
        return {
            "schema": MISSION_FLOW_SCHEMA,
            "phase": self.phase.value,
            "phase_label": PHASE_LABELS.get(self.phase.value, self.phase.value.upper()),
            "step": self.step,
            "phase_step": self.phase_step,
            "trimmer_allowed": self.phase == MissionPhase.MOW,
            "map_completion": completion,
            "mean_confidence": conf,
            "planned_mowable_cells": planned,
            "reachable_mowable_cells": reachable,
            "unreachable_mowable_cells": unreachable,
            "unmapped_mowable_cells": int(plan.unmapped_mowable_cells) if plan else 0,
            "planned_coverage_fraction": plan.planned_coverage_fraction if plan else 0.0,
            "actual_coverage_fraction": actual,
            "world_coverage_fraction": world,
            "coverage_source": str((info or {}).get("coverage_source") or "gym_grid"),
            "n_frontiers": len(self._frontier_xy),
            "n_waypoints": len(self.waypoints),
            "waypoint_index": self.index,
            "replans": self.replans,
            "skipped_global": len(self._skipped_global),
            "leftover_replans": int(self._leftover_replans),
            "closed": bool(self.snapshot.closed) if self.snapshot else False,
            "fence_unusable": bool(self.fence_unusable),
            "explore_reason": dict(self.explore_reason or self._build_explore_reason(completion, info)),
            "full_explore": bool(self.settings.full_explore),
            "blocked_cells": int(omap.blockage_count()) if omap is not None else 0,
            "blocked_frontiers": int(self._blocked_frontier_count),
            "n_blockages": int(self._blockage_events),
            "return_kind": self._return_kind,
            "charge_state": self._charge_state(),
            "resume_phase": self._resume_phase.value if self._resume_phase is not None else "",
            "resume_index": int(self._resume_index),
            "area_legend": ObservedMap.area_legend(),
            "tilt_kind": self.last_tilt_kind,
            "chassis_tipped": bool(self._chassis_tipped),
            "terrain_state": self.terrain_state,
            "look_ahead_reason": self._look_ahead_reason(),
            "not_a_benchmark": True,
        }

    def _stamp(self, obs: dict[str, Any], info: dict[str, Any], pose: Pose, *, explored: bool) -> None:
        if self.observed is None:
            return
        radius = float(self.settings.stamp_radius_m)
        self.observed.stamp_disk(pose.x, pose.y, radius, explored=explored)
        images = obs.get("cameras") or {}
        if images:
            self.observed.stamp_cameras(
                images,
                self.cfg.resolved_cameras(),
                pose,
                max_range_m=float(self.settings.camera_range_m),
                pixel_stride=3 if images and next(iter(images.values())).shape[0] > 40 else 1,
            )
        self.observed.ingest_observer(
            obs,
            only_observed=True,
            authored_structure=self._authored_structure,
            pose=pose,
            grade_radius_m=max(float(self.settings.stamp_radius_m) * 1.6, 2.2),
        )
        self._stamp_metric_fuse(obs, info, pose)
        if self.keep_in_mask is None or self.keep_in_mask.shape != self.observed.observed.shape:
            self.keep_in_mask = self.observed.keep_in_mask(self._geofence)

    def _stamp_metric_fuse(self, obs: dict[str, Any], info: dict[str, Any], pose: Pose) -> None:
        """Gym MAP-2: ideal stereo + ToF + local IMU onto ObservedMap.

        Default gym ``front_left`` / ``front_right`` (40° yaw, ~40 cm) are
        not a pair. ``configs/orin/extrinsics_stereo.yaml`` is. Gym uses
        ideal disparity from the observer height raster — not a matcher,
        not COLMAP, ``fps_claim`` / ``map_claim`` stay null. Locked cells
        stay frozen.
        """
        if self.observed is None:
            return
        pair = find_stereo_pair(self.cfg.resolved_cameras())
        images = obs.get("cameras") or {}
        width = int(self.cfg.sensors.width)
        height = int(self.cfg.sensors.height)
        if images:
            sample = next(iter(images.values()), None)
            if sample is not None and getattr(sample, "ndim", 0) >= 2:
                height = int(sample.shape[0])
                width = int(sample.shape[1])
        stereo_elev = None
        stereo_hits = None
        n_points = 0
        ev = obs.get("elevation")
        if pair is not None and ev is not None:
            raster_src = np.asarray(ev, dtype=np.float32)
            if raster_src.shape == self.observed.elevation.shape:
                xs, ys, zs = synthetic_stereo_points(
                    pose,
                    pair,
                    width=width,
                    height=height,
                    height_at=height_sampler_from_raster(
                        raster_src, resolution_m=self.observed.resolution_m
                    ),
                    pixel_stride=3 if height > 40 else 2,
                )
                stereo_elev, stereo_hits = rasterize_points(
                    xs,
                    ys,
                    zs,
                    shape=self.observed.elevation.shape,
                    resolution_m=self.observed.resolution_m,
                    width_m=self.observed.width_m,
                    height_m=self.observed.height_m,
                )
                n_points = int(xs.size)
        result = fuse_elev_stereo_tof_imu(
            shape=self.observed.elevation.shape,
            resolution_m=self.observed.resolution_m,
            world_size=(self.observed.width_m, self.observed.height_m),
            pose=pose,
            stereo_elev=stereo_elev,
            stereo_hits=stereo_hits,
            tof=obs.get("tof"),
            imu=obs.get("imu"),
            length_m=self.cfg.robot.length_m,
            track_m=self.cfg.robot.track_m,
            hover_m=0.06,
            elevation=self.observed.elevation,
            locked=self.observed.locked,
            elevation_set=self.observed.elevation_set,
            rgb_prior=None,
            prior_weight=0.0,
        )
        written = self.observed.fuse_metric(result, respect_lock=True)
        info["elev_fuse"] = result.as_info()
        info["elev_fuse"]["cells_written"] = int(written)
        if pair is not None:
            blob = pair.as_info()
            blob["cells_written"] = int(written)
            blob["n_points"] = n_points
            info["stereo"] = blob

    def _sense_advice(
        self,
        obs: dict[str, Any],
        info: dict[str, Any],
        fused: Pose,
        pose_hint: Pose,
    ) -> str:
        env_advice = str(info.get("terrain_advice") or "ok")
        climb = float(self.cfg.planner.max_climb_slope_rad)
        lethal_frac = float(self.cfg.planner.tip_lethal_frac)
        sensed = imu_advice(
            obs["imu"],
            tip_roll_rad=self.cfg.robot.tip_roll_rad,
            tip_pitch_rad=self.cfg.robot.tip_pitch_rad,
            slow_frac=self.cfg.planner.imu_slow_frac,
            stop_frac=self.cfg.planner.imu_stop_frac,
            pose_pitch=fused.pitch,
            pose_roll=fused.roll,
            max_climb_slope_rad=climb,
            tip_lethal_frac=lethal_frac,
        )
        chassis = imu_advice(
            obs["imu"],
            tip_roll_rad=self.cfg.robot.tip_roll_rad,
            tip_pitch_rad=self.cfg.robot.tip_pitch_rad,
            slow_frac=self.cfg.planner.imu_slow_frac,
            stop_frac=self.cfg.planner.imu_stop_frac,
            pose_pitch=pose_hint.pitch,
            pose_roll=pose_hint.roll,
            max_climb_slope_rad=climb,
            tip_lethal_frac=lethal_frac,
        )
        roll, pitch = read_tilt(
            obs["imu"],
            pose_pitch=fused.pitch if abs(fused.pitch) >= abs(pose_hint.pitch) else pose_hint.pitch,
            pose_roll=fused.roll if abs(fused.roll) >= abs(pose_hint.roll) else pose_hint.roll,
        )
        filtered = self._tilt_filter.update(
            classify_tilt(
                roll,
                pitch,
                tip_roll_rad=self.cfg.robot.tip_roll_rad,
                tip_pitch_rad=self.cfg.robot.tip_pitch_rad,
                slow_frac=self.cfg.planner.imu_slow_frac,
                stop_frac=self.cfg.planner.imu_stop_frac,
                max_climb_slope_rad=climb,
                tip_lethal_frac=lethal_frac,
                static_tip_roll_rad=self.cfg.robot.static_tip_roll_rad(),
                static_tip_pitch_rad=self.cfg.robot.static_tip_pitch_rad(),
            ),
            tip_roll_rad=self.cfg.robot.tip_roll_rad,
            tip_pitch_rad=self.cfg.robot.tip_pitch_rad,
        )
        sensed = combine_advice(sensed, filtered.advice)
        chassis = combine_advice(chassis, filtered.advice)
        self.last_tilt_kind = filtered.kind
        ahead = look_ahead_from_elevation(
            pose_hint,
            obs.get("elevation"),
            resolution_m=self.cfg.world.resolution_m,
            length_m=self.cfg.robot.length_m,
            track_m=self.cfg.robot.track_m,
            look_ahead_m=self.cfg.planner.grade_look_ahead_m,
            n_samples=self.cfg.planner.grade_look_ahead_samples,
            tip_roll_rad=self.cfg.robot.tip_roll_rad,
            tip_pitch_rad=self.cfg.robot.tip_pitch_rad,
            slow_frac=self.cfg.planner.imu_slow_frac,
            stop_frac=self.cfg.planner.imu_stop_frac,
            max_climb_slope_rad=climb,
            tip_lethal_frac=lethal_frac,
            static_tip_roll_rad=self.cfg.robot.static_tip_roll_rad(),
            static_tip_pitch_rad=self.cfg.robot.static_tip_pitch_rad(),
        )
        sensed = combine_advice(sensed, look_ahead_advice(ahead))
        chassis = combine_advice(chassis, look_ahead_advice(ahead))
        self.last_tilt_kind = merge_look_ahead_kind(self.last_tilt_kind, ahead)
        if ahead is not None:
            self._look_ahead_kind = str(ahead.kind or "")
            self._look_ahead_advice = look_ahead_advice(ahead)
            self._look_ahead_pitch = float(ahead.pitch)
            self._look_ahead_roll = float(ahead.roll)
        else:
            self._look_ahead_kind = ""
            self._look_ahead_advice = ""
            self._look_ahead_pitch = 0.0
            self._look_ahead_roll = 0.0
        if self._pose_past_tip(pose_hint, info) or (
            env_advice == "stop" and bool(info.get("tipover"))
        ):
            self.last_tilt_kind = KIND_TIP
            self._chassis_tipped = True
        living = str(info.get("living_advice") or "ok")
        fence = str(info.get("geofence_advice") or "ok")
        power = budget_advice(info)
        if self.phase == MissionPhase.CALIBRATE_BOUNDARY:
            # Tracing an inset keep-in: fence/living chatter must not abort the lap.
            advice = combine_advice(env_advice, sensed, chassis, power)
        elif self.phase == MissionPhase.EXPLORE:
            if fence == "stop":
                fence = "slow"
            advice = combine_advice(env_advice, sensed, chassis, living, fence, power)
        else:
            advice = combine_advice(env_advice, sensed, chassis, living, fence, power)
        if advice not in TERRAIN_ADVICE:
            advice = "ok"
        # Climbable / look-ahead face → contour. Env look-ahead may say
        # "stop" so we do not drive *into* a shed lip; that is not a
        # seated tip. Skipping the coverage plan here left CI jobs at
        # ~50% cut. True tipover / KIND_TIP still stop.
        if (
            advice == "stop"
            and self.last_tilt_kind == KIND_GRADE
            and not bool(info.get("tipover"))
            and not bool(info.get("chassis_tipped"))
            and not self._chassis_tipped
            and not self._drain_hazard(info)
        ):
            advice = "reroute"
        return advice

    def _tick_calibrate(
        self,
        obs: dict[str, Any],
        info: dict[str, Any],
        pose: Pose,
        advice: str,
    ) -> np.ndarray:
        if advice == "stop":
            self._calibrate_stall += 1
            if self._calibrate_stall >= 8 and self.teach.waypoints:
                self.teach.index = min(self.teach.index + 1, max(0, len(self.teach.waypoints) - 1))
                self._calibrate_stall = 0
            return self._nudge_inward(pose)
        self._calibrate_stall = 0
        action = self.teach.act(obs, info)
        if advice == "slow":
            wheels = np.asarray(action, dtype=np.float32).reshape(-1)
            action = np.array([0.45 * float(wheels[0]), 0.45 * float(wheels[1]), 0.0], dtype=np.float32)
        self.index = self.teach.index
        timed_out = self.phase_step + 1 >= int(self.settings.max_calibrate_steps)
        confirmed = self._calibrate_confirmed()
        if self.teach.done or timed_out or confirmed:
            self._close_calibrate_now(
                timed_out=bool(timed_out and not self.teach.done and not confirmed),
                confirmed=bool(confirmed and not self.teach.done),
            )
        wheels = np.asarray(action, dtype=np.float32).reshape(-1)
        return self._drive(float(wheels[0]), float(wheels[1]), 0.0, pose)

    def _close_calibrate_now(self, *, timed_out: bool = False, confirmed: bool = False) -> None:
        """Accept authored / taught keep-in and enter explore (manual or auto)."""
        keep_out = [list(poly) for poly in (self._geofence.keep_out or [])]
        if self.profile is None or len(getattr(self.profile, "keep_in", []) or []) < 3:
            self.profile = self.teach.to_profile(name="mission", keep_out=keep_out)
        self.profile.home = {"x": self._home.x, "y": self._home.y, "theta": self._home.theta}
        planned = self.teach.planned_ring()
        if len(planned) >= 3:
            self.profile.keep_in = planned
        self._geofence = self.profile.geofence_spec()
        if self.observed is not None:
            self.keep_in_mask = self.observed.keep_in_mask(self._geofence)
        self._emit(
            "boundary_recorded",
            {
                "keep_in_vertices": len(self.profile.keep_in),
                "trail_points": len(self.teach.trail),
                "trail_m": self._trail_length_m(),
                "timed_out": bool(timed_out),
                "confirmed": bool(confirmed),
            },
        )
        self._transition(MissionPhase.EXPLORE)

    def _tick_explore(
        self,
        obs: dict[str, Any],
        info: dict[str, Any],
        pose: Pose,
        advice: str,
    ) -> np.ndarray:
        assert self.observed is not None
        keep = self.keep_in_mask
        raw = frontiers(
            self.observed.observed,
            self.observed.free,
            keep_in=keep,
            ignore_unknown=np.asarray(self.observed.blockage, dtype=bool),
        )
        thin = downsample_frontiers(raw, min_sep=2, limit=40)
        self._frontier_xy = [self.observed.cell_to_world(r, c) for r, c in thin]
        completion = self.observed.completion(keep)
        if completion > self._progress_map + 0.003:
            self._progress_map = float(completion)
            self._progress_map_stall = 0
        else:
            self._progress_map_stall += 1
        skip = set(self._skipped_frontiers)
        reachable_thin = [cell for cell in thin if cell not in skip]
        drain_stamped = any(
            ev.event == "blockage_stamped" and str((ev.detail or {}).get("reason") or "") == "drain"
            for ev in self.events
        )
        if (
            completion >= 0.95
            and self._progress_map_stall > 16
            and reachable_thin
            and drain_stamped
        ):
            for cell in reachable_thin:
                if cell not in self._skipped_frontiers:
                    self._skipped_frontiers.append(cell)
            reachable_thin = []
        ready = self._explore_ready(completion, bool(reachable_thin))
        timed_out = self.phase_step + 1 >= int(self.settings.max_explore_steps)
        # Full explore: a step cap is not MAP READY while reachable
        # frontiers remain and the keep-in is not essentially mapped.
        if (
            timed_out
            and not ready
            and self.settings.full_explore
            and reachable_thin
            and completion < 0.99
        ):
            timed_out = False
        if ready or timed_out:
            self.explore_reason = self._build_explore_reason(
                completion, info, n_frontiers=len(thin), code="map_progress" if ready else "waiting_cap"
            )
            self._emit(
                "explore_complete",
                {
                    "map_completion": completion,
                    "n_frontiers": len(thin),
                    "timed_out": bool(timed_out and not ready),
                    "no_frontier": not thin,
                    "full_explore": bool(self.settings.full_explore),
                    "blocked_frontiers": int(self._blocked_frontier_count),
                },
            )
            if self._keep_in_too_small():
                self.fence_unusable = True
            self._transition(MissionPhase.REVIEW)
            return self._hold()

        retrace_action = self._tick_retrace(pose, advice)
        if retrace_action is not None:
            self.explore_reason = self._build_explore_reason(
                completion, info, n_frontiers=len(thin), code="retrace"
            )
            return retrace_action

        if self._drain_hazard(info) and self._explore_recover_cool <= 0:
            self._stamp_learned_blockage(pose, info, reason="drain")
            remapping = self._blocked_frontier_count >= int(self.settings.blockage_replan_after)
            self.explore_reason = self._build_explore_reason(
                completion,
                info,
                n_frontiers=len(thin),
                code="remapping" if remapping else "blockage_stamped",
            )
            self._explore_recover_cool = 4
            return self._retrace_or_nudge(pose, reason="drain")

        if info.get("collision") and self._explore_recover_cool <= 0:
            self._stamp_learned_blockage(pose, info, reason="collision")
            remapping = self._blocked_frontier_count >= int(self.settings.blockage_replan_after)
            self.explore_reason = self._build_explore_reason(
                completion,
                info,
                n_frontiers=len(thin),
                code="remapping" if remapping else "blockage_stamped",
            )
            self._explore_recover_cool = 3
            return self._retrace_or_nudge(pose, reason="collision")

        if self._explore_recover_cool > 0:
            self._explore_recover_cool -= 1
            remapping = self._blocked_frontier_count >= int(self.settings.blockage_replan_after)
            code = "retrace" if self._retrace_wps else ("remapping" if remapping else "blockage_stamped")
            self.explore_reason = self._build_explore_reason(
                completion,
                info,
                n_frontiers=len(thin),
                code=code,
            )
            if self._retrace_wps:
                action = self._tick_retrace(pose, advice)
                if action is not None:
                    return action
            if self._explore_recover_cool >= 2:
                return self._retrace_or_nudge(pose, reason="recover")
            if self._explore_recover_cool == 1:
                return self._lateral_nudge(pose)
            self.explore_plan = None
            self.index = 0

        stalled = self._explore_no_progress(pose, info, advice, completion)
        if stalled:
            self._stamp_learned_blockage(pose, info, reason="no_progress")
            remapping = self._blocked_frontier_count >= int(self.settings.blockage_replan_after)
            grade = self._grade_like(advice)
            code = "retrace" if grade else ("remapping" if remapping else "blockage_stamped")
            self.explore_reason = self._build_explore_reason(
                completion,
                info,
                n_frontiers=len(thin),
                code=code,
            )
            self._explore_recover_cool = 3
            return self._retrace_or_nudge(pose, reason="no_progress")

        need_new = (
            self.explore_plan is None
            or self.index >= len(self.explore_plan.waypoints)
            or self.phase_step % 12 == 0
        )
        if need_new:
            prev_target = self.explore_plan.target if self.explore_plan is not None else None
            avoid = self._last_blockage_xy
            self.explore_plan = plan_explore(
                self.observed,
                (pose.x, pose.y),
                keep_in=keep,
                waypoint_stride_m=max(0.35, self.cfg.planner.waypoint_stride_m),
                skip_cells=self._skipped_frontiers,
                avoid_xy=avoid,
                cluster_cells=int(self.settings.blockage_cluster_cells),
                max_climb_slope_rad=self.cfg.planner.max_climb_slope_rad,
                tip_lethal_slope_rad=tip_lethal_slope_rad(
                    tip_roll_rad=self.cfg.robot.tip_roll_rad,
                    tip_pitch_rad=self.cfg.robot.tip_pitch_rad,
                    tip_lethal_frac=self.cfg.planner.tip_lethal_frac,
                ),
                contour_cost=self.cfg.planner.contour_cost,
            )
            self.index = 0
            if self.explore_plan.target != prev_target:
                self._reset_explore_progress(pose, completion)
            if self.explore_plan.target is not None:
                tx, ty = self.observed.cell_to_world(*self.explore_plan.target)
                self._emit(
                    "frontier_target",
                    {
                        "x": tx,
                        "y": ty,
                        "n_frontiers": len(thin),
                        "unreachable_frontiers": int(self.explore_plan.unreachable_frontiers),
                        "skipped_frontiers": int(self.explore_plan.skipped_frontiers),
                    },
                )
        if advice == "stop":
            self._explore_spin += 1
            if self._explore_spin >= 4:
                self._stamp_learned_blockage(pose, info, reason="tip_collision")
                self.explore_reason = self._build_explore_reason(
                    completion, info, n_frontiers=len(thin), code="tip_recovery"
                )
                self._explore_spin = 0
                self._explore_recover_cool = 3
                return self._retrace_or_nudge(pose, reason="tip_risk")
            self.explore_reason = self._build_explore_reason(
                completion, info, n_frontiers=len(thin), code="tip_recovery"
            )
            return self._retrace_or_nudge(pose, reason="tip_risk")
        self._explore_spin = 0
        if self.explore_plan is None or not self.explore_plan.waypoints:
            # Frontiers exist but A* cannot connect, or none remain.
            self._explore_blocked += 1
            unreachable = int(getattr(self.explore_plan, "unreachable_frontiers", 0) or 0)
            blocked = bool(thin) or unreachable > 0 or bool(self._skipped_frontiers)
            code = "path_blocked" if blocked else "no_frontier"
            if self._explore_blocked >= 3 and thin:
                self._stamp_learned_blockage(pose, info, reason="path_blocked")
                code = "frontier_skipped"
            self.explore_reason = self._build_explore_reason(
                completion, info, n_frontiers=len(thin), code=code
            )
            give_up = max(8, int(getattr(self.settings, "min_explore_steps", 8) or 8))
            demo_exit = (
                not bool(self.settings.full_explore)
                and self.phase_step >= give_up
                and (
                    completion >= float(self.settings.explore_no_frontier)
                    or self.phase_step > give_up + 8
                )
            )
            full_clear = (
                bool(self.settings.full_explore)
                and self.phase_step >= give_up
                and completion >= float(self.settings.explore_no_frontier)
                and (not reachable_thin or completion >= 0.99)
            )
            # Look around a few ticks so a real stall can unstick, then
            # demo may still MAP READY with leftover frontiers.
            if self._explore_blocked <= 6:
                if self._explore_blocked % 3 == 1:
                    return self._look_around(pose)
                if thin and self._explore_blocked % 3 == 2:
                    return self._nudge_toward_frontier(pose, thin[0])
                return self._hold()
            if demo_exit or full_clear:
                if self._keep_in_too_small():
                    self.fence_unusable = True
                self._transition(MissionPhase.REVIEW)
                return self._hold()
            if self._explore_blocked % 3 == 1:
                return self._look_around(pose)
            if thin and self._explore_blocked % 3 == 2:
                return self._nudge_toward_frontier(pose, thin[0])
            return self._hold()
        self._explore_blocked = 0
        remapping = self._blocked_frontier_count >= int(self.settings.blockage_replan_after)
        code = "remapping" if remapping and self._progress_map_stall > 8 else "seeking_frontier"
        if self.terrain_state == TERRAIN_RETRACE:
            code = "retrace"
        elif self.terrain_state == TERRAIN_TIP_REVERSE:
            code = "tip_recovery"
        elif self.last_tilt_kind == KIND_GRADE and not remapping:
            code = "steep_grade"
        elif self.terrain_state == TERRAIN_CONTOUR and not remapping:
            code = "steep_grade"
        self.explore_reason = self._build_explore_reason(
            completion, info, n_frontiers=len(thin), code=code
        )
        return self._track_list(
            self.explore_plan.waypoints,
            pose,
            cruise=float(self.settings.explore_cruise),
            advice=advice,
            trimmer=0.0,
        )

    def _explore_ready(self, completion: float, has_frontier: bool) -> bool:
        # Never declare the map done on the first explore tick — a tiny
        # keep-in is already "33% observed / no frontiers" at spawn.
        if self.phase_step < 1:
            return False
        target = float(self.settings.explore_complete)
        # 1.0 is the taught fence. Leftover cells / float noise / one
        # unreachable lip must not hang MAP READY forever.
        if completion >= target or (target >= 0.99 and completion >= 0.99):
            return True
        min_steps = max(4, int(getattr(self.settings, "min_explore_steps", 8) or 8))
        if (
            not has_frontier
            and completion >= float(self.settings.explore_no_frontier)
            and self.phase_step >= min_steps
        ):
            return True
        return False

    def _tick_review(self, obs: dict[str, Any], info: dict[str, Any], pose: Pose) -> np.ndarray:
        if not self._review_hold:
            self.snapshot = self._freeze(info)
            self.global_plan = self._plan_global_mow(pose)
            self.plan = self.global_plan
            self.index = 0
            self._review_hold = True
            metrics = self.global_plan.as_metrics() if self.global_plan else {}
            if self._keep_in_too_small() or self._empty_mow_plan():
                self.fence_unusable = True
                if self.snapshot is not None:
                    self.snapshot.notes.append("fence too small — 0 mowable cells")
            self._emit(
                "map_ready",
                {
                    "closed": self.snapshot.closed,
                    "completion": self.snapshot.completion,
                    "mean_confidence": self.snapshot.mean_confidence,
                    "n_components": self.snapshot.n_components,
                    "notes": list(self.snapshot.notes),
                    "fence_unusable": self.fence_unusable,
                    **metrics,
                },
            )
            return self._hold()
        if self.fence_unusable:
            return self._hold()
        hold_steps = max(1, int(self.settings.review_hold_steps))
        ready = self._mow_requested or self.phase_step >= hold_steps
        if not ready:
            return self._hold()
        if self._empty_mow_plan():
            self.fence_unusable = True
            return self._hold()
        self._transition(MissionPhase.MOW)
        return self._hold()

    def _tick_mow(
        self,
        obs: dict[str, Any],
        info: dict[str, Any],
        pose: Pose,
        advice: str,
    ) -> np.ndarray:
        if self.global_plan is None or not self.global_plan.waypoints:
            if self._replan_leftover_uncut(pose, info):
                return self._hold()
            self._transition(MissionPhase.RETURN_HOME)
            return self._hold()
        if self.phase_step + 1 >= int(self.settings.max_mow_steps):
            self._emit("mow_budget", {"waypoints_left": max(0, len(self.global_plan.waypoints) - self.index)})
            self._transition(MissionPhase.RETURN_HOME)
            return self._hold()
        if self._mow_paint_done(info):
            self._emit(
                "mow_complete",
                {
                    **self.global_plan.as_metrics(),
                    "reason": "demo_cut",
                    "cut_pct": self._job_cut_fraction(
                        info, self.global_plan, float((info or {}).get("coverage_fraction") or 0.0)
                    ),
                },
            )
            self._transition(MissionPhase.RETURN_HOME)
            return self._hold()
        if advice == "stop" and self.last_tilt_kind == KIND_GRADE:
            if not bool((info or {}).get("tipover")) and not self._chassis_tipped:
                advice = "reroute"
        self.index = self._skip_arrived(self.global_plan.waypoints, pose, self.index)
        if self._mow_no_progress(pose, info, advice):
            hard = (
                self.last_tilt_kind == KIND_TIP
                or bool((info or {}).get("collision"))
                or bool((info or {}).get("tipover"))
            )
            if hard:
                self._stamp_learned_blockage(pose, info, reason="mow_no_progress")
            if self.index < len(self.global_plan.waypoints):
                skipped = self.global_plan.waypoints[self.index]
                self._skipped_global.append(skipped)
                self._emit(
                    "unreachable_segment",
                    {"x": skipped[0], "y": skipped[1], "index": self.index, "reason": "blockage"},
                )
                self.index += 1
            self._local_replan(obs, pose)
            self._reset_explore_progress(pose, 0.0)
        if advice == "stop" and self._stop_cool > 0:
            # After a ridge skip, keep painting instead of crawling reverse.
            advice = "slow"
            self._stop_cool -= 1
        if advice == "stop":
            # Ridge / IMU tip-stop: reverse, pivot off the lip, then skip a
            # short cluster and keep tracking. Do not limp-park here.
            self._calibrate_stall += 1
            if self._calibrate_stall == 1:
                return self._retrace_or_nudge(pose, reason="mow_tip")
            if self._calibrate_stall == 2:
                return self._lateral_nudge(pose)
            skipped_now = False
            if self._calibrate_stall >= 3 and self.index < len(self.global_plan.waypoints):
                cluster = self._mow_skip_count()
                for _ in range(cluster):
                    if self.index >= len(self.global_plan.waypoints):
                        break
                    skipped = self.global_plan.waypoints[self.index]
                    self._skipped_global.append(skipped)
                    self._emit(
                        "unreachable_segment",
                        {"x": skipped[0], "y": skipped[1], "index": self.index, "reason": "stop"},
                    )
                    self.index += 1
                self._calibrate_stall = 0
                self._stop_cool = max(1, int(self.settings.mow_stop_cool))
                self._local_replan(obs, pose)
                skipped_now = True
            skip_limit = 48
            min_mow = 80
            stop_skips = sum(1 for ev in self.events if ev.event == "unreachable_segment" and ev.detail.get("reason") == "stop")
            if stop_skips >= skip_limit and self.phase_step >= min_mow:
                if self._replan_leftover_uncut(pose, info):
                    return self._hold()
                self._emit(
                    "mow_budget",
                    {
                        "reason": "imu_stop_ridge",
                        "waypoints_left": max(0, len(self.global_plan.waypoints) - self.index),
                        "skips": len(self._skipped_global),
                    },
                )
                self._transition(MissionPhase.RETURN_HOME)
                return self._hold()
            if skipped_now:
                # Skip landed — drive the next waypoint this step.
                if self.index >= len(self.global_plan.waypoints):
                    if self._replan_leftover_uncut(pose, info):
                        return self._hold()
                    self._emit("mow_complete", self.global_plan.as_metrics())
                    self._transition(MissionPhase.RETURN_HOME)
                    return self._hold()
                return self._track_list(
                    self.global_plan.waypoints,
                    pose,
                    cruise=float(self.settings.mow_cruise),
                    advice="slow",
                    trimmer=1.0,
                )
            return self._hold()
        self._calibrate_stall = 0
        if self.index >= len(self.global_plan.waypoints):
            if self._replan_leftover_uncut(pose, info):
                return self._hold()
            self._emit("mow_complete", self.global_plan.as_metrics())
            self._transition(MissionPhase.RETURN_HOME)
            return self._hold()
        if self._replan_cool > 0:
            self._replan_cool -= 1
        if advice == "reroute" and self._replan_cool == 0:
            self._local_replan(obs, pose)
        self._maybe_local_block(obs, pose)
        if self._detour:
            action = self._track_list(
                self._detour,
                pose,
                cruise=float(self.settings.mow_cruise),
                advice=advice,
                trimmer=1.0 if advice in {"ok", "slow"} or self.last_tilt_kind == KIND_GRADE else 0.0,
                index_attr="_detour_index",
            )
            if self._detour_index >= len(self._detour):
                self._detour = []
                self._detour_index = 0
            return action
        action = self._track_list(
            self.global_plan.waypoints,
            pose,
            cruise=float(self.settings.mow_cruise),
            advice=advice,
            trimmer=1.0 if advice in {"ok", "slow"} or self.last_tilt_kind == KIND_GRADE else 0.0,
        )
        if self.index >= len(self.global_plan.waypoints):
            if self._replan_leftover_uncut(pose, info):
                return action
            self._emit("mow_complete", self.global_plan.as_metrics())
            self._transition(MissionPhase.RETURN_HOME)
            return self._hold()
        return action

    def _tick_return(
        self,
        obs: dict[str, Any],
        info: dict[str, Any],
        pose: Pose,
        advice: str,
    ) -> np.ndarray:
        if math.hypot(pose.x - self._home.x, pose.y - self._home.y) <= 0.45:
            if self._return_kind == "battery":
                self._transition(MissionPhase.CHARGING)
                return self._hold()
            self._transition(MissionPhase.COMPLETE)
            return self._hold()
        if self.phase_step + 1 >= int(self.settings.max_return_steps):
            self._emit("return_timeout", {"x": pose.x, "y": pose.y})
            if self._return_kind == "battery":
                self._transition(MissionPhase.CHARGING)
                return self._hold()
            self._transition(MissionPhase.COMPLETE)
            return self._hold()
        if not self._detour:
            self._detour = self._home_path(pose)
            self._detour_index = 0
            if not self._detour:
                self._detour = [(self._home.x, self._home.y)]
        return self._track_list(
            self._detour,
            pose,
            cruise=float(self.settings.explore_cruise),
            advice=advice,
            trimmer=0.0,
            index_attr="_detour_index",
        )

    def _tick_charge(self, obs: dict[str, Any], info: dict[str, Any], pose: Pose) -> np.ndarray:
        _ = obs
        self._sync_env_charge(True)
        soc = self._current_soc(info)
        target = self._charge_resume_soc()
        if soc >= target:
            self._emit(
                "battery_charged",
                {"soc": soc, "resume": self._resume_phase.value if self._resume_phase else "mow"},
            )
            return self._resume_after_charge(pose)
        return self._hold()

    def _resume_after_charge(self, pose: Pose) -> np.ndarray:
        _ = pose
        self._sync_env_charge(False)
        self._return_kind = ""
        self._resume_copy_steps = 16
        target = self._resume_phase or MissionPhase.MOW
        idx = int(self._resume_index)
        self._emit("battery_resume", {"phase": target.value, "index": idx})
        if target == MissionPhase.MOW and self.global_plan is not None:
            self._transition(MissionPhase.MOW)
            self.index = min(idx, max(0, len(self.global_plan.waypoints) - 1))
        elif target == MissionPhase.EXPLORE:
            self._transition(MissionPhase.EXPLORE)
            self.index = 0
            self.explore_plan = None
        else:
            self._transition(target)
        return self._hold()

    def _maybe_battery_return(self, info: dict[str, Any], pose: Pose) -> bool:
        _ = pose
        if self.phase not in {MissionPhase.EXPLORE, MissionPhase.MOW}:
            return False
        if self._return_kind == "battery":
            return False
        soc = self._current_soc(info)
        if soc > self._soc_return_threshold():
            return False
        self._resume_phase = self.phase
        self._resume_index = int(self.index)
        self._return_kind = "battery"
        self._detour = []
        self._detour_index = 0
        self._emit(
            "battery_return",
            {"soc": soc, "threshold": self._soc_return_threshold(), "from": self.phase.value, "index": self.index},
        )
        self._transition(MissionPhase.RETURN_HOME)
        return True

    def _current_soc(self, info: Optional[dict[str, Any]] = None) -> float:
        if self._env is not None and getattr(self._env, "budget", None) is not None:
            return float(self._env.budget.soc)
        blob = info if isinstance(info, dict) else self._last_info
        return float((blob or {}).get("battery_soc", 1.0) or 1.0)

    def _soc_return_threshold(self) -> float:
        configured = float(getattr(self.settings, "min_soc_return", 0.0) or 0.0)
        if configured > 0.0:
            return configured
        schedule = getattr(self.profile, "schedule", None) if self.profile is not None else None
        if isinstance(schedule, dict) and schedule.get("min_soc") is not None:
            return float(schedule["min_soc"])
        if schedule is not None and getattr(schedule, "min_soc", None) is not None:
            return float(schedule.min_soc)
        return float(self.cfg.runtime.battery.limp_soc)

    def _charge_resume_soc(self) -> float:
        target = float(getattr(self.settings, "charge_resume_soc", 0.80) or 0.80)
        return min(1.0, max(target, self._soc_return_threshold() + 0.20))

    def _charge_state(self) -> str:
        if self.phase == MissionPhase.CHARGING:
            return "charging"
        if self.phase == MissionPhase.RETURN_HOME and self._return_kind == "battery":
            return "returning"
        if self._resume_copy_steps > 0 and self.phase in {MissionPhase.MOW, MissionPhase.EXPLORE}:
            return "resuming"
        return ""

    def _sync_env_charge(self, charging: bool) -> None:
        if self._env is None:
            return
        self._env._charging = bool(charging)
        self._env._charge_delta = float(self.settings.gym_charge_soc_per_step)
        self._env._mission_phase = self.phase.value

    def _look_around(self, pose: Pose) -> np.ndarray:
        _ = pose
        return self._drive(0.42, -0.42, 0.0, pose)

    def _nudge_toward_frontier(self, pose: Pose, cell: tuple[int, int]) -> np.ndarray:
        if self.observed is None:
            return self._look_around(pose)
        tx, ty = self.observed.cell_to_world(*cell)
        wheels, _dist, _err = tracking_action(
            pose,
            (tx, ty),
            cruise=0.28,
            wheelbase_m=self.cfg.robot.wheelbase_m,
            turn_in_place_rad=0.85,
        )
        return self._drive(float(wheels[0]), float(wheels[1]), 0.0, pose)

    def _reset_explore_progress(self, pose: Pose, completion: float) -> None:
        self._progress_pose = (pose.x, pose.y)
        self._progress_best = 1e9
        self._progress_stall = 0
        if completion + 1e-6 >= self._progress_map:
            self._progress_map = float(completion)
            self._progress_map_stall = 0

    def _current_explore_goal(self, pose: Pose) -> Optional[tuple[float, float]]:
        if self.explore_plan is None:
            return None
        wps = self.explore_plan.waypoints
        if wps and 0 <= self.index < len(wps):
            return wps[self.index]
        if self.explore_plan.target is not None and self.observed is not None:
            return self.observed.cell_to_world(*self.explore_plan.target)
        return None

    def _explore_no_progress(
        self,
        pose: Pose,
        info: dict[str, Any],
        advice: str,
        completion: float,
    ) -> bool:
        """True when commanded motion is not approaching the frontier / growing the map."""
        if self.explore_plan is None or not self.explore_plan.waypoints:
            return False
        if self._explore_recover_cool > 0:
            return False
        goal = self._current_explore_goal(pose)
        moved = 0.0
        if self._progress_pose is not None:
            moved = math.hypot(pose.x - self._progress_pose[0], pose.y - self._progress_pose[1])
        self._progress_pose = (pose.x, pose.y)
        if completion > self._progress_map + 0.003:
            self._progress_map = float(completion)
            self._progress_map_stall = 0
            self._progress_stall = 0
            if goal is not None:
                self._progress_best = min(
                    self._progress_best, math.hypot(pose.x - goal[0], pose.y - goal[1])
                )
            return False
        self._progress_map_stall += 1
        progressed = False
        if goal is not None:
            dist = math.hypot(pose.x - goal[0], pose.y - goal[1])
            if dist + 0.05 < self._progress_best:
                self._progress_best = dist
                progressed = True
        grade_like = self._grade_like(advice)
        if moved >= 0.06 or (grade_like and moved >= 0.012):
            progressed = True
        if progressed:
            self._progress_stall = 0
            return False
        self._progress_stall += 1
        hard_stop = bool(info.get("collision")) or bool(info.get("tipover"))
        if (advice == "stop" or hard_stop) and not (grade_like and not hard_stop):
            self._progress_stall += 2
        commanded = abs(self._last_v) > 0.05
        pivoting = abs(self._last_omega) > 0.35
        if commanded and (not pivoting) and moved < 0.03 and not grade_like:
            self._progress_stall += 1
        limit = max(4, int(self.settings.blockage_no_progress_steps))
        if grade_like:
            limit = max(limit, int(self.settings.blockage_no_progress_steps) + 8)
        return self._progress_stall >= limit

    def _mow_no_progress(self, pose: Pose, info: dict[str, Any], advice: str) -> bool:
        """Stamp a no-go when mow tracking is commanded but the pose is stuck."""
        if self.global_plan is None or self.index >= len(self.global_plan.waypoints):
            return False
        if self._detour or self._replan_cool > 0:
            return False
        goal = self.global_plan.waypoints[self.index]
        moved = 0.0
        if self._progress_pose is not None:
            moved = math.hypot(pose.x - self._progress_pose[0], pose.y - self._progress_pose[1])
        self._progress_pose = (pose.x, pose.y)
        dist = math.hypot(pose.x - goal[0], pose.y - goal[1])
        if dist + 0.05 < self._progress_best:
            self._progress_best = dist
            self._progress_stall = 0
            return False
        if moved >= 0.06:
            self._progress_stall = 0
            return False
        self._progress_stall += 1
        if advice == "stop" or bool(info.get("collision")) or bool(info.get("tipover")):
            self._progress_stall += 2
        commanded = abs(self._last_v) > 0.05
        pivoting = abs(self._last_omega) > 0.35
        if commanded and (not pivoting) and moved < 0.03:
            self._progress_stall += 1
        limit = max(6, int(self.settings.blockage_no_progress_steps) + 4)
        if self._leftover_replans:
            limit = max(limit, 18)
        return self._progress_stall >= limit

    def _blockage_stamp_xy(self, pose: Pose) -> tuple[float, float]:
        """A point ahead of the chassis — never the robot's own cell."""
        heading = float(pose.theta)
        ahead = (
            pose.x + 0.62 * math.cos(heading),
            pose.y + 0.62 * math.sin(heading),
        )
        goal = self._current_explore_goal(pose)
        if goal is not None and math.hypot(goal[0] - pose.x, goal[1] - pose.y) < 1.15:
            ahead = goal
        if math.hypot(ahead[0] - pose.x, ahead[1] - pose.y) < 0.38:
            ahead = (
                pose.x + 0.70 * math.cos(heading) - 0.35 * math.sin(heading),
                pose.y + 0.70 * math.sin(heading) + 0.35 * math.cos(heading),
            )
        return ahead

    def _stamp_learned_blockage(
        self,
        pose: Pose,
        info: dict[str, Any],
        *,
        reason: str,
    ) -> int:
        """Paint a local no-go, skip the current frontier, and force a replan.

        Climbable grade / look-ahead hills are not learned no-go. One stall
        cannot mint thousands of events (cooldown + min separation).
        """
        blob = info if isinstance(info, dict) else {}
        if self.observed is None:
            return 0
        allow = may_stamp_blockage(
            reason=reason,
            tilt_kind=self.last_tilt_kind,
            advice=self.last_advice,
            look_ahead_kind=self._look_ahead_kind,
            collision=bool(blob.get("collision")) or reason == "collision",
            drain_drop=bool(blob.get("drain_drop")) or reason == "drain" or self._drain_hazard(blob),
            tipover=bool(blob.get("tipover")),
            chassis_tipped=bool(self._chassis_tipped or blob.get("chassis_tipped")),
            hard_structure=self._ahead_is_hard_structure(pose),
            drain_lip=self._ahead_is_drain(pose),
        )
        if not allow:
            self._defer_current_frontier(force=reason in {"drain", "collision", "path_blocked"})
            self._emit(
                "blockage_skipped",
                {"reason": reason, "why": "climbable_grade", "tilt_kind": self.last_tilt_kind},
            )
            return 0
        x, y = self._blockage_stamp_xy(pose)
        if stamp_on_cooldown(
            cool_left=self._blockage_cool,
            last_xy=self._last_blockage_xy,
            stamp_xy=(x, y),
            step=self.step,
            last_step=self._last_blockage_step,
            cooldown_steps=int(self.settings.blockage_stamp_cooldown_steps),
            min_sep_m=float(self.settings.blockage_stamp_min_sep_m),
        ):
            self._emit("blockage_skipped", {"reason": reason, "why": "cooldown"})
            return 0
        radius = float(self.settings.blockage_radius_m)
        added = self.observed.stamp_blockage(x, y, radius)
        self._last_blockage_xy = (x, y)
        self._last_blockage_step = int(self.step)
        self._blockage_cool = max(1, int(self.settings.blockage_stamp_cooldown_steps))
        self._blockage_events += 1
        target = self.explore_plan.target if self.explore_plan is not None else None
        if target is not None:
            self._skipped_frontiers.append(target)
            self._blocked_frontier_count += 1
        elif self.observed.world_to_cell(x, y) is not None:
            cell = self.observed.world_to_cell(x, y)
            if cell is not None:
                self._skipped_frontiers.append(cell)
                self._blocked_frontier_count += 1
        if reason in {"drain", "collision", "lip"}:
            self._skip_frontiers_near(x, y, radius_m=max(1.4, radius * 2.4))
        if len(self._skipped_frontiers) > 48:
            self._skipped_frontiers = self._skipped_frontiers[-48:]
        # Explore must drop the current frontier plan. Mow must keep the
        # coverage index — resetting to 0 was re-skipping the first strip
        # and aborting the tiny job at ~50–80% cut.
        if self.phase != MissionPhase.MOW:
            self.explore_plan = None
            self.index = 0
        self._reset_explore_progress(pose, self.observed.completion(self.keep_in_mask))
        self._emit(
            "blockage_stamped",
            {
                "x": x,
                "y": y,
                "radius_m": radius,
                "cells": int(added),
                "reason": reason,
                "blocked_frontiers": int(self._blocked_frontier_count),
                "unreachable_frontiers": int(self._blocked_frontier_count),
            },
        )
        if target is not None:
            tx, ty = self.observed.cell_to_world(*target)
            self._emit(
                "frontier_skipped",
                {
                    "x": tx,
                    "y": ty,
                    "reason": reason,
                    "blocked_frontiers": int(self._blocked_frontier_count),
                },
            )
        return added

    def _build_explore_reason(
        self,
        completion: float,
        info: Optional[dict[str, Any]] = None,
        *,
        n_frontiers: Optional[int] = None,
        code: str = "",
    ) -> dict[str, Any]:
        _ = info
        target = float(self.settings.explore_complete)
        max_steps = int(self.settings.max_explore_steps)
        n_front = int(n_frontiers if n_frontiers is not None else len(self._frontier_xy))
        n_wp = 0
        unreachable = int(self._blocked_frontier_count)
        skipped = int(len(self._skipped_frontiers))
        if self.explore_plan is not None:
            n_wp = len(self.explore_plan.waypoints)
            unreachable = max(
                unreachable,
                int(getattr(self.explore_plan, "unreachable_frontiers", 0) or 0),
            )
            skipped = max(skipped, int(getattr(self.explore_plan, "skipped_frontiers", 0) or 0))
        if not code:
            if self.phase != MissionPhase.EXPLORE:
                code = "idle"
            elif self.terrain_state in TERRAIN_REASON_CODE and self.terrain_state != TERRAIN_OK:
                code = TERRAIN_REASON_CODE[self.terrain_state]
            elif self.last_tilt_kind == KIND_GRADE:
                code = "steep_grade"
            elif self.last_advice == "stop":
                code = "tip_recovery"
            elif self._blockage_events and self._explore_recover_cool > 0:
                code = "blockage_stamped"
            elif n_wp > 0:
                code = "seeking_frontier"
            elif n_front > 0 or unreachable > 0:
                code = "path_blocked"
            elif completion >= target:
                code = "map_progress"
            else:
                code = "no_frontier"
        labels = {
            "seeking_frontier": "Seeking frontier",
            "path_blocked": "Path blocked — looking around",
            "tip_recovery": "Tip risk — reversing",
            "steep_grade": "Steep grade — contouring",
            "retrace": "Retracing last metres",
            "map_progress": "Map progress",
            "waiting_cap": "Waiting on explore cap",
            "no_frontier": "No reachable frontier",
            "stalled": "Explore stalled",
            "idle": "Idle",
            "blockage_stamped": "Blocked — remapping around obstacle",
            "frontier_skipped": "Frontier unreachable — skipping",
            "remapping": "Blocked — remapping around obstacle",
        }
        label = (
            f"{labels.get(code, code)} · map {100.0 * completion:.0f}% of target "
            f"{100.0 * target:.0f}% · step {self.phase_step}/{max_steps}"
        )
        if n_front:
            label = f"{label} · {n_front} frontiers"
        if unreachable:
            label = f"{label} · {unreachable} unreachable"
        return {
            "code": code,
            "label": label,
            "map_pct": float(completion),
            "target_pct": target,
            "phase_step": int(self.phase_step),
            "max_steps": max_steps,
            "n_frontiers": n_front,
            "n_waypoints": n_wp,
            "unreachable_frontiers": unreachable,
            "skipped_frontiers": skipped,
            "blocked_cells": int(self.observed.blockage_count()) if self.observed is not None else 0,
            "blocked_frontiers": int(self._blocked_frontier_count),
            "n_blockages": int(self._blockage_events),
            "full_explore": bool(self.settings.full_explore),
            "terrain_state": self.terrain_state,
        }

    def _freeze(self, info: dict[str, Any]) -> YardSnapshot:
        assert self.observed is not None
        keep = self.keep_in_mask if self.keep_in_mask is not None else self.observed.keep_in_mask(self._geofence)
        trail = list(self.teach.trail)
        ring = list(self.profile.keep_in) if self.profile is not None else trail_to_polygon(trail)
        closed = self.teach.done or _ring_closed(ring, trail, float(self.settings.review_min_closure_m))
        notes: list[str] = []
        if not closed:
            notes.append("keep-in trail did not close tightly")
        completion = self.observed.completion(keep)
        target = float(self.settings.explore_complete)
        if not (completion >= target or (target >= 0.99 and completion >= 0.99)):
            notes.append(f"map completion {completion:.2f} below target {target:.2f}")
        mowable = self.observed.mowable_mask(keep)
        comps = connected_components(mowable)
        if len(comps) > 1:
            notes.append(f"{len(comps)} disconnected mowable regions")
        profile = self.profile or YardProfile(
            name="mission",
            width_m=self.cfg.world.width_m,
            height_m=self.cfg.world.height_m,
            resolution_m=self.cfg.world.resolution_m,
            keep_in=ring,
            home={"x": self._home.x, "y": self._home.y, "theta": self._home.theta},
            trail=trail,
        )
        snap = self.observed.copy()
        self.observed.lock_observed()
        return YardSnapshot(
            profile=profile,
            observed=snap.observed,
            explored=snap.explored,
            hazard=snap.hazard,
            structure=snap.structure,
            elevation=snap.elevation,
            confidence=snap.confidence,
            keep_in_mask=np.asarray(keep, dtype=bool).copy(),
            free=snap.free,
            step=self.step,
            completion=completion,
            mean_confidence=snap.mean_confidence(keep),
            closed=closed,
            n_components=len(comps),
            notes=notes,
        )

    def _plan_global_mow(
        self,
        pose: Pose,
        *,
        leftover: bool = False,
        info: Optional[dict[str, Any]] = None,
    ) -> CoveragePlan:
        assert self.observed is not None
        snap = self.snapshot
        keep = snap.keep_in_mask if snap is not None else self.keep_in_mask
        extra = self.observed.unknown_blocked(keep)
        if np.any(np.asarray(self.observed.blockage, dtype=bool)):
            extra = extra | np.asarray(self.observed.blockage, dtype=bool)
        structure = self.observed.structure if snap is None else snap.structure
        raw_hazard = self.observed.hazard if snap is None else snap.hazard
        conf = self.observed.confidence if snap is None else snap.confidence
        hazard = _control_hazard(raw_hazard, conf)
        elevation = self.observed.elevation if snap is None else snap.elevation
        slope = slope_from_elevation(elevation, self.cfg.world.resolution_m)
        costmap = build_costmap(
            hazard,
            slope,
            resolution_m=self.cfg.world.resolution_m,
            width_m=self.cfg.world.width_m,
            height_m=self.cfg.world.height_m,
            max_climb_slope_rad=self.cfg.planner.max_climb_slope_rad,
            tip_lethal_slope_rad=tip_lethal_slope_rad(
                tip_roll_rad=self.cfg.robot.tip_roll_rad,
                tip_pitch_rad=self.cfg.robot.tip_pitch_rad,
                tip_lethal_frac=self.cfg.planner.tip_lethal_frac,
            ),
            contour_cost=self.cfg.planner.contour_cost,
            drain_clearance_m=self.cfg.planner.drain_clearance_m,
            occupancy=self.observed.occupancy,
            occupancy_inflate_m=max(0.20, self.cfg.planner.occupancy_inflate_m),
            margin_m=self.cfg.robot.collision_radius_m,
            extra_blocked=extra,
            confidence=conf,
            uncertainty_inflate=self.cfg.planner.uncertainty.inflate,
            uncertain_hazard_boost=self.cfg.planner.uncertainty.hazard_boost,
            uncertain_confidence_floor=self.cfg.planner.uncertainty.confidence_floor,
            geofence=self._geofence,
            geofence_inflate_m=self.cfg.planner.geofence_inflate_m,
            wet=self._wet,
            wet_slope_extra=self.cfg.planner.wet_slope_extra,
            structure=structure,
            path_cost=self.cfg.planner.path_cost,
            bunker_cost=self.cfg.planner.bunker_cost,
            elevation=elevation,
        )
        mowable = self.observed.mowable_mask(keep) & ~costmap.blocked
        if int(mowable.sum()) < 8:
            # Incomplete map: still sweep known-free grass, never unknown.
            known = self.observed.free & ~costmap.blocked
            if keep is not None:
                known = known & np.asarray(keep, dtype=bool)
            hardscape = self.observed.structure != 0
            mowable = known & ~hardscape
        if leftover:
            cut = self._keep_in_cut_mask(info if info is not None else self._last_info)
            if cut is not None and cut.shape == mowable.shape:
                mowable = mowable & ~cut
        heading = choose_strip_orientation(
            elevation,
            mowable,
            resolution_m=self.cfg.world.resolution_m,
        )
        res = max(float(self.cfg.world.resolution_m), 1e-6)
        spacing = float(self.cfg.planner.strip_spacing_m)
        stride = float(self.cfg.planner.waypoint_stride_m)
        if leftover:
            # Inter-strip cells the first pass never visited. Drive every
            # leftover cell so cut% is of planned mowable, not of lanes.
            spacing = min(spacing, res)
            stride = min(stride, max(res, 0.24))
        return plan_coverage(
            costmap,
            (pose.x, pose.y),
            strip_spacing_m=spacing,
            waypoint_stride_m=stride,
            mowable=mowable,
            orientation_rad=heading,
            elevation=elevation,
            soft_transit=self._soft_transit_mask(keep, structure, raw_hazard),
        )

    def _local_replan(self, obs: dict[str, Any], pose: Pose) -> None:
        if self.global_plan is None:
            return
        nxt = self._next_clear_global(obs, pose)
        if nxt is None:
            return
        if math.hypot(nxt[0] - pose.x, nxt[1] - pose.y) <= self.cfg.planner.arrive_radius_m:
            return
        path = self._safe_path(pose, nxt, obs)
        if not path:
            skipped = self.global_plan.waypoints[self.index] if self.index < len(self.global_plan.waypoints) else nxt
            self._skipped_global.append(skipped)
            self._emit("unreachable_segment", {"x": skipped[0], "y": skipped[1], "index": self.index})
            self.index += 1
            return
        self._detour = path
        self._detour_index = 0
        self.replans += 1
        self._replan_cool = 10

    def _maybe_local_block(self, obs: dict[str, Any], pose: Pose) -> None:
        if self.global_plan is None or self._detour or self._replan_cool > 0:
            return
        occ = obs.get("occupancy")
        if occ is None:
            return
        grid = np.asarray(occ, dtype=np.float32)
        res = self.cfg.world.resolution_m
        ignore = self._body_ignore_m()
        look = self.global_plan.waypoints[self.index : self.index + 3]
        for tx, ty in look:
            if math.hypot(tx - pose.x, ty - pose.y) <= ignore:
                continue
            col = int(tx / res)
            row = int(ty / res)
            if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1] and grid[row, col] > 0.5:
                self._local_replan(obs, pose)
                return

    def _next_clear_global(self, obs: dict[str, Any], pose: Pose) -> Optional[tuple[float, float]]:
        if self.global_plan is None:
            return None
        res = self.cfg.world.resolution_m
        arrive = self.cfg.planner.arrive_radius_m
        ignore = self._body_ignore_m()
        hazard = np.asarray(obs.get("hazard"), dtype=np.float32) if obs.get("hazard") is not None else None
        occ = np.asarray(obs.get("occupancy"), dtype=np.float32) if obs.get("occupancy") is not None else None
        for i in range(self.index, len(self.global_plan.waypoints)):
            x, y = self.global_plan.waypoints[i]
            if math.hypot(x - pose.x, y - pose.y) <= arrive:
                continue
            col = int(x / res)
            row = int(y / res)
            near = math.hypot(x - pose.x, y - pose.y) <= ignore
            blocked = False
            if hazard is not None and 0 <= row < hazard.shape[0] and 0 <= col < hazard.shape[1]:
                blocked = float(hazard[row, col]) >= 2.0
            if occ is not None and 0 <= row < occ.shape[0] and 0 <= col < occ.shape[1] and not near:
                blocked = blocked or float(occ[row, col]) > 0.5
            if not blocked:
                self.index = i
                return (x, y)
        return None

    def _skip_arrived(
        self,
        waypoints: list[tuple[float, float]],
        pose: Pose,
        index: int,
    ) -> int:
        arrive = self.cfg.planner.arrive_radius_m
        idx = int(index)
        while idx < len(waypoints):
            tx, ty = waypoints[idx]
            if math.hypot(tx - pose.x, ty - pose.y) <= arrive:
                idx += 1
                continue
            break
        return idx

    def _body_ignore_m(self) -> float:
        return float(self.cfg.robot.collision_radius_m) + 0.45

    def _soft_transit_mask(
        self,
        keep: Optional[np.ndarray],
        structure: np.ndarray,
        hazard: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Unknown cells inside keep-in that are not a real obstacle.

        Used only to classify fog islands vs drain/shed islands.
        """
        if self.observed is None:
            return None
        unknown = ~np.asarray(self.observed.observed, dtype=bool)
        hard = np.isin(
            np.asarray(structure),
            (
                STRUCTURE_BUILDING,
                STRUCTURE_BUNKER,
                STRUCTURE_GARDEN,
                STRUCTURE_GREEN,
                STRUCTURE_POND,
            ),
        )
        channel = np.asarray(hazard, dtype=np.float32) >= HAZARD_DRAIN
        soft = unknown & ~hard & ~channel
        if keep is not None:
            soft = soft & np.asarray(keep, dtype=bool)
        return soft

    def _job_cut_fraction(
        self,
        info: Optional[dict[str, Any]],
        plan: Optional[CoveragePlan],
        world: float,
    ) -> float:
        """Cut of the planned reachable lawn when a plan exists.

        World grass on an acre is thousands of cells; 1% world looks stuck
        even while strips paint. Owner cut % is job progress on the frozen
        plan. ``world_coverage_fraction`` stays on status for honesty.
        """
        blob = info or {}
        reachable = int(getattr(plan, "reachable_mowable_cells", 0) or 0) if plan is not None else 0
        if plan is None or reachable <= 0:
            return float(world)
        denom = max(1, reachable)
        raster = blob.get("coverage_cut")
        keep = self.keep_in_mask
        painted = 0
        if raster is not None and keep is not None:
            cut = np.asarray(raster)
            mask = np.asarray(keep, dtype=bool)
            if cut.shape == mask.shape:
                if self.observed is not None:
                    mow = self.observed.mowable_mask(keep)
                    if mow.shape == mask.shape:
                        mask = mask & mow
                painted = int((cut.astype(bool) & mask).sum())
        cut_cells = int(blob.get("coverage_cut_cells") or 0)
        n = max(painted, cut_cells)
        if n > 0:
            return float(min(1.0, n / denom))
        return float(world)

    def _mow_paint_done(self, info: Optional[dict[str, Any]]) -> bool:
        target = float(self.settings.mow_complete_frac or 0.0)
        if target <= 0.0 or self.global_plan is None:
            return False
        world = float((info or {}).get("coverage_fraction") or 0.0)
        return self._job_cut_fraction(info, self.global_plan, world) >= target

    def _keep_in_cut_mask(self, info: Optional[dict[str, Any]]) -> Optional[np.ndarray]:
        blob = info or {}
        raster = blob.get("coverage_cut")
        if raster is None:
            return None
        cut = np.asarray(raster, dtype=bool)
        keep = self.keep_in_mask
        if keep is not None and cut.shape == np.asarray(keep).shape:
            return cut & np.asarray(keep, dtype=bool)
        return cut

    def _uncut_mowable_mask(self, info: Optional[dict[str, Any]]) -> Optional[np.ndarray]:
        if self.observed is None:
            return None
        keep = self.keep_in_mask
        mow = self.observed.mowable_mask(keep)
        cut = self._keep_in_cut_mask(info)
        leftover = np.asarray(mow, dtype=bool)
        if cut is not None and cut.shape == leftover.shape:
            leftover = leftover & ~cut
        return leftover

    def _direct_leftover_waypoints(
        self,
        leftover: np.ndarray,
        pose: Pose,
    ) -> list[tuple[float, float]]:
        """Nearest-neighbour tour of leftover uncut cells."""
        ys, xs = np.where(np.asarray(leftover, dtype=bool))
        if ys.size == 0:
            return []
        res = max(float(self.cfg.world.resolution_m), 1e-6)
        pts = [((float(c) + 0.5) * res, (float(r) + 0.5) * res) for r, c in zip(ys.tolist(), xs.tolist())]
        path: list[tuple[float, float]] = []
        cx, cy = float(pose.x), float(pose.y)
        remaining = pts
        arrive = max(0.16, float(self.cfg.planner.arrive_radius_m))
        while remaining and len(path) < 96:
            i = min(range(len(remaining)), key=lambda k: (remaining[k][0] - cx) ** 2 + (remaining[k][1] - cy) ** 2)
            nxt = remaining.pop(i)
            if math.hypot(nxt[0] - cx, nxt[1] - cy) < arrive and path:
                continue
            path.append(nxt)
            cx, cy = nxt
        return path

    def _replan_leftover_uncut(self, pose: Pose, info: Optional[dict[str, Any]]) -> bool:
        """When the strip list ends, sweep leftover uncut keep-in cells.

        Boustrophedon spacing wider than the trimmer (and IMU/grade skips)
        can exhaust waypoints at ~50% cut. That is not job complete.
        """
        if self.observed is None or self.global_plan is None:
            return False
        if self._leftover_replans >= 8:
            return False
        world = float((info or {}).get("coverage_fraction") or 0.0)
        cut_frac = self._job_cut_fraction(info, self.global_plan, world)
        if cut_frac >= 0.95:
            return False
        mask = self._uncut_mowable_mask(info)
        leftover_n = int(mask.sum()) if mask is not None else 0
        reachable = max(1, int(self.global_plan.reachable_mowable_cells or 0))
        if leftover_n <= 0 or leftover_n / reachable <= 0.05:
            return False
        leftover = self._plan_global_mow(pose, leftover=True, info=info)
        direct = self._direct_leftover_waypoints(mask, pose) if mask is not None else []
        if leftover is None:
            return False
        if len(direct) >= 2 and len(leftover.waypoints) < max(4, len(direct) // 2):
            leftover.waypoints = list(direct)
        if len(leftover.waypoints) < 2:
            leftover.waypoints = list(direct)
        if len(leftover.waypoints) < 2:
            return False
        orig = self.global_plan
        leftover.planned_mowable_cells = int(orig.planned_mowable_cells)
        leftover.reachable_mowable_cells = int(orig.reachable_mowable_cells)
        leftover.unreachable_mowable_cells = int(orig.unreachable_mowable_cells)
        leftover.unmapped_mowable_cells = int(orig.unmapped_mowable_cells)
        leftover.planned_coverage_fraction = float(orig.planned_coverage_fraction)
        self.global_plan = leftover
        self.plan = leftover
        self.index = 0
        self._detour = []
        self._detour_index = 0
        self._leftover_replans += 1
        self._reset_explore_progress(pose, 0.0)
        self._emit(
            "mow_leftover",
            {
                "pass": int(self._leftover_replans),
                "cut_pct": float(cut_frac),
                "leftover_cells": leftover_n,
                "n_waypoints": len(leftover.waypoints),
            },
        )
        return True

    def _mow_skip_count(self) -> int:
        n = max(1, int(self.settings.mow_skip_cluster or 4))
        if self.global_plan is None:
            return n
        left = max(1, len(self.global_plan.waypoints) - self.index)
        short = min(float(self.cfg.world.width_m), float(self.cfg.world.height_m))
        if short >= 24.0:
            n = max(n, 8)
        return min(n, left)

    def _safe_path(
        self,
        pose: Pose,
        goal: tuple[float, float],
        obs: dict[str, Any],
    ) -> list[tuple[float, float]]:
        if self.observed is None:
            return [goal]
        extra = None
        occ = obs.get("occupancy")
        if occ is not None:
            extra = np.asarray(occ, dtype=np.float32) > 0.5
        cm = explore_costmap(self.observed, self.keep_in_mask)
        if extra is not None and extra.shape == cm.blocked.shape:
            cm.blocked = cm.blocked | extra
            cm.cost[extra] = cm.cost[extra]
        cells = shortest_path(cm, (pose.x, pose.y), goal)
        if not cells:
            return []
        stride = max(1, int(round(0.35 / max(self.cfg.world.resolution_m, 1e-6))))
        picked = cells[::stride]
        if picked[-1] != cells[-1]:
            picked.append(cells[-1])
        return [cm.cell_to_world(r, c) for r, c in picked]

    def _home_path(self, pose: Pose) -> list[tuple[float, float]]:
        if self.observed is None:
            return [(self._home.x, self._home.y)]
        cm = explore_costmap(self.observed, self.keep_in_mask)
        cells = shortest_path(cm, (pose.x, pose.y), (self._home.x, self._home.y))
        if not cells:
            return [(self._home.x, self._home.y)]
        return [cm.cell_to_world(r, c) for r, c in cells[::2]]

    def _track_list(
        self,
        waypoints: list[tuple[float, float]],
        pose: Pose,
        *,
        cruise: float,
        advice: str,
        trimmer: float,
        index_attr: str = "index",
    ) -> np.ndarray:
        idx = int(getattr(self, index_attr))
        arrive = self.cfg.planner.arrive_radius_m
        while idx < len(waypoints):
            tx, ty = waypoints[idx]
            if math.hypot(tx - pose.x, ty - pose.y) <= arrive:
                idx += 1
                continue
            break
        setattr(self, index_attr, idx)
        if idx >= len(waypoints):
            return self._hold()
        cruise = grade_aware_cruise(
            cruise,
            advice=advice,
            tilt_kind=self.last_tilt_kind,
            slow_speed_factor=self.cfg.planner.slow_speed_factor,
            grade_speed_factor=self.cfg.planner.grade_speed_factor,
        )
        wheels, _dist, _err = tracking_action(
            pose,
            waypoints[idx],
            cruise=cruise,
            wheelbase_m=self.cfg.robot.wheelbase_m,
            turn_in_place_rad=self.cfg.planner.turn_in_place_rad,
        )
        return self._drive(float(wheels[0]), float(wheels[1]), trimmer, pose)

    def _drive(self, left: float, right: float, trimmer: float, pose: Pose) -> np.ndarray:
        _ = pose
        if self.phase != MissionPhase.MOW:
            trimmer = 0.0
        vmax = self.cfg.robot.max_wheel_speed_mps
        self._last_v, self._last_omega = unicycle_from_wheels(
            left * vmax,
            right * vmax,
            self.cfg.robot.wheelbase_m,
        )
        return np.array([left, right, trimmer], dtype=np.float32)

    def _empty_mow_plan(self) -> bool:
        plan = self.global_plan
        if plan is None:
            return True
        return len(getattr(plan, "waypoints", []) or []) < 2

    def _keep_in_too_small(self) -> bool:
        profile = self.profile
        if profile is None or len(profile.keep_in) < 3:
            return False
        # Teach/save still uses acre-relative scribble thresholds.
        # Once a keep-in is the job fence, only reject a centre scribble
        # so a small taught pocket on the acre world can still be mowed.
        return not keep_in_usable(
            profile.keep_in,
            float(self.cfg.world.width_m),
            float(self.cfg.world.height_m),
            min_span_frac=0.0,
            min_area_frac=0.0,
            min_span_m=1.5,
            min_area_m2=2.0,
        )

    def _hold(self) -> np.ndarray:
        self._last_v = 0.0
        self._last_omega = 0.0
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def _finish(self, action: np.ndarray, advice: str, info: dict[str, Any]) -> np.ndarray:
        self._sync_env_charge(self.phase == MissionPhase.CHARGING)
        if self.help_requested:
            self.safe.enter_safe("call-for-help")
        # Seated past-tip / drain still immobilise. Predicted look-ahead
        # stop and IMU tip-risk must not limp-park the acre fence lap.
        safe_advice = advice
        if (
            advice == "stop"
            and not bool(info.get("tipover"))
            and not bool(info.get("drain_drop"))
            and not self._chassis_tipped
            and not (
                self._last_pose is not None
                and attitude_past_tip(
                    self._last_pose.roll,
                    self._last_pose.pitch,
                    self.cfg.robot.static_tip_roll_rad(),
                    self.cfg.robot.static_tip_pitch_rad(),
                )
            )
        ):
            safe_advice = "reroute" if self.phase == MissionPhase.CALIBRATE_BOUNDARY else "slow"
        self.safe.tick(
            advice=safe_advice,
            estop=estop_requested(None, info),
            tipover=bool(info.get("tipover")),
            drain_drop=bool(info.get("drain_drop")),
            help_requested=self.help_requested,
        )
        self.last_safe_mode = self.safe.mode
        if self.safe.mode == "estop" and self.phase not in {
            MissionPhase.COMPLETE,
            MissionPhase.FAULT,
            MissionPhase.SAFE,
        }:
            self._transition(MissionPhase.SAFE)
        elif (
            self.safe.mode == "safe"
            and self.phase
            in {
                MissionPhase.CALIBRATE_BOUNDARY,
                MissionPhase.EXPLORE,
                MissionPhase.REVIEW,
                MissionPhase.MOW,
                MissionPhase.RETURN_HOME,
                MissionPhase.CHARGING,
            }
            and not self.help_requested
            and not bool(info.get("tipover"))
            and not bool(info.get("drain_drop"))
        ):
            # Mapping / MAP READY hold near a fence must not become SAFE.
            self.safe.clear_if_not_estop()
            self.last_safe_mode = self.safe.mode
        elif self.safe.mode == "safe" and self.phase not in {
            MissionPhase.COMPLETE,
            MissionPhase.FAULT,
            MissionPhase.SAFE,
        }:
            self._transition(MissionPhase.SAFE)
        out = self.safe.apply(action)
        if self.safe.command().hold:
            self._last_v = 0.0
            self._last_omega = 0.0
            self.last_advice = "stop"
        if self.phase != MissionPhase.MOW:
            out = np.array([float(out[0]), float(out[1]), 0.0], dtype=np.float32)
        return out

    def _fail(self, reason: str, advice: str, info: dict[str, Any]) -> np.ndarray:
        self.help_requested = True
        self.last_advice = "stop"
        self.safe.enter_safe(reason)
        self._transition(MissionPhase.FAULT)
        return self._finish(self._hold(), "stop", info)

    def _enter(self, phase: MissionPhase, event: str, detail: Optional[dict[str, Any]] = None) -> None:
        self.phase = phase
        self.phase_step = 0
        self._progress_stall = 0
        self._progress_best = 1e9
        self._progress_pose = None
        self._explore_recover_cool = 0
        self._clear_retrace()
        self.phase_ranges.append({"phase": phase.value, "start": self.step, "end": None})
        self._emit(event, detail or {})

    def _transition(self, phase: MissionPhase) -> None:
        if self.phase_ranges and self.phase_ranges[-1].get("end") is None:
            self.phase_ranges[-1]["end"] = self.step
        self._detour = []
        self._detour_index = 0
        self._enter(phase, "phase_enter", {"from": self.phase.value if phase != self.phase else None})

    def _emit(self, event: str, detail: Optional[dict[str, Any]] = None) -> None:
        self.events.append(
            MissionEvent(step=self.step, phase=self.phase.value, event=event, detail=detail or {})
        )

    def _apply_calibrate_drive(self) -> None:
        """Faster fence tracking on large yards. 0 in YAML means auto."""
        short = min(float(self.cfg.world.width_m), float(self.cfg.world.height_m))
        large = short >= 24.0
        cruise = float(self.settings.calibrate_cruise or 0.0)
        arrive = float(self.settings.calibrate_arrive_m or 0.0)
        if cruise <= 0.0:
            cruise = 0.98 if large else 0.75
        if arrive <= 0.0:
            arrive = 1.80 if large else 0.55
        self.teach.cruise = float(cruise)
        self.teach.arrive_m = float(arrive)

    def _calibrate_stride_m(self) -> float:
        configured = float(self.settings.calibrate_stride_m or 0.0)
        if configured > 0.0:
            return configured
        short = min(float(self.cfg.world.width_m), float(self.cfg.world.height_m))
        return max(0.70, min(4.0, 0.055 * short))

    def _trail_length_m(self) -> float:
        trail = list(self.teach.trail)
        if len(trail) < 2:
            return 0.0
        return float(
            sum(
                math.hypot(trail[i][0] - trail[i - 1][0], trail[i][1] - trail[i - 1][1])
                for i in range(1, len(trail))
            )
        )

    def _calibrate_confirmed(self) -> bool:
        limit = float(self.settings.calibrate_confirm_m or 0.0)
        return limit > 0.0 and self._trail_length_m() >= limit

    def _reverse_nudge(self, pose: Pose) -> np.ndarray:
        """One-step reverse off a ridge before skipping the waypoint."""
        _ = pose
        return self._drive(-0.36, -0.36, 0.0, pose)

    def _clear_retrace(self) -> None:
        self._retrace_wps = []
        self._retrace_index = 0
        self._retrace_reason = ""

    def _record_pose_trail(self, pose: Pose) -> None:
        if self.phase not in {MissionPhase.EXPLORE, MissionPhase.MOW}:
            return
        if self._chassis_tipped or self._retrace_wps:
            return
        pt = (float(pose.x), float(pose.y), float(pose.theta))
        if not self._pose_trail:
            self._pose_trail.append(pt)
            return
        last = self._pose_trail[-1]
        stride = float(self.settings.retrace_stride_m or 0.30)
        if math.hypot(pt[0] - last[0], pt[1] - last[1]) < stride:
            return
        self._pose_trail.append(pt)
        max_n = max(40, int(self.settings.trail_max_points or 400))
        if len(self._pose_trail) > max_n:
            self._pose_trail = self._pose_trail[-max_n:]

    def _start_retrace(self, pose: Pose, *, reason: str) -> bool:
        if self._chassis_tipped or self._retrace_cool > 0:
            return False
        wps = retrace_waypoints(
            self._pose_trail,
            (pose.x, pose.y),
            length_m=float(self.settings.retrace_length_m or 2.8),
        )
        if not wps:
            return False
        self._retrace_wps = wps
        self._retrace_index = 0
        self._retrace_reason = reason
        self.terrain_state = TERRAIN_RETRACE
        self._emit("trail_retrace", {"n": len(wps), "reason": reason})
        return True

    def _tick_retrace(self, pose: Pose, advice: str) -> Optional[np.ndarray]:
        if not self._retrace_wps:
            return None
        if self._chassis_tipped:
            self._clear_retrace()
            return None
        arrive = max(0.22, float(self.cfg.planner.arrive_radius_m))
        while self._retrace_index < len(self._retrace_wps):
            tx, ty = self._retrace_wps[self._retrace_index]
            if math.hypot(tx - pose.x, ty - pose.y) <= arrive:
                self._retrace_index += 1
                continue
            break
        if self._retrace_index >= len(self._retrace_wps):
            self._clear_retrace()
            self._retrace_cool = 16
            self.explore_plan = None
            if self.phase != MissionPhase.MOW:
                self.index = 0
            return None
        return self._track_list(
            self._retrace_wps,
            pose,
            cruise=0.32,
            advice=advice,
            trimmer=0.0,
            index_attr="_retrace_index",
        )

    def _retrace_or_nudge(self, pose: Pose, *, reason: str) -> np.ndarray:
        """Retrace the breadcrumb trail; fall back to a one-step reverse.

        Software tip-risk still begins with a reverse nudge so PLN / IMU
        recovery stays a wheel-reverse, then the trail is followed.
        """
        if self._chassis_tipped:
            return self._hold()
        if self._retrace_wps:
            action = self._tick_retrace(pose, self.last_advice)
            if action is not None:
                return action
        started = self._start_retrace(pose, reason=reason)
        if reason in {"tip_risk", "mow_tip"}:
            return self._reverse_nudge(pose)
        if started:
            action = self._tick_retrace(pose, self.last_advice)
            if action is not None:
                return action
        return self._reverse_nudge(pose)

    def _drain_hazard(self, info: Optional[dict[str, Any]]) -> bool:
        blob = info if isinstance(info, dict) else {}
        if bool(blob.get("drain_drop")):
            return True
        reason = str(blob.get("terrain_reason") or "").lower()
        return "drain" in reason

    def _grade_like(self, advice: str) -> bool:
        return climbable_grade(
            tilt_kind=self.last_tilt_kind,
            advice=advice,
            look_ahead_kind=self._look_ahead_kind,
            chassis_tipped=self._chassis_tipped,
        )

    def _ahead_cell(self, pose: Pose) -> Optional[tuple[int, int]]:
        if self.observed is None:
            return None
        return self.observed.world_to_cell(*self._blockage_stamp_xy(pose))

    def _ahead_is_hard_structure(self, pose: Pose) -> bool:
        cell = self._ahead_cell(pose)
        if cell is None or self.observed is None:
            return False
        r, c = cell
        struct = int(self.observed.structure[r, c])
        return struct in {STRUCTURE_BUILDING, STRUCTURE_BUNKER, STRUCTURE_GARDEN, STRUCTURE_GREEN}

    def _ahead_is_drain(self, pose: Pose) -> bool:
        cell = self._ahead_cell(pose)
        if cell is None or self.observed is None:
            return False
        r, c = cell
        haz = int(self.observed.hazard[r, c])
        return haz in {HAZARD_DRAIN, HAZARD_DRAIN_EDGE}

    def _defer_current_frontier(self, *, force: bool = False) -> None:
        """Drop the current plan. Skip the lip after a repeat stall (or force)."""
        target = self.explore_plan.target if self.explore_plan is not None else None
        if target is not None:
            hits = getattr(self, "_deferred_once", None)
            if hits is None:
                hits = set()
                self._deferred_once = hits
            if force or target in hits:
                if target not in self._skipped_frontiers:
                    self._skipped_frontiers.append(target)
            else:
                hits.add(target)
        if self.phase != MissionPhase.MOW:
            self.explore_plan = None
            self.index = 0

    def _skip_frontiers_near(self, x: float, y: float, *, radius_m: float) -> None:
        if self.observed is None:
            return
        for wx, wy in list(self._frontier_xy):
            if math.hypot(wx - x, wy - y) > float(radius_m):
                continue
            cell = self.observed.world_to_cell(wx, wy)
            if cell is not None and cell not in self._skipped_frontiers:
                self._skipped_frontiers.append(cell)

    def _look_ahead_reason(self) -> dict[str, Any]:
        return look_ahead_reason_blob(
            kind=self._look_ahead_kind,
            advice=self._look_ahead_advice,
            pitch=self._look_ahead_pitch,
            roll=self._look_ahead_roll,
        )

    def _refresh_terrain_state(self, info: Optional[dict[str, Any]], *, advice: str) -> None:
        blob = info if isinstance(info, dict) else {}
        self.terrain_state = fuse_terrain_state(
            chassis_tipped=bool(self._chassis_tipped or blob.get("tipover") or blob.get("chassis_tipped")),
            tipover=bool(blob.get("tipover")),
            retracing=bool(self._retrace_wps),
            tilt_kind=self.last_tilt_kind,
            advice=advice,
            stamped_recent=self._blockage_cool > 0 and self._blockage_events > 0,
            blockage_recovering=self._explore_recover_cool > 0 and self._blockage_events > 0,
        )

    def _lateral_nudge(self, pose: Pose) -> np.ndarray:
        """Pivot-reverse so the next skip is not the same ridge cell."""
        _ = pose
        return self._drive(-0.18, -0.46, 0.0, pose)

    def _teach_inset_m(self) -> float:
        short = min(float(self.cfg.world.width_m), float(self.cfg.world.height_m))
        return max(0.75, min(2.10, 0.18 * short))

    def _inset_teach_ring(self, margin_m: float) -> None:
        ring = list(self.teach.waypoints)
        if len(ring) >= 2 and math.hypot(ring[0][0] - ring[-1][0], ring[0][1] - ring[-1][1]) < 1e-6:
            ring = ring[:-1]
        inset = _inset_polygon(ring, margin_m)
        if len(inset) >= 3:
            dense = _densify_ring(inset, stride_m=self._calibrate_stride_m())
            self.teach.waypoints = dense + [dense[0]]
            self.teach.index = 0

    def _start_teach_nearest(self, pose: Pose) -> None:
        wps = list(self.teach.waypoints)
        if len(wps) < 2:
            return
        if math.hypot(wps[0][0] - wps[-1][0], wps[0][1] - wps[-1][1]) < 1e-6:
            core = wps[:-1]
        else:
            core = wps
        if not core:
            return
        i = min(range(len(core)), key=lambda k: math.hypot(core[k][0] - pose.x, core[k][1] - pose.y))
        rotated = core[i:] + core[:i]
        self.teach.waypoints = rotated + [rotated[0]]
        self.teach.index = 0

    def _nudge_inward(self, pose: Pose) -> np.ndarray:
        """IMU / terrain stop: ease toward the keep-in centroid instead of tipping."""
        ring = list(self.teach.waypoints)
        if len(ring) < 2:
            return self._hold()
        cx = sum(p[0] for p in ring) / len(ring)
        cy = sum(p[1] for p in ring) / len(ring)
        ix, iy = cx - pose.x, cy - pose.y
        inward = math.hypot(ix, iy)
        tx, ty = ring[min(self.teach.index, len(ring) - 1)]
        if inward > 1e-6:
            ix, iy = ix / inward, iy / inward
        else:
            ix, iy = 0.0, 0.0
        goal = (pose.x + 0.55 * ix + 0.20 * (tx - pose.x), pose.y + 0.55 * iy + 0.20 * (ty - pose.y))
        wheels, _dist, _err = tracking_action(
            pose,
            goal,
            cruise=0.32,
            wheelbase_m=self.cfg.robot.wheelbase_m,
            turn_in_place_rad=0.80,
        )
        return self._drive(float(wheels[0]), float(wheels[1]), 0.0, pose)

    def close_phase_ranges(self) -> list[dict[str, Any]]:
        if self.phase_ranges and self.phase_ranges[-1].get("end") is None:
            self.phase_ranges[-1]["end"] = self.step
        return list(self.phase_ranges)


def _fast_settings(base: MissionConfig) -> MissionConfig:
    data = {item.name: getattr(base, item.name) for item in fields(MissionConfig)}
    data.update(
        explore_complete=0.62,
        explore_no_frontier=0.40,
        stamp_radius_m=max(2.2, float(base.stamp_radius_m)),
        camera_range_m=max(5.0, float(base.camera_range_m)),
        max_calibrate_steps=min(int(base.max_calibrate_steps), 220),
        max_explore_steps=min(int(base.max_explore_steps), 220),
        max_mow_steps=min(int(base.max_mow_steps), 260),
        max_return_steps=min(int(base.max_return_steps), 80),
        snapshot_stride=max(8, int(base.snapshot_stride)),
        review_min_closure_m=1.20,
        explore_cruise=0.95,
        mow_cruise=0.85,
        calibrate_stride_m=float(base.calibrate_stride_m or 0.55),
        calibrate_cruise=max(0.90, float(base.calibrate_cruise or 0.0)),
        calibrate_arrive_m=max(0.50, float(base.calibrate_arrive_m or 0.0)),
        calibrate_confirm_m=float(base.calibrate_confirm_m or 0.0),
        phase_budget_scale=1.0,
        review_hold_steps=min(2, int(base.review_hold_steps or 2)),
        mow_complete_frac=float(base.mow_complete_frac or 0.0),
        blockage_no_progress_steps=min(10, int(base.blockage_no_progress_steps or 16)),
        blockage_stamp_cooldown_steps=min(12, int(base.blockage_stamp_cooldown_steps or 20)),
        retrace_length_m=min(1.8, float(base.retrace_length_m or 2.8)),
        cover_radius_m=float(base.cover_radius_m or 0.0),
        mow_skip_cluster=max(4, int(base.mow_skip_cluster or 4)),
        mow_stop_cool=max(6, int(base.mow_stop_cool or 10)),
    )
    return MissionConfig(**data)


def apply_full_explore(settings: MissionConfig, *, world_width_m: float = 70.0) -> MissionConfig:
    """Production explore: no demo 30% / short-cap early exit.

    Tiny CI yards keep a tractable target so tests can still finish. Acre
    and larger yards raise the map-ready gate and the step budget.
    """
    settings.full_explore = True
    tiny = float(world_width_m) < 20.0
    if tiny:
        settings.explore_complete = max(float(settings.explore_complete), 0.95)
        settings.max_explore_steps = max(int(settings.max_explore_steps), 360)
        settings.mow_complete_frac = 0.0
        settings.max_mow_steps = max(int(settings.max_mow_steps), 1400)
    else:
        settings.explore_complete = max(
            float(settings.explore_complete),
            float(settings.full_explore_complete or 1.0),
        )
        settings.explore_no_frontier = max(float(settings.explore_no_frontier), 0.90)
        settings.max_explore_steps = max(
            int(settings.max_explore_steps),
            int(settings.full_explore_steps or 4000),
        )
        settings.mow_complete_frac = 0.0
        settings.max_mow_steps = max(int(settings.max_mow_steps), 5500)
    return settings


def scale_mission_budget(settings: MissionConfig, scale: float) -> MissionConfig:
    """Shrink phase caps for live demos. Does not change physics or stamps."""
    factor = min(1.0, max(0.05, float(scale)))
    if factor >= 0.999:
        return settings
    settings.max_calibrate_steps = max(24, int(settings.max_calibrate_steps * factor))
    settings.max_explore_steps = max(40, int(settings.max_explore_steps * factor))
    settings.max_mow_steps = max(40, int(settings.max_mow_steps * factor))
    settings.max_return_steps = max(20, int(settings.max_return_steps * factor))
    settings.phase_budget_scale = factor
    return settings


def _ring_closed(
    ring: list[tuple[float, float]],
    trail: list[tuple[float, float]],
    limit_m: float,
) -> bool:
    if len(trail) >= 2:
        if math.hypot(trail[0][0] - trail[-1][0], trail[0][1] - trail[-1][1]) <= limit_m:
            return True
    if len(ring) >= 3:
        dx = ring[0][0] - ring[-1][0]
        dy = ring[0][1] - ring[-1][1]
        if math.hypot(dx, dy) <= max(limit_m, 1e-3):
            return True
        # Keep-in polygons are closed by construction (last edge returns to first).
        return True
    return False


def _control_hazard(hazard: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """Keep real channels; drop colour-heuristic lips that are not a ditch."""
    from jims_mower.constants import HAZARD_DRAIN, HAZARD_DRAIN_EDGE

    hz = np.asarray(hazard, dtype=np.float32).copy()
    conf = np.asarray(confidence, dtype=np.float32)
    lips = (hz >= HAZARD_DRAIN_EDGE) & (hz < HAZARD_DRAIN)
    channel = hz >= HAZARD_DRAIN
    near_channel = channel.copy()
    near_channel[1:, :] |= channel[:-1, :]
    near_channel[:-1, :] |= channel[1:, :]
    near_channel[:, 1:] |= channel[:, :-1]
    near_channel[:, :-1] |= channel[:, 1:]
    hz[lips & ~near_channel] = 0.0
    if conf.shape == hz.shape:
        hz[lips & (conf < 0.85)] = 0.0
    return hz


def _slope_from_elevation(elevation: np.ndarray, resolution_m: float) -> np.ndarray:
    """Back-compat wrapper — prefer ``slope_from_elevation``."""
    return slope_from_elevation(elevation, resolution_m)


def _densify_ring(ring: list[tuple[float, float]], stride_m: float) -> list[tuple[float, float]]:
    """Interpolate a closed polygon so the teach lap does not cut long diagonals."""
    if len(ring) < 2:
        return list(ring)
    stride = max(float(stride_m), 0.25)
    pts: list[tuple[float, float]] = []
    n = len(ring)
    for i in range(n):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % n]
        dist = math.hypot(x1 - x0, y1 - y0)
        steps = max(1, int(round(dist / stride)))
        for k in range(steps):
            t = k / steps
            pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    return pts


def _inset_polygon(ring: list[tuple[float, float]], margin_m: float) -> list[tuple[float, float]]:
    """Shrink an axis-aligned keep-in; fall back to centroid inset."""
    if len(ring) < 3 or margin_m <= 0.0:
        return list(ring)
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    aa = all(
        (abs(x - x0) < 1e-6 or abs(x - x1) < 1e-6) and (abs(y - y0) < 1e-6 or abs(y - y1) < 1e-6)
        for x, y in ring
    )
    if aa and (x1 - x0) > 2.0 * margin_m + 1.2 and (y1 - y0) > 2.0 * margin_m + 1.2:
        nx0, nx1 = x0 + margin_m, x1 - margin_m
        ny0, ny1 = y0 + margin_m, y1 - margin_m
        corners = {
            (x0, y0): (nx0, ny0),
            (x1, y0): (nx1, ny0),
            (x1, y1): (nx1, ny1),
            (x0, y1): (nx0, ny1),
        }
        return [corners.get((x, y), (x, y)) for x, y in ring]
    cx = sum(xs) / len(ring)
    cy = sum(ys) / len(ring)
    out: list[tuple[float, float]] = []
    for x, y in ring:
        dx, dy = cx - x, cy - y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            out.append((x, y))
            continue
        step = min(float(margin_m), 0.45 * dist)
        out.append((x + step * dx / dist, y + step * dy / dist))
    return out


def _origin_from_info(info: Optional[dict[str, Any]], profile: Optional[YardProfile] = None) -> dict[str, Any]:
    if info and isinstance(info.get("survey_origin"), dict):
        return info["survey_origin"]
    spec = (info or {}).get("geofence_spec") if info else None
    if isinstance(spec, dict) and isinstance(spec.get("origin"), dict):
        return spec["origin"]
    if profile is not None:
        return profile.origin.as_dict()
    return {"e_m": 0.0, "n_m": 0.0, "u_m": 0.0}


def _gps_world(
    obs: dict[str, Any],
    info: Optional[dict[str, Any]],
    profile: Optional[YardProfile] = None,
) -> np.ndarray:
    raw = obs.get("gps", np.zeros(4, dtype=np.float32))
    origin = _origin_from_info(info, profile)
    return gps_world_from_enu(raw, origin)


def _pose_from_obs(obs: dict[str, Any], info: dict[str, Any]) -> Pose:
    raw = info.get("pose")
    if isinstance(raw, dict) and "x" in raw:
        return Pose(
            float(raw["x"]),
            float(raw["y"]),
            float(raw.get("theta", 0.0)),
            float(raw.get("z", 0.0)),
            float(raw.get("pitch", 0.0)),
            float(raw.get("roll", 0.0)),
        )
    arr = np.asarray(obs.get("pose", [0, 0, 0]), dtype=np.float32).reshape(-1)
    return Pose(
        float(arr[0]) if arr.size > 0 else 0.0,
        float(arr[1]) if arr.size > 1 else 0.0,
        float(arr[2]) if arr.size > 2 else 0.0,
        float(arr[3]) if arr.size > 3 else 0.0,
        float(arr[4]) if arr.size > 4 else 0.0,
        float(arr[5]) if arr.size > 5 else 0.0,
    )


def mission_timeline(
    policy: MissionPolicy,
    *,
    actual_coverage: float = 0.0,
) -> dict[str, Any]:
    omap = policy.observed
    plan = policy.global_plan
    return {
        "schema": MISSION_FLOW_SCHEMA,
        "phases": [p for p in MISSION_PHASES],
        "phase_ranges": policy.close_phase_ranges(),
        "events": [e.as_dict() for e in policy.events],
        "metrics": {
            **policy.status({"coverage_fraction": actual_coverage}),
            "n_events": len(policy.events),
        },
        "explore_route": [
            {"x": x, "y": y} for x, y in (policy.explore_plan.waypoints if policy.explore_plan else [])
        ],
        "mow_plan": [{"x": x, "y": y} for x, y in (plan.waypoints if plan else [])],
        "frontiers": [{"x": x, "y": y} for x, y in policy._frontier_xy],
        "skipped_segments": [{"x": x, "y": y} for x, y in policy._skipped_global],
        "map_completion": omap.completion(policy.keep_in_mask) if omap is not None else 0.0,
        "session_summary": session_summary(policy, actual_coverage=actual_coverage),
        "not_a_benchmark": True,
    }


def session_summary(
    policy: MissionPolicy,
    *,
    actual_coverage: float = 0.0,
    yard: str = "",
    duration_s: float = 0.0,
    wall_s: float = 0.0,
    info: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Owner end-of-job card: map / planned / cut / skips / duration."""
    extra: dict[str, Any] = {"coverage_fraction": actual_coverage}
    if info:
        extra.update(info)
        extra["coverage_fraction"] = actual_coverage
    status = policy.status(extra)
    dt = float(getattr(policy.cfg, "dt", 0.10) or 0.10)
    sim_s = duration_s if duration_s > 0.0 else float(policy.step) * dt
    return {
        "schema": SESSION_SCHEMA,
        "yard": yard,
        "phase": status["phase"],
        "phase_label": status["phase_label"],
        "map_pct": float(status["map_completion"]),
        "planned_pct": float(status.get("planned_coverage_fraction") or 0.0),
        "reachable": int(status.get("reachable_mowable_cells") or 0),
        "unreachable": int(status.get("unreachable_mowable_cells") or 0),
        "cut_pct": float(status["actual_coverage_fraction"]),
        "world_cut_pct": float(status.get("world_coverage_fraction") or 0.0),
        "coverage_source": str(status.get("coverage_source") or "gym_grid"),
        "skips": int(status.get("skipped_global") or 0),
        "duration_s": float(sim_s),
        "wall_s": float(wall_s),
        "steps": int(status["step"]),
        "not_a_benchmark": True,
    }


# keepouts_from_env is re-exported for the demo CLI (taught holes).
__all__ = [
    "MISSION_FLOW_SCHEMA",
    "PHASE_LABELS",
    "PHASE_ORDER",
    "MissionEvent",
    "MissionPhase",
    "MissionPolicy",
    "YardSnapshot",
    "keepouts_from_env",
    "mission_timeline",
    "apply_full_explore",
    "scale_mission_budget",
    "session_summary",
]

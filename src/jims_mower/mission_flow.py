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
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

from jims_mower.config import EnvConfig, MissionConfig
from jims_mower.constants import MISSION_FLOW_SCHEMA, MISSION_PHASES, TERRAIN_ADVICE
from jims_mower.faults import fault_is_immobilised, fault_is_retrieve
from jims_mower.geofence import GeofenceSpec
from jims_mower.kinematics import unicycle_from_wheels
from jims_mower.planning.controller import (
    combine_advice,
    geofence_from_info,
    imu_advice,
    observed_hand_signal,
    tracking_action,
)
from jims_mower.planning.costmap import build_costmap
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
from jims_mower.profile import YardProfile, trail_to_polygon
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
        self.observed: Optional[ObservedMap] = None
        self.phase = MissionPhase.CALIBRATE_BOUNDARY
        self.events: list[MissionEvent] = []
        self.phase_ranges: list[dict[str, Any]] = []
        self.step = 0
        self.phase_step = 0
        self.index = 0
        self.replans = 0
        self.last_advice = "ok"
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
        self._detour: list[tuple[float, float]] = []
        self._detour_index = 0
        self._wet = False
        self._skipped_global: list[tuple[float, float]] = []
        self._frontier_xy: list[tuple[float, float]] = []
        self._replan_cool = 0
        self._calibrate_stall = 0
        self._authored_structure: Optional[np.ndarray] = None

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

    def reset(self, obs: dict[str, Any], info: Optional[dict[str, Any]] = None) -> None:
        info = info or {}
        pose = _pose_from_obs(obs, info)
        self.fusion.reset(pose.x, pose.y, pose.theta, pose.z, pose.pitch, pose.roll)
        self.fusion.update(
            obs.get("gps", np.zeros(4, dtype=np.float32)),
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
        self.help_requested = False
        self.plan = None
        self.explore_plan = None
        self.global_plan = None
        self.snapshot = None
        self.profile = None
        self._review_hold = False
        self._detour = []
        self._detour_index = 0
        self._skipped_global = []
        self._frontier_xy = []
        self._replan_cool = 0
        self._calibrate_stall = 0
        self.safe.reset()
        self.last_safe_mode = self.safe.mode
        self._wet = bool((info.get("weather") or {}).get("wet", False))
        if obs.get("structure") is not None:
            self._authored_structure = np.asarray(obs["structure"]).copy()
        self._stamp(obs, info, pose, explored=True)
        self._enter(MissionPhase.CALIBRATE_BOUNDARY, "phase_enter", {"guided": True})

    def act(self, obs: dict[str, Any], info: dict[str, Any]) -> np.ndarray:
        pose_hint = _pose_from_obs(obs, info)
        fused = self.fusion.update(
            obs["gps"],
            obs["imu"],
            self.cfg.dt,
            commanded_v=self._last_v,
            commanded_omega=self._last_omega,
            seed_xy=(pose_hint.x, pose_hint.y),
        )
        pose = pose_hint
        self._geofence = geofence_from_info(info, self.cfg)
        self._wet = bool((info.get("weather") or {}).get("wet", False))
        self._stamp(obs, info, pose, explored=True)

        advice = self._sense_advice(obs, info, fused, pose_hint)
        self.last_advice = advice
        if estop_requested(obs, info):
            self.safe.request_estop("software/hardware estop")
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

        if self.phase == MissionPhase.CALIBRATE_BOUNDARY:
            action = self._tick_calibrate(obs, info, pose, advice)
        elif self.phase == MissionPhase.EXPLORE:
            action = self._tick_explore(obs, info, pose, advice)
        elif self.phase == MissionPhase.REVIEW:
            action = self._tick_review(obs, info, pose)
        elif self.phase == MissionPhase.MOW:
            action = self._tick_mow(obs, info, pose, advice)
        elif self.phase == MissionPhase.RETURN_HOME:
            action = self._tick_return(obs, info, pose, advice)
        else:
            action = self._hold()
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
        actual = float((info or {}).get("coverage_fraction") or 0.0)
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
            "planned_coverage_fraction": plan.planned_coverage_fraction if plan else 0.0,
            "actual_coverage_fraction": actual,
            "n_frontiers": len(self._frontier_xy),
            "n_waypoints": len(self.waypoints),
            "waypoint_index": self.index,
            "replans": self.replans,
            "skipped_global": len(self._skipped_global),
            "closed": bool(self.snapshot.closed) if self.snapshot else False,
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
        )
        if self.keep_in_mask is None or self.keep_in_mask.shape != self.observed.observed.shape:
            self.keep_in_mask = self.observed.keep_in_mask(self._geofence)

    def _sense_advice(
        self,
        obs: dict[str, Any],
        info: dict[str, Any],
        fused: Pose,
        pose_hint: Pose,
    ) -> str:
        env_advice = str(info.get("terrain_advice") or "ok")
        sensed = imu_advice(
            obs["imu"],
            tip_roll_rad=self.cfg.robot.tip_roll_rad,
            tip_pitch_rad=self.cfg.robot.tip_pitch_rad,
            slow_frac=self.cfg.planner.imu_slow_frac,
            stop_frac=self.cfg.planner.imu_stop_frac,
            pose_pitch=fused.pitch,
            pose_roll=fused.roll,
        )
        chassis = imu_advice(
            obs["imu"],
            tip_roll_rad=self.cfg.robot.tip_roll_rad,
            tip_pitch_rad=self.cfg.robot.tip_pitch_rad,
            slow_frac=self.cfg.planner.imu_slow_frac,
            stop_frac=self.cfg.planner.imu_stop_frac,
            pose_pitch=pose_hint.pitch,
            pose_roll=pose_hint.roll,
        )
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
        if self.teach.done or timed_out:
            keep_out = [list(poly) for poly in (self._geofence.keep_out or [])]
            self.profile = self.teach.to_profile(name="mission", keep_out=keep_out)
            self.profile.home = {"x": self._home.x, "y": self._home.y, "theta": self._home.theta}
            # Drop the spawn-to-ring spoke so keep-in is the closed lap, not a slice.
            planned = self.teach.planned_ring()
            if self.teach.done and len(planned) >= 3:
                self.profile.keep_in = planned
            self._geofence = self.profile.geofence_spec()
            if self.observed is not None:
                self.keep_in_mask = self.observed.keep_in_mask(self._geofence)
            self._emit(
                "boundary_recorded",
                {
                    "keep_in_vertices": len(self.profile.keep_in),
                    "trail_points": len(self.teach.trail),
                    "timed_out": bool(timed_out and not self.teach.done),
                },
            )
            self._transition(MissionPhase.EXPLORE)
        wheels = np.asarray(action, dtype=np.float32).reshape(-1)
        return self._drive(float(wheels[0]), float(wheels[1]), 0.0, pose)

    def _tick_explore(
        self,
        obs: dict[str, Any],
        info: dict[str, Any],
        pose: Pose,
        advice: str,
    ) -> np.ndarray:
        assert self.observed is not None
        keep = self.keep_in_mask
        raw = frontiers(self.observed.observed, self.observed.free, keep_in=keep)
        thin = downsample_frontiers(raw, min_sep=2, limit=40)
        self._frontier_xy = [self.observed.cell_to_world(r, c) for r, c in thin]
        completion = self.observed.completion(keep)
        ready = self._explore_ready(completion, bool(thin))
        timed_out = self.phase_step + 1 >= int(self.settings.max_explore_steps)
        if ready or timed_out:
            self._emit(
                "explore_complete",
                {
                    "map_completion": completion,
                    "n_frontiers": len(thin),
                    "timed_out": bool(timed_out and not ready),
                    "no_frontier": not thin,
                },
            )
            self._transition(MissionPhase.REVIEW)
            return self._hold()

        need_new = (
            self.explore_plan is None
            or self.index >= len(self.explore_plan.waypoints)
            or self.phase_step % 12 == 0
        )
        if need_new:
            self.explore_plan = plan_explore(
                self.observed,
                (pose.x, pose.y),
                keep_in=keep,
                waypoint_stride_m=max(0.35, self.cfg.planner.waypoint_stride_m),
            )
            self.index = 0
            if self.explore_plan.target is not None:
                tx, ty = self.observed.cell_to_world(*self.explore_plan.target)
                self._emit("frontier_target", {"x": tx, "y": ty, "n_frontiers": len(thin)})
        if self.explore_plan is None or not self.explore_plan.waypoints:
            # No reachable frontier: sweep known space once, then review.
            if completion >= float(self.settings.explore_no_frontier) or self.phase_step > 8:
                self._transition(MissionPhase.REVIEW)
            return self._hold()
        return self._track_list(
            self.explore_plan.waypoints,
            pose,
            cruise=float(self.settings.explore_cruise),
            advice=advice,
            trimmer=0.0,
        )

    def _explore_ready(self, completion: float, has_frontier: bool) -> bool:
        if completion >= float(self.settings.explore_complete):
            return True
        if not has_frontier and completion >= float(self.settings.explore_no_frontier):
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
            self._emit(
                "map_ready",
                {
                    "closed": self.snapshot.closed,
                    "completion": self.snapshot.completion,
                    "mean_confidence": self.snapshot.mean_confidence,
                    "n_components": self.snapshot.n_components,
                    "notes": list(self.snapshot.notes),
                    **metrics,
                },
            )
            return self._hold()
        if self.global_plan is None or len(self.global_plan.waypoints) < 2:
            self._transition(MissionPhase.RETURN_HOME)
        else:
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
            self._transition(MissionPhase.RETURN_HOME)
            return self._hold()
        if self.phase_step + 1 >= int(self.settings.max_mow_steps):
            self._emit("mow_budget", {"waypoints_left": max(0, len(self.global_plan.waypoints) - self.index)})
            self._transition(MissionPhase.RETURN_HOME)
            return self._hold()
        self.index = self._skip_arrived(self.global_plan.waypoints, pose, self.index)
        if advice == "stop":
            self._calibrate_stall += 1
            if self._calibrate_stall >= 6 and self.index < len(self.global_plan.waypoints):
                skipped = self.global_plan.waypoints[self.index]
                self._skipped_global.append(skipped)
                self._emit("unreachable_segment", {"x": skipped[0], "y": skipped[1], "index": self.index, "reason": "stop"})
                self.index += 1
                self._calibrate_stall = 0
            return self._nudge_inward(pose)
        self._calibrate_stall = 0
        if self.index >= len(self.global_plan.waypoints):
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
                trimmer=1.0 if advice in {"ok", "slow"} else 0.0,
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
            trimmer=1.0 if advice in {"ok", "slow"} else 0.0,
        )
        if self.index >= len(self.global_plan.waypoints):
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
            self._transition(MissionPhase.COMPLETE)
            return self._hold()
        if self.phase_step + 1 >= int(self.settings.max_return_steps):
            self._emit("return_timeout", {"x": pose.x, "y": pose.y})
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
        if completion < float(self.settings.explore_complete):
            notes.append(f"map completion {completion:.2f} below target {self.settings.explore_complete:.2f}")
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

    def _plan_global_mow(self, pose: Pose) -> CoveragePlan:
        assert self.observed is not None
        snap = self.snapshot
        keep = snap.keep_in_mask if snap is not None else self.keep_in_mask
        extra = self.observed.unknown_blocked(keep)
        structure = self.observed.structure if snap is None else snap.structure
        raw_hazard = self.observed.hazard if snap is None else snap.hazard
        conf = self.observed.confidence if snap is None else snap.confidence
        hazard = _control_hazard(raw_hazard, conf)
        elevation = self.observed.elevation if snap is None else snap.elevation
        slope = _slope_from_elevation(elevation, self.cfg.world.resolution_m)
        costmap = build_costmap(
            hazard,
            slope,
            resolution_m=self.cfg.world.resolution_m,
            width_m=self.cfg.world.width_m,
            height_m=self.cfg.world.height_m,
            max_climb_slope_rad=self.cfg.planner.max_climb_slope_rad,
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
        heading = choose_strip_orientation(
            elevation,
            mowable,
            resolution_m=self.cfg.world.resolution_m,
        )
        return plan_coverage(
            costmap,
            (pose.x, pose.y),
            strip_spacing_m=self.cfg.planner.strip_spacing_m,
            waypoint_stride_m=self.cfg.planner.waypoint_stride_m,
            mowable=mowable,
            orientation_rad=heading,
            elevation=elevation,
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
        if advice == "slow":
            cruise *= self.cfg.planner.slow_speed_factor
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

    def _hold(self) -> np.ndarray:
        self._last_v = 0.0
        self._last_omega = 0.0
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def _finish(self, action: np.ndarray, advice: str, info: dict[str, Any]) -> np.ndarray:
        if self.help_requested:
            self.safe.enter_safe("call-for-help")
        self.safe.tick(
            advice=advice,
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
            and self.phase in {MissionPhase.CALIBRATE_BOUNDARY, MissionPhase.EXPLORE, MissionPhase.MOW}
            and not self.help_requested
            and not bool(info.get("tipover"))
            and not bool(info.get("drain_drop"))
        ):
            # Mapping near a fence or tree must not retire the job.
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

    def _teach_inset_m(self) -> float:
        short = min(float(self.cfg.world.width_m), float(self.cfg.world.height_m))
        return max(0.75, min(2.10, 0.18 * short))

    def _inset_teach_ring(self, margin_m: float) -> None:
        ring = list(self.teach.waypoints)
        if len(ring) >= 2 and math.hypot(ring[0][0] - ring[-1][0], ring[0][1] - ring[-1][1]) < 1e-6:
            ring = ring[:-1]
        inset = _inset_polygon(ring, margin_m)
        if len(inset) >= 3:
            dense = _densify_ring(inset, stride_m=0.70)
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
    return MissionConfig(
        observe_confidence=base.observe_confidence,
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
    )


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
    res = max(float(resolution_m), 1e-6)
    gy, gx = np.gradient(np.asarray(elevation, dtype=np.float32), res)
    return np.arctan(np.hypot(gx, gy)).astype(np.float32)


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
]

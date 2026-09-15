"""Zero-turn waypoint tracker that maps terrain_advice → slow / reroute / stop."""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from jims_mower.config import EnvConfig
from jims_mower.constants import GRAVITY_MPS2, HAZARD_DRAIN_EDGE, TERRAIN_ADVICE
from jims_mower.kinematics import unicycle_from_wheels, wheels_from_unicycle, wrap_angle
from jims_mower.planning.costmap import build_costmap
from jims_mower.planning.coverage import CoveragePlan, plan_coverage
from jims_mower.planning.fusion import ComplementaryPoseFilter, attitude_from_accel
from jims_mower.types import Pose

_ADVICE_RANK = {name: i for i, name in enumerate(("ok", "slow", "reroute", "stop"))}


def combine_advice(*advices: str) -> str:
    """Highest-severity advice wins (stop > reroute > slow > ok)."""
    best = "ok"
    best_rank = 0
    for raw in advices:
        name = raw if raw in _ADVICE_RANK else "ok"
        rank = _ADVICE_RANK[name]
        if rank > best_rank:
            best, best_rank = name, rank
    return best


def imu_advice(
    imu: np.ndarray,
    *,
    tip_roll_rad: float,
    tip_pitch_rad: float,
    slow_frac: float,
    stop_frac: float,
    pose_pitch: float = 0.0,
    pose_roll: float = 0.0,
) -> str:
    """Cross-check: accel tilt *or* fused/true pitch-roll vs tip thresholds."""
    roll, pitch = pose_roll, pose_pitch
    imu_arr = np.asarray(imu, dtype=np.float32).reshape(-1)
    spec = float(np.linalg.norm(imu_arr[:3])) if imu_arr.size >= 3 else GRAVITY_MPS2
    if abs(spec - GRAVITY_MPS2) < 0.75:
        roll_a, pitch_a = attitude_from_accel(imu_arr)
        roll = roll_a if abs(roll_a) >= abs(roll) else roll
        pitch = pitch_a if abs(pitch_a) >= abs(pitch) else pitch
    if abs(roll) >= stop_frac * tip_roll_rad or abs(pitch) >= stop_frac * tip_pitch_rad:
        return "stop"
    if abs(roll) >= slow_frac * tip_roll_rad or abs(pitch) >= slow_frac * tip_pitch_rad:
        return "slow"
    return "ok"


def tracking_action(
    pose: Pose,
    target: tuple[float, float],
    *,
    cruise: float,
    wheelbase_m: float,
    turn_in_place_rad: float,
    max_omega: float = 2.8,
    k_heading: float = 2.4,
) -> tuple[np.ndarray, float, float]:
    """Differential-drive command toward ``target``. Wheels in [-1, 1]."""
    dx = target[0] - pose.x
    dy = target[1] - pose.y
    dist = math.hypot(dx, dy)
    heading = math.atan2(dy, dx)
    err = wrap_angle(heading - pose.theta)
    if dist < 1e-4:
        return np.array([0.0, 0.0], dtype=np.float32), dist, err
    if abs(err) > turn_in_place_rad:
        omega = float(np.clip(k_heading * err, -max_omega, max_omega))
        v = 0.0
    else:
        v = float(cruise) * max(0.15, 1.0 - 0.55 * abs(err) / math.pi)
        omega = float(np.clip(k_heading * err, -max_omega, max_omega))
    left, right = wheels_from_unicycle(v, omega, wheelbase_m)
    peak = max(abs(left), abs(right), 1e-6)
    if peak > 1.0:
        left /= peak
        right /= peak
    return np.array([left, right], dtype=np.float32), dist, err


def _mowable_mask(coverage: Optional[np.ndarray], blocked_shape: tuple[int, int]) -> Optional[np.ndarray]:
    if coverage is None:
        return None
    cov = np.asarray(coverage)
    if cov.shape != blocked_shape:
        return None
    return cov >= 0.0


def _cone_mask(
    shape: tuple[int, int],
    pose: Pose,
    resolution_m: float,
    *,
    reach_m: float = 0.95,
    half_width_m: float = 0.38,
) -> np.ndarray:
    """Cells in a short forward rectangle — used to force a reroute."""
    mask = np.zeros(shape, dtype=bool)
    rows, cols = shape
    res = max(resolution_m, 1e-6)
    c = math.cos(pose.theta)
    s = math.sin(pose.theta)
    for t in np.linspace(0.12, reach_m, 10):
        for w in np.linspace(-half_width_m, half_width_m, 7):
            x = pose.x + t * c - w * s
            y = pose.y + t * s + w * c
            col = int(x / res)
            row = int(y / res)
            if 0 <= row < rows and 0 <= col < cols:
                mask[row, col] = True
    return mask


class TerrainPolicy:
    """Coverage planner + zero-turn controller for the headless demo / agents."""

    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg
        self.fusion = ComplementaryPoseFilter(
            gps_blend=cfg.planner.gps_blend,
            accel_blend=cfg.planner.accel_blend,
        )
        self.plan: Optional[CoveragePlan] = None
        self.index = 0
        self.replans = 0
        self.last_advice = "ok"
        self._last_v = 0.0
        self._last_omega = 0.0
        self._reroute_cool = 0
        self._width_m = cfg.world.width_m
        self._height_m = cfg.world.height_m
        self._resolution_m = cfg.world.resolution_m
        self._drain_cells = 0

    def reset(self, obs: dict[str, Any], info: Optional[dict[str, Any]] = None) -> CoveragePlan:
        info = info or {}
        pose = _pose_from_obs(obs, info)
        self.fusion.reset(pose.x, pose.y, pose.theta, pose.z, pose.pitch, pose.roll)
        self.fusion.update(
            obs.get("gps", np.zeros(4, dtype=np.float32)),
            obs.get("imu", np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0], dtype=np.float32)),
            self.cfg.dt,
            seed_xy=(pose.x, pose.y),
        )
        start = self.fusion.pose()
        self._last_v = 0.0
        self._last_omega = 0.0
        self.replans = 0
        self._reroute_cool = 0
        self.last_advice = "ok"
        self._remember_map_size(obs)
        self._drain_cells = _drain_cell_count(obs.get("hazard"))
        self.plan = self._build_plan(obs, start, extra_blocked=None)
        self.index = _skip_arrived(self.plan.waypoints, start, self.cfg.planner.arrive_radius_m)
        return self.plan

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
        # Plan is in the map / world frame. Track the observed pose (the same
        # contract an RL policy sees). Fusion still owns planner start + IMU
        # attitude; swap `pose = fused` on the Orin when this *is* localization.
        pose = pose_hint
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
        advice = combine_advice(env_advice, sensed, chassis)
        if advice not in TERRAIN_ADVICE:
            advice = "ok"
        self.last_advice = advice

        if advice == "stop":
            self._last_v = 0.0
            self._last_omega = 0.0
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        if self.plan is None:
            self.reset(obs, info)

        self._maybe_replan_new_hazards(obs, pose)

        if advice == "reroute":
            self._handle_reroute(obs, pose)

        waypoints = self.plan.waypoints if self.plan is not None else []
        arrive = self.cfg.planner.arrive_radius_m
        while self.index < len(waypoints):
            tx, ty = waypoints[self.index]
            if math.hypot(tx - pose.x, ty - pose.y) <= arrive:
                self.index += 1
                continue
            break
        if self.index >= len(waypoints):
            self._replan(obs, pose, extra_blocked=None)
            waypoints = self.plan.waypoints if self.plan is not None else []
            self.index = _skip_arrived(waypoints, pose, arrive)
            if self.index >= len(waypoints):
                self._last_v = 0.0
                self._last_omega = 0.0
                return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        target = waypoints[self.index]
        cruise = self.cfg.planner.cruise_speed
        if advice == "slow":
            cruise *= self.cfg.planner.slow_speed_factor
        wheels, _dist, _err = tracking_action(
            pose,
            target,
            cruise=cruise,
            wheelbase_m=self.cfg.robot.wheelbase_m,
            turn_in_place_rad=self.cfg.planner.turn_in_place_rad,
        )
        vmax = self.cfg.robot.max_wheel_speed_mps
        self._last_v, self._last_omega = unicycle_from_wheels(
            float(wheels[0]) * vmax,
            float(wheels[1]) * vmax,
            self.cfg.robot.wheelbase_m,
        )
        trimmer = 1.0 if advice in {"ok", "slow"} else 0.0
        return np.array([wheels[0], wheels[1], trimmer], dtype=np.float32)

    @property
    def waypoints(self) -> list[tuple[float, float]]:
        if self.plan is None:
            return []
        return self.plan.waypoints

    def _remember_map_size(self, obs: dict[str, Any]) -> None:
        hazard = obs.get("hazard")
        if hazard is None:
            return
        rows, cols = np.asarray(hazard).shape[:2]
        res = self._resolution_m
        self._height_m = rows * res
        self._width_m = cols * res

    def _build_plan(
        self,
        obs: dict[str, Any],
        pose: Pose,
        extra_blocked: Optional[np.ndarray],
    ) -> CoveragePlan:
        hazard = np.asarray(obs["hazard"], dtype=np.float32)
        slope = np.asarray(obs["slope"], dtype=np.float32)
        occupancy = np.asarray(obs["occupancy"], dtype=np.float32) if "occupancy" in obs else None
        coverage = np.asarray(obs["coverage"], dtype=np.float32) if "coverage" in obs else None
        costmap = build_costmap(
            hazard,
            slope,
            resolution_m=self._resolution_m,
            width_m=self._width_m,
            height_m=self._height_m,
            max_climb_slope_rad=self.cfg.planner.max_climb_slope_rad,
            drain_clearance_m=self.cfg.planner.drain_clearance_m,
            occupancy=occupancy,
            occupancy_inflate_m=self.cfg.planner.occupancy_inflate_m,
            margin_m=self.cfg.robot.collision_radius_m,
            extra_blocked=extra_blocked,
        )
        mowable = _mowable_mask(coverage, costmap.blocked.shape)
        return plan_coverage(
            costmap,
            (pose.x, pose.y),
            strip_spacing_m=self.cfg.planner.strip_spacing_m,
            waypoint_stride_m=self.cfg.planner.waypoint_stride_m,
            mowable=mowable,
        )

    def _replan(
        self,
        obs: dict[str, Any],
        pose: Pose,
        extra_blocked: Optional[np.ndarray],
    ) -> None:
        if self.replans >= self.cfg.planner.max_replans and extra_blocked is not None:
            return
        self.plan = self._build_plan(obs, pose, extra_blocked)
        self.index = _skip_arrived(self.plan.waypoints, pose, self.cfg.planner.arrive_radius_m)
        self.replans += 1

    def _maybe_replan_new_hazards(self, obs: dict[str, Any], pose: Pose) -> None:
        """Rebuild the coverage path when the vision map grows new lips/channels."""
        n = _drain_cell_count(obs.get("hazard"))
        if n >= self._drain_cells + 6 and self.replans < self.cfg.planner.max_replans:
            self._replan(obs, pose, extra_blocked=None)
        self._drain_cells = max(self._drain_cells, n)

    def _handle_reroute(self, obs: dict[str, Any], pose: Pose) -> None:
        waypoints = self.waypoints
        hazard = np.asarray(obs["hazard"], dtype=np.float32)
        res = self._resolution_m
        while self.index < len(waypoints):
            tx, ty = waypoints[self.index]
            col = int(tx / res)
            row = int(ty / res)
            on_lip = (
                0 <= row < hazard.shape[0]
                and 0 <= col < hazard.shape[1]
                and float(hazard[row, col]) >= HAZARD_DRAIN_EDGE
            )
            if on_lip:
                self.index += 1
                continue
            break
        if self._reroute_cool > 0:
            self._reroute_cool -= 1
            return
        extra = _cone_mask(hazard.shape, pose, res)
        self._replan(obs, pose, extra_blocked=extra)
        self._reroute_cool = 10


def _skip_arrived(
    waypoints: list[tuple[float, float]],
    pose: Pose,
    arrive: float,
) -> int:
    i = 0
    while i < len(waypoints):
        if math.hypot(waypoints[i][0] - pose.x, waypoints[i][1] - pose.y) <= arrive:
            i += 1
            continue
        break
    return i


def _drain_cell_count(hazard: Any) -> int:
    if hazard is None:
        return 0
    arr = np.asarray(hazard)
    if arr.size == 0:
        return 0
    return int((arr >= HAZARD_DRAIN_EDGE).sum())


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

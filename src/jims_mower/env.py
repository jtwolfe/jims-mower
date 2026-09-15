"""Gymnasium environment for the camera-driven zero-turn trimmer mower."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from jims_mower.cameras import camera_world_pose
from jims_mower.config import EnvConfig, load_config
from jims_mower.constants import (
    DET_FEATURES,
    HAND_SIGNALS,
    LABEL_TO_ID,
    MAX_DETECTIONS,
    SIGNAL_TO_ID,
)
from jims_mower.kinematics import integrate_pose, trimmer_xy
from jims_mower.maps import GrassCoverageMap, occupancy_from_detections
from jims_mower.perception import ColorGrassObserver, HandSignalCurriculum, MockDetector
from jims_mower.perception.base import Detector, GrassObserver
from jims_mower.renderer import render_camera, render_topdown
from jims_mower.reward import compute_reward
from jims_mower.safety import first_collision, in_yard, trimmer_interlock
from jims_mower.types import CameraSpec, Detection, PerceptionContext, Pose
from jims_mower.world import robot_start_pose, spawn_yard, step_movers


def encode_detections(
    detections: list[Detection],
    camera_index: dict[str, int],
    max_det: int = MAX_DETECTIONS,
) -> np.ndarray:
    buf = np.zeros((max_det, DET_FEATURES), dtype=np.float32)
    for i, det in enumerate(detections[:max_det]):
        x, y, w, h = det.bbox
        buf[i, 0] = LABEL_TO_ID.get(det.label, 0)
        buf[i, 1] = camera_index.get(det.camera, -1)
        buf[i, 2] = x
        buf[i, 3] = y
        buf[i, 4] = w
        buf[i, 5] = h
        buf[i, 6] = det.confidence
        buf[i, 7] = SIGNAL_TO_ID.get(det.hand_signal or "", 0)
    return buf


class MowerEnv(gym.Env):
    """Zero-turn mower with a front string-trimmer and 4–6 RGB cameras.

    Action (Box 3):
        [0] left wheel speed in [-1, 1] (fraction of max)
        [1] right wheel speed in [-1, 1]
        [2] trimmer request in [0, 1] (enabled if > 0.5 and interlock allows)

    Observation (Dict):
        cameras: per-camera uint8 RGB
        coverage: grass map (1 cut, 0 uncut, -1 non-grass)
        occupancy: detection-rasterized occupancy
        detections: padded [label, cam, u, v, w, h, conf, signal]
        pose: (x, y, theta)
        trimmer_enabled: {0, 1}
        hand_signal: 0=none, 1=stop, 2=go, 3=follow, 4=back
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        config: Optional[Union[str, Path, dict, EnvConfig]] = None,
        render_mode: Optional[str] = None,
        detector: Optional[Detector] = None,
        grass_observer: Optional[GrassObserver] = None,
        hand_signals: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.cfg = load_config(config)
        if hand_signals is not None:
            self.cfg.curriculum.hand_signals = bool(hand_signals)
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode
        self.cameras: list[CameraSpec] = self.cfg.resolved_cameras()
        self.camera_index = {c.name: i for i, c in enumerate(self.cameras)}
        self.detector: Detector = detector or MockDetector()
        self.grass_observer: GrassObserver = grass_observer or ColorGrassObserver()
        self._signals = HandSignalCurriculum(
            self.cfg.curriculum.hand_signals,
            self.cfg.curriculum.signal_hold_steps,
        )

        w = int(self.cfg.sensors.width)
        h = int(self.cfg.sensors.height)
        cam_spaces = {
            cam.name: spaces.Box(0, 255, shape=(h, w, 3), dtype=np.uint8)
            for cam in self.cameras
        }
        cov_shape = (
            max(1, int(round(self.cfg.world.height_m / self.cfg.world.resolution_m))),
            max(1, int(round(self.cfg.world.width_m / self.cfg.world.resolution_m))),
        )
        self.observation_space = spaces.Dict(
            {
                "cameras": spaces.Dict(cam_spaces),
                "coverage": spaces.Box(-1.0, 1.0, shape=cov_shape, dtype=np.float32),
                "occupancy": spaces.Box(0.0, 1.0, shape=cov_shape, dtype=np.float32),
                "detections": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(MAX_DETECTIONS, DET_FEATURES),
                    dtype=np.float32,
                ),
                "pose": spaces.Box(
                    low=np.array([-2.0, -2.0, -np.pi], dtype=np.float32),
                    high=np.array(
                        [self.cfg.world.width_m + 2.0, self.cfg.world.height_m + 2.0, np.pi],
                        dtype=np.float32,
                    ),
                    dtype=np.float32,
                ),
                "trimmer_enabled": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "hand_signal": spaces.Discrete(1 + len(HAND_SIGNALS)),
            }
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._pose = robot_start_pose(self.cfg.world.width_m, self.cfg.world.height_m)
        self._coverage = GrassCoverageMap(
            self.cfg.world.width_m,
            self.cfg.world.height_m,
            self.cfg.world.resolution_m,
        )
        self._yard = spawn_yard(
            np.random.default_rng(0),
            self.cfg.world.width_m,
            self.cfg.world.height_m,
            {},
            (self._pose.x, self._pose.y, 1.2),
        )
        self._steps = 0
        self._trimmer_on = False
        self._last_images: dict[str, np.ndarray] = {}
        self._last_detections: list[Detection] = []
        self._last_topdown: Optional[np.ndarray] = None

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        self._steps = 0
        self._trimmer_on = False
        self._pose = robot_start_pose(self.cfg.world.width_m, self.cfg.world.height_m)
        counts = {
            "person": self.cfg.world.n_people,
            "dog": self.cfg.world.n_dogs,
            "cat": self.cfg.world.n_cats,
            "bird": self.cfg.world.n_birds,
            "tree": self.cfg.world.n_trees,
            "furniture": self.cfg.world.n_furniture,
            "toy": self.cfg.world.n_toys,
        }
        if "counts" in options:
            counts.update(options["counts"])
        keepout_r = max(1.4, self.cfg.robot.trimmer.safety_radius_m * 0.7)
        self._yard = spawn_yard(
            self.np_random,
            self.cfg.world.width_m,
            self.cfg.world.height_m,
            counts,
            (self._pose.x, self._pose.y, keepout_r),
        )
        self._coverage.reset()
        for obst in self._yard.static():
            if obst.kind in {"tree", "furniture"}:
                self._coverage.exclude_circle(obst.x, obst.y, obst.radius)
        self._signals.assign(self._yard.obstacles, self.np_random)
        obs, info = self._observe()
        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 3:
            raise ValueError("action must have shape (3,)")
        left_n = float(np.clip(action[0], -1.0, 1.0))
        right_n = float(np.clip(action[1], -1.0, 1.0))
        requested = bool(action[2] > 0.5)
        vmax = self.cfg.robot.max_wheel_speed_mps
        self._pose = integrate_pose(
            self._pose,
            left_n * vmax,
            right_n * vmax,
            self.cfg.robot.wheelbase_m,
            self.cfg.dt,
            vmax,
        )
        step_movers(self._yard, self.cfg.dt, self.np_random)
        self._signals.maybe_rotate(self._yard.obstacles, self.np_random)

        decision = trimmer_interlock(
            requested,
            self._pose,
            self._yard.obstacles,
            offset_m=self.cfg.robot.trimmer.offset_m,
            safety_radius_m=self.cfg.robot.trimmer.safety_radius_m,
        )
        self._trimmer_on = decision.trimmer_enabled

        newly = 0
        if self._trimmer_on:
            hx, hy = trimmer_xy(self._pose, self.cfg.robot.trimmer.offset_m)
            newly = self._coverage.mark_circle(hx, hy, self.cfg.robot.trimmer.radius_m)

        hit = first_collision(
            self._pose, self._yard.obstacles, self.cfg.robot.collision_radius_m
        )
        oob = not in_yard(
            self._pose,
            self.cfg.world.width_m,
            self.cfg.world.height_m,
            self.cfg.robot.collision_radius_m,
        )
        breakdown = compute_reward(
            self.cfg.reward,
            newly_cut=newly,
            grass_cells=self._coverage.grass_cell_count(),
            coverage_fraction=self._coverage.coverage_fraction(),
            collision_kind=hit.kind if hit is not None else None,
            out_of_bounds=oob,
        )
        self._steps += 1
        terminated = bool(hit is not None or oob or breakdown.done_success)
        truncated = bool(self._steps >= self.cfg.max_steps and not terminated)

        obs, info = self._observe()
        info.update(
            {
                "newly_cut": newly,
                "collision": hit.kind if hit is not None else None,
                "out_of_bounds": oob,
                "trimmer_requested": requested,
                "trimmer_enabled": self._trimmer_on,
                "trimmer_block": decision.blocked_reason,
                "nearest_living_m": decision.nearest_living_m,
                "reward_coverage": breakdown.coverage,
                "reward_collision": breakdown.collision,
                "success": breakdown.done_success,
            }
        )
        return obs, float(breakdown.total), terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            if self._last_topdown is None:
                self._last_topdown = self._topdown()
            return self._last_topdown
        return None

    def _topdown(self) -> np.ndarray:
        return render_topdown(
            self._pose,
            self._coverage,
            self._yard.obstacles,
            trimmer_xy=trimmer_xy(self._pose, self.cfg.robot.trimmer.offset_m),
            trimmer_on=self._trimmer_on,
        )

    def _render_cameras(self) -> dict[str, np.ndarray]:
        yard = (self.cfg.world.width_m, self.cfg.world.height_m)
        images = {}
        for cam in self.cameras:
            images[cam.name] = render_camera(
                self._pose,
                cam,
                self._coverage,
                self._yard.obstacles,
                self.cfg.sensors.width,
                self.cfg.sensors.height,
                yard,
            )
        return images

    def _observe(self) -> tuple[dict[str, Any], dict[str, Any]]:
        images = self._render_cameras()
        self._last_images = images
        context = PerceptionContext(
            pose=self._pose,
            cameras=self.cameras,
            obstacles=list(self._yard.obstacles),
            image_size=(self.cfg.sensors.width, self.cfg.sensors.height),
            hand_signals_enabled=self.cfg.curriculum.hand_signals,
        )
        detections = self.detector.detect(images, context)
        self._last_detections = detections
        occupancy = occupancy_from_detections(
            self._coverage.cut.shape,
            detections,
            self.cfg.world.resolution_m,
        )
        vision = self.grass_observer.estimate(images)
        signal_name = self._signals.nearest_person_signal(
            self._yard.obstacles, (self._pose.x, self._pose.y)
        )
        signal_id = SIGNAL_TO_ID.get(signal_name or "", 0)
        self._last_topdown = self._topdown()
        obs = {
            "cameras": images,
            "coverage": self._coverage.as_float(),
            "occupancy": occupancy.astype(np.float32),
            "detections": encode_detections(detections, self.camera_index),
            "pose": np.array(
                [self._pose.x, self._pose.y, self._pose.theta], dtype=np.float32
            ),
            "trimmer_enabled": np.array(
                [1.0 if self._trimmer_on else 0.0], dtype=np.float32
            ),
            "hand_signal": int(signal_id),
        }
        info = {
            "pose": {"x": self._pose.x, "y": self._pose.y, "theta": self._pose.theta},
            "coverage_fraction": self._coverage.coverage_fraction(),
            "detections": [d.as_dict() for d in detections],
            "grass_vision": vision,
            "camera_poses": [
                _cam_pose_dict(self._pose, cam) for cam in self.cameras
            ],
            "hand_signals_enabled": self.cfg.curriculum.hand_signals,
            "steps": self._steps,
        }
        return obs, info


def _cam_pose_dict(pose: Pose, cam: CameraSpec) -> dict[str, Any]:
    wp = camera_world_pose(pose, cam)
    return {
        "name": cam.name,
        "x": wp.x,
        "y": wp.y,
        "z": wp.z,
        "yaw": wp.yaw,
        "pitch": wp.pitch,
    }

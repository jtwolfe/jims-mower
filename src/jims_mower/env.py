"""Gymnasium environment for the camera-driven zero-turn trimmer mower."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from jims_mower.appearance import Appearance, appearance_rng, sample_appearance
from jims_mower.cameras import camera_world_pose
from jims_mower.config import EnvConfig, load_config
from jims_mower.renderer import apply_weather_rgb
from jims_mower.scenarios import Scenario, load_source
from jims_mower.constants import (
    DET_FEATURES,
    HAND_SIGNALS,
    LABEL_TO_ID,
    MAX_DETECTIONS,
    SEMANTIC_NAMES,
    SIGNAL_TO_ID,
    STRUCTURE_NAMES,
)
from jims_mower.structures import layer_from_terrain_labels, no_mow_from_labels
from jims_mower.kinematics import (
    integrate_pose,
    sit_on_terrain,
    trimmer_xy,
    trimmer_xyz,
    unicycle_from_wheels,
    wheel_clearances,
)
from jims_mower.geofence import GeofenceSpec, geofence_advice
from jims_mower.maps import GrassCoverageMap, occupancy_from_detections
from jims_mower.mapping import LoopClosureStub, PersistentOccupancy, fuse_height_rgb_tof
from jims_mower.mission import apply_mission, load_mission, save_mission
from jims_mower.perception import HandSignalCurriculum
from jims_mower.perception.base import Detector, GrassObserver
from jims_mower.perception.detect import detector_from_mode
from jims_mower.perception.grass import (
    ClassAwareGrassObserver,
    grass_observer_from_mode,
    observer_coverage_fraction,
)
from jims_mower.perception.semantic import semantic_raster
from jims_mower.perception.temporal import DetectionTracklets
from jims_mower.perception.terrain import TerrainObserver, terrain_observer_from_mode
from jims_mower.perception.trt import TrtDetector, TrtTerrainObserver
from jims_mower.renderer import render_camera, render_scalar_map, render_topdown
from jims_mower.reward import compute_reward
from jims_mower.safety import (
    cutter_risk_hit,
    first_collision,
    in_yard,
    living_advice,
    nearest_living,
    terrain_hazards,
    trimmer_interlock,
)
from jims_mower.faults import FaultBus
from jims_mower.hardware_estop import HardwareEstop, is_hw_estop_kind, is_hw_reset_mode
from jims_mower.radio import RadioSim
from jims_mower.runtime.budget import OrinBudget, budget_from_config
from jims_mower.runtime.capture import adapter_kind, ensure_contract_rgb
from jims_mower.runtime.watchdog import SensorWatchdog
from jims_mower.sensors import (
    imu_to_array,
    sample_accel_bias,
    simulate_gps,
    simulate_imu,
    simulate_tof,
    slide_board_under_wheel,
)
from jims_mower.terrain import HeightField, generate_terrain
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
        pose: (x, y, theta, z, pitch, roll)
        imu: (ax, ay, az, gx, gy, gz) body-frame specific force + gyro
        gps: (x, y, z, valid)
        tof: downward ranges at FL, FR, RL, RR
        elevation / slope / hazard / confidence: terrain maps (oracle, heuristic, or blind)
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
        terrain_observer: Optional[TerrainObserver] = None,
        hand_signals: Optional[bool] = None,
        scenario: Optional[Scenario] = None,
    ) -> None:
        super().__init__()
        if isinstance(config, Scenario):
            self.cfg = config.config
            self.scenario: Optional[Scenario] = config
        elif scenario is not None:
            self.cfg = load_config(config) if config is not None else scenario.config
            self.scenario = scenario
        else:
            self.cfg, self.scenario = load_source(config)
        if hand_signals is not None:
            self.cfg.curriculum.hand_signals = bool(hand_signals)
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode
        self.cameras: list[CameraSpec] = self.cfg.resolved_cameras()
        self.camera_index = {c.name: i for i, c in enumerate(self.cameras)}
        self._camera_adapter = None
        backend = str(self.cfg.perception.detector_backend or "mock").strip().lower()
        if detector is not None:
            inner_det: Detector = detector
        else:
            inner_det = detector_from_mode(
                backend,
                onnx_path=self.cfg.perception.onnx_path or None,
                engine_path=self.cfg.perception.engine_path or None,
            )
        if backend in {"trt", "tensorrt"} and not isinstance(inner_det, TrtDetector):
            self.detector = TrtDetector(self.cfg.perception.engine_path or None, inner_det)
        else:
            self.detector = inner_det
        self.grass_observer: GrassObserver = grass_observer or grass_observer_from_mode(
            self.cfg.perception.grass_mode,
            self.cfg.perception.weights_path,
        )
        inner_terrain = terrain_observer or terrain_observer_from_mode(
            self.cfg.perception.terrain_mode,
            weights_path=self.cfg.perception.weights_path or None,
            onnx_path=self.cfg.perception.onnx_path or None,
            engine_path=self.cfg.perception.engine_path or None,
            temporal=self.cfg.perception.temporal or None,
        )
        if self.cfg.perception.engine_path and not isinstance(inner_terrain, TrtTerrainObserver):
            self.terrain_observer = TrtTerrainObserver(
                self.cfg.perception.engine_path or None,
                inner=inner_terrain,
                weights_path=self.cfg.perception.weights_path or None,
            )
        else:
            self.terrain_observer = inner_terrain
        self._tracklets = DetectionTracklets()
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
                    -1.0e6,
                    1.0e6,
                    shape=(MAX_DETECTIONS, DET_FEATURES),
                    dtype=np.float32,
                ),
                "pose": spaces.Box(
                    low=np.array(
                        [-2.0, -2.0, -np.pi, -5.0, -np.pi / 2, -np.pi / 2],
                        dtype=np.float32,
                    ),
                    high=np.array(
                        [
                            self.cfg.world.width_m + 2.0,
                            self.cfg.world.height_m + 2.0,
                            np.pi,
                            5.0,
                            np.pi / 2,
                            np.pi / 2,
                        ],
                        dtype=np.float32,
                    ),
                    dtype=np.float32,
                ),
                "imu": spaces.Box(-80.0, 80.0, shape=(6,), dtype=np.float32),
                "gps": spaces.Box(
                    low=np.array([-50.0, -50.0, -20.0, 0.0], dtype=np.float32),
                    high=np.array(
                        [
                            self.cfg.world.width_m + 50.0,
                            self.cfg.world.height_m + 50.0,
                            20.0,
                            1.0,
                        ],
                        dtype=np.float32,
                    ),
                    dtype=np.float32,
                ),
                "tof": spaces.Box(
                    0.0,
                    max(self.cfg.sensors.tof.max_range_m, 0.1),
                    shape=(4,),
                    dtype=np.float32,
                ),
                "elevation": spaces.Box(-5.0, 5.0, shape=cov_shape, dtype=np.float32),
                "elevation_prior": spaces.Box(-5.0, 5.0, shape=cov_shape, dtype=np.float32),
                "slope": spaces.Box(0.0, np.pi / 2, shape=cov_shape, dtype=np.float32),
                "hazard": spaces.Box(0.0, 3.0, shape=cov_shape, dtype=np.float32),
                "structure": spaces.Box(0.0, 8.0, shape=cov_shape, dtype=np.float32),
                "confidence": spaces.Box(0.0, 1.0, shape=cov_shape, dtype=np.float32),
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
        self._terrain = HeightField.empty(
            self.cfg.world.width_m,
            self.cfg.world.height_m,
            self.cfg.world.resolution_m,
        )
        self._structure = layer_from_terrain_labels(
            self._terrain.labels,
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
        self._prev_pose = self._pose
        self._prev_v = 0.0
        self._last_v = 0.0
        self._last_omega = 0.0
        self._accel_bias = np.zeros(3, dtype=np.float64)
        self._last_imu = np.zeros(6, dtype=np.float32)
        self._last_gps = np.zeros(4, dtype=np.float32)
        self._last_tof = np.zeros(4, dtype=np.float32)
        self._last_terrain_est = None
        self._appearance = Appearance.neutral()
        self._had_episode = False
        self._grass_save_path: Optional[str] = None
        self._grass_loaded = False
        self._mission_save_path: Optional[str] = None
        self._mission_loaded = False
        self._episode_seed: Optional[int] = None
        self.budget: OrinBudget = budget_from_config(self.cfg)
        self.watchdog = SensorWatchdog.from_config(self.cfg, dt=self.cfg.dt)
        self.hw_estop = HardwareEstop()
        self._imu_stamp_s = 0.0
        self._vision_stamp_s = 0.0
        self.fault_bus = FaultBus.from_config(self.cfg)
        self.radio = RadioSim.from_config(self.cfg)
        self._occ_persist: Optional[PersistentOccupancy] = None
        self._loop = LoopClosureStub()
        self._fused_elev: Optional[np.ndarray] = None
        self._observer_coverage: Optional[np.ndarray] = None
        self._last_semantic: Optional[np.ndarray] = None
        self._yard_profile = None

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        self._steps = 0
        self._trimmer_on = False
        grow_cfg = self.cfg.world.grass
        persist = options.get("save_grass", grow_cfg.persist_path)
        load_path = options.get("load_grass", grow_cfg.persist_path)
        self._grass_save_path = str(persist) if persist else None
        mission_save = options.get("save_mission")
        mission_load = options.get("load_mission")
        self._mission_save_path = str(mission_save) if mission_save else None
        self._mission_loaded = False
        self._episode_seed = seed
        self._yard_profile = None
        raw_profile = options.get("yard_profile") or options.get("profile")
        if raw_profile:
            from jims_mower.profile import YardProfile, apply_profile_to_scenario, load_yard_profile
            from jims_mower.scenarios import empty_scenario

            if isinstance(raw_profile, YardProfile):
                self._yard_profile = raw_profile
            else:
                self._yard_profile = load_yard_profile(raw_profile)
            if self.scenario is None:
                self.scenario = empty_scenario(self.cfg)
            apply_profile_to_scenario(
                self.scenario,
                self._yard_profile,
                resize_world=bool(options.get("resize_world", True)),
            )
        previous_cut = None
        if grow_cfg.enabled and self._had_episode:
            previous_cut = self._coverage.cut.copy()
        self._pose = robot_start_pose(self.cfg.world.width_m, self.cfg.world.height_m)
        if self._yard_profile is not None:
            self._pose = self._yard_profile.home_pose()
        counts = {
            "person": self.cfg.world.n_people,
            "dog": self.cfg.world.n_dogs,
            "cat": self.cfg.world.n_cats,
            "bird": self.cfg.world.n_birds,
            "tree": self.cfg.world.n_trees,
            "furniture": self.cfg.world.n_furniture,
            "toy": self.cfg.world.n_toys,
            "hose": self.cfg.world.n_hoses,
            "cord": self.cfg.world.n_cords,
        }
        if "counts" in options:
            counts.update(options["counts"])
        keepout_r = max(1.4, self.cfg.robot.trimmer.safety_radius_m * 0.7)
        terr_cfg = self.cfg.world.terrain
        robot_keep = (self._pose.x, self._pose.y, max(keepout_r, terr_cfg.keepout_m))
        grad = terr_cfg.base_gradient
        self._terrain = generate_terrain(
            self.np_random,
            self.cfg.world.width_m,
            self.cfg.world.height_m,
            self.cfg.world.resolution_m,
            n_drains=terr_cfg.n_drains,
            n_banks=terr_cfg.n_banks,
            drain_width_m=terr_cfg.drain_width_m,
            drain_depth_m=terr_cfg.drain_depth_m,
            drain_length_m=terr_cfg.drain_length_m,
            drain_side_slope=terr_cfg.drain_side_slope,
            bank_height_m=terr_cfg.bank_height_m,
            bank_width_m=terr_cfg.bank_width_m,
            bank_length_m=terr_cfg.bank_length_m,
            max_slope_rad=terr_cfg.max_slope_rad,
            noise_amp_m=terr_cfg.noise_amp_m,
            keepout=[robot_keep],
            enabled=terr_cfg.enabled,
            layout=self.cfg.world.layout,
            n_puddles=self.cfg.world.n_puddles,
            puddle_radius_m=terr_cfg.puddle_radius_m,
            puddle_depth_m=terr_cfg.puddle_depth_m,
            explicit_drains=list(self.scenario.drains) if self.scenario else None,
            explicit_banks=list(self.scenario.banks) if self.scenario else None,
            gradient_slope_rad=grad.effective_slope_rad(),
            gradient_yaw_rad=grad.yaw_rad,
            gradient_undulation_m=grad.undulation_m,
            multi_scale_amp_m=terr_cfg.multi_scale_amp_m,
            swale_amp_m=terr_cfg.swale_amp_m,
            dem_path=terr_cfg.dem_path,
            paths=list(self.scenario.paths) if self.scenario else None,
            buildings=(
                list(self.scenario.buildings) + list(self.scenario.greens)
                if self.scenario
                else None
            ),
            bunkers=list(self.scenario.bunkers) if self.scenario else None,
            garden_beds=list(self.scenario.garden_beds) if self.scenario else None,
            ponds=list(self.scenario.ponds) if self.scenario else None,
        )
        self._structure = layer_from_terrain_labels(
            self._terrain.labels,
            self.cfg.world.width_m,
            self.cfg.world.height_m,
            self.cfg.world.resolution_m,
        )
        self._yard = spawn_yard(
            self.np_random,
            self.cfg.world.width_m,
            self.cfg.world.height_m,
            counts,
            robot_keep,
            extra_keepout=self._terrain.feature_keepouts(),
            layout=self.cfg.world.layout,
            orchard_rows=self.cfg.world.orchard_rows,
            orchard_cols=self.cfg.world.orchard_cols,
            explicit=list(self.scenario.obstacles) if self.scenario else None,
            mover_mode=self.cfg.world.movers.default_mode,
            density=self.cfg.world.movers.density,
        )
        self._coverage.reset()
        for obst in self._yard.static():
            if obst.kind in {"tree", "furniture"}:
                self._coverage.exclude_circle(obst.x, obst.y, obst.radius)
        self._coverage.exclude_mask(~no_mow_from_labels(self._terrain.labels))
        loaded = False
        mission_pose = None
        if mission_load and Path(mission_load).is_file():
            state = load_mission(mission_load)
            mission_pose = apply_mission(self._coverage, state)
            loaded = True
            self._mission_loaded = True
        elif load_path and Path(load_path).is_file():
            self._coverage.load_state(load_path)
            for obst in self._yard.static():
                if obst.kind in {"tree", "furniture"}:
                    self._coverage.exclude_circle(obst.x, obst.y, obst.radius)
            self._coverage.exclude_mask(~no_mow_from_labels(self._terrain.labels))
            loaded = True
        elif previous_cut is not None:
            self._coverage.cut = previous_cut & self._coverage.grass
            self._coverage.regenerate(self.np_random, grow_cfg.regenerate_frac)
        if self.cfg.domain_randomization.enabled:
            look_rng = appearance_rng(self.np_random, self.cfg.domain_randomization)
        else:
            look_rng = np.random.default_rng(0)
        self._appearance = sample_appearance(
            self.cfg,
            look_rng,
            (self.cfg.world.width_m, self.cfg.world.height_m),
            self._yard.obstacles,
        )
        self._had_episode = True
        self._grass_loaded = loaded
        if mission_pose is not None:
            self._pose = mission_pose
        self._pose = sit_on_terrain(
            self._pose,
            self._terrain,
            self.cfg.robot.length_m,
            self.cfg.robot.track_m,
        )
        self._prev_pose = self._pose
        self._prev_v = 0.0
        self._last_v = 0.0
        self._last_omega = 0.0
        self._accel_bias = sample_accel_bias(self.np_random, self.cfg.sensors.imu.accel_bias_std)
        self.budget = budget_from_config(self.cfg)
        self.watchdog = SensorWatchdog.from_config(self.cfg, dt=self.cfg.dt)
        self.watchdog.reset()
        self.hw_estop = HardwareEstop()
        self._imu_stamp_s = 0.0
        self._vision_stamp_s = 0.0
        self._camera_adapter = None
        self.fault_bus = FaultBus.from_config(self.cfg)
        if options.get("inject_fault"):
            self.fault_bus.inject(**_inject_kwargs(options["inject_fault"]))
        for extra in options.get("inject_faults") or []:
            self.fault_bus.inject(**_inject_kwargs(extra))
        self.radio = RadioSim.from_config(self.cfg, seed=int(seed or 0))
        self.radio.reset(seed=int(seed or 0))
        shape = self._coverage.cut.shape
        self._occ_persist = PersistentOccupancy(
            shape[0],
            shape[1],
            decay=self.cfg.perception.occupancy_decay,
        )
        self._loop.reset()
        fence = self.geofence_spec()
        if fence.keep_in:
            self._loop.teach_vertices(fence.keep_in)
        self._fused_elev = np.zeros(shape, dtype=np.float32)
        self._observer_coverage: Optional[np.ndarray] = None
        self._last_semantic = None
        self._signals.assign(self._yard.obstacles, self.np_random)
        reset_obs = getattr(self.terrain_observer, "reset", None)
        if callable(reset_obs):
            reset_obs()
        self._tracklets.reset()
        obs, info = self._observe()
        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 3:
            raise ValueError("action must have shape (3,)")
        self.watchdog.observe(
            self._last_imu,
            self._last_images,
            dt=self.cfg.dt,
            imu_stamp_s=self._imu_stamp_s,
            vision_stamp_s=self._vision_stamp_s,
        )
        action = self.watchdog.filter_action(action)
        self.fault_bus.tick(self._steps)
        self.radio.tick(self.cfg.dt, rng=self.np_random)
        if self.radio.enabled and self.radio.lost:
            self.fault_bus.note_radio_loss(self.radio.on_loss)
        left_n = float(np.clip(action[0], -1.0, 1.0))
        right_n = float(np.clip(action[1], -1.0, 1.0))
        requested = bool(action[2] > 0.5)
        left_n, right_n, requested = self.fault_bus.apply_drive(left_n, right_n, requested)
        if self.radio.enabled and self.radio.lost and self.radio.on_loss == "stop_beacon":
            left_n, right_n, requested = 0.0, 0.0, False
        elif self.radio.enabled and self.radio.lost and self.radio.on_loss == "limp_home":
            scale = float(self.cfg.planner.safe_state.limp_scale)
            left_n *= scale
            right_n *= scale
            requested = False
        # Hardware paddle is the last rail filter — policy / software ESTOP
        # / limp-home cannot soft-override a latched kill.
        left_n, right_n, requested = self.hw_estop.apply(left_n, right_n, requested)
        self.fault_bus.last_applied = (left_n, right_n, 1.0 if requested else 0.0)
        vmax = self.cfg.robot.max_wheel_speed_mps
        self._prev_pose = self._pose
        self._prev_v = self._last_v
        self._pose = integrate_pose(
            self._pose,
            left_n * vmax,
            right_n * vmax,
            self.cfg.robot.wheelbase_m,
            self.cfg.dt,
            vmax,
        )
        self._pose = sit_on_terrain(
            self._pose,
            self._terrain,
            self.cfg.robot.length_m,
            self.cfg.robot.track_m,
        )
        v, omega = unicycle_from_wheels(left_n * vmax, right_n * vmax, self.cfg.robot.wheelbase_m)
        self._last_v = v
        self._last_omega = omega
        step_movers(
            self._yard,
            self.cfg.dt,
            self.np_random,
            heading_jitter=self.cfg.world.movers.heading_jitter,
        )
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
        cord_hit = None
        if self._trimmer_on:
            hx, hy = trimmer_xy(self._pose, self.cfg.robot.trimmer.offset_m)
            radius = float(self.cfg.robot.trimmer.radius_m)
            cover = float(getattr(self.cfg.mission, "cover_radius_m", 0.0) or 0.0)
            if cover > 0.0:
                radius = max(radius, cover)
            newly = self._coverage.mark_circle(hx, hy, radius)
            cord_hit = cutter_risk_hit(
                (hx, hy),
                self._yard.obstacles,
                self.cfg.robot.trimmer.radius_m,
            )

        hit = first_collision(
            self._pose, self._yard.obstacles, self.cfg.robot.collision_radius_m
        )
        oob = not in_yard(
            self._pose,
            self.cfg.world.width_m,
            self.cfg.world.height_m,
            self.cfg.robot.collision_radius_m,
            spec=self.geofence_spec(),
        )
        terrain_ev = terrain_hazards(
            self._pose,
            self._terrain,
            length_m=self.cfg.robot.length_m,
            track_m=self.cfg.robot.track_m,
            tip_roll_rad=self.cfg.robot.tip_roll_rad,
            tip_pitch_rad=self.cfg.robot.tip_pitch_rad,
            wheel_drop_m=self.cfg.robot.wheel_drop_m,
            steep_slope_rad=self.cfg.robot.steep_slope_rad,
        )
        breakdown = compute_reward(
            self.cfg.reward,
            newly_cut=newly,
            grass_cells=self._coverage.grass_cell_count(),
            coverage_fraction=self._coverage.coverage_fraction(),
            collision_kind=hit.kind if hit is not None else None,
            out_of_bounds=oob,
            tipover=terrain_ev.tipover,
            drain_drop=terrain_ev.drain_drop,
            steep=terrain_ev.steep,
        )
        self._steps += 1
        self.budget.step(
            self.cfg.dt,
            np.array([left_n, right_n, 1.0 if self._trimmer_on else 0.0], dtype=np.float32),
            n_cameras=len(self.cameras),
        )
        terminated = bool(
            hit is not None
            or oob
            or terrain_ev.tipover
            or terrain_ev.drain_drop
            or breakdown.done_success
        )
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
                "reward_terrain": breakdown.terrain,
                "success": breakdown.done_success,
                "tipover": terrain_ev.tipover,
                "drain_drop": terrain_ev.drain_drop,
                "steep": terrain_ev.steep,
                "terrain_advice": terrain_ev.advice,
                "terrain_reason": terrain_ev.reason,
                "wheels_in_drain": list(terrain_ev.wheels_in_drain),
                "cutter_risk": cord_hit is not None,
                "cutter_risk_kind": cord_hit.kind if cord_hit is not None else None,
                "nearest_person_m": self._nearest_person_m(),
                "scenario": self.scenario.name if self.scenario else "",
                "weather": _weather_dict(self.scenario),
                "fault": self.fault_bus.as_info(
                    self._pose,
                    gnss_dropped=float(np.asarray(obs.get("gps", [0, 0, 0, 1])).reshape(-1)[-1])
                    < 0.5,
                ),
            }
        )
        if terminated or truncated:
            self._persist_grass()
            self._persist_mission()
        return obs, float(breakdown.total), terminated, truncated, info

    def inject_fault(
        self,
        kind: str,
        *,
        mode: str = "open_circuit",
        cameras: Optional[list[str]] = None,
        at_step: Optional[int] = None,
    ) -> None:
        """Kill a motor / sensor mid-episode, or hit / reset the HW paddle."""
        if is_hw_estop_kind(kind):
            if is_hw_reset_mode(mode):
                self.reset_hw_estop()
            else:
                self.hit_hw_estop(str(kind))
            return
        self.fault_bus.inject(kind, mode=mode, cameras=cameras, at_step=at_step)

    def hit_hw_estop(self, reason: str = "paddle") -> None:
        """Sim paddle hit — traction + trimmer rails drop without Python."""
        self.hw_estop.hit(reason)

    def reset_hw_estop(self) -> None:
        """Documented hardware reset. Software ESTOP clear must not call this."""
        self.hw_estop.reset()

    def close(self) -> None:
        self._persist_grass()
        self._persist_mission()
        super().close()

    def geofence_spec(self) -> GeofenceSpec:
        if self.scenario is None:
            return GeofenceSpec()
        return self.scenario.geofence_spec(self.cfg.planner.geofence_inflate_m)

    def _persist_grass(self) -> None:
        if self._grass_save_path:
            self._coverage.save_state(self._grass_save_path)

    def _persist_mission(self) -> None:
        if not self._mission_save_path:
            return
        save_mission(
            self._mission_save_path,
            self._coverage,
            self._pose,
            scenario=self.scenario.name if self.scenario else "",
            seed=self._episode_seed,
            steps=self._steps,
        )

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
            terrain=self._terrain,
        )

    def _ensure_camera_adapter(self) -> None:
        """Build FakeGst / GstNvmm when ``runtime.cameras.adapter`` is set."""
        kind = adapter_kind(getattr(self.cfg.runtime.cameras, "adapter", ""))
        if kind == "renderer":
            self._camera_adapter = None
            return
        if self._camera_adapter is not None:
            return
        names = [cam.name for cam in self.cameras]
        width = int(self.cfg.sensors.width)
        height = int(self.cfg.sensors.height)
        if kind in {"fake_gst", "fake_csi"}:
            from jims_mower.runtime.gstreamer import FakeGstAdapter

            self._camera_adapter = FakeGstAdapter(
                names, width=width, height=height, dt=self.cfg.dt
            )
            return
        if kind == "gst":
            from jims_mower.runtime.gstreamer import GstNvmmAdapter

            self._camera_adapter = GstNvmmAdapter(
                names, width=width, height=height, dt=self.cfg.dt
            )

    def _camera_images(self) -> dict[str, np.ndarray]:
        """Named RGB at the ICD contract size (downsample in ``runtime.capture``)."""
        width = int(self.cfg.sensors.width)
        height = int(self.cfg.sensors.height)
        kind = adapter_kind(getattr(self.cfg.runtime.cameras, "adapter", ""))
        if kind in {"fake_gst", "fake_csi", "gst"}:
            self._ensure_camera_adapter()
            assert self._camera_adapter is not None
            result = self._camera_adapter.grab()
            images = {
                name: ensure_contract_rgb(frame, width, height)
                for name, frame in result.cameras.items()
            }
            return self.fault_bus.apply_cameras(images)
        images = {
            name: ensure_contract_rgb(frame, width, height)
            for name, frame in self._render_cameras().items()
        }
        return self.fault_bus.apply_cameras(images)

    def slide_board_under_wheel(
        self, corner: str = "FL", thickness_m: float = 0.04
    ) -> dict[str, Any]:
        """Gym fixture: raise ground under one wheel and refresh ToF.

        Not a real VL53 board. Addresses stay documentation. Re-observe
        without sitting so that corner's downward range shrinks.
        """
        slide_board_under_wheel(
            self._terrain,
            self._pose,
            corner,
            length_m=self.cfg.robot.length_m,
            track_m=self.cfg.robot.track_m,
            thickness_m=thickness_m,
        )
        obs, info = self._observe()
        return {"obs": obs, "info": info, "corner": str(corner).upper()}

    def _render_cameras(self) -> dict[str, np.ndarray]:
        yard = (self.cfg.world.width_m, self.cfg.world.height_m)
        images = {}
        weather = self.scenario.weather if self.scenario is not None else None
        for cam in self.cameras:
            frame = render_camera(
                self._pose,
                cam,
                self._coverage,
                self._yard.obstacles,
                self.cfg.sensors.width,
                self.cfg.sensors.height,
                yard,
                terrain=self._terrain,
                appearance=self._appearance,
            )
            if weather is not None and (weather.night or weather.dawn or weather.wet):
                frame = apply_weather_rgb(
                    frame,
                    night=weather.night,
                    dawn=weather.dawn,
                    wet=weather.wet,
                )
            images[cam.name] = frame
        return images

    def _nearest_person_m(self) -> float:
        best = math.inf
        for obst in self._yard.obstacles:
            if obst.kind != "person":
                continue
            d = math.hypot(self._pose.x - obst.x, self._pose.y - obst.y)
            if d < best:
                best = d
        return float(best)

    def oracle_labels(self) -> dict[str, np.ndarray]:
        """True height-field / grass rasters (exporter labels, not the observer)."""
        if self._terrain.slope is None:
            self._terrain.recompute_slope()
        assert self._terrain.slope is not None
        return {
            "elevation": self._terrain.elevation.astype(np.float32).copy(),
            "slope": self._terrain.slope.astype(np.float32).copy(),
            "hazard": self._terrain.hazard_map(self.cfg.robot.steep_slope_rad),
            "grass": self._coverage.as_float(),
        }

    def terrain_layer_images(self) -> dict[str, np.ndarray]:
        """False-color elevation / slope / hazard rasters the policy sees."""
        if self._last_terrain_est is not None:
            elev = self._last_terrain_est.elevation
            slope = self._last_terrain_est.slope
            hazard = self._last_terrain_est.hazard
        else:
            if self._terrain.slope is None:
                self._terrain.recompute_slope()
            assert self._terrain.slope is not None
            elev = self._terrain.elevation
            slope = self._terrain.slope
            hazard = self._terrain.hazard_map(self.cfg.robot.steep_slope_rad)
        return {
            "elevation": render_scalar_map(elev, cmap="elev"),
            "slope": render_scalar_map(slope, vmin=0.0, vmax=0.7, cmap="slope"),
            "hazard": render_scalar_map(hazard, vmin=0.0, vmax=3.0, cmap="hazard"),
        }

    def _observe(self) -> tuple[dict[str, Any], dict[str, Any]]:
        images = self._camera_images()
        self._last_images = images
        imu_sample = simulate_imu(
            self._pose,
            self._prev_pose,
            v=self._last_v,
            v_prev=self._prev_v,
            omega=self._last_omega,
            dt=self.cfg.dt,
            rng=self.np_random,
            accel_noise_std=self.cfg.sensors.imu.accel_noise_std,
            gyro_noise_std=self.cfg.sensors.imu.gyro_noise_std,
            accel_bias=self._accel_bias,
            enabled=self.cfg.sensors.imu.enabled,
        )
        imu = self.fault_bus.apply_imu(imu_to_array(imu_sample))
        now_s = float(self._steps) * float(self.cfg.dt)
        if not self.fault_bus.imu_frozen:
            self._imu_stamp_s = now_s
        if not self.fault_bus.cam_blind:
            adapter = self._camera_adapter
            if adapter is not None and hasattr(adapter, "last_stamp_s"):
                self._vision_stamp_s = float(adapter.last_stamp_s)
            else:
                self._vision_stamp_s = now_s
        gps_sample = simulate_gps(
            self._pose,
            self.np_random,
            horiz_noise_std_m=self.cfg.sensors.gps.horiz_noise_std_m,
            vert_noise_std_m=self.cfg.sensors.gps.vert_noise_std_m,
            dropout_prob=self.cfg.sensors.gps.dropout_prob,
            include_altitude=self.cfg.sensors.gps.include_altitude,
            enabled=self.cfg.sensors.gps.enabled,
        )
        gps = self.fault_bus.apply_gps(gps_sample.as_array())
        tof = simulate_tof(
            wheel_clearances(
                self._pose,
                self._terrain,
                self.cfg.robot.length_m,
                self.cfg.robot.track_m,
            ),
            self.np_random,
            noise_std_m=self.cfg.sensors.tof.noise_std_m,
            max_range_m=self.cfg.sensors.tof.max_range_m,
            enabled=self.cfg.sensors.tof.enabled,
            count=int(self.cfg.sensors.tof.count),
        )
        self._last_imu = imu
        self._last_gps = gps
        self._last_tof = tof

        context = PerceptionContext(
            pose=self._pose,
            cameras=self.cameras,
            obstacles=list(self._yard.obstacles),
            image_size=(self.cfg.sensors.width, self.cfg.sensors.height),
            hand_signals_enabled=self.cfg.curriculum.hand_signals,
            hand_signal_classifier=self.cfg.curriculum.hand_signal_classifier,
            imu=imu,
            gps=gps,
            terrain=self._terrain,
            map_shape=self._coverage.cut.shape,
            resolution_m=self.cfg.world.resolution_m,
            world_size=(self.cfg.world.width_m, self.cfg.world.height_m),
            steep_slope_rad=self.cfg.robot.steep_slope_rad,
            tof=tof,
            length_m=self.cfg.robot.length_m,
            track_m=self.cfg.robot.track_m,
            chassis_hover_m=0.06,
        )
        detections = self.detector.detect(images, context)
        self._last_detections = detections
        tracklets = self._tracklets.update(detections)
        if self.cfg.perception.persistent_occupancy:
            if self._occ_persist is None or self._occ_persist.grid.shape != self._coverage.cut.shape:
                self._occ_persist = PersistentOccupancy(
                    *self._coverage.cut.shape,
                    decay=self.cfg.perception.occupancy_decay,
                )
            occupancy = self._occ_persist.update(
                detections,
                resolution_m=self.cfg.world.resolution_m,
                tof=tof,
                pose=self._pose,
                length_m=self.cfg.robot.length_m,
                track_m=self.cfg.robot.track_m,
            )
        else:
            occupancy = occupancy_from_detections(
                self._coverage.cut.shape,
                detections,
                self.cfg.world.resolution_m,
            )
        vision = self.grass_observer.estimate(images)
        observer_cut = None
        if isinstance(self.grass_observer, ClassAwareGrassObserver):
            self._observer_coverage = self.grass_observer.stamp_bev(
                images,
                self.cameras,
                self._pose,
                shape=self._coverage.cut.shape,
                resolution_m=self.cfg.world.resolution_m,
                world_size=(self.cfg.world.width_m, self.cfg.world.height_m),
            )
            observer_cut = observer_coverage_fraction(self._observer_coverage)
        terrain_est = self.terrain_observer.estimate(images, imu, gps, context)
        self._last_terrain_est = terrain_est
        if self.cfg.perception.height_fusion:
            if self._fused_elev is None or self._fused_elev.shape != self._coverage.cut.shape:
                self._fused_elev = np.zeros(self._coverage.cut.shape, dtype=np.float32)
            prior = (
                terrain_est.elevation_prior
                if terrain_est.elevation_prior is not None
                else terrain_est.elevation
            )
            self._fused_elev = fuse_height_rgb_tof(
                images,
                self.cameras,
                self._pose,
                tof,
                shape=self._coverage.cut.shape,
                resolution_m=self.cfg.world.resolution_m,
                world_size=(self.cfg.world.width_m, self.cfg.world.height_m),
                length_m=self.cfg.robot.length_m,
                track_m=self.cfg.robot.track_m,
                elevation=self._fused_elev,
                prior=prior,
                imu=imu,
            )
        if self.cfg.curriculum.hand_signal_classifier:
            signal_name = _nearest_detection_signal(detections, (self._pose.x, self._pose.y))
        else:
            signal_name = self._signals.nearest_person_signal(
                self._yard.obstacles, (self._pose.x, self._pose.y)
            )
        signal_id = SIGNAL_TO_ID.get(signal_name or "", 0)
        structure = _merge_structure(self._structure.grid, terrain_est.structure)
        self._last_topdown = self._topdown()
        hx, hy, hz = trimmer_xyz(
            self._pose,
            self.cfg.robot.trimmer.offset_m,
            self._terrain,
            self.cfg.robot.trimmer.height_m,
        )
        terrain_ev = terrain_hazards(
            self._pose,
            self._terrain,
            length_m=self.cfg.robot.length_m,
            track_m=self.cfg.robot.track_m,
            tip_roll_rad=self.cfg.robot.tip_roll_rad,
            tip_pitch_rad=self.cfg.robot.tip_pitch_rad,
            wheel_drop_m=self.cfg.robot.wheel_drop_m,
            steep_slope_rad=self.cfg.robot.steep_slope_rad,
        )
        spec = self.geofence_spec()
        fence_advice = geofence_advice(
            self._pose,
            spec,
            slow_m=self.cfg.planner.geofence_slow_m,
            stop_m=self.cfg.planner.geofence_stop_m,
        )
        hub = trimmer_xy(self._pose, self.cfg.robot.trimmer.offset_m)
        dist, living_obst = nearest_living(hub, self._yard.obstacles)
        living = living_advice(
            dist,
            living_obst.kind if living_obst is not None else None,
            slow_m=self.cfg.planner.living_slow_m,
            reroute_m=self.cfg.planner.living_reroute_m,
            stop_m=self.cfg.planner.living_stop_m,
        )
        obs = {
            "cameras": images,
            "coverage": self._coverage.as_float(),
            "occupancy": occupancy.astype(np.float32),
            "detections": encode_detections(detections, self.camera_index),
            "pose": np.array(
                [
                    self._pose.x,
                    self._pose.y,
                    self._pose.theta,
                    self._pose.z,
                    self._pose.pitch,
                    self._pose.roll,
                ],
                dtype=np.float32,
            ),
            "imu": imu,
            "gps": gps,
            "tof": tof,
            "elevation": terrain_est.elevation.astype(np.float32),
            "elevation_prior": (
                terrain_est.elevation_prior.astype(np.float32)
                if terrain_est.elevation_prior is not None
                else terrain_est.elevation.astype(np.float32)
            ),
            "slope": terrain_est.slope.astype(np.float32),
            "hazard": terrain_est.hazard.astype(np.float32),
            "structure": structure,
            "confidence": _terrain_confidence(terrain_est),
            "trimmer_enabled": np.array(
                [1.0 if self._trimmer_on else 0.0], dtype=np.float32
            ),
            "hand_signal": int(signal_id),
        }
        info = {
            "pose": {
                "x": self._pose.x,
                "y": self._pose.y,
                "theta": self._pose.theta,
                "z": self._pose.z,
                "pitch": self._pose.pitch,
                "roll": self._pose.roll,
            },
            "coverage_fraction": self._coverage.coverage_fraction(),
            "gym_coverage_fraction": self._coverage.coverage_fraction(),
            "coverage_source": str(self.cfg.perception.coverage_source or "gym_grid"),
            "coverage_cut_cells": self._coverage.cut_cell_count(),
            "coverage_grass_cells": self._coverage.grass_cell_count(),
            "coverage_cut": (self._coverage.cut & self._coverage.grass),
            "detections": [d.as_dict() for d in detections],
            "tracklets": [t.as_dict() for t in tracklets],
            "grass_vision": vision,
            "camera_poses": [
                _cam_pose_dict(self._pose, cam) for cam in self.cameras
            ],
            "hand_signals_enabled": self.cfg.curriculum.hand_signals,
            "steps": self._steps,
            "imu": imu.tolist(),
            "gps": gps.tolist(),
            "tof": tof.tolist(),
            "trimmer_xyz": [hx, hy, hz],
            "n_drains": len(self._terrain.drains),
            "n_banks": len(self._terrain.banks),
            "n_puddles": len(self._terrain.puddles),
            "layout": self.cfg.world.layout,
            "weather_pack": self.cfg.weather.pack,
            "grass_loaded": self._grass_loaded,
            "mission_loaded": self._mission_loaded,
            "terrain_source": terrain_est.source,
            "terrain_advice": terrain_ev.advice,
            "terrain_reason": terrain_ev.reason,
            "tipover": terrain_ev.tipover,
            "drain_drop": terrain_ev.drain_drop,
            "steep": terrain_ev.steep,
            "nearest_person_m": self._nearest_person_m(),
            "scenario": self.scenario.name if self.scenario else "",
            "weather": _weather_dict(self.scenario),
            "geofence": [list(p) for p in spec.keep_in],
            "geofence_spec": spec.as_info(),
            "geofence_advice": fence_advice,
            "structure": structure.copy(),
            "structure_names": STRUCTURE_NAMES,
            "living_advice": living.advice,
            "living_reason": living.reason,
            **self.budget.as_info(),
            **self.watchdog.as_info(),
            **self.hw_estop.as_info(),
            **self.radio.as_info(),
            "imu_stamp_s": float(self._imu_stamp_s),
            "vision_stamp_s": float(self._vision_stamp_s),
        }
        gym_frac = float(self._coverage.coverage_fraction())
        src = str(self.cfg.perception.coverage_source or "gym_grid").strip().lower()
        info["gym_coverage_fraction"] = gym_frac
        info["observer_coverage_fraction"] = observer_cut
        if src == "observer" and observer_cut is not None:
            info["coverage_fraction"] = float(observer_cut)
            info["coverage_source"] = "observer"
        else:
            info["coverage_fraction"] = gym_frac
            info["coverage_source"] = "gym_grid"
            if src == "observer" and observer_cut is None:
                info["coverage_source_note"] = (
                    "observer requested but no class-aware BEV; using gym_grid"
                )
        info["fault"] = self.fault_bus.as_info(
            self._pose, gnss_dropped=float(gps[3]) < 0.5
        )
        if self.cfg.perception.semantic:
            sem = semantic_raster(
                self._coverage.as_float(),
                terrain_est.hazard,
                occupancy,
                structure,
            )
            self._last_semantic = sem
            info["semantic"] = sem
            info["semantic_names"] = SEMANTIC_NAMES
        if self.cfg.perception.height_fusion and self._fused_elev is not None:
            info["height_fused"] = self._fused_elev
            info["height_fusion_stub"] = False
            info["height_fusion"] = {
                "source": "gym_stereo_tof_imu",
                "not_matcher": True,
                "fps_claim": None,
                "map_claim": None,
            }
        if self.cfg.perception.loop_closure:
            self._loop.update(occupancy, self._pose, self.cfg.world.resolution_m)
            info.update(self._loop.as_info())
        return obs, info


def _inject_kwargs(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("inject_fault must be a mapping")
    kind = item.get("kind") or item.get("component")
    if not kind:
        raise ValueError("inject_fault needs kind or component")
    kwargs: dict[str, Any] = {"kind": kind}
    if "mode" in item:
        kwargs["mode"] = item["mode"]
    if "cameras" in item:
        kwargs["cameras"] = item["cameras"]
    if "at_step" in item:
        kwargs["at_step"] = item["at_step"]
    return kwargs


def _nearest_detection_signal(
    detections: list[Detection], xy: tuple[float, float]
) -> Optional[str]:
    best = None
    best_d = float("inf")
    for det in detections:
        if det.label != "person" or not det.hand_signal or det.world_xy is None:
            continue
        d = (det.world_xy[0] - xy[0]) ** 2 + (det.world_xy[1] - xy[1]) ** 2
        if d < best_d:
            best_d = d
            best = det.hand_signal
    return best


def _weather_dict(scenario: Optional[Scenario]) -> dict[str, Any]:
    if scenario is None:
        return {"night": False, "dawn": False, "wet": False, "lighting": "day"}
    w = scenario.weather
    return {"night": w.night, "dawn": w.dawn, "wet": w.wet, "lighting": w.lighting}


def _merge_structure(authored: np.ndarray, observed: Optional[np.ndarray]) -> np.ndarray:
    base = np.asarray(authored, dtype=np.float32)
    if observed is None:
        return base
    extra = np.asarray(observed, dtype=np.float32)
    if extra.shape != base.shape:
        return base
    return np.maximum(base, extra)


def _terrain_confidence(est: Any) -> np.ndarray:
    raw = getattr(est, "confidence", None)
    if raw is None:
        fill = 1.0 if getattr(est, "source", "") == "oracle" else 0.0
        return np.full(np.asarray(est.hazard).shape, fill, dtype=np.float32)
    return np.clip(np.asarray(raw, dtype=np.float32), 0.0, 1.0)


def _cam_pose_dict(pose: Pose, cam: CameraSpec) -> dict[str, Any]:
    wp = camera_world_pose(pose, cam)
    return {
        "name": cam.name,
        "x": wp.x,
        "y": wp.y,
        "z": wp.z,
        "yaw": wp.yaw,
        "pitch": wp.pitch,
        "roll": wp.roll,
    }

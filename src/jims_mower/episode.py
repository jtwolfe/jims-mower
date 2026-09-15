"""Episode record / replay: raw obs + actions + maps on disk."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import numpy as np

from jims_mower.config import EnvConfig, load_config
from jims_mower.contract import CONTRACT_VERSION, WheelCommand
from jims_mower.env import MowerEnv
from jims_mower.latency import LatencyDelays, LatencyHarness
from jims_mower.planning import TerrainPolicy

MANIFEST_NAME = "manifest.json"
STEPS_NAME = "steps.jsonl"
FRAMES_DIR = "frames"
SCORECARD_NAME = "scorecard.json"
RESET_FRAME = "reset.npz"

_MAP_KEYS = (
    "coverage",
    "occupancy",
    "elevation",
    "slope",
    "hazard",
    "confidence",
    "detections",
    "pose",
    "imu",
    "gps",
    "tof",
    "trimmer_enabled",
    "hand_signal",
)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def config_snapshot(cfg: EnvConfig) -> dict[str, Any]:
    return _jsonable(asdict(cfg))


def _dump_obs_npz(path: Path, obs: dict[str, Any], action: Optional[np.ndarray] = None) -> None:
    payload: dict[str, np.ndarray] = {}
    cameras = obs.get("cameras") or {}
    for name, frame in cameras.items():
        payload[f"cam_{name}"] = np.asarray(frame)
    for key in _MAP_KEYS:
        if key in obs:
            payload[key] = np.asarray(obs[key])
    if action is not None:
        payload["action"] = np.asarray(action, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _load_obs_npz(path: Path) -> tuple[dict[str, Any], Optional[np.ndarray]]:
    data = np.load(path, allow_pickle=False)
    obs: dict[str, Any] = {"cameras": {}}
    action = None
    for key in data.files:
        arr = data[key]
        if key == "action":
            action = np.asarray(arr, dtype=np.float32)
        elif key.startswith("cam_"):
            obs["cameras"][key[4:]] = np.asarray(arr)
        elif key == "hand_signal":
            obs[key] = int(np.asarray(arr).reshape(-1)[0])
        else:
            obs[key] = np.asarray(arr)
    if "hand_signal" in data.files:
        obs["hand_signal"] = int(np.asarray(data["hand_signal"]).reshape(-1)[0])
    if "trimmer_enabled" in obs:
        obs["trimmer_enabled"] = np.asarray(obs["trimmer_enabled"], dtype=np.float32)
    return obs, action


class EpisodeRecorder:
    """Write one episode directory: manifest, JSONL metadata, npz frames."""

    def __init__(self, out_dir: Union[str, Path], manifest: dict[str, Any]) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / FRAMES_DIR).mkdir(parents=True, exist_ok=True)
        self._steps = (self.out_dir / STEPS_NAME).open("w", encoding="utf-8")
        self.n_steps = 0
        payload = dict(manifest)
        payload.setdefault("version", CONTRACT_VERSION)
        payload.setdefault("contract_version", CONTRACT_VERSION)
        (self.out_dir / MANIFEST_NAME).write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        self.manifest = payload

    def write_reset(self, obs: dict[str, Any], info: dict[str, Any]) -> None:
        frame = self.out_dir / FRAMES_DIR / RESET_FRAME
        _dump_obs_npz(frame, obs)
        meta = {
            "kind": "reset",
            "frame": f"{FRAMES_DIR}/{RESET_FRAME}",
            "info": _jsonable(_info_slim(info)),
        }
        self._steps.write(json.dumps(meta) + "\n")

    def write_step(
        self,
        step: int,
        obs: dict[str, Any],
        action: np.ndarray,
        info: dict[str, Any],
        *,
        reward: float = 0.0,
        terminated: bool = False,
        truncated: bool = False,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        name = f"{step:06d}.npz"
        frame = self.out_dir / FRAMES_DIR / name
        _dump_obs_npz(frame, obs, action)
        cmd = WheelCommand(
            left=float(action[0]),
            right=float(action[1]),
            trimmer=float(action[2]) if np.asarray(action).size > 2 else 0.0,
        )
        meta = {
            "kind": "step",
            "step": int(step),
            "frame": f"{FRAMES_DIR}/{name}",
            "action": [float(action[0]), float(action[1]), float(action[2])],
            "command": cmd.to_dict(),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": _jsonable(_info_slim(info)),
        }
        if extra:
            meta["extra"] = _jsonable(extra)
        self._steps.write(json.dumps(meta) + "\n")
        self.n_steps += 1

    def write_scorecard(self, scorecard: dict[str, Any]) -> Path:
        path = self.out_dir / SCORECARD_NAME
        path.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
        return path

    def close(self) -> None:
        self.manifest["n_steps"] = self.n_steps
        (self.out_dir / MANIFEST_NAME).write_text(
            json.dumps(self.manifest, indent=2),
            encoding="utf-8",
        )
        self._steps.close()

    def __enter__(self) -> "EpisodeRecorder":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _info_slim(info: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "pose",
        "coverage_fraction",
        "detections",
        "terrain_advice",
        "terrain_source",
        "terrain_reason",
        "living_advice",
        "geofence_advice",
        "nearest_person_m",
        "imu",
        "gps",
        "tof",
        "tipover",
        "drain_drop",
        "steep",
        "n_drains",
        "n_banks",
        "steps",
        "trimmer_enabled",
        "collision",
        "battery_soc",
        "thermal_c",
        "budget_advice",
        "budget_reason",
        "fault",
        "radio_lost",
        "radio_channel",
        "radio_on_loss",
    )
    return {k: info[k] for k in keep if k in info}


class EpisodeReader:
    def __init__(self, episode_dir: Union[str, Path]) -> None:
        self.episode_dir = Path(episode_dir)
        man_path = self.episode_dir / MANIFEST_NAME
        if not man_path.is_file():
            raise FileNotFoundError(f"episode manifest missing: {man_path}")
        self.manifest = json.loads(man_path.read_text(encoding="utf-8"))
        self.reset_obs: Optional[dict[str, Any]] = None
        self.reset_info: dict[str, Any] = {}
        self.steps: list[dict[str, Any]] = []
        with (self.episode_dir / STEPS_NAME).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                frame = self.episode_dir / rec["frame"]
                obs, action = _load_obs_npz(frame)
                if rec.get("kind") == "reset":
                    self.reset_obs = obs
                    self.reset_info = rec.get("info") or {}
                    continue
                if action is None and "action" in rec:
                    action = np.asarray(rec["action"], dtype=np.float32)
                rec["obs"] = obs
                rec["action_arr"] = action
                self.steps.append(rec)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.steps)

    def config(self) -> EnvConfig:
        raw = self.manifest.get("config")
        if raw:
            return load_config(raw)
        path = self.manifest.get("config_path")
        if path:
            return load_config(path)
        return load_config()


def record_episode(
    out_dir: Union[str, Path],
    *,
    steps: int = 40,
    seed: int = 7,
    config: Optional[Union[str, Path, dict, EnvConfig]] = None,
    policy: str = "terrain",
    cameras: Optional[int] = None,
    delays: Optional[LatencyDelays] = None,
    terrain_observer: Optional[str] = None,
) -> dict[str, Any]:
    """Run the env, log obs/actions/maps, write a latency scorecard."""
    from jims_mower.demo import POLICIES, _random_action, _scripted_action

    name = (policy or "terrain").strip().lower()
    if name not in POLICIES or name == "bc":
        raise ValueError(f"record policy must be terrain|scripted|random; got {policy!r}")
    cfg = load_config(config)
    if cameras is not None:
        cfg.sensors.camera_count = cameras
        cfg.sensors.cameras = []
    if terrain_observer:
        key = terrain_observer.strip().lower()
        if key not in {"oracle", "heuristic", "blind"}:
            raise ValueError(f"terrain_observer must be oracle|heuristic|blind; got {key!r}")
        cfg.perception.terrain_mode = key
    env = MowerEnv(config=cfg, render_mode=None)
    obs, info = env.reset(seed=seed)
    harness = LatencyHarness(delays or LatencyDelays())
    terrain_policy: Optional[TerrainPolicy] = None
    rng = np.random.default_rng(seed)
    if name == "terrain":
        terrain_policy = TerrainPolicy(env.cfg)
        terrain_policy.reset(obs, info)

    manifest = {
        "version": CONTRACT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "seed": int(seed),
        "dt": float(env.cfg.dt),
        "policy": name,
        "terrain_mode": env.cfg.perception.terrain_mode,
        "requested_steps": int(steps),
        "config": config_snapshot(env.cfg),
    }
    recorder = EpisodeRecorder(out_dir, manifest)
    recorder.write_reset(obs, info)
    poses: list[Any] = [info.get("pose") or {}]
    dump_at = {0, max(0, int(steps) // 2), max(0, int(steps) - 1)}
    try:
        for t in range(int(steps)):
            harness.start_cycle()
            harness.after_camera()
            if name == "scripted":
                action = _scripted_action()
            elif name == "random":
                action = _random_action(env, rng)
            else:
                assert terrain_policy is not None
                action = terrain_policy.act(obs, info)
            harness.after_plan()
            harness.after_cmd()
            extra = {
                "policy_advice": terrain_policy.last_advice if terrain_policy else None,
                "waypoint_index": terrain_policy.index if terrain_policy else None,
                "safe_mode": getattr(terrain_policy, "last_safe_mode", None)
                if terrain_policy
                else None,
            }
            if terrain_policy is not None and hasattr(terrain_policy.fusion, "pose"):
                fp = terrain_policy.fusion.pose()
                extra["fused_pose"] = {
                    "x": fp.x,
                    "y": fp.y,
                    "theta": fp.theta,
                    "z": fp.z,
                    "pitch": fp.pitch,
                    "roll": fp.roll,
                }
                cov = getattr(terrain_policy.fusion, "variance", None)
                if callable(cov):
                    extra["fused_variance"] = [float(v) for v in cov()]
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            poses.append(next_info.get("pose") or {})
            if t in dump_at and obs.get("cameras"):
                from jims_mower.viewer import dump_step_frames

                dump_step_frames(Path(out_dir) / f"step_{t:03d}", obs)
            recorder.write_step(
                t,
                obs,
                action,
                info,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                extra=extra,
            )
            obs, info = next_obs, next_info
            if terminated or truncated:
                break
    finally:
        scorecard = harness.scorecard(
            {
                "policy": name,
                "steps_run": recorder.n_steps,
                "seed": int(seed),
            }
        )
        recorder.write_scorecard(scorecard)
        recorder.close()
        from jims_mower.viewer import write_viewer_bundle

        write_viewer_bundle(
            out_dir,
            env=env,
            poses=poses,
            policy=name,
            cameras=list((obs.get("cameras") or {}).keys()),
        )
        env.close()
    return {
        "out_dir": str(Path(out_dir)),
        "n_steps": recorder.n_steps,
        "scorecard": scorecard,
        "manifest": recorder.manifest,
    }


def replay_offline(
    episode_dir: Union[str, Path],
    *,
    delays: Optional[LatencyDelays] = None,
    out_dir: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """Drive the planner/controller from logged obs (no physics)."""
    reader = EpisodeReader(episode_dir)
    cfg = reader.config()
    policy = TerrainPolicy(cfg)
    if reader.reset_obs is None:
        raise ValueError("episode has no reset snapshot")
    policy.reset(reader.reset_obs, reader.reset_info)
    harness = LatencyHarness(delays or LatencyDelays())
    actions: list[list[float]] = []
    recorded: list[list[float]] = []
    max_abs = 0.0
    for rec in reader.steps:
        harness.start_cycle()
        harness.after_camera()
        action = policy.act(rec["obs"], rec.get("info") or {})
        harness.after_plan()
        harness.after_cmd()
        actions.append([float(action[0]), float(action[1]), float(action[2])])
        rec_act = np.asarray(rec["action_arr"], dtype=np.float32).reshape(-1)
        recorded.append([float(rec_act[0]), float(rec_act[1]), float(rec_act[2])])
        max_abs = max(max_abs, float(np.max(np.abs(action[:3] - rec_act[:3]))))
    scorecard = harness.scorecard(
        {
            "mode": "offline",
            "source": str(Path(episode_dir)),
            "steps_run": len(actions),
            "max_abs_action_delta": max_abs,
        }
    )
    summary = {
        "mode": "offline",
        "n_steps": len(actions),
        "max_abs_action_delta": max_abs,
        "actions": actions,
        "recorded_actions": recorded,
        "scorecard": scorecard,
    }
    if out_dir is not None:
        dest = Path(out_dir)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "replay.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (dest / SCORECARD_NAME).write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
    return summary


def replay_env(
    episode_dir: Union[str, Path],
    *,
    out_dir: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """Reset the env with the logged seed and play recorded actions."""
    reader = EpisodeReader(episode_dir)
    cfg = reader.config()
    env = MowerEnv(config=cfg, render_mode=None)
    seed = int(reader.manifest.get("seed", 0))
    _obs, info = env.reset(seed=seed)
    rewards: list[float] = []
    try:
        for rec in reader.steps:
            action = np.asarray(rec["action_arr"], dtype=np.float32)
            _obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(reward))
            if terminated or truncated:
                break
    finally:
        env.close()
    summary = {
        "mode": "env",
        "n_steps": len(rewards),
        "final_coverage_fraction": float(info.get("coverage_fraction") or 0.0),
        "terminated_drain_drop": bool(info.get("drain_drop")),
        "terminated_tipover": bool(info.get("tipover")),
        "rewards": rewards,
    }
    if out_dir is not None:
        dest = Path(out_dir)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "replay.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

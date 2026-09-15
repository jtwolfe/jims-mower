"""Pre-mow self-test: unloaded wheel spin, IMU still, camera entropy.

Software checks against the gym / fake drivers. Not a claimed hardware
ATE or RF calibration. ``jims-mower-selftest`` prints a JSON report.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from jims_mower.config import EnvConfig, load_config
from jims_mower.constants import GRAVITY_MPS2, SELFTEST_SCHEMA
from jims_mower.env import MowerEnv


def frame_entropy(image: np.ndarray) -> float:
    """Shannon entropy of an 8-bit histogram (bits). Black frames ≈ 0."""
    arr = np.asarray(image)
    if arr.size == 0:
        return 0.0
    hist = np.bincount(arr.astype(np.uint8).reshape(-1), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    p = hist[hist > 0] / total
    return float(-(p * np.log2(p)).sum())


def imu_still_ok(
    imu: np.ndarray,
    *,
    gyro_limit: float = 0.25,
    accel_tol: float = 1.5,
) -> tuple[bool, str]:
    arr = np.asarray(imu, dtype=np.float32).reshape(-1)
    if arr.size < 6:
        return False, "imu vector shorter than 6"
    accel = arr[:3]
    gyro = arr[3:6]
    spec = float(np.linalg.norm(accel))
    gyro_n = float(np.linalg.norm(gyro))
    if gyro_n > gyro_limit:
        return False, f"gyro moving at rest ({gyro_n:.3f} rad/s)"
    if abs(spec - GRAVITY_MPS2) > accel_tol:
        return False, f"accel norm {spec:.3f} not near g"
    return True, "ok"


def _tiny_selftest_config(source: Optional[Union[str, Path, dict, EnvConfig]]) -> EnvConfig:
    cfg = load_config(source)
    cfg.sensors.camera_count = 4
    cfg.sensors.cameras = []
    cfg.sensors.width = 32
    cfg.sensors.height = 24
    cfg.world.width_m = 8.0
    cfg.world.height_m = 8.0
    cfg.world.resolution_m = 0.20
    cfg.world.n_people = 0
    cfg.world.n_dogs = 0
    cfg.world.n_cats = 0
    cfg.world.n_birds = 0
    cfg.world.n_trees = 0
    cfg.world.n_furniture = 0
    cfg.world.n_toys = 0
    cfg.world.terrain.enabled = False
    cfg.perception.terrain_mode = "oracle"
    cfg.max_steps = 40
    cfg.sensors.gps.dropout_prob = 0.0
    return cfg


def check_unloaded_spin(env: MowerEnv, *, steps: int = 6) -> dict[str, Any]:
    """Command both wheels; pose must move. Then kill a motor — must not drag."""
    obs, info = env.reset(seed=3)
    start = (float(info["pose"]["x"]), float(info["pose"]["y"]), float(info["pose"]["theta"]))
    moved = False
    for _ in range(int(steps)):
        obs, _rew, term, trunc, info = env.step(np.array([0.7, 0.7, 0.0], dtype=np.float32))
        dx = float(info["pose"]["x"]) - start[0]
        dy = float(info["pose"]["y"]) - start[1]
        if math.hypot(dx, dy) > 0.04:
            moved = True
            break
        if term or trunc:
            break
    healthy = {
        "name": "unloaded_wheel_spin",
        "ok": bool(moved),
        "detail": "pose advanced under equal wheel command" if moved else "no motion under command",
        "start": list(start),
        "end": [float(info["pose"]["x"]), float(info["pose"]["y"]), float(info["pose"]["theta"])],
    }
    return healthy


def check_dead_motor_no_drag(env: MowerEnv) -> dict[str, Any]:
    """After a left-motor kill, commanded wheels must not translate the chassis."""
    obs, info = env.reset(seed=4)
    env.fault_bus.inject("motor_left", mode="open_circuit")
    before = (float(info["pose"]["x"]), float(info["pose"]["y"]))
    last_info = info
    for _ in range(4):
        obs, _rew, _term, _trunc, last_info = env.step(
            np.array([0.9, 0.9, 1.0], dtype=np.float32)
        )
    after = (float(last_info["pose"]["x"]), float(last_info["pose"]["y"]))
    drift = math.hypot(after[0] - before[0], after[1] - before[1])
    fault = last_info.get("fault") or {}
    ok = (
        str(fault.get("code")) == "FAULT_IMMOBILISED"
        and bool(fault.get("retrieve"))
        and drift < 1e-4
        and not bool(last_info.get("trimmer_enabled"))
    )
    return {
        "name": "dead_motor_no_drag",
        "ok": ok,
        "detail": "immobilised + SOS, chassis held" if ok else "dead motor still dragged or no SOS",
        "drift_m": drift,
        "fault": fault,
    }


def check_imu_still(env: MowerEnv) -> dict[str, Any]:
    obs, info = env.reset(seed=5)
    imu = np.asarray(obs["imu"], dtype=np.float32)
    ok, detail = imu_still_ok(imu)
    return {
        "name": "imu_still",
        "ok": ok,
        "detail": detail,
        "imu": imu.tolist(),
    }


def check_cam_entropy(env: MowerEnv, *, min_entropy: float = 1.0) -> dict[str, Any]:
    obs, _info = env.reset(seed=6)
    entropies = {name: frame_entropy(frame) for name, frame in (obs.get("cameras") or {}).items()}
    live_ok = bool(entropies) and min(entropies.values()) >= float(min_entropy)
    env.fault_bus.inject("cam_blind")
    obs2, _r, _t, _tr, _i = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    blind = {name: frame_entropy(frame) for name, frame in (obs2.get("cameras") or {}).items()}
    blind_ok = bool(blind) and max(blind.values()) < 0.15
    ok = live_ok and blind_ok
    return {
        "name": "cam_frame_entropy",
        "ok": ok,
        "detail": "live frames have entropy; blind frames are black" if ok else "entropy check failed",
        "live": entropies,
        "blind": blind,
        "min_entropy": float(min_entropy),
    }


def run_selftest(
    config: Optional[Union[str, Path, dict, EnvConfig]] = None,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the three software checks. Always ``not_a_benchmark``."""
    cfg = _tiny_selftest_config(config)
    env = MowerEnv(config=cfg)
    try:
        checks = [
            check_unloaded_spin(env),
            check_dead_motor_no_drag(env),
            check_imu_still(env),
            check_cam_entropy(env),
        ]
    finally:
        env.close()
    ok = all(bool(c.get("ok")) for c in checks)
    return {
        "schema": SELFTEST_SCHEMA,
        "ok": ok,
        "seed": int(seed),
        "checks": checks,
        "not_a_benchmark": True,
        "not_hardware_ate": True,
    }


def write_selftest(path: Union[str, Path], payload: dict[str, Any]) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Jim's Mower software self-test (no RF hardware)")
    p.add_argument("--config", default=None, help="optional YAML / scenario (yard is still shrunk)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_selftest(args.config, seed=args.seed)
    text = json.dumps(report, indent=2)
    if args.out is not None:
        write_selftest(args.out, report)
        print(f"self-test {'PASS' if report['ok'] else 'FAIL'} → {args.out}")
    else:
        print(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

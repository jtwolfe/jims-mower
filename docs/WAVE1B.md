# WAVE 1B

Parallel to WAVE 1A. This wave does **not** edit `ROADMAP.md` / `ICD.md`.

## What landed

1. **EKF pose filter** (`EkfPoseFilter`) — loosely coupled 6-state
   `[x, y, z, yaw, pitch, roll]`. Wheel odometry + gyro predict; GPS XY
   (optional Z) and accel tilt update. Noise is YAML-configurable under
   `planner.ekf`. `ComplementaryPoseFilter` stays as the stub
   (`planner.pose_filter: complementary`). The terrain policy path uses
   the EKF by default.
2. **Uncertainty-aware costmap** — observers emit a per-cell
   `confidence` raster (oracle ≈ 1, heuristic observed cells high /
   unobserved low, blind ≈ 0). `build_costmap` inflates finite costs when
   confidence is low (`planner.uncertainty.inflate` /
   `hazard_boost` / `confidence_floor`). Uncertain free cells stay
   traversable.
3. **Runtime contract** — [`runtime_contract.md`](runtime_contract.md)
   plus `jims_mower.contract` dataclasses and JSON Schema dicts. Every
   message has a `version` field (currently `"1"`).
4. **Record / replay** — `jims-mower-record` writes an episode directory
   (manifest, `steps.jsonl`, `frames/*.npz` with cameras + maps +
   action). `jims-mower-replay` `--mode offline` drives the planner /
   controller from logged obs (no physics). `--mode env` resets the gym
   with the logged seed and applies recorded actions.
5. **Latency harness** — optional injected camera / plan / cmd delays.
   `scorecard.json` reports camera→plan→cmd host timings. Assumptions
   are Orin Nano *class* budgets, not measured onboard FPS. The scorecard
   sets `not_a_benchmark: true` and `fps_claim: null`.

## Config knobs

```yaml
planner:
  pose_filter: ekf          # or complementary
  ekf:
    q_xy: 0.04
    q_z: 0.08
    q_yaw: 0.03
    q_tilt: 0.04
    q_odom: 0.06
    r_gps_xy: 1.2
    r_gps_z: 2.0
    r_tilt: 0.10
    use_gps_z: true
    gps_gate_m: 5.0
    gyro_yaw_mix: 0.30
  uncertainty:
    inflate: 1.5
    hazard_boost: 4.0
    confidence_floor: 0.25
```

## CLI

```bash
jims-mower-record --out /tmp/ep --steps 20 --cameras 4
jims-mower-replay /tmp/ep --mode offline --out /tmp/ep-off
jims-mower-replay /tmp/ep --mode env --out /tmp/ep-env
```

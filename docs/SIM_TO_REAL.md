# Sim-to-real protocol

Same [ICD](../ICD.md) observation keys on the laptop gym and on the Orin.
The on-box loop **must not** import `jims_mower.renderer` or call
`env.render()`.

## ICD keys that must match

| Key | On-box source | Notes |
| --- | --- | --- |
| `cameras` | CSI / GStreamer / NVMM → `obs["cameras"][name]` | Downsample to the contract size (sim default 80×60) |
| `imu` | I2C 6-axis | Body specific force + gyro; rest ≈ `(0,0,9.81,0,0,0)` |
| `gps` | UART GNSS | `(x, y, z, valid)`; `valid=0` on dropout |
| `tof` | I2C downward corners | FL, FR, RL, RR; unused stay 0 |
| `occupancy` | Detector + ToF persist | **Not** a god-view occupancy |
| `elevation` / `slope` / `hazard` / `confidence` | `TerrainObserver` + optional stereo stamp | Heuristic / learned / TensorRT placeholder. Metric near-field elev from a true stereo pair (`perception/stereo.py`); default gym look-arounds skip it. Not COLMAP. |
| `detections` | `Detector` | Padded `[label, cam, u, v, w, h, conf, signal]` |
| `pose` | `EkfPoseFilter` / complementary | Planner start / attitude |
| `coverage` | Your cut-map, or zeros | Gym writes the grass grid; on-box you own it |
| `hand_signal` | Classifier stub or none | Discrete 0–4 |
| `trimmer_enabled` | Interlock | After living-thing radius |

`info` extras (`terrain_advice`, `living_advice`, `geofence_advice`,
`budget_advice`, `watchdog_*`, `hw_estop`) are controller-facing, not
space keys. Hardware ESTOP is a rail filter under the command, not an
obs key.

## What stays on the laptop

- Gymnasium `Mower-v0`, geometric renderer, `OracleTerrainObserver`
- `jims-mower-farm`, `jims-mower-study`, dataset export
- Domain-randomised lighting / wet / dawn

## What you swap on the Orin

1. `FakeCsiDriver` / `FakeGstAdapter` → `GstNvmmAdapter` (when GStreamer exists)
2. `MockDetector` → your head (`TrtDetector` is a load-weights hook)
3. `HeuristicTerrainObserver` → your head (`TrtTerrainObserver` same)
4. Keep `SensorWatchdog` in front of wheel commands; enable it on the
   bench (`runtime.watchdog.enabled` / `configs/orin/bench.yaml`)
4b. Keep `HardwareEstop` as the last rail filter (paddle / `hw_reset`)
5. Load [`configs/orin/extrinsics_stereo.yaml`](../configs/orin/extrinsics_stereo.yaml)
   (forward 6–12 cm pair + side/rear mono). The look-around file
   [`extrinsics_6cam.yaml`](../configs/orin/extrinsics_6cam.yaml) is
   the gym default and is **not** a stereo pair.

See [`JETSON.md`](JETSON.md) and [`runtime_contract.md`](runtime_contract.md).
No FPS / mAP numbers belong in that swap.

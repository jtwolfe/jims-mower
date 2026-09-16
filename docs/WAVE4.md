# WAVE 4 — remaining ROADMAP hooks

Builds on latest `main` after WAVE 3A + 3B. Breadth of working hooks, not
one deep unfinished feature. **No claimed mAP / FPS / IoU / SLAM quality.**

## Perception

- `MockDetector` still projects obstacles, then a numpy appearance model
  (`perception.classify.refine_category`) labels **person / animal / toy /
  static** from crop palette stats.
- `HandSignalClassifier` sits behind `curriculum.hand_signals` **and**
  `curriculum.hand_signal_classifier` (crop brightness / aspect / red bias).
  Default curriculum still uses oracle person labels.
- **CV-5 gym:** when `detector_backend: appearance` and
  `curriculum.hand_signals` is on, a red-biased person crop fills ICD
  `hand_signal` (`stop` / optional `go`) even without `world_xy`. KIND_RGB
  person is already red → `stop`. Gym-only. No confusion matrix. Drop on
  real clips if unreliable.
- `FeatureGrassObserver` (`perception.grass_mode: feature`) is a six-stat
  linear mix. Optional `weights_path` `.npz` (`w`, `b`). Default stays
  `ColorGrassObserver`.
- `ClassAwareGrassObserver` (`perception.grass_mode: class`) uses terrain
  seg classes (grass vs drain/lip/bank) when labels exist, else the
  palette heuristic. Gym painted-strip error is **not** field mAP.
  Phone cut % default is `coverage_source: gym_grid`.
- `info["semantic"]` — uint8 grass / non-grass / drain / bank / static
  (`perception.semantic`, default on).

## Mapping

- Persistent BEV occupancy from **detections + short ToF hits**, decaying
  (`perception.persistent_occupancy`). Not the god-view obstacle list.
- Height-map fusion: gym ideal stereo + ToF corners + local IMU
  (`info["height_fused"]`, `info["elev_fuse"]`). Not a field matcher.
- Loop-closure **stub**: coarse occupancy fingerprint, optional taught
  fence pull. `info["loop_closure"]`. `not_slam: true`. No pose-graph.

## Planning

- `weather.wet` → extra cost on steep corridors (`planner.wet_slope_extra`).
- Energy-aware strip order when SOC is near limp (`planner.energy_aware_strips`
  + `OrinBudget`).
- `jims-mower-sequence --yards paddock,suburban`.

## World

- Seasonal overlays: `configs/overlays/long_grass.yaml`, `leaf_clutter.yaml`
  (`season:` on a scenario or `--` apply via `overlays.apply_overlay`).
- [`narrow_gate`](../configs/scenarios/narrow_gate.yaml),
  [`fence_line`](../configs/scenarios/fence_line.yaml),
  [`property_scale`](../configs/scenarios/property_scale.yaml) (48×40 m @ 0.40 m),
  [`acre_yard`](../configs/scenarios/acre_yard.yaml) (70×58 m @ 0.50 m ≈ 1 acre),
  [`flat`](../configs/scenarios/flat.yaml).
- `jims-mower-import-yard` — survey JSON → geofence + drain polylines.

## Orin

- `SensorWatchdog` zeros wheels if IMU / vision **stamps** freeze
  (`runtime.watchdog.enabled`, off by default). Bench overlay:
  [`configs/orin/bench.yaml`](../configs/orin/bench.yaml).
- [`configs/orin/extrinsics_6cam.yaml`](../configs/orin/extrinsics_6cam.yaml)
  — same `CameraSpec` as the gym (look-around, **not** a stereo pair).
- [`configs/orin/extrinsics_stereo.yaml`](../configs/orin/extrinsics_stereo.yaml)
  — EXAMPLE 6–12 cm pair (`calibration.measured: false`). Bench:
  [`CALIBRATION.md`](CALIBRATION.md), `jims-mower-calibrate`.
- `GstNvmmAdapter` raises without Gst; CI / bench use `FakeGstAdapter` /
  `FakeCsiDriver` to fill named `obs["cameras"]` at the ICD size
  (`runtime.capture.downsample_rgb`). Prefer `stereo_left` /
  `stereo_right` + mono. GStreamer is **not** a dependency. No FPS.
- `TrtDetector` / `TrtTerrainObserver` load-weights placeholders
  (`perception.detector_backend: trt`, `perception.terrain_mode: trt`).
  Software train→ONNX exists (`jims-mower-train-terrain --onnx`); no
  field-ready engine shipped; `fps_claim` / `iou_claim`: null.
- `AppearanceDetector` (`detector_backend: appearance|onnx`) ignores
  `context.obstacles`. MockDetector stays the gym default. No fake mAP.

## Ops / learning

- Farm `--flake-budget` / `--quarantine` records skips instead of swallowing
  exceptions.
- Exporter `meta.json` **must** have `schema: jims_mower.dataset.v1`
  (`jims_mower.dataset.validate_dataset_meta`); train refuses other values.
- `jims-mower-study --kind pitch|clearance|trimmer|observer` (plus existing
  cameras). Reports leftover % = `100 - coverage`, not mAP.
- Curriculum: `jims-mower-curriculum` → flat → suburban → wet → night.
- Sim-to-real protocol: [`SIM_TO_REAL.md`](SIM_TO_REAL.md).

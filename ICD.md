# ICD — Jim's Mower interface control

Short contract for the gym, plugins, and (later) the Orin runtime.
Implementation: [`src/jims_mower/env.py`](src/jims_mower/env.py),
[`src/jims_mower/perception/`](src/jims_mower/perception/),
[`src/jims_mower/planning/`](src/jims_mower/planning/).
Gymnasium id: `jims_mower/Mower-v0`.

## Action space

`Box(3,)` float32:

| Index | Range | Meaning |
| --- | --- | --- |
| 0 | [−1, 1] | Left wheel, fraction of `robot.max_wheel_speed_mps` |
| 1 | [−1, 1] | Right wheel |
| 2 | [0, 1] | Trimmer request; **on** if `> 0.5` *and* the interlock allows |

Equal-and-opposite wheels are a zero-radius pivot. The interlock can refuse
the trimmer when a living thing is inside `robot.trimmer.safety_radius_m` of
the hub.

## Observation keys

`Dict` space (see `MowerEnv.observation_space`):

| Key | Shape / type | Meaning |
| --- | --- | --- |
| `cameras` | dict name → `uint8 (H, W, 3)` | Pinhole RGB, body-frame rig. Field preference: `stereo_left` / `stereo_right` (6–12 cm, shared yaw/pitch) + side/rear mono. Default gym `front_left` / `front_right` are look-arounds, not a metric pair. Downsample to `sensors.width` × `sensors.height` (sim default 80×60) in `runtime.capture.downsample_rgb` before this key is filled. |
| `coverage` | `float32 (R, C)` | `1` cut grass, `0` uncut, `−1` non-grass |
| `occupancy` | `float32 (R, C)` | Detection-rasterized, not god-view |
| `detections` | `float32 (24, 8)` | Padded `[label, cam, u, v, w, h, conf, signal]` |
| `pose` | `float32 (6,)` | `(x, y, theta, z, pitch, roll)` metres / rad |
| `imu` | `float32 (6,)` | Body specific force + gyro; rest ≈ `(0,0,9.81,0,0,0)` |
| `gps` | `float32 (4,)` | `(x, y, z, valid)` in **local ENU metres** from `YardProfile.origin` (default peg = gym SW corner, so ENU == world); `valid=0` on dropout |
| `tof` | `float32 (4,)` | Downward ranges FL, FR, RL, RR |
| `elevation` | `float32 (R, C)` | Observer height estimate (m) |
| `slope` | `float32 (R, C)` | Observer slope (rad, 0–π/2) |
| `hazard` | `float32 (R, C)` | Observer hazard classes (below) |
| `confidence` | `float32 (R, C)` | Relative observer merge weight 0–1 (not a published score) |
| `trimmer_enabled` | `float32 (1,)` | `0` or `1` after the interlock |
| `hand_signal` | `Discrete(5)` | `0` none, `1` stop, `2` go, `3` follow, `4` back |

`info` (not in the space) also carries `terrain_advice`, `living_advice`,
`geofence_advice`, `budget_advice` (`ok` / `slow` / `stop` from the
Orin-class battery / thermal stub when `runtime.enabled`), `battery_soc`,
`thermal_c`, `capacity_wh`, `charge_time_h`, `pack_measured` (50 Wh gym
stub unless `runtime.battery.measured`; no acre-runtime claim),
`detections` as dicts, `weather`, `scenario`, `geofence`
(keep-in vertices), `geofence_spec` (`keep_in` / `keep_out`),
`nearest_person_m`, IMU/GPS lists.

Maps `R×C` align with the grass grid: `resolution_m`, origin at world `(0,0)`,
row = y, col = x.

## Hazard labels

Same integers on the observer raster and on oracle PNGs from the exporter:

| Value | Name | Policy meaning |
| --- | --- | --- |
| 0 | free | Cost 1 |
| 1 | steep | Slow corridor if `slope < planner.max_climb_slope_rad`, else blocked |
| 2 | drain lip | Reroute; inflated by `planner.drain_clearance_m` |
| 3 | drain channel | Forbidden; same inflation |

Physics uses the **true** height field. `info["terrain_advice"]` is
`ok` | `slow` | `reroute` | `stop` from that field (plus IMU cross-check in
the controller).

## Detector

```python
class Detector(Protocol):
    def detect(self, images: dict[str, np.ndarray], context: PerceptionContext) -> list[Detection]: ...
```

- `images[name]` is `uint8 (H, W, 3)`.
- A real head **must ignore** `context.obstacles` (sim-only).
- `Detection`: `label`, `camera`, `bbox=(u,v,w,h)`, `confidence`, optional
  `world_xy`, `hand_signal`, `category`, `depth_m`.
- Labels: `person`, `dog`, `cat`, `bird`, `tree`, `furniture`, `toy`
  ([`constants.LABEL_TO_ID`](src/jims_mower/constants.py)).
- Categories: `person` / `animal` / `toy` / `static` (WAVE 4 appearance refine).
- Gym default: `MockDetector`. Stub: `BlindDetector` (always `[]`).
- `AppearanceDetector` (`detector_backend: appearance|onnx`) ignores
  `context.obstacles` and returns `[]` until a box ONNX exists.
- `TrtDetector` is a load-weights placeholder (`fps_claim: null`).

## TerrainObserver

```python
class TerrainObserver(Protocol):
    def estimate(self, images, imu, gps, context) -> TerrainEstimate: ...
    def reset(self) -> None: ...  # optional; env calls it if present
```

- Returns `TerrainEstimate(elevation, slope, hazard, source)` at `context.map_shape`.
  Optional `elevation_prior` (slow plane for the costmap; not a flopping
  owner mesh) and `structure`.
- A real head **must ignore** `context.terrain` (god-view `HeightField`).
- Modes: `heuristic` (default, RGB+ToF+local IMU tilt, no `context.terrain`), `oracle`
  (training), `blind` (zeros), `learned` (exporter-trained numpy stub;
  weights via `perception.weights_path` or `LearnedTerrainObserver(...)`),
  `onnx` (`OnnxTerrainObserver`; onnxruntime if present, else numpy /
  heuristic), `trt` (engine if present, else heuristic / numpy).
  Sim ONNX is `sim_only`. Not field-ready. `iou_claim` is null.
- IMU pitch/roll is local chassis attitude. `elevation` is a neighborhood
  sample; already-mapped cells stay put. See `docs/TERRAIN_MAPS.md`.
- Heuristic isolated drain-lip stamps are gated unless they sit next to a
  channel (or a ToF drop). Not a published detector score.
- Optional `TerrainEstimate.confidence` is a **relative merge weight** from
  the multi-camera BEV fuse — not a published score.
- `imu` is the 6-vector; `gps` is the 4-vector.
- `info["tracklets"]` — person/dog association stub (`id`, `x`, `y`, `hits`).
- `obs["structure"]` / `info["structure"]` — path / building / bunker / garden / green.

## GrassObserver

```python
class GrassObserver(Protocol):
    def estimate(self, images: dict[str, np.ndarray]) -> dict[str, float]: ...
```

Per-camera uncut-grass fraction. Default: `ColorGrassObserver`.
`FeatureGrassObserver` (`perception.grass_mode: feature`) is a numpy stub.
`ClassAwareGrassObserver` (`perception.grass_mode: class`) uses terrain-seg
classes when provided, else the palette heuristic. Owner cut % default is
the gym grass grid (`info["coverage_source"] == "gym_grid"`). Opt-in
`perception.coverage_source: observer` uses the class-aware BEV. Not field mAP.
Optional `info["semantic"]` raster: grass / non-grass / drain / bank / static
plus path / building / bunker / garden when a structure layer is present.

## Planner / controller

`TerrainPolicy` consumes **observation** maps (whatever the observer wrote),
not the height field:

1. `build_costmap(hazard, slope, occupancy, structure, elevation_prior, …)`
2. Boustrophedon strips + A* (`plan_coverage`)
3. Zero-turn tracker; `terrain_advice` + `living_advice` + `geofence_advice`
   + IMU tilt → slow / reroute / stop. Repeated tip / channel advice runs
   reverse → pivot → call-for-help. With `curriculum.hand_signals`, obs
   `hand_signal` (`stop` / `go` / `follow` / `back`) overrides wheels even
   when the detector is the mock / oracle.

`ComplementaryPoseFilter` is a GPS+IMU stub for planner start / attitude, not
a published EKF. The controller wraps wheel commands in
[`SafeStateMachine`](src/jims_mower/safe_state.py): **RUN** / **LIMP** /
**ESTOP** / **SAFE**. `info["budget_advice"]` from `OrinBudget` is combined
with the other advice strings so a hot / low-SOC stub can limp or hold.

On a laptop, `jims_mower.runtime.drivers` publish the same IMU / GNSS /
camera / ToF contract kinds onto in-process queues (fake I2C / UART / CSI).
`FakeGstAdapter` / `FakeCsiDriver` also write named `obs["cameras"]` at the
contract size with fresh stamps (`runtime.cameras.adapter: fake_csi`).
`GstNvmmAdapter` raises without GStreamer. Addresses in `drivers.py` are
documentation.
`jims-mower-bridge` replays a recorded episode through those queues. CI
does not install ROS 2; optional `[ros2]` node stubs are import-guarded.
`sensors.tof.count` is one of `{0, 2, 4}`. `runtime.enabled` defaults
false so gym tests do not limp.

## Software ESTOP / limp / safe (WAVE 3A)

This is a **software latch**, not a claimed hardware SIL rating.

| Mode | Wheels | Trimmer | How you get there |
| --- | --- | --- | --- |
| `run` | as commanded | interlock still applies | default |
| `limp` | scaled by `planner.safe_state.limp_scale` | off | repeated `stop` advice / tip / drain |
| `estop` | zero | off | `info["estop"]` or `obs["estop"]` true |
| `safe` | zero | off | limp exhausted or controller call-for-help |

`estop` stays latched until an operator `SafeStateMachine.clear()`. Hand-signal
`go` clears limp/safe only. The ICD action is still `Box(3,)`; the machine
rewrites the command the controller sends.

**Hardware ESTOP** is a separate latch (`HardwareEstop` on `MowerEnv`).
When the sim paddle is hit, traction + trimmer **rails** go dead
underneath the policy. `info["hw_estop"]` / live `estop_kind` distinguish
it from software ESTOP. `clear()` / owner Start do **not** restore
rails — only `env.reset_hw_estop()` / live `hw_reset`. Not a claimed
SIL rating and not a physical paddle. See [`docs/ESTOP.md`](docs/ESTOP.md).

Contract message: `SafeState` (`jims_mower.contract`) — `mode`, `scale`,
`hold`, `trimmer_allowed`, `help_requested`.

## Owner app API (WAVE UX-C)

Local stdlib HTTP JSON API served by `jims-mower-app` (same process as the
phone shell). Not a cloud account and not a claimed RF / mapping score.

| Method | Path | Contract |
| --- | --- | --- |
| `GET` | `/status` | `jims_mower.app_status.v1` — `pose`, `battery`, `state` (`mission` + SafeState `machine`), `radio`, `faults` |
| `GET` / `PUT` | `/yard` | full [`YardProfile`](src/jims_mower/profile.py) (`jims_mower.yard.v1`) — `schedule` is evaluated by [`ScheduleEngine`](src/jims_mower/schedule.py) |
| `POST` | `/command` | `{cmd}` ∈ `start` / `stop` / `return` / `estop` / `teach` (live also `save_yard` / `load_yard` / `pair`) |
| `GET` | `/map/mesh` | UX-A `mesh_to_payload` + `ux_a_href: /viewer` |
| `GET` | `/map/coverage` | downsampled cut / uncut / non-grass |
| `GET` | `/events` | SSE of `/status` |
| `GET` | `/api/live` | live SSE when `--live` (`jims_mower.live.v1`) |
| `POST` | `/api/live/control` | same cmds as `jims-mower-live` (`start` / `pause` / `teach` / `save_yard` / `load_yard` / …; no second mission loop) |

`--live` `/status` adds `backend`, `robot` (idle / pairing / live /
fault), `owner_copy`, `radio_path`, map/cut %, `session_summary`.
`start` after `estop` is the operator clear of the software latch.
`/status` also carries `schedule` (`jims_mower.schedule.v1`) and
`weather.rain`. The weekly window arms `start` / duration-stops when
`YardProfile.schedule.enabled` is true (SOC / rain / fault gates).
Not a cloud calendar. See [`docs/SCHEDULE.md`](docs/SCHEDULE.md) and
[`docs/UX_C.md`](docs/UX_C.md).

## Learning stubs (WAVE 3A)

- `jims-mower-bc collect` / `train` — (obs→action) from `TerrainPolicy`, tiny
  numpy MLP. `jims-mower-demo --policy bc` loads `bc_weights.npz` if present.
- `jims-mower-rl` — 1-episode CPU REINFORCE / random-search. Optional
  `pip install -e ".[rl]"` for SB3. Hazard action mask zeros forward wheels
  into drain-lip / channel cells.
- No claimed imitation / return / FPS numbers.

## Scenario extras (WAVE 1A)

Not observation keys. Loaded via [`scenarios.load_source`](src/jims_mower/scenarios.py):

- `weather.night` / `weather.dawn` / `weather.wet` — camera tint only
- `geofence` — keep-in polygon (legacy vertex list) or
  `{keep_in, keep_out}`; `in_yard` fails outside keep-in or inside keep-out
- `keepout` — extra no-go polygons
- explicit `drains` / `banks` / `obstacles` (obstacles may carry a
  `trajectory`: `wander` / `patrol` / `loop` / `line`)
- mission resume — `reset(options={"load_mission","save_mission"})` and
  `jims-mower-mission` write map + uncut + pose + optional ObservedMap
  fog + YardProfile (`jims_mower.mission.v1` / session bundle). Cold
  load is files on disk, not the same sim process.
- `sensors.tof.count` — `0` / `2` / `4` downward corners (unused stay 0)
- `runtime.enabled` — Orin-class battery / thermal limp stub (off by default)
- `runtime.watchdog.enabled` — stop wheels if IMU / vision **stamps** freeze (off by default; bench overlay `configs/orin/bench.yaml`)
- `info["hw_estop"]` — hardware paddle latch (rails dead; not software ESTOP)
- `weather.wet` — extra steep-corridor cost (`planner.wet_slope_extra`)
- `season` — `none` / `long_grass` / `leaf_clutter`
- dataset export `meta.json` **requires** `schema: jims_mower.dataset.v1`

## Exporter labels

[`export.py`](src/jims_mower/export.py) writes **oracle** rasters from
`MowerEnv.oracle_labels()` (true height field + grass), even if the env
observer is heuristic. Layout is in the dump's `LAYOUT.md`. `meta.json`
includes `camera_specs` and a `domain_randomization` snapshot so
[`perception.train`](src/jims_mower/perception/train.py) can back-project
pixels. `--domain-rand` turns renderer DR on for training dumps — see
[`docs/WAVE2B.md`](docs/WAVE2B.md).

## Faults + SOS + radio (WAVE UX-B)

Additive. Notes: [`docs/UX_B.md`](docs/UX_B.md). Off by default
(`faults.enabled` / `radio.enabled`). No claimed RF / SIL / mAP / FPS.

`info["fault"]` (always present):

| Field | Meaning |
| --- | --- |
| `code` | `ok` / `STUCK` / `FAULT_IMMOBILISED` / `TRIMMER_JAM` / `CAM_BLIND` / `IMU_FREEZE` / `GNSS_DROPOUT` / `RADIO_LOSS` |
| `component` | `drive_left` / `drive_right` / `trimmer` / `camera` / `imu` / `gnss` / `chassis` / `radio` |
| `pose` | `{x,y,theta,z,pitch,roll}` at the report |
| `retrieve` | `true` only for dead-motor immobilised or radio `stop_beacon` |

A dead left/right drive motor (`cmd_ignored` / `encoder_stuck` /
`open_circuit`) latches **`FAULT_IMMOBILISED`**: both wheels + trimmer
are zeroed (no one-sided drag across a drain). **Stuck** (terrain lip /
`code=STUCK`) still uses reverse → pivot → help.

`FaultBus` also unifies trimmer jam, black camera frames, IMU freeze,
and GNSS `valid=0` (including the existing `gps.dropout_prob` bit).

`jims-mower-telemetry` includes `sos` / `sos_steps`. The incident
scrubber shows an SOS banner when `retrieve` is set.

Radio sim (no hardware): command preference **Wi-Fi → BT → LoRa**.
`radio.on_loss` is `stop_beacon` (SOS hold) or `limp_home` (limp scale).
`jims-mower-selftest` checks unloaded wheel spin, IMU still, and camera
entropy.

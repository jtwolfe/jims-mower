# Product → hardware: honest inventory

This document is the master checklist for taking Jim's Mower from a
**working sim owner loop** to **ready to fab hardware and field-test**.

It is **not** a claim that ROADMAP checkboxes are production computer
vision. Many WAVE 2–4 items are checked because a hook exists
(`MockDetector`, numpy terrain stub, TensorRT placeholder, loop-closure
stub, radio sim, schedule *fields*). Checked ≠ field-ready.

No mAP, IoU, FPS, RF range, or pack-runtime numbers belong here unless
they were **measured on hardware**. Physics honesty in the gym is
unchanged.

Construction math: [`HARDWARE_DESIGN.md`](HARDWARE_DESIGN.md).
ICD keys: [`../ICD.md`](../ICD.md). Sim-to-real swap:
[`SIM_TO_REAL.md`](SIM_TO_REAL.md).

**Software path is complete for fab + field.** Build-order §1–§17 are
gym/bench hooks. §18–§20 in this PR are procedures + config hooks +
docs — not measured Wh, not a fabbed chassis, not a run acre.

Remaining work is **human**:

- Fab the chassis — [`FAB_CHECKLIST.md`](FAB_CHECKLIST.md)
- Measure pack Wh / charge hours / board °C — [`PACK_THERMAL.md`](PACK_THERMAL.md)
- Hang-measure mass / CG and revise HARDWARE_DESIGN math
- Commit measured extrinsics — [`CALIBRATION.md`](CALIBRATION.md)
- Collect real labels — [`DATASET.md`](DATASET.md)
- Run the residential-acre scorecard — [`FIELD_TEST.md`](FIELD_TEST.md)
  (practice first: `jims-mower-field-dryrun`)

Do **not** start another WAVE of gym stubs.

---

## Already solid (sim owner loop)

These pieces work end-to-end in the laptop gym / phone shell. They are
the product loop Jamie can demo today. They are **not** a flashed Orin
image.

| Piece | What is actually true | Where |
| --- | --- | --- |
| Live fog map | Unknown cells stay dark; observed terrain grows as the robot looks | `ObservedMap`, `jims-mower-live`, `#/map` |
| Observed elevation freeze | Explore stamps a local elev estimate (observer + optional gym stereo); MAP READY locks cells so they do not flop; mow plans on the frozen map, not god-view | `docs/MISSION_FLOW.md`, `docs/TERRAIN_MAPS.md`, `perception/stereo.py` |
| teach → explore → mow → home → done on phone | Pair stub → teach keep-in → Start → MAP READY → mow → return → idle; cut % rises | `jims-mower-owner --live`, `acre_yard_demo` |
| Acre world | `acre_yard` / `acre_yard_demo` ~70×58 m @ 0.50 m; physics grade / drains / banks | `configs/scenarios/acre_yard*.yaml` |
| Faults / SOS UX | Immobilised (dead motor, retrieve) vs stuck (reverse / pivot / help); software ESTOP vs hardware paddle latch (sim) | `FaultBus`, `HardwareEstop`, `#/fault` |
| ICD action / obs contract | `Box(3,)` wheels + trimmer; dict obs keys listed in the ICD | [`ICD.md`](../ICD.md), `docs/runtime_contract.md` |
| YardProfile geofence / home | Taught keep-in / keep-out / home persist as `jims_mower.yard.v1` | `profile.py`, `/yard` |
| Weekly schedule **engine** | `YardProfile.schedule` arms / skips / duration-stops a job (this PR) | [`SCHEDULE.md`](SCHEDULE.md) |

If a row above is the only thing you need to *show*, stop. Everything
below is required before a spinning trimmer leaves the bench.

---

## Broken / stubbed / missing for a real robot

Each item: **status** (`stub` | `partial` | `missing`), why it matters,
an acceptance test you can fail honestly, and what it depends on.

Status key:

- **stub** — named type or ROADMAP box exists; behavior is fake, palette,
  or `pass`.
- **partial** — real control flow in sim; sensors / models / hardware
  are not the field article.
- **missing** — no implementation (or only a comment).

### 1. Computer vision

Do **not** treat exporter-oracle labels or `MockDetector` confidence as
mAP. `fps_claim` / `map_claim` are `null` on purpose.

#### Photogrammetry — robot-shaped, not drone SfM

This is a **~0.5 m** multi-cam zero-turn, not a nadir survey drone.
Do **not** plan classic offline COLMAP-style full-yard SfM as the live
control map. Grass is low-texture / repetitive; mower vibration kills
naive VO; monocular SfM has no metric scale or slope without IMU / GPS /
GCPs.

Recommended stack (photogrammetry *ideas*, robot-shaped):

1. **Near-field metric depth (primary for tip / obstacle).** Fixed
   forward **stereo pair** (or wide-baseline temporal stereo from one
   forward cam across a known wheel baseline) → dense disparity → local
   height / occupancy in the **0.8–4 m** band. Depth resolution worsens
   with range (\(\delta Z \approx (Z^2 / fB)\,\delta d\)) — plan cell
   size from that, not a published score. Gym path: ideal disparity
   from known ray range (`perception/stereo.py`). That is **not** a
   matcher.
2. **Multi-view / SfM concepts (secondary, sparse).** Track features
   across the 4–6 cam rig + over time for **pose assist** and sparse 3D
   landmarks — not a dense DEM each frame. Fuse with wheel odom + IMU
   tilt (later RTK). RTK-VIO papers exist specifically because lawn
   mowers drift on repetitive texture.
3. **Semantic photogrammetry.** Terrain **seg head** (drain / lip /
   bank / grass) on RGB is still required. Geometry alone will not
   label sand / pond / path.
4. **Offline survey mode (optional, later).** When docked / idle, a
   heavier multi-view densify can refine the owner mesh (true
   photogrammetry). It must **never** block the live mow loop.
5. **Learned monocular depth** (Orin-deployable nets) only as a
   **prior** fused with stereo / ToF — never the sole metric source.

Default gym `front_left` / `front_right` (40° yaw, ~40 cm) are
look-arounds, **not** a stereo pair. Field preference: calibrated
forward baseline **~6–12 cm** on a 50 cm body + side / rear mono. See
[`configs/orin/extrinsics_stereo.yaml`](../configs/orin/extrinsics_stereo.yaml)
and [`HARDWARE_DESIGN.md`](HARDWARE_DESIGN.md) §7.

| ID | Item | Status | Why it matters | Acceptance test | Depends on |
| --- | --- | --- | --- | --- | --- |
| CV-1 | Terrain segmentation (drain / lip / bank / grass) | **partial** (software path this PR) | Trainable gym path: `jims_mower.dataset.v1` → numpy (or optional torch) MLP → optional ONNX. `OnnxTerrainObserver` loads via onnxruntime when present. Heuristic remains the **live default**. Sim weights are `sim_only` / not field-ready. Palette heuristic will not survive daylight grass. | Held-out **real** frames; report IoU per class only after a locked test set. Fail if you only have sim loss. **No invented IoU.** `iou_claim` stays null. Software tests: `tests/test_onnx.py`. | CV-8 dataset, RT-1 capture, HD-cam extrinsics |
| CV-2 | Grass coverage observer | **partial** (gym this PR) | `ClassAwareGrassObserver` (`grass_mode: class`) uses terrain-seg classes (grass vs drain/lip/bank) when labels exist, else the palette heuristic. `ColorGrassObserver` / `FeatureGrassObserver` stay as fallbacks. Gym painted-strip error is in `tests/test_grass_coverage.py` — **not** field mAP. Phone cut % default is still the *gym grass grid* (`coverage_source: gym_grid`). Opt-in `coverage_source: observer` uses the class-aware BEV and sets `info["coverage_source"]`. Field strip test still needed. | Gym: coverage drift vs a painted cut/uncut/path fixture (report error in tests). Field: on-box vs painted/measured strips on one lawn, same day. No invented IoU. | CV-1, MAP-1 |
| CV-3 | Person / animal / obstacle detect | **partial** (gym RGB this PR) | `MockDetector` still **projects** `context.obstacles` (gym default). `AppearanceDetector` (`detector_backend: appearance`) ignores that list and finds KIND_RGB **blobs in the camera image**. Gym living interlock can consume those dets (`interlock_source: detections` / `auto`). Not a field head. `map_claim` stays null. | Gym: painted person blob in the front camera trips trimmer-off; oracle person behind / out of view does not. Field: recorded walk-through. No fake mAP. | CV-8, RT-1 |
| CV-4 | Tracking / tracklets | **partial** (IoU/centroid this PR) | Associate consecutive dets by world XY **or** bbox IoU / image centroid when `world_xy` is missing (appearance). Not MOT. No mAP. | Gym: overlapping boxes keep one id. Field: ID-switch on a 30 s clip. | CV-3 |
| CV-5 | Hand signals | **partial** (gym this PR) | Optional red-bias stop/go on an **appearance** person crop fills ICD `hand_signal` and the policy hold when `curriculum.hand_signals` is on. KIND_RGB person is red → `stop`. Gym-only. Unreliable on real clips — drop it then. Classifier stub still exists. **No confusion matrix. No fake numbers.** | Gym: painted red-bias person blob → `obs["hand_signal"]=stop` → wheels hold (`tests/test_field_dryrun.py`). Field: confusion matrix on real stop/go/back clips, or drop. | CV-3 |
| CV-6 | Train → ONNX → TensorRT | **partial** (software path this PR) | `jims-mower-train-terrain --onnx` writes a sim_only Gemm graph when the `onnx` extra is installed. `jims-mower-export-trt --dry-run` still prints `trtexec`. `TrtTerrainObserver` / `TrtDetector` load an engine **if you provide one**, else heuristic / mock. No production ONNX in git. | `trtexec` builds your engine on the Orin; `fps_claim` stays null until you measure. Software pipeline ≠ field head. | CV-1 or CV-3 weights |
| CV-7 | Domain gap (wet / dawn / night) | **partial** | Renderer tints only. Not HDR, IR, or wet-lens. | Same route at noon vs dusk vs wet; hazard stamps must not invert drain vs grass. | CV-1, RT-1 |
| CV-8 | Dataset (real) | **partial** (harness this PR) | Exporter writes `jims_mower.dataset.v1` (oracle PNG + COCO-like index) from renderer **or** FakeCsi/Gst (`--adapter`). Train/val is last-frac (`meta.split` / `split.json`). Label protocol: [`DATASET.md`](DATASET.md). Fake CSI is OK in CI — not real photos. No field bag yet. No published mAP. | N frames from the rig, labeled, versioned, train/val documented. CI: FakeCsi → v1 + split. | RT-1, HD-cam |

### 2. Mapping

| ID | Item | Status | Why it matters | Acceptance test | Depends on |
| --- | --- | --- | --- | --- | --- |
| MAP-1 | ObservedMap | **partial** | Works in sim: unknown ≠ safe; camera hits + body/ToF disk. Stamps **observer** rasters, so garbage in → garbage map. | After a real explore, fog holes match what the cameras saw; no authored shed leaked outside the mask. | CV-1, RT-1, RT-2 |
| MAP-2 | Near-field metric stereo + frozen elev fuse | **partial** (gym fuse this PR) | `fuse_elev_stereo_tof_imu` / `fuse_height_rgb_tof` unify gym **ideal** stereo (when a 6–12 cm pair exists) + ToF corners + local IMU grade onto `ObservedMap`. MAP READY `lock_observed()` so later stamps do not flop frozen cells. Default gym look-around rig is a **no-op** (not a pair). **Not** a field stereo matcher. Not COLMAP. `fps_claim` / `map_claim` stay null. | Gym: kerb/lip step vs known geometry (`tests/test_elev_fuse.py`); lock holds after a flopping observer raster. Field: kerb cross-section vs tape + IMU; no invented mAP / FPS. | CV-1, HD-cam stereo, RT-2 |
| MAP-2b | Learned mono depth prior | **stub** (interface this PR) | `MonoDepthPrior` / `apply_mono_prior` fill *gaps* only. Valid stereo cells stay. No Orin depth net shipped. | Gym: prior cannot win when stereo is valid (`tests/test_elev_fuse.py`). Field: ablation stereo-only vs stereo+prior on one kerb. | MAP-2, RT-3 |
| MAP-3 | Loop closure / revisit | **stub** | `LoopClosureStub` occupancy fingerprint + optional taught-vertex pull. `not_slam: true`. No pose-graph optimizer. | Gym: fence vertices stay inside a stated metre error on a loop (`tests/test_pose_assist.py`, 0.75 m). Field: return to dock after 1 acre explore. | RT-2 GNSS/IMU, MAP-1 |
| MAP-3b | Sparse multi-view / pose assist | **stub** (gym this PR) | Landmark revisit on taught fence vertices (2-D translation, capped). Not feature tracks across the 4–6 cam rig. Not a dense DEM. | Gym: drifted revisit pulled toward teach vertices. Field: RTK-VIO later. | MAP-3, RT-2 |
| MAP-3c | Offline docked densify | **missing** | Heavier multi-view pass while idle. True photogrammetry for the owner mesh only. Must not block the live mow loop. | Docked job writes a refined mesh; mow loop FPS / cycle time unchanged (measure later; do not invent). | MAP-2, RT-7 |
| MAP-4 | Multi-session persistence | **partial** (software this PR) | `save_mission` writes ObservedMap fog + uncut + pose + YardProfile to disk. `MissionPolicy.restore_session` + `load_mission` work in a **new process**. Live pause/close writes `session.npz`. Not a flashed Orin day-2 image. | Gym: save → tear down → load; uncut still planned; fog preserved (`tests/test_mission.py`, `tests/test_pln_regression.py`). Field: power-cycle the Orin on a surveyed peg. | MAP-3, MAP-5 |
| MAP-5 | Geofence on Earth | **partial** (surveyed-origin model this PR) | `YardProfile.origin` is a local-ENU peg (`e_m/n_m/u_m`, optional lat/lon). Keep-in is metres relative to it. ICD `gps` is ENU + `valid`. **No WGS84 field survey in this repo.** | Gym: GNSS/pose near the fence edge stops short of the tape (`tests/test_survey_origin.py`). Field: RTK the peg, tape the fence, stop before the tape — **not claimed**. | RT-2 GNSS, UX teach |

### 3. Planning / control

| ID | Item | Status | Why it matters | Acceptance test | Depends on |
| --- | --- | --- | --- | --- | --- |
| PLN-1 | Explore | **partial** (gym regression this PR) | Frontier walk on ObservedMap. `tests/test_pln_regression.py` runs explore → MAP READY on `mission_tiny` with a taught profile. Needs real maps in the field. | MAP READY on a real lawn without counting fog islands as unreachable (already true in sim). | MAP-1 |
| PLN-2 | Coverage | **partial** (gym regression this PR) | Boustrophedon + A* on observer costmap; energy strip order uses stub SOC. Regression freezes the observed map then mows. | Leftover uncut vs planned cells on a marked 10×10 m patch. | MAP-1, CV-2 |
| PLN-3 | Tip recovery | **partial** (gym regression this PR) | Reverse → pivot → help; IMU tip-skip continues paint in sim. Thresholds (`tip_roll_rad=0.40`) are gym constants — **not retuned**. | Tip the chassis to the software trip on a known ramp; wheels stop; recover without drain entry. | RT-2 IMU, HD-mass |
| PLN-4 | Living-thing interlock | **partial** (dets-from-camera this PR) | Mock + `interlock_source: obstacles` (default) still uses the oracle list. `appearance` / `interlock_source: detections` trips on a living blob in a **forward camera** only — a person on the oracle list behind the robot does not fire. No invented metres. | Gym: painted front-camera person → trimmer off; behind / out of view → trimmer stays. Field: person in the radius on the rig. | CV-3, CV-4 |
| PLN-5 | Resume after stop | **partial** (this PR) | Mission save/load + owner Pause/Resume in live + cold `restore_session`. Schedule duration-stop unchanged. Field “battery died mid-strip” is untested. | Pause 10 min, resume; uncut cells still planned. Gym: `tests/test_pln_regression.py`. | MAP-4, SCH-1 |

### 4. Scheduling

| ID | Item | Status | Why it matters | Acceptance test | Depends on |
| --- | --- | --- | --- | --- | --- |
| SCH-1 | Arm / stop from `YardProfile.schedule` | **partial** (this PR) | Engine + Health toggle + SOC / rain / fault gates. Not a cloud calendar. Not a rain *service*. | Enable Mon 09:00 UTC on FrozenClock → mission `mowing`; SOC 0.10 → `soc_low`; rain flag → skip; duration expires → idle. See [`SCHEDULE.md`](SCHEDULE.md). | Owner app (done) |
| SCH-2 | Timezone | **partial** | `local` or IANA. Orin must set a real zone (e.g. `Australia/Sydney`). | Job starts at 09:00 in that zone, not UTC-by-accident. | SCH-1 |
| SCH-3 | Rain skip | **partial** | Uses env `weather.wet` or `status.weather.rain`. No BOM/radar. | Set rain flag; window is consumed as skip; next week can still run. | SCH-1 |
| SCH-4 | SOC gate | **partial** | Compares `battery.soc` to `schedule.min_soc` (default 0.25). OrinBudget / `/status` read `runtime.battery.capacity_wh` (50 Wh gym stub unless `measured: true`). Remaining Wh = soc × capacity. **No acre-runtime claim.** | Real fuel gauge below min_soc skips. | RT-5 pack telemetry |
| SCH-5 | Notifications | **partial** (in-app this PR) | `NotificationLog` + `/notifications`. Skip/finish reasons on the LAN list. Webhook is a **stub** (`delivered: false`). No SMS / vendor push. | Owner sees a skip reason on `/notifications` without a second SSE tab. Field SMS still missing. | SCH-1, UX-2 |

### 5. Runtime (Orin)

| ID | Item | Status | Why it matters | Acceptance test | Depends on |
| --- | --- | --- | --- | --- | --- |
| RT-1 | CSI / GStreamer capture | **partial** (software path this PR) | `FakeGstAdapter` / `FakeCsiDriver` fill named `obs["cameras"]` at the ICD contract size with fresh stamps. Downsample is `runtime.capture.downsample_rgb`. `GstNvmmAdapter` still raises without Gst. **Not** physical CSI — JetPack + real cameras still required. No FPS. | Named cameras (prefer `stereo_left` / `stereo_right` + mono) fill `obs["cameras"]` at `sensors.width` × `sensors.height`; `SensorWatchdog` stays happy on fresh stamps; freeze stamps → wheels zero. Field: plug CSI on the Orin (needs JetPack). | HD-cam, JetPack |
| RT-2 | IMU / GNSS / ToF drivers | **partial** (gym stubs this PR) | Fake I2C/UART publishers still copy gym vectors onto in-process queues. Level-rest IMU, GNSS `valid` toggle, and ToF board-under-wheel fixture are writable. Addresses in `drivers.py` remain documentation. | Gym: rest IMU ≈ `(0,0,9.81,0,0,0)`; `valid` bit toggles; ToF corner shortens when a board is slid under a wheel. Field: same ICD keys from real BMI/ICM + GNSS + VL53-class parts. | HD-place |
| RT-3 | TensorRT load | **stub** | See CV-6. Software ONNX export exists; deserialize still needs an Orin engine you build. Fallback stays heuristic / mock if the path is missing (keep that). | Engine deserializes; fallback still mock if path missing (keep that). | CV-6 |
| RT-4 | Watchdog | **partial** (bench config + gym stamp stall this PR) | Zeros wheels if IMU/vision **stamps** freeze (`runtime.watchdog.enabled`). Off in default gym tests. Bench overlay: [`configs/orin/bench.yaml`](../configs/orin/bench.yaml). Fake adapters — not real CSI/IMU. | Gym: freeze IMU or camera stamps → wheels zero within `vision_stall_s` / configured stall (`tests/test_hardware_estop.py`, `tests/test_wave4_ops.py`). Field: unplug a camera on the wired rig (needs RT-1). | RT-1, RT-2 |
| RT-5 | Battery / thermal telemetry | **partial** (procedure this PR) | `OrinBudget` still an RC. Default `capacity_wh: 50`, `measured: false`. Template: [`configs/orin/pack_measured.template.yaml`](../configs/orin/pack_measured.template.yaml). Procedure: [`PACK_THERMAL.md`](PACK_THERMAL.md). `measured: true` refuses the silent 50 Wh default. Numbers **null** until bench. Not a BMS. | SOC and board °C from hardware; limp/stop match measured limits. Software: `tests/test_pack.py`. | HD-batt |
| RT-6 | Hardware ESTOP | **partial** (sim + doc this PR) | Gym `HardwareEstop` drops traction + trimmer **rails** underneath policy / `SafeStateMachine`. Wiring + reset: [`ESTOP.md`](ESTOP.md), [`HARDWARE_DESIGN.md`](HARDWARE_DESIGN.md) §10. No physical paddle. | Gym: dummy load commanding wheels+trimmer; paddle latch zeros outputs; software clear does **not** restore; only `hw_reset` does (`tests/test_hardware_estop.py`). Field: hit a real paddle while a dummy load spins — **not claimed**. | HD-wire |
| RT-7 | On-box loop (no renderer) | **partial** (this PR) | `jims-mower-onbox` loads bench/Orin YAML, Fake* / Gst adapters, watchdog, HW ESTOP, black box. Never imports `jims_mower.renderer` (guard + test). Systemd example: [`deploy/jims-mower.service`](../deploy/jims-mower.service). [`docs/ONBOX.md`](ONBOX.md). | Process runs without importing `jims_mower.renderer`. Loop steps with fake sensors. Field: real CSI still later. | RT-1…RT-4 |

### 6. Safety / ops

| ID | Item | Status | Why it matters | Acceptance test | Depends on |
| --- | --- | --- | --- | --- | --- |
| SAF-1 | Self-test | **partial** | `jims-mower-selftest`: unloaded spin, IMU still, cam entropy — against **gym** streams. Unchanged. | Same checks on the wired robot before first mow. | RT-1, RT-2 |
| SAF-2 | Immobilised vs stuck | **partial** | FaultBus semantics are right in sim. `BlackBox.retrieve_sos()` pulls IMU+cmds after immobilised. Real motor open-circuit must raise `FAULT_IMMOBILISED`. | Kill one drive encoder; both wheels + trimmer hold; SOS retrieve. | RT-2, HD-drive |
| SAF-3 | Incident logs | **partial** (gym/bench black box this PR) | Rotating JSONL (`BlackBox`) when `reset(options={"blackbox": path})`. After a tip, IMU + wheel cmds are on disk. Not an Orin always-on daemon. | After a tip, pull IMU + advice + wheel cmds from onboard storage. Gym: `tests/test_session_ops.py`. | RT-7 |
| SAF-4 | OTA | **stub** | `jims_mower.ota.ota_status()` / `ota_apply()` are a documented no-op. No updater, no signed image, no A/B slot. `/ota` says `available: false`. | Documented no-op or a real A/B slot — do not pretend. | UX-2 Wi-Fi |
| SAF-5 | SIL / cert | **missing** | Software ESTOP is explicitly **not** a SIL rating. | Do not claim one. | legal / field |

### 7. Owner UX

| ID | Item | Status | Why it matters | Acceptance test | Depends on |
| --- | --- | --- | --- | --- | --- |
| UX-1 | Schedule UI | **partial** (this PR) | Health toggle + next run + skip reason. Days/time still JSON. | Toggle On, FrozenClock in window → Start without Map. | SCH-1 |
| UX-2 | Notifications | **partial** (in-app this PR) | `/notifications` list + webhook stub. Not SMS. Not a vendor push. | See SCH-5. | SCH-5 |
| UX-3 | Multi-yard | **partial** (this PR) | `YardStore` + `/yards` + `POST /yards/select`. Switching replaces keep-in/home; no fence bleed. Sequence CLI remains a sim farm tool. | Switch yards on the phone; home/geofence/schedule swap; no fence bleed. Gym: `tests/test_session_ops.py`. | MAP-4 |
| UX-4 | Radio | **stub** | Wi-Fi → BT → LoRa **sim**. No RF. | Field: BT pair required, LoRa command at the far fence — measure or don't claim range. | hardware radios |
| UX-5 | Pairing | **stub** | Phone “Pair Bluetooth” sets a bool. | Real BT pairing before first Start. | UX-4 |

### 8. Sim-to-real

| ID | Item | Status | Why it matters | Acceptance test | Depends on |
| --- | --- | --- | --- | --- | --- |
| S2R-1 | ICD key match | **partial** | Contract is written. Camera software path fills `obs["cameras"]`; IMU/GNSS/ToF/Gst chips are still fakes. | Laptop gym and Orin process the same key set; renderer never imported on-box. | RT-1…RT-3 |
| S2R-2 | Extrinsics YAML | **partial** (this PR) | `extrinsics_stereo.yaml` is the documented **EXAMPLE** (`calibration.measured: false`). Load + baseline-cm helper rejects non-pairs. Prefer it (6–12 cm forward pair + mono) for the field article. `extrinsics_6cam.yaml` stays the gym look-around. **Neither is taped.** | Reproject a checkerboard / drain lip; stereo pair verifies baseline + disparity vs tape. Gym: `jims-mower-calibrate` / `tests/test_calibration.py`. Field: still measure. | HD-cam |
| S2R-3 | Calibration bench | **partial** (procedure + gym this PR) | Written steps: [`CALIBRATION.md`](CALIBRATION.md). Gym lip / checkerboard fixture stamps the right ObservedMap cells; ideal disparity matches tape in 0.8–4 m. MEASURED template: `extrinsics_stereo_measured.template.yaml`. Physical measure-and-commit still required. | Human tapes the baseline, writes `extrinsics_stereo_measured.yaml`, `calibration.measured: true`. Not claimed here. | S2R-2 |
| S2R-4 | Wheel / trimmer scale | **partial** (procedure + gym this PR) | YAML `robot.drive.scale` / `robot.trimmer.scale`, `measured: false` default. Gym: 1.0 command moves at `max_wheel_speed_mps × scale`. Procedure: [`SCALE.md`](SCALE.md). **No invented Kv.** | Gym: scale 0.5 halves distance (`tests/test_scale.py`). Field: tape 1.0 command → m/s; tach RPM. | HD-drive |

ICD keys that **must** match (do not rename): `cameras`, `imu`, `gps`,
`tof`, `occupancy`, `elevation`, `slope`, `hazard`, `confidence`,
`detections`, `pose`, `coverage`, `hand_signal`, `trimmer_enabled`.
See [`SIM_TO_REAL.md`](SIM_TO_REAL.md).

---

## Numbered build order

One element at a time: **build → test → review → revise**. Do not start
the next row until the acceptance test is failed honestly or passed.
Stop when only **fab + field test** remain.

1. **Schedule engine (this PR).** Owner can leave a weekly window armed
   in sim / app. Justification: it was the only owner-loop item that was
   *fields only*; CV first would have been another incomplete head.  
   *Test:* `pytest tests/test_schedule.py tests/test_app_api.py`.  
   *Revise:* timezone on the Orin, rain from a real sensor later (SCH-3/4).

2. **CV terrain = stereo + seg + frozen elev fuse (this PR).** Gym
   slice only. Synthetic stereo (or a real pair later) stamps metric
   local elev into `ObservedMap`; MAP READY locks those cells so they
   do not flop. Terrain seg remains the exporter → numpy stub path
   (real train hook, not a published head). **Not** live COLMAP.  
   *Test:* `pytest tests/test_stereo.py tests/test_learn.py tests/test_cv_terrain.py`.  
   *Honesty:* no claimed mAP / IoU / FPS.  
   *Revise:* real matcher + calibrated baseline on the rig (row 7+).

3. **Hardware ESTOP + power kill (RT-6, HD-wire).** Sim model + design
   doc (this PR). Gym `HardwareEstop` is the last rail filter — policy
   cannot soft-override a latched paddle. Field paddle + dummy-load
   spin is still required.  
   *Test (gym):* `pytest tests/test_hardware_estop.py tests/test_safe_state.py`.  
   *Test (field):* paddle while a dummy load spins. Not claimed here.  
   *Revise:* fuse map vs what actually opened on the bench.

4. **Watchdog enabled on the bench loop (RT-4).** Gym/bench config
   (`runtime.watchdog.enabled`, [`configs/orin/bench.yaml`](../configs/orin/bench.yaml)).
   Freeze IMU or camera stamps → wheels zero within the stall window.
   Fake adapters; no real CSI/IMU.  
   *Test:* `pytest tests/test_hardware_estop.py tests/test_wave4_ops.py -k watchdog`.  
   *Honesty:* stamps in the gym; field unplug still needs RT-1 / RT-2.

5. **Real CSI / GStreamer → `obs["cameras"]` (RT-1).** Software path
   (this PR): same names as `CameraSpec` (prefer `stereo_left` /
   `stereo_right` + mono). Downsample to the contract size in
   `runtime.capture.downsample_rgb`. `GstNvmmAdapter` raises without
   Gst; CI / bench use `FakeGstAdapter` / `FakeCsiDriver`. Physical
   CSI on Orin still needs JetPack + real cameras.  
   *Test:* `pytest tests/test_capture.py tests/test_wave4_ops.py -k gst`.
   Named live frames; watchdog happy on fresh stamps; freeze stamps →
   wheels zero (reuses §4). No FPS claim.

6. **Real IMU + GNSS + ToF (RT-2).** Gym stubs (this PR): Fake
   publishers + a board-under-wheel fixture so the acceptance tests
   are writable. Addresses stay documentation. Hardware chips still
   required for the physical line.  
   *Test:* `pytest tests/test_drivers.py tests/test_sensors.py -k "imu or gnss or tof or board"`.
   Level rest IMU ≈ `(0,0,9.81,0,0,0)`; GNSS `valid` bit toggles; ToF
   corners change when a board is slid under a wheel.

7. **Extrinsics + stereo calibration bench (S2R-2, S2R-3).** Procedure
   + gym checks (this PR). [`CALIBRATION.md`](CALIBRATION.md),
   `jims-mower-calibrate`, gym lip / tape fixture. EXAMPLE YAML stays
   unmeasured. Physical tape-and-commit still required on the rig.  
   *Test:* `pytest tests/test_calibration.py tests/test_stereo.py`.  
   *Honesty:* no FPS / mAP. Human still measures.

8. **Dataset harness on the rig (CV-8).** Record + label protocol
   (this PR, focused). [`DATASET.md`](DATASET.md). FakeCsi/Gst →
   `jims_mower.dataset.v1` with a documented train/val split. Train
   stub may still run. No trained production head.  
   *Test:* `pytest tests/test_export.py tests/test_calibration.py -k dataset`.  
   *Honesty:* Fake CSI in CI; real CSI + human labels after JetPack.
   Do not publish mAP.

9. **Terrain seg train → ONNX → TRT (CV-1, CV-6).** Software path
   (this PR). Exporter / FakeCsi → numpy (or optional torch) train →
   optional ONNX under `artifacts/` or `models/`. Config can select
   `terrain_mode: onnx|trt|learned|heuristic` without breaking default
   gym demos (heuristic stays live). `OnnxTerrainObserver` uses
   onnxruntime when present, else numpy / heuristic. `trtexec` dry-run
   and load-if-present stay. Sim weights are `sim_only` / not
   field-ready.  
   *Test (software):* `pytest tests/test_onnx.py tests/test_learn.py tests/test_bridge.py`.  
   *Test (field):* held-out **real** IoU. If you cannot measure it, do
   **not** ship the head. `iou_claim` / `map_claim` / `fps_claim` stay
   null. Real labels: [`DATASET.md`](DATASET.md).

10. **Detector + tracker (CV-3, CV-4)** (gym RGB this PR).
    `AppearanceDetector` finds KIND_RGB blobs in the camera image and
    does **not** read `context.obstacles`. `MockDetector` remains the
    gym default. Tracklets associate by IoU / centroid when `world_xy`
    is missing. Living interlock can consume those dets (row 15).  
    *Test (software):* painted blob → det; black frame + oracle list →
    no det (`tests/test_appearance_interlock.py`, `tests/test_onnx.py`).  
    *Test (field):* recorded walk-through. No fake mAP. `map_claim` null.

11. **Grass coverage observer (CV-2)** (gym this PR). Class-aware
    observer + painted-strip fixture. Phone cut % default remains
    `gym_grid`; opt-in `observer`. Field strip test still needed.  
    *Test:* `pytest tests/test_grass_coverage.py tests/test_perception.py`.  
    *Honesty:* no field mAP / IoU. `map_claim` / `iou_claim` stay null.

12. **Gym elev fuse (MAP-2 gym this PR).** Stereo (ideal) + ToF + local
    IMU onto `ObservedMap`; lock holds. Field stereo matcher still
    later. MAP-2b is an interface only (prior cannot override stereo).  
    *Test:* `pytest tests/test_elev_fuse.py tests/test_stereo.py`.  
    *Honesty:* no matcher / FPS / mAP. Kerb vs tape in gym only.

13. **Sparse pose assist / loop-closure (focused, this PR).** Taught
    fence vertices + occupancy fingerprint; 2-D pull, `not_slam: true`.
    Offline docked densify (MAP-3c) stays optional and off the live
    loop.  
    *Test:* `pytest tests/test_pose_assist.py tests/test_wave4_mapping.py`.  
    *Honesty:* not SLAM. Stated gym band 0.75 m.

14. **Geofence in a surveyed frame (MAP-5)** + multi-session load
    (MAP-4) (this PR). Surveyed-origin *model*: peg + metre keep-in.
    Cold file load of ObservedMap + uncut + yard. No WGS84 survey here.  
    *Test:* `pytest tests/test_survey_origin.py tests/test_mission.py tests/test_pln_regression.py`.  
    *Honesty:* tape-stop on a real RTK peg still required.  
    *Revise:* field survey + Orin power-cycle.

15. **Re-test explore + coverage + tip recovery + resume (PLN-1…5)**
    (gym regression this PR) on observed maps (`mission_tiny`). Gym
    constants were **not** retuned. Do not “fix” physics.  
    *Test:* `pytest tests/test_pln_regression.py tests/test_mission_flow.py`.  
    *Honesty:* BlindDetector detections stay `[]`. Appearance +
    `interlock_source: detections` trips only on a forward-camera blob
    (`tests/test_appearance_interlock.py`). Field maps still later.

16. **Self-test on hardware, incident black box, immobilised vs stuck
    (SAF-1…3)** (gym/bench black box this PR). Self-test still gym
    streams. OTA is an honest no-op stub (SAF-4).  
    *Test:* `pytest tests/test_session_ops.py tests/test_selftest.py tests/test_incident.py`.

17. **Owner notifications + multi-yard (UX-2, UX-3)** (in-app this PR).
    `/notifications` + `/yards` switch; webhook stub; no SMS.  
    *Test:* `pytest tests/test_session_ops.py tests/test_app_api.py`.

18. **Measure thermal / pack (RT-5, HD-batt)** (procedure this PR).
    How to measure Wh / charge hours / board °C:
    [`PACK_THERMAL.md`](PACK_THERMAL.md). Config hooks:
    `runtime.battery.capacity_wh`, `charge_time_h`, `measured: false`
    by default. Gym keeps the 50 Wh stub when unmeasured. `measured:
    true` requires a filled template. **Numbers null until bench. No
    claimed acre runtime.**  
    *Test:* `pytest tests/test_pack.py tests/test_budget.py tests/test_config.py`.

19. **Fab the chassis** (checklist this PR) per
    [`HARDWARE_DESIGN.md`](HARDWARE_DESIGN.md) (or revise the math from
    measured mass / CG). [`FAB_CHECKLIST.md`](FAB_CHECKLIST.md) +
    [`configs/hardware/bom.yaml`](../configs/hardware/bom.yaml). Not a
    pretend chassis.  
    *Test:* `pytest tests/test_bom.py`.

20. **Field test scorecard** (template + gym dry-run this PR) on a
    taught residential acre: ESTOP, living interlock, rain/SOC skip,
    return-to-home. Score tips / drain entries / leftover uncut /
    ESTOP pulls — not mAP. [`FIELD_TEST.md`](FIELD_TEST.md). Practice
    first: `jims-mower-field-dryrun` (laptop, `domain: gym_dryrun`,
    `field_ready: false`). Scorecard ready; field not run.  
    *Test:* `pytest tests/test_field_scorecard.py tests/test_field_dryrun.py`.

After row 20 the remaining work is **human fab, measure pack/CG,
commit measured extrinsics, collect real labels, field scorecard** —
not another WAVE of gym stubs.

### Post-§20 software leftovers (this PR)

§18–§20 shipped **procedures**. The software leftovers that still
blocked a first Orin boot and Jamie's original CV ask (people /
animals / obstacles) are here — still **not** a license to invent
field numbers.

| Leftover | What shipped | Still later |
| --- | --- | --- |
| On-box unit (RT-7) | `jims-mower-onbox` + systemd example + [`ONBOX.md`](ONBOX.md). No renderer. | Real CSI, real IMU, flashed image |
| Appearance gym (CV-3/4, PLN-4) | RGB blob dets + dets-from-camera interlock + IoU tracklets | Real dets, real CSI, field mAP (never invent) |
| Hand signals (CV-5) | Gym red-bias stop/go on an appearance person crop → ICD / policy. No confusion matrix. | Real clips, or drop the feature |
| Field-scorecard gym dry-run | `jims-mower-field-dryrun` fills `field_scorecard` (`domain: gym_dryrun`, `field_ready: false`) | The real acre — [`FIELD_TEST.md`](FIELD_TEST.md) |
| Scale cal (S2R-4) | YAML + gym × scale + [`SCALE.md`](SCALE.md) | Tape m/s + tach RPM on the rig |
| Bring-up | `jims-mower-bringup` PASS/FAIL/SKIP | Physical paddle, JetPack CSI |
| Field RF / OTA / IoU | Honest stubs (UX-4, SAF-4). Do not claim SIL | BT/LoRa, A/B OTA, held-out IoU |

---

## What this PR ships

- **FIELD_TEST gym dry-run:** `jims-mower-field-dryrun` runs a short
  `mission_tiny` fixture and writes a filled scorecard
  (`domain: gym_dryrun`, `field_ready: false`). Preflight via
  bring-up, teach/load + origin flag, explore → MAP READY → mow →
  return-home (capped OK), appearance living interlock, tip inject,
  rain/SOC schedule skips, day-2 new-process restore, HW ESTOP sim
  latch. Scores tips / drains / leftover / ESTOP pulls — not mAP.
  [`FIELD_TEST.md`](FIELD_TEST.md). **Not a field test.**
- **CV-5 gym:** red-bias stop/go on an appearance person crop fills
  ICD `hand_signal` and the policy hold. Gym-only. No confusion
  matrix. Drop on real clips if it is unreliable.
- Honest leftover: the real acre, real CSI, real dets, field RF /
  OTA / IoU. Do **not** claim SIL or acre runtime.

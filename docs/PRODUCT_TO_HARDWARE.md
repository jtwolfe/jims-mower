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
| CV-2 | Grass coverage observer | **stub** | `ColorGrassObserver` matches synthetic green; `FeatureGrassObserver` is a 6-stat mix. Cut % on the phone today is the *gym grass grid*, not this net. | On-box coverage drift vs painted/measured strips on one lawn, same day. | CV-1, MAP-1 |
| CV-3 | Person / animal / obstacle detect | **stub** | `MockDetector` **projects** `context.obstacles` (sim-only) and stays the gym default. `AppearanceDetector` (`detector_backend: appearance\|onnx`) ignores that list and returns `[]` until a real ONNX box head exists. `BlindDetector` returns `[]`. | Precision/recall on a recorded real walk-through with a person + dog + chair. Publish the set size. No fake mAP. | CV-8, RT-1 |
| CV-4 | Tracking / tracklets | **stub** | Temporal association on whatever `detections` the detector emitted (`info["tracklets"]`). Not MOT. Still consumes the ICD `detections` key for living interlock. | ID-switch count on a 30 s real clip with one crossing. | CV-3 |
| CV-5 | Hand signals | **stub** | Oracle person labels or crop brightness / red-bias classifier. | Confusion matrix on real stop/go/back clips, or **drop the feature** until CV-3 works. | CV-3 |
| CV-6 | Train → ONNX → TensorRT | **partial** (software path this PR) | `jims-mower-train-terrain --onnx` writes a sim_only Gemm graph when the `onnx` extra is installed. `jims-mower-export-trt --dry-run` still prints `trtexec`. `TrtTerrainObserver` / `TrtDetector` load an engine **if you provide one**, else heuristic / mock. No production ONNX in git. | `trtexec` builds your engine on the Orin; `fps_claim` stays null until you measure. Software pipeline ≠ field head. | CV-1 or CV-3 weights |
| CV-7 | Domain gap (wet / dawn / night) | **partial** | Renderer tints only. Not HDR, IR, or wet-lens. | Same route at noon vs dusk vs wet; hazard stamps must not invert drain vs grass. | CV-1, RT-1 |
| CV-8 | Dataset (real) | **partial** (harness this PR) | Exporter writes `jims_mower.dataset.v1` (oracle PNG + COCO-like index) from renderer **or** FakeCsi/Gst (`--adapter`). Train/val is last-frac (`meta.split` / `split.json`). Label protocol: [`DATASET.md`](DATASET.md). Fake CSI is OK in CI — not real photos. No field bag yet. No published mAP. | N frames from the rig, labeled, versioned, train/val documented. CI: FakeCsi → v1 + split. | RT-1, HD-cam |

### 2. Mapping

| ID | Item | Status | Why it matters | Acceptance test | Depends on |
| --- | --- | --- | --- | --- | --- |
| MAP-1 | ObservedMap | **partial** | Works in sim: unknown ≠ safe; camera hits + body/ToF disk. Stamps **observer** rasters, so garbage in → garbage map. | After a real explore, fog holes match what the cameras saw; no authored shed leaked outside the mask. | CV-1, RT-1, RT-2 |
| MAP-2 | Near-field metric stereo + frozen elev fuse | **partial** (this PR) | `find_stereo_pair` + gym synthetic stereo stamps local elev in the 0.8–4 m band onto `ObservedMap`. MAP READY `lock_observed()` so later stamps do not flop frozen cells. Default gym look-around rig is a **no-op** (not a pair). `fuse_height_rgb_tof` is still the old RGB-label + ToF-corner stub — not the live metric path. Not COLMAP. | Gym: stereo YAML stamps cells; lock holds after a flopping observer raster (`tests/test_stereo.py`). Field: kerb cross-section vs tape + IMU; no invented mAP / FPS. | CV-1, HD-cam stereo, RT-2 |
| MAP-2b | Learned mono depth prior | **missing** | Orin-class depth net only as a prior fused with stereo / ToF. Never the sole metric source. | Ablation: stereo-only vs stereo+prior on one kerb; prior must not win when stereo is valid. | MAP-2, RT-3 |
| MAP-3 | Loop closure / revisit | **stub** | `LoopClosureStub` occupancy fingerprint. `not_slam: true`. No pose-graph. | Return to dock after 1 acre explore; fence vertices stay inside a stated metre error vs teach. | RT-2 GNSS/IMU, MAP-1 |
| MAP-3b | Sparse multi-view / pose assist | **missing** | Track features across the 4–6 cam rig + time; landmarks + wheel odom + IMU tilt (later RTK). Not a dense DEM each frame. | Drift vs teach vertices after a repetitive-grass loop; RTK-VIO later. | MAP-3, RT-2 |
| MAP-3c | Offline docked densify | **missing** | Heavier multi-view pass while idle. True photogrammetry for the owner mesh only. Must not block the live mow loop. | Docked job writes a refined mesh; mow loop FPS / cycle time unchanged (measure later; do not invent). | MAP-2, RT-7 |
| MAP-4 | Multi-session persistence | **partial** | `jims-mower-mission` saves map + uncut + pose for the **same sim process**. No day-2 load on a cold Orin with GNSS origin. | Power cycle, reload yesterday's yard, resume uncut without reteaching. | MAP-3, MAP-5 |
| MAP-5 | Geofence on Earth | **partial** | Taught polygon in the gym metre frame. GNSS is a noisy `(x,y,z,valid)` in that frame, not WGS84. | Keep-in vertices + a surveyed origin; robot stops before the tape, not 3 m past. | RT-2 GNSS, UX teach |

### 3. Planning / control

| ID | Item | Status | Why it matters | Acceptance test | Depends on |
| --- | --- | --- | --- | --- | --- |
| PLN-1 | Explore | **partial** | Frontier walk on ObservedMap. Needs real maps to be useful. | MAP READY on a real lawn without counting fog islands as unreachable (already true in sim). | MAP-1 |
| PLN-2 | Coverage | **partial** | Boustrophedon + A* on observer costmap; energy strip order uses stub SOC. | Leftover uncut vs planned cells on a marked 10×10 m patch. | MAP-1, CV-2 |
| PLN-3 | Tip recovery | **partial** | Reverse → pivot → help; IMU tip-skip continues paint in sim. Thresholds (`tip_roll_rad=0.40`) are gym constants. | Tip the chassis to the software trip on a known ramp; wheels stop; recover without drain entry. | RT-2 IMU, HD-mass |
| PLN-4 | Living-thing interlock | **partial** | Trimmer off inside `safety_radius_m` of a **detection**. With MockDetector this is god-view. With BlindDetector it never fires. | Person steps into the radius on the real rig; trimmer request is refused within one control cycle. | CV-3, CV-4 |
| PLN-5 | Resume after stop | **partial** | Mission save/load + owner Pause/Resume in live. Schedule duration-stop is new. Field “battery died mid-strip” is untested. | Pause 10 min, resume; uncut cells still planned. | MAP-4, SCH-1 |

### 4. Scheduling

| ID | Item | Status | Why it matters | Acceptance test | Depends on |
| --- | --- | --- | --- | --- | --- |
| SCH-1 | Arm / stop from `YardProfile.schedule` | **partial** (this PR) | Engine + Health toggle + SOC / rain / fault gates. Not a cloud calendar. Not a rain *service*. | Enable Mon 09:00 UTC on FrozenClock → mission `mowing`; SOC 0.10 → `soc_low`; rain flag → skip; duration expires → idle. See [`SCHEDULE.md`](SCHEDULE.md). | Owner app (done) |
| SCH-2 | Timezone | **partial** | `local` or IANA. Orin must set a real zone (e.g. `Australia/Sydney`). | Job starts at 09:00 in that zone, not UTC-by-accident. | SCH-1 |
| SCH-3 | Rain skip | **partial** | Uses env `weather.wet` or `status.weather.rain`. No BOM/radar. | Set rain flag; window is consumed as skip; next week can still run. | SCH-1 |
| SCH-4 | SOC gate | **partial** | Compares `battery.soc` to `schedule.min_soc` (default 0.25). Gym SOC is the OrinBudget **stub**. | Real fuel gauge below min_soc skips. | RT-5 pack telemetry |
| SCH-5 | Notifications | **missing** | No SMS / push when a run skips or finishes. | Owner gets a skip reason without opening the LAN page. | SCH-1, UX-2 |

### 5. Runtime (Orin)

| ID | Item | Status | Why it matters | Acceptance test | Depends on |
| --- | --- | --- | --- | --- | --- |
| RT-1 | CSI / GStreamer capture | **partial** (software path this PR) | `FakeGstAdapter` / `FakeCsiDriver` fill named `obs["cameras"]` at the ICD contract size with fresh stamps. Downsample is `runtime.capture.downsample_rgb`. `GstNvmmAdapter` still raises without Gst. **Not** physical CSI — JetPack + real cameras still required. No FPS. | Named cameras (prefer `stereo_left` / `stereo_right` + mono) fill `obs["cameras"]` at `sensors.width` × `sensors.height`; `SensorWatchdog` stays happy on fresh stamps; freeze stamps → wheels zero. Field: plug CSI on the Orin (needs JetPack). | HD-cam, JetPack |
| RT-2 | IMU / GNSS / ToF drivers | **partial** (gym stubs this PR) | Fake I2C/UART publishers still copy gym vectors onto in-process queues. Level-rest IMU, GNSS `valid` toggle, and ToF board-under-wheel fixture are writable. Addresses in `drivers.py` remain documentation. | Gym: rest IMU ≈ `(0,0,9.81,0,0,0)`; `valid` bit toggles; ToF corner shortens when a board is slid under a wheel. Field: same ICD keys from real BMI/ICM + GNSS + VL53-class parts. | HD-place |
| RT-3 | TensorRT load | **stub** | See CV-6. Software ONNX export exists; deserialize still needs an Orin engine you build. Fallback stays heuristic / mock if the path is missing (keep that). | Engine deserializes; fallback still mock if path missing (keep that). | CV-6 |
| RT-4 | Watchdog | **partial** (bench config + gym stamp stall this PR) | Zeros wheels if IMU/vision **stamps** freeze (`runtime.watchdog.enabled`). Off in default gym tests. Bench overlay: [`configs/orin/bench.yaml`](../configs/orin/bench.yaml). Fake adapters — not real CSI/IMU. | Gym: freeze IMU or camera stamps → wheels zero within `vision_stall_s` / configured stall (`tests/test_hardware_estop.py`, `tests/test_wave4_ops.py`). Field: unplug a camera on the wired rig (needs RT-1). | RT-1, RT-2 |
| RT-5 | Battery / thermal telemetry | **stub** | `OrinBudget` 50 Wh class-scale RC. Not a BMS. | SOC and board °C from hardware; limp/stop match measured limits. | HD-batt |
| RT-6 | Hardware ESTOP | **partial** (sim + doc this PR) | Gym `HardwareEstop` drops traction + trimmer **rails** underneath policy / `SafeStateMachine`. Wiring + reset: [`ESTOP.md`](ESTOP.md), [`HARDWARE_DESIGN.md`](HARDWARE_DESIGN.md) §10. No physical paddle. | Gym: dummy load commanding wheels+trimmer; paddle latch zeros outputs; software clear does **not** restore; only `hw_reset` does (`tests/test_hardware_estop.py`). Field: hit a real paddle while a dummy load spins — **not claimed**. | HD-wire |
| RT-7 | On-box loop (no renderer) | **partial** | Documented; not a shipped systemd unit. | Process runs without importing `jims_mower.renderer`. | RT-1…RT-4 |

### 6. Safety / ops

| ID | Item | Status | Why it matters | Acceptance test | Depends on |
| --- | --- | --- | --- | --- | --- |
| SAF-1 | Self-test | **partial** | `jims-mower-selftest`: unloaded spin, IMU still, cam entropy — against **gym** streams. | Same checks on the wired robot before first mow. | RT-1, RT-2 |
| SAF-2 | Immobilised vs stuck | **partial** | FaultBus semantics are right in sim. Real motor open-circuit must raise `FAULT_IMMOBILISED`. | Kill one drive encoder; both wheels + trimmer hold; SOS retrieve. | RT-2, HD-drive |
| SAF-3 | Incident logs | **partial** | Record / incident scrubber / telemetry JSON exist for **episodes**. No always-on black box on Orin. | After a tip, pull IMU + advice + wheel cmds from onboard storage. | RT-7 |
| SAF-4 | OTA | **missing** | UX-C mentions optional Wi-Fi for OTA. No updater, no signed image. | Documented no-op or a real A/B slot — do not pretend. | UX-2 Wi-Fi |
| SAF-5 | SIL / cert | **missing** | Software ESTOP is explicitly **not** a SIL rating. | Do not claim one. | legal / field |

### 7. Owner UX

| ID | Item | Status | Why it matters | Acceptance test | Depends on |
| --- | --- | --- | --- | --- | --- |
| UX-1 | Schedule UI | **partial** (this PR) | Health toggle + next run + skip reason. Days/time still JSON. | Toggle On, FrozenClock in window → Start without Map. | SCH-1 |
| UX-2 | Notifications | **missing** | LAN SSE only. | See SCH-5. | SCH-5 |
| UX-3 | Multi-yard | **stub** | Sequence CLI (`paddock` then `suburban`) is a sim farm tool. One YardProfile per app process. | Switch yards on the phone; home/geofence/schedule swap; no fence bleed. | MAP-4 |
| UX-4 | Radio | **stub** | Wi-Fi → BT → LoRa **sim**. No RF. | Field: BT pair required, LoRa command at the far fence — measure or don't claim range. | hardware radios |
| UX-5 | Pairing | **stub** | Phone “Pair Bluetooth” sets a bool. | Real BT pairing before first Start. | UX-4 |

### 8. Sim-to-real

| ID | Item | Status | Why it matters | Acceptance test | Depends on |
| --- | --- | --- | --- | --- | --- |
| S2R-1 | ICD key match | **partial** | Contract is written. Camera software path fills `obs["cameras"]`; IMU/GNSS/ToF/Gst chips are still fakes. | Laptop gym and Orin process the same key set; renderer never imported on-box. | RT-1…RT-3 |
| S2R-2 | Extrinsics YAML | **partial** (this PR) | `extrinsics_stereo.yaml` is the documented **EXAMPLE** (`calibration.measured: false`). Load + baseline-cm helper rejects non-pairs. Prefer it (6–12 cm forward pair + mono) for the field article. `extrinsics_6cam.yaml` stays the gym look-around. **Neither is taped.** | Reproject a checkerboard / drain lip; stereo pair verifies baseline + disparity vs tape. Gym: `jims-mower-calibrate` / `tests/test_calibration.py`. Field: still measure. | HD-cam |
| S2R-3 | Calibration bench | **partial** (procedure + gym this PR) | Written steps: [`CALIBRATION.md`](CALIBRATION.md). Gym lip / checkerboard fixture stamps the right ObservedMap cells; ideal disparity matches tape in 0.8–4 m. MEASURED template: `extrinsics_stereo_measured.template.yaml`. Physical measure-and-commit still required. | Human tapes the baseline, writes `extrinsics_stereo_measured.yaml`, `calibration.measured: true`. Not claimed here. | S2R-2 |
| S2R-4 | Wheel / trimmer scale | **partial** | Action is ±1 of `max_wheel_speed_mps` (1.2). Real motors have different Kv / gearing. | 1.0 command → measured m/s within a stated %. | HD-drive |

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

10. **Detector + tracker (CV-3, CV-4).** Scaffolding (this PR, focused).
    `AppearanceDetector` can load ONNX later and does **not** read
    `context.obstacles`. `MockDetector` remains the gym default.
    Tracklets stay a stub. Living interlock still consumes `detections`.  
    *Test (software):* empty dets when appearance/onnx; mock default
    unchanged (`tests/test_onnx.py`).  
    *Test (field):* recorded walk-through; living interlock (PLN-4) on
    those dets. No fake mAP.

11. **Grass coverage observer (CV-2)** once terrain classes exist.

12. **Field stereo matcher + ToF / IMU fuse (MAP-2 on hardware).**
    Replace gym ideal disparity. Learned mono depth only as a prior
    (MAP-2b).  
    *Test:* kerb step vs tape; locked cells still do not flop.

13. **Sparse pose assist / loop-closure good enough to hold the taught
    fence (MAP-3, MAP-3b).** Still not “we shipped SLAM.” Offline
    docked densify (MAP-3c) stays optional and off the live loop.

14. **Geofence in a surveyed frame (MAP-5)** + multi-session load (MAP-4).

15. **Re-test explore + coverage + tip recovery + resume (PLN-1…5)** on
    the real maps. Retune gym constants; do not “fix” physics.

16. **Self-test on hardware, incident black box, immobilised vs stuck
    (SAF-1…3).** OTA stays missing or an honest stub (SAF-4).

17. **Owner notifications + multi-yard (UX-2, UX-3)** after the robot
    can finish a job without a laptop SSE tab.

18. **Measure thermal / pack (RT-5, HD-batt).** Replace the 50 Wh stub
    with a measured Wh and charge time. **No claimed acre runtime** until
    this row.

19. **Fab the chassis per [`HARDWARE_DESIGN.md`](HARDWARE_DESIGN.md)**
    (or revise the math from measured mass / CG).

20. **Field test** on a taught residential acre: ESTOP, living interlock,
    rain/SOC skip, return-to-home. Scorecard is tips / drain entries /
    leftover uncut — not mAP.

After row 20 the remaining work is **fab revisions + more field tests**,
not another WAVE of gym stubs.

---

## What this PR ships

- Build-order **§9** (CV-1 / CV-6): train → ONNX → TRT **software
  pipeline**. `jims-mower-train-terrain --onnx` writes a sim_only
  Gemm graph when `pip install -e ".[onnx]"` is available.
  `OnnxTerrainObserver` / `terrain_mode: onnx|trt|learned|heuristic`.
  Heuristic remains the live mow default. `trtexec` dry-run unchanged.
  **Not** a field-ready head. No held-out real IoU. `iou_claim` /
  `map_claim` / `fps_claim` stay null.
- Build-order **§10** (CV-3 / CV-4, focused scaffolding):
  `AppearanceDetector` ignores `context.obstacles` and can load ONNX
  later. `MockDetector` stays the gym default. Tracklets remain a stub.
  Living interlock still reads the `detections` ICD key. No fake mAP.
- Honest leftover: collect real labels per [`DATASET.md`](DATASET.md)
  before any IoU claim. Do not start §11 (grass coverage field net) or
  §12 (field stereo matcher) from this PR.

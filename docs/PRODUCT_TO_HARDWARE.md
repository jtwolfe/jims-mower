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
| Observed elevation freeze | Explore stamps a local elev estimate; mow plans on the frozen observed map, not god-view | `docs/MISSION_FLOW.md`, `docs/TERRAIN_MAPS.md` |
| teach → explore → mow → home → done on phone | Pair stub → teach keep-in → Start → MAP READY → mow → return → idle; cut % rises | `jims-mower-owner --live`, `acre_yard_demo` |
| Acre world | `acre_yard` / `acre_yard_demo` ~70×58 m @ 0.50 m; physics grade / drains / banks | `configs/scenarios/acre_yard*.yaml` |
| Faults / SOS UX | Immobilised (dead motor, retrieve) vs stuck (reverse / pivot / help); software ESTOP | `FaultBus`, `#/fault` |
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

| ID | Item | Status | Why it matters | Acceptance test | Depends on |
| --- | --- | --- | --- | --- | --- |
| CV-1 | Terrain segmentation (drain / lip / bank / grass) | **stub** | Heuristic RGB + `LearnedTerrainObserver` numpy MLP trained on *sim oracle* rasters. Palette will not survive daylight grass. Wrong lip → wheel in channel. | Held-out **real** frames; report IoU per class only after a locked test set. Fail if you only have sim loss. No invented IoU. | CV-8 dataset, RT-1 capture, HD-cam extrinsics |
| CV-2 | Grass coverage observer | **stub** | `ColorGrassObserver` matches synthetic green; `FeatureGrassObserver` is a 6-stat mix. Cut % on the phone today is the *gym grass grid*, not this net. | On-box coverage drift vs painted/measured strips on one lawn, same day. | CV-1, MAP-1 |
| CV-3 | Person / animal / obstacle detect | **stub** | `MockDetector` **projects** `context.obstacles` (sim-only). Appearance refine is crop-palette stats. `BlindDetector` returns `[]`. | Precision/recall on a recorded real walk-through with a person + dog + chair. Publish the set size. No fake mAP. | CV-8, RT-1 |
| CV-4 | Tracking / tracklets | **stub** | Temporal association on projected blobs (`info["tracklets"]`). Not MOT. | ID-switch count on a 30 s real clip with one crossing. | CV-3 |
| CV-5 | Hand signals | **stub** | Oracle person labels or crop brightness / red-bias classifier. | Confusion matrix on real stop/go/back clips, or **drop the feature** until CV-3 works. | CV-3 |
| CV-6 | Train → ONNX → TensorRT | **stub** | `jims-mower-export-trt --dry-run` prints a `trtexec` line. `TrtDetector` / `TrtTerrainObserver` load an engine **if you provide one**, else mock/heuristic. No ONNX in-repo. | `trtexec` builds your engine on the Orin; `fps_claim` stays null until you measure. | CV-1 or CV-3 weights |
| CV-7 | Domain gap (wet / dawn / night) | **partial** | Renderer tints only. Not HDR, IR, or wet-lens. | Same route at noon vs dusk vs wet; hazard stamps must not invert drain vs grass. | CV-1, RT-1 |
| CV-8 | Dataset (real) | **partial** | Exporter writes oracle PNG + COCO-like index (`jims_mower.dataset.v1`). That is **sim**. No field bag, no label protocol for drain/lip on real CSI. | N frames from the rig, labeled, versioned, train/val split documented. | RT-1, HD-cam |

### 2. Mapping

| ID | Item | Status | Why it matters | Acceptance test | Depends on |
| --- | --- | --- | --- | --- | --- |
| MAP-1 | ObservedMap | **partial** | Works in sim: unknown ≠ safe; camera hits + body/ToF disk. Stamps **observer** rasters, so garbage in → garbage map. | After a real explore, fog holes match what the cameras saw; no authored shed leaked outside the mask. | CV-1, RT-1, RT-2 |
| MAP-2 | Local elev fusion (cameras + ToF + IMU tilt) | **stub** | `fuse_height_rgb_tof` back-projects RGB drain/bank labels onto a plane + ToF corners. Not a depth net or stereo. | Cross-section of a known kerb: fused step within a stated cm budget vs tape + IMU. | RT-2, HD-ext |
| MAP-3 | Loop closure / revisit | **stub** | `LoopClosureStub` occupancy fingerprint. `not_slam: true`. No pose-graph. | Return to dock after 1 acre explore; fence vertices stay inside a stated metre error vs teach. | RT-2 GNSS/IMU, MAP-1 |
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
| RT-1 | CSI / GStreamer capture | **stub** | `GstNvmmAdapter` raises without Gst; CI uses `FakeGstAdapter` / `FakeCsiDriver`. | Six named cameras fill `obs["cameras"]` at the ICD size; `SensorWatchdog` sees fresh stamps. | HD-cam, JetPack |
| RT-2 | IMU / GNSS / ToF drivers | **stub** | Fake I2C/UART publishers copy **gym** vectors onto in-process queues. Addresses in `drivers.py` are documentation. | Same ICD keys from real BMI/ICM + GNSS + VL53-class parts; dropout sets `gps[3]=0`. | HD-place |
| RT-3 | TensorRT load | **stub** | See CV-6. | Engine deserializes; fallback still mock if path missing (keep that). | CV-6 |
| RT-4 | Watchdog | **partial** | Zeros wheels if IMU/vision stall (`runtime.watchdog.enabled`, off in gym tests). | Unplug a camera; wheels stop within `vision_stall_s`. | RT-1, RT-2 |
| RT-5 | Battery / thermal telemetry | **stub** | `OrinBudget` 50 Wh class-scale RC. Not a BMS. | SOC and board °C from hardware; limp/stop match measured limits. | HD-batt |
| RT-6 | Hardware ESTOP | **partial** | Software latch (`SafeStateMachine`) zeros wheels + trimmer. No paddle → FET/contactor wiring. | Hit the paddle; traction + trimmer rails go dead **without** Python. | HD-wire |
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
| S2R-1 | ICD key match | **partial** | Contract is written. On-box sources are still fakes. | Laptop gym and Orin process the same key set; renderer never imported on-box. | RT-1…RT-3 |
| S2R-2 | Extrinsics YAML | **partial** | `configs/orin/extrinsics_6cam.yaml` is an **example**, not a calibrated file. | Reproject a checkerboard / drain lip to the observed map within a stated pixel/metre error. | HD-cam |
| S2R-3 | Calibration bench | **missing** | No procedure beyond “copy, measure, replace the numbers.” | Written steps + a fixture; saved YAML committed as *measured*. | S2R-2 |
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
   *fields only*; CV first would be another incomplete head.  
   *Test:* `pytest tests/test_schedule.py tests/test_app_api.py`.  
   *Revise:* timezone on the Orin, rain from a real sensor later (SCH-3/4).

2. **Hardware ESTOP + power kill (RT-6, HD-wire).** Paddle drops traction
   and trimmer rails without Python.  
   *Test:* paddle while a dummy load spins.  
   *Revise:* fuse map vs what actually opened.

3. **Watchdog enabled on the bench loop (RT-4).**  
   *Test:* freeze IMU or camera stamps → wheels zero.

4. **Real CSI / GStreamer → `obs["cameras"]` (RT-1).** Same names as
   `CameraSpec`. Downsample to the contract size.  
   *Test:* six (or four) live frames; watchdog happy. No FPS claim.

5. **Real IMU + GNSS + ToF (RT-2).**  
   *Test:* level rest IMU ≈ `(0,0,9.81,0,0,0)`; GNSS `valid` bit; ToF
   corners change when you slide a board under a wheel.

6. **Extrinsics + calibration bench (S2R-2, S2R-3).** Measure, write YAML.  
   *Test:* lip / checkerboard lands in the right ObservedMap cells.

7. **Dataset harness on the rig (CV-8).** Record + label protocol only.
   No trained production head yet.  
   *Test:* `jims_mower.dataset.v1` from **real** cameras; train stub may
   still run; do not publish mAP.

8. **Terrain seg train → ONNX → TRT (CV-1, CV-6).** Replace heuristic /
   numpy stub.  
   *Test:* held-out **real** IoU. If you cannot measure it, do not ship
   the head.

9. **Detector + tracker (CV-3, CV-4).** Replace `MockDetector`.  
   *Test:* recorded walk-through; living interlock (PLN-4) on those dets.

10. **Grass coverage observer (CV-2)** once terrain classes exist.

11. **Local elev fusion (MAP-2)** with real ToF + IMU + cameras.

12. **Loop-closure / revisit good enough to hold the taught fence
    (MAP-3).** Still not “we shipped SLAM.”

13. **Geofence in a surveyed frame (MAP-5)** + multi-session load (MAP-4).

14. **Re-test explore + coverage + tip recovery + resume (PLN-1…5)** on
    the real maps. Retune gym constants; do not “fix” physics.

15. **Self-test on hardware, incident black box, immobilised vs stuck
    (SAF-1…3).** OTA stays missing or an honest stub (SAF-4).

16. **Owner notifications + multi-yard (UX-2, UX-3)** after the robot
    can finish a job without a laptop SSE tab.

17. **Measure thermal / pack (RT-5, HD-batt).** Replace the 50 Wh stub
    with a measured Wh and charge time. **No claimed acre runtime** until
    this row.

18. **Fab the chassis per [`HARDWARE_DESIGN.md`](HARDWARE_DESIGN.md)**
    (or revise the math from measured mass / CG).

19. **Field test** on a taught residential acre: ESTOP, living interlock,
    rain/SOC skip, return-to-home. Scorecard is tips / drain entries /
    leftover uncut — not mAP.

After row 19 the remaining work is **fab revisions + more field tests**,
not another WAVE of gym stubs.

---

## What this PR ships

- This checklist.
- [`HARDWARE_DESIGN.md`](HARDWARE_DESIGN.md) — construction math, BOM
  *classes*, wiring sketch. No claimed field runtime.
- **One** executable slice: the schedule engine (build-order §1).
  Phone Health next-run + enable toggle. Tests in
  `tests/test_schedule.py`.

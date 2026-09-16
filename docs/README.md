# docs

Wave notes and the on-box message contract live here.

| File | What it is |
| --- | --- |
| [`WAVE1B.md`](WAVE1B.md) | EKF pose filter, uncertainty costmap, record/replay, latency harness |
| [`WAVE1C.md`](WAVE1C.md) | Scenario library, domain randomisation, weather, grass stub |
| [`WAVE2A.md`](WAVE2A.md) | Movers, geofences, recovery, hand-signal hooks, mission resume |
| [`WAVE2B.md`](WAVE2B.md) | Numpy terrain stub, BEV fuse, temporal filters, train CLI |
| [`WAVE3A.md`](WAVE3A.md) | BC stub, RL scaffold, ESTOP/limp, incident/telemetry, owner overlay |
| [`WAVE3B.md`](WAVE3B.md) | Fake drivers, multiprocessing bridge, studies, Orin budget stub |
| [`WAVE4.md`](WAVE4.md) | Appearance categories, persist occupancy, wet/energy plan, world/Orin hooks |
| [`TERRAIN_MAPS.md`](TERRAIN_MAPS.md) | Observer-vs-physics grade, paths/structures, golf + acre yards |
| [`MISSION_FLOW.md`](MISSION_FLOW.md) | Calibrate → explore → freeze map → global mow; live SSE + fog-of-war |
| [`UX.md`](UX.md) | World Viewer + Teach Boundary (low-poly mesh, no mAP/FPS) |
| [`UX_B.md`](UX_B.md) | FaultBus, dead-motor SOS, radio sim, `jims-mower-selftest` |
| [`UX_C.md`](UX_C.md) | Thin owner app, YardProfile API, buy→mow (BT / optional Wi-Fi / LoRa) |
| [`SCHEDULE.md`](SCHEDULE.md) | Weekly window actually arms / skips / stops jobs |
| [`PRODUCT_TO_HARDWARE.md`](PRODUCT_TO_HARDWARE.md) | Honest stub inventory + numbered build order. Software path complete for fab+field; §18–§20 are procedures; post-§20 leftovers: on-box + appearance gym + scale |
| [`ONBOX.md`](ONBOX.md) | RT-7 on-box loop: `jims-mower-onbox`, systemd, journal. ESTOP still hardware. No renderer |
| [`SCALE.md`](SCALE.md) | S2R-4: measure 1.0 command → m/s and trimmer RPM. `measured: false`. No invented Kv |
| [`SURVEY_ORIGIN.md`](SURVEY_ORIGIN.md) | MAP-5 peg + metre keep-in; MAP-4 cold session load. No WGS84 survey in this repo. |
| [`CALIBRATION.md`](CALIBRATION.md) | Stereo / extrinsics bench: EXAMPLE vs MEASURED, tape procedure, gym lip / disparity checks |
| [`DATASET.md`](DATASET.md) | CV-8 record + label protocol; FakeCsi → `jims_mower.dataset.v1` + train/val; real labels required before any IoU |
| [`PACK_THERMAL.md`](PACK_THERMAL.md) | §18 pack Wh / charge hours / board °C procedure. 50 Wh gym stub; `measured: false` until bench |
| [`FAB_CHECKLIST.md`](FAB_CHECKLIST.md) | §19 chassis fab steps + BOM freeze (part classes, no SKUs). Hang-measure → revise tip math |
| [`FIELD_TEST.md`](FIELD_TEST.md) | §20 residential-acre scorecard (tips / drains / leftover / ESTOP — not mAP). Field not run |
| [`HARDWARE_DESIGN.md`](HARDWARE_DESIGN.md) | Construction math, BOM classes, forward stereo + ESTOP sketch (no claimed runtime) |
| [`ESTOP.md`](ESTOP.md) | Hardware paddle → FET/contactor, fuse map, reset; gym `HardwareEstop` |
| [`SIM_TO_REAL.md`](SIM_TO_REAL.md) | Same ICD keys on-box; no gym renderer |
| [`JETSON.md`](JETSON.md) | What not to run on-box; Dockerfile.aarch64; TensorRT placeholder |
| [`runtime_contract.md`](runtime_contract.md) | Versioned CameraFrame / IMU / GPS / ToF / maps / Plan / WheelCommand / SafeState |

Repo-root [`ROADMAP.md`](../ROADMAP.md) and [`ICD.md`](../ICD.md) landed
with WAVE 1A. Later waves add sections; they do not invent mAP / FPS.

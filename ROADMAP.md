# ROADMAP — no-hardware backlog

Structured checklist for work that does **not** require the physical mower.
Hardware bring-up (wiring, JetPack flash, field calibration) lives elsewhere.
There are no claimed mAP / FPS numbers on this list.

WAVE **1A** is the foundation package. WAVE **1C** expanded yards / renderer.
WAVE **1B** (EKF / record-replay), WAVE **2A** (movers, geofences,
recovery, hand-signal hooks, mission resume), WAVE **2B** (sim-only
perception: exporter → numpy terrain stub, BEV fuse, temporal filters),
and WAVE **3A** (behaviour cloning, RL scaffold, incident / telemetry /
ESTOP, owner overlay) are on `main`. WAVE **3B** (runtime / Orin packaging /
design studies) is on `main`. WAVE **4** closes remaining perception /
mapping / planning / world / Orin / ops hooks. WAVE **UX-A** (World Viewer
+ Teach Boundary) and WAVE **UX-B** (FaultBus, radio sim, self-test) are
on `main`. WAVE **UX-C** (this PR) is the thin owner app + YardProfile API;
it reuses the UX-A mesh + three.js viewer and UX-B fault / radio status.
Later items stay unchecked until they land. No claimed mAP / FPS.

## First-run teach → save yard → run job (this PR)

- [x] Phone / live first-run: pair → teach keep-in (drive or edit vertices) → save `YardProfile`
- [x] Live Start uses the taught profile as geofence/home (skip authored `calibrate_confirm_m`)
- [x] After teach, job still does explore fog → MAP READY → mow
- [x] One command: `jims-mower-owner --live` (Teach → Save → Start) or `--first-run`
- [x] Reuse UX-A `YardProfile` / teach trail — no second fence format
- [x] Acre-scale physics world stays (`acre_yard_demo`); `--fast` is CI
- [x] Tests: teach → profile → live start smoke (no full-acre mow)
- [x] Docs: first-run vs demo confirm fence; `acre_yard` vs `acre_yard_demo`
- [x] Teach Save rejects/repairs scribble keep-ins; Start actually explores; empty mow plan ≠ Hold-safe

## WAVE UX-C — owner app shell + YardProfile API (done, on main)

- [x] YardProfile JSON (`jims_mower.yard.v1`) — home, keep-in/out, mesh, radio prefs, schedule stub
- [x] Load / save / validate on the UX-A `YardProfile` (radio / schedule extras)
- [x] Local HTTP JSON API: `/status`, `/yard`, `/command`, `/map/mesh`, `/map/coverage`
- [x] SSE `/events` for live status
- [x] Thin phone shell: unbox → pair → place home → teach → first mow; map; health; SOS
- [x] Buy→mow checklist (BT required, Wi-Fi optional, LoRa long-range)
- [x] `jims-mower-app` serves API+UI against sim env or recorded episode
- [x] Reuse UX-A `viewer_static` + `mesh_to_payload` (no second three.js stack)
- [x] Surface UX-B FaultBus / RadioSim on `/status` (no second radio stack)
- [x] YardProfile + API smoke tests
- [x] ROADMAP checkboxes for the items above

## WAVE UX-A — World Viewer + Teach Boundary (done, on main)

- [x] World Viewer local web UI (`jims-mower-viewer`) — three.js CDN,
      low-poly mesh, coverage / hazard / occupancy toggles, plan overlay,
      live/scrub pose, camera PiP, health/radio placeholders
- [x] Low-poly mesh export (`jims-mower-mesh`) — elevation → decimated
      GLB + OBJ + JSON (no extra mesh deps)
- [x] Teach Boundary (`--policy teach` / `jims-mower-teach`) — perimeter
      trail → smoothed polygon → viewer vertex edit → `YardProfile` JSON
- [x] Load `YardProfile` into env / scenario (`--profile` or `--config`)
- [x] Demo / record write a viewer bundle (`viewer.json`, `yard.glb`, maps)
- [x] [`docs/UX.md`](docs/UX.md)
- [x] Tests: mesh non-empty, trail → polygon, viewer static assets, CI

See [`docs/UX.md`](docs/UX.md). No claimed mAP / FPS.

## WAVE 4 — remaining ROADMAP hooks (done, on main)

- [x] Person / animal / toy categories + appearance model on `MockDetector`
- [x] Hand-signal classifier stub behind `curriculum.hand_signal_classifier`
- [x] `FeatureGrassObserver` grass-coverage stub (numpy features)
- [x] Semantic layer raster (grass / non-grass / drain / bank / static)
- [x] Persistent BEV occupancy (detections + ToF, not god-view)
- [x] Height-map fusion stub (RGB back-proj + downward ToF)
- [x] Loop-closure stub (revisit fingerprint, not SLAM)
- [x] Wet-slope extra cost when `weather.wet`
- [x] Energy / battery-aware strip order (`OrinBudget` SOC)
- [x] Multi-yard sequence (`paddock` then `suburban`)
- [x] Seasonal overlays (long grass / leaf clutter)
- [x] Narrow-gate + fence-line scenarios
- [x] Property-scale (40 m+) coarse yard
- [x] Real-yard import stub (survey polygon JSON)
- [x] Watchdog: stop wheels if IMU / vision stall
- [x] YAML extrinsics for a 4–6 cam rig
- [x] GStreamer / NVMM adapter stub (not required in CI)
- [x] TensorRT load-weights placeholder behind Detector / TerrainObserver
- [x] Farm flake budget / quarantine (no silent skip)
- [x] Dataset `schema: jims_mower.dataset.v1` enforced
- [x] Design studies: front pitch, `drain_clearance_m` vs coverage, trimmer leftover
- [x] Curriculum schedule flat → suburban → wet → night
- [x] Sim-to-real protocol doc (ICD keys, no renderer on-box)
- [x] ROADMAP checkboxes for the items above

## WAVE 3B — runtime / Orin packaging / design studies (done)

- [x] Fake I2C IMU / UART GNSS / CSI camera (and I2C ToF) drivers that
      publish gym or recorded streams onto in-process queues matching
      [`runtime_contract.md`](docs/runtime_contract.md)
- [x] ROS 2-lite **stdlib multiprocessing** bridge (no ZMQ / gRPC in the
      default install); optional `[ros2]` extra is a marker with
      import-guarded node stubs
- [x] Record / replay compatibility: `jims-mower-bridge` consumes a
      `jims-mower-record` episode directory
- [x] Jetson packaging notes — [`docs/JETSON.md`](docs/JETSON.md),
      [`docker/Dockerfile.aarch64`](docker/Dockerfile.aarch64),
      TensorRT export **placeholder** (`jims-mower-export-trt --dry-run`;
      no weights, `fps_claim: null`)
- [x] What **not** to run on-box: gym renderer, `OracleTerrainObserver`,
      farm / study sweeps, desktop OpenCV GUI
- [x] Design-study scripts — camera 4/5/6, ToF 0/2/4, IMU noise on
      frozen seeds; markdown/CSV of **tip_rate** / **drain_entry_rate**
      (physics counts, not mAP)
- [x] Orin-class battery / thermal stub that can limp / stop
      `TerrainPolicy` when hot or low SOC (`runtime.enabled`)
- [x] ROADMAP checkboxes for the items above

## WAVE 3A — learning + ops tooling (done, on main)

- [x] Behaviour cloning hook — log (obs→action) from terrain-policy demos;
      train a tiny numpy MLP stub; `jims-mower-demo --policy bc` loads
      weights if present
- [x] RL fine-tune scaffold — numpy REINFORCE / random-search plus optional
      `[rl]` extra (SB3/torch); 1-episode CPU smoke; hazard action mask
- [x] Incident replay viewer — `jims-mower-incident` scrubs cameras +
      hazard + advice from a recorded episode
- [x] Telemetry summary JSON — coverage, tip rate, drain entries, living
      near-misses (`jims-mower-telemetry`)
- [x] ESTOP / limp / safe-state machine used by the controller; ICD
      software e-stop contract
- [x] Owner UX stub — phone-sized HTML overlay of yard + geofence + plan
- [x] ROADMAP checkboxes for the items above

## WAVE 2B — sim perception pipeline (done, on main)

- [x] Exporter → numpy terrain stub (`LearnedTerrainObserver`, no claimed accuracy)
- [x] Multi-camera BEV fuse for hazard stamps
- [x] Temporal hysteresis / decay on hazard + person/dog tracklets stub
- [x] `jims-mower-train-terrain` / `scripts/train_terrain_seg.py`
- [x] Optional `[torch]` extra (not used in CI)

## WAVE 2A — dynamic world + behaviour (done)

- [x] Moving people / animals with simple trajectories (`patrol` / `loop` /
      `line` / `wander`); occupancy updates each step
- [x] Planner / controller slow / stop / reroute around living things
      (extends the trimmer interlock via `living_advice`)
- [x] Geofences — GPS polygon keep-in / keep-out from scenario YAML;
      costmap blocks outside; pre-touch slow; demo
      [`geofence_movers`](configs/scenarios/geofence_movers.yaml)
- [x] Recovery behaviours — reverse off a lip, pivot, then call-for-help
      when tip / wheel-in-channel advice fires repeatedly
- [x] Hand-signal policy hooks — `stop` / `go` / `follow` / `back` override
      the controller when `curriculum.hand_signals` is on (mock / oracle labels)
- [x] Multi-session resume — `jims-mower-mission` + demo
      `--save-mission` / `--load-mission` (map + uncut + pose)
- [x] ROADMAP checkboxes for the items above

## WAVE 1A — foundation (done)

- [x] [`ROADMAP.md`](ROADMAP.md) — this backlog
- [x] [`ICD.md`](ICD.md) — observation / action / plugin contracts
- [x] [Scenario DSL](src/jims_mower/scenarios.py) + schema [`configs/scenarios/schema.yaml`](configs/scenarios/schema.yaml)
      - yards: [`suburban`](configs/scenarios/suburban.yaml),
        [`paddock`](configs/scenarios/paddock.yaml),
        [`playground`](configs/scenarios/playground.yaml),
        [`orchard`](configs/scenarios/orchard.yaml),
        [`night_dawn`](configs/scenarios/night_dawn.yaml) (dawn flag),
        [`wet_slope`](configs/scenarios/wet_slope.yaml) (wet + banks)
      - plus existing [`configs/steep_yard.yaml`](configs/steep_yard.yaml)
- [x] [Dataset exporter](src/jims_mower/export.py) — `python -m jims_mower.export`
- [x] [Metrics / scorecards](src/jims_mower/metrics.py) — coverage %, tips, drains, near-miss, advice histogram
- [x] [Overnight farm](src/jims_mower/farm.py) — `python -m jims_mower.farm` + optional [nightly workflow](.github/workflows/farm.yml)
- [x] [BEV / visual debugger](src/jims_mower/bev.py) — demo writes `bev.png` / `bev_final.png`

## Perception

- [x] Replace `classify_terrain_rgb` with a learned drain / lip / bank / grass head (train on exporter labels) — WAVE 2B **numpy stub**, no claimed accuracy ([`docs/WAVE2B.md`](docs/WAVE2B.md))
- [x] Temporal consistency on hazard stamps (hysteresis / decay) — WAVE 2B
- [x] Multi-camera BEV fuse for hazard stamps + person/dog tracklets stub — WAVE 2B (not published MOT / NMS scores)
- [x] Person / animal / toy categories beyond `MockDetector` blobs — WAVE 4 appearance model (not mAP)
- [x] Hand-signal classifier behind `HandSignalCurriculum` (optional) — WAVE 4 numpy crop stub
- [x] Grass coverage net behind `GrassObserver` (replace color heuristic) — WAVE 4 `FeatureGrassObserver` stub
- [x] Domain-randomised lighting / wet / dawn from scenario flags — WAVE 1C renderer + WAVE 2B `--domain-rand` export note
- [x] Uncertainty maps on `TerrainEstimate` (relative merge weights, not published scores) — WAVE 1B costmap + WAVE 2B BEV fuse

## Mapping

- [x] Persistent BEV occupancy (detections + ToF, not god-view) (WAVE 4)
- [x] Height-map fusion from RGB back-projection + downward ToF (WAVE 4 stub)
- [x] Geofence as a first-class map layer (inflate, visualize, plan) (WAVE 2A)
- [x] Multi-session yard memory (same scenario, new seed) (WAVE 2A)
- [x] Loop-closure *stub* only — do not drop a full SLAM stack in-repo (WAVE 4)
- [x] Semantic layers: grass / non-grass / drain / bank / static (WAVE 4)

## Planning

- [x] Online completeness: resume uncut cells after a person forces a stop (WAVE 2A)
- [x] Dynamic people: temporary block + replan (not just collision terminate) (WAVE 2A)
- [x] Battery / thermal limp stub (WAVE 3B; not strip reordering)
- [x] Energy / battery-aware strip order (WAVE 4; uses OrinBudget SOC)
- [x] Wet-slope cost (scenario `weather.wet` → extra slow corridor) (WAVE 4)
- [x] Geofence-aware `in_yard` already exists; planner treats the polygon as blocked margin (WAVE 2A)
- [x] Multi-yard coverage sequence (paddock then suburban) (WAVE 4)

## Learning

- [x] Offline behaviour cloning from terrain-policy demos / episode logs (WAVE 3A)
- [x] RL scaffold on the Gymnasium env (REINFORCE / random-search; optional SB3) (WAVE 3A)
- [x] Imitation of `TerrainPolicy` as a baseline, not a claimed SOTA (WAVE 3A)
- [x] Sim-to-real protocol: same ICD obs keys, no gym renderer on-box ([`docs/SIM_TO_REAL.md`](docs/SIM_TO_REAL.md))
- [x] Curriculum: flat → suburban → wet_slope → night_dawn (WAVE 4 schedule)

## World library

- [x] Scenario DSL + 6 authored yards beyond `steep_yard` (WAVE 1A)
- [x] Seasonal variants (long grass, leaf clutter) as scenario overlays (WAVE 4)
- [x] Real-yard import (survey polygon → geofence + drain polylines) (WAVE 4 stub)
- [x] Moving-animal density presets (`world.movers.density`: sparse / default / busy) (WAVE 2A)
- [x] Narrow-gate / fence-line scenarios (WAVE 4)
- [x] Property-scale (40 m+) yards at coarser resolution (WAVE 4)

## Orin runtime

- [x] Fake I2C / UART / CSI publishers matching the runtime contract (WAVE 3B)
- [x] GStreamer / NVMM capture adapter that fills `obs["cameras"]` (WAVE 4 stub; Gst not in CI)
- [x] TensorRT export **placeholder** (no ONNX shipped, no FPS) (WAVE 3B)
- [x] TensorRT (or similar) behind `Detector` / `TerrainObserver` (WAVE 4 load-weights placeholder)
- [x] Complementary filter / small EKF behind `ComplementaryPoseFilter` (WAVE 1B)
- [x] Watchdog process: stop wheels if vision or IMU stalls (WAVE 4)
- [x] YAML extrinsics for the real 4–6 cam rig (same `CameraSpec`) (WAVE 4)
- [x] No desktop OpenCV GUI / no gym renderer on-box (see [`docs/JETSON.md`](docs/JETSON.md))
- [x] Stdlib multiprocessing bridge; optional `[ros2]` stubs (WAVE 3B)

## Safety / ops

- [x] Scorecard gates (tip / drain counts) + farm thresholds (WAVE 1A)
- [x] E-stop contract in the ICD (software latch; hardware is still field) (WAVE 3A)
- [x] Black-box log: IMU, GPS valid bit, advice, wheel commands (WAVE 1B record + WAVE 3A incident)
- [x] Geofence violation → `stop` (already OOB terminate; pre-touch slow) (WAVE 2A)
- [x] Trimmer / living / tip telemetry JSON (headless) (WAVE 3A)
- [x] Farm flake budget / quarantine a scenario instead of silent skip (WAVE 4)

## Data tooling

- [x] Dataset exporter (PNG + JSON sidecars + COCO-like index) (WAVE 1A)
- [x] Episode scorecards + JSON report (WAVE 1A)
- [x] Overnight farm + dry-run (WAVE 1A)
- [x] BEV debugger composites in demo output (WAVE 1A)
- [x] Train stub from export (`scripts/train_terrain_seg.py` / `python -m jims_mower.perception.train`) — WAVE 2B
- [x] Replay a recorded episode without re-simulating (`jims-mower-replay --mode offline`; WAVE 1B + WAVE 3A incident scrubber)
- [x] Label / hazard overlay on the incident viewer (WAVE 3A)
- [x] Replay recorded frames through fake drivers (`jims-mower-bridge`; WAVE 3B)
- [x] Dataset versioning (schema field `jims_mower.dataset.v1` enforced) (WAVE 4)

## Design studies

- [x] Camera count (4 / 5 / 6) vs tip / drain-entry rates on frozen seeds (WAVE 3B; not mAP)
- [x] ToF 0 / 2 / 4 vs RGB-only lips (ablate unused corners; WAVE 3B)
- [x] IMU noise scale vs tip / drain-entry rates (WAVE 3B)
- [x] Front pitch (−12° vs −22°) vs channel visibility (`jims-mower-study --kind pitch`; tip/drain/coverage, not mAP)
- [x] `drain_clearance_m` vs coverage % (scorecard, not mAP) (`--kind clearance`)
- [x] Trimmer offset / radius vs leftover strips (`--kind trimmer`; leftover = 100 − coverage)
- [x] Heuristic vs oracle coverage gap (already sketched in README; `--kind observer` keeps it honest)

## Verify WAVE 1A + 2B

```bash
python -m pip install -e ".[dev]"
pytest
python -m jims_mower.export --steps 8 --seed 7 --cameras 4 --out dataset_out
python -m jims_mower.export --steps 8 --seed 7 --cameras 4 --domain-rand --out dataset_dr
python scripts/train_terrain_seg.py --dataset dataset_out --out terrain_mlp.npz --epochs 12
python -m jims_mower.farm --dry-run --out farm_out
python -m jims_mower.demo --config suburban --steps 12 --out demo_out
# expect demo_out/bev_final.png and demo_out/step_000/bev.png

# WAVE 2A
python -m jims_mower.demo --config geofence_movers --steps 16 --out demo_fence
python -m jims_mower.demo --hand-signals --steps 12 --out demo_signals
python -m jims_mower.demo --steps 16 --save-mission /tmp/mission.npz --out demo_save
python -m jims_mower.mission resume --in /tmp/mission.npz --steps 8 --out demo_resume

# WAVE 3A
jims-mower-bc collect --steps 16 --cameras 4 --out bc_logs
jims-mower-bc train --in bc_logs --out bc_weights.npz
jims-mower-demo --policy bc --bc-weights bc_weights.npz --steps 12 --out demo_bc
jims-mower-rl --smoke --algo reinforce --episodes 1 --steps 4
jims-mower-record --out /tmp/ep --steps 8 --cameras 4
jims-mower-incident /tmp/ep --out /tmp/incident
jims-mower-telemetry /tmp/ep --out /tmp/telemetry.json
jims-mower-owner --config geofence_movers --out owner_overlay.html

# WAVE 3B
jims-mower-study --dry-run --out study_out
jims-mower-bridge /tmp/ep --out /tmp/ep-bridge.json
jims-mower-export-trt --dry-run --out /tmp/trt.json

# WAVE 4
jims-mower-sequence --yards paddock,suburban --dry-run --out seq_out
jims-mower-curriculum
jims-mower-import-yard configs/surveys/example_yard.json
jims-mower-study --kind pitch --dry-run --out study_pitch
jims-mower-farm --dry-run --scenarios suburban --quarantine playground --out farm_q
python -m jims_mower.demo --config narrow_gate --steps 8 --out demo_gate

# WAVE UX-A
jims-mower-demo --steps 12 --out demo_out
jims-mower-viewer --episode demo_out --prepare-only
jims-mower-mesh --out /tmp/yard.glb --seed 7 --cameras 4
jims-mower-teach --steps 8 --out teach_out --cameras 4
jims-mower-demo --policy teach --steps 8 --out demo_teach
jims-mower-demo --profile teach_out/profile.json --steps 4 --out demo_taught

# WAVE UX-C
jims-mower-app --help
jims-mower-app --backend memory --yard configs/yards/example_profile.json --port 8765
jims-mower-owner --live
```

## WAVE UX-C slice 4 — phone owns the live job

- [x] `jims-mower-app --live` wraps `LiveSession` (same `POST /api/live/control` + SSE)
- [x] Phone chrome: status pill, Start / Pause / ESTOP, speed, phase copy, map/cut %, session card
- [x] One command: `jims-mower-owner --live` (port 8766, `acre_yard_demo`)
- [x] Desktop `jims-mower-live` unchanged
- [x] Radio path chips (BT teach / Wi-Fi map / LoRa sparse, simulated)
- [x] Pairing stub before Start; stuck vs dead-motor SOS inject
- [x] App live control contract smoke + existing live tests

## WAVE UX-B — faults + radio sim (append)

Parallel to UX-A. Do not rewrite the WAVE 4 boxes above. Notes:
[`docs/UX_B.md`](docs/UX_B.md). No claimed mAP / FPS / RF range.

- [x] `FaultBus` — kill left/right drive mid-episode (`cmd_ignored` /
      `encoder_stuck` / `open_circuit`); trimmer jam, cam blind, IMU
      freeze, GNSS dropout unified
- [x] `FAULT_IMMOBILISED` (dead motor: zero wheels+trimmer, SOS retrieve)
      vs **stuck** (recovery reverse / pivot / help still runs)
- [x] `info["fault"]` + telemetry `sos` + incident SOS banner
- [x] `jims-mower-selftest` — unloaded spin, IMU still, cam entropy
- [x] Radio sim Wi-Fi → BT → LoRa; heartbeat loss → `limp_home` or
      `stop_beacon` (no RF hardware)


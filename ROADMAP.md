# ROADMAP — no-hardware backlog

Structured checklist for work that does **not** require the physical mower.
Hardware bring-up (wiring, JetPack flash, field calibration) lives elsewhere.
There are no claimed mAP / FPS numbers on this list.

WAVE **1A** is the foundation package. WAVE **1C** expanded yards / renderer.
WAVE **1B** (EKF / record-replay), WAVE **2A** (movers, geofences,
recovery, hand-signal hooks, mission resume), and WAVE **2B** (sim-only
perception: exporter → numpy terrain stub, BEV fuse, temporal filters)
are on `main`. WAVE **3A** (this PR) is learning + ops tooling:
behaviour cloning, an RL scaffold, incident / telemetry / ESTOP, and a
phone overlay stub. Later waves stay unchecked until they land.

## WAVE 3A — learning + ops tooling (done)

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
- [ ] Person / animal / toy categories beyond `MockDetector` blobs
- [ ] Hand-signal classifier behind `HandSignalCurriculum` (optional)
- [ ] Grass coverage net behind `GrassObserver` (replace color heuristic)
- [x] Domain-randomised lighting / wet / dawn from scenario flags — WAVE 1C renderer + WAVE 2B `--domain-rand` export note
- [x] Uncertainty maps on `TerrainEstimate` (relative merge weights, not published scores) — WAVE 1B costmap + WAVE 2B BEV fuse

## Mapping

- [ ] Persistent BEV occupancy (detections + ToF, not god-view)
- [ ] Height-map fusion from RGB back-projection + downward ToF
- [x] Geofence as a first-class map layer (inflate, visualize, plan) (WAVE 2A)
- [x] Multi-session yard memory (same scenario, new seed) (WAVE 2A)
- [ ] Loop-closure *stub* only — do not drop a full SLAM stack in-repo
- [ ] Semantic layers: grass / non-grass / drain / bank / static

## Planning

- [x] Online completeness: resume uncut cells after a person forces a stop (WAVE 2A)
- [x] Dynamic people: temporary block + replan (not just collision terminate) (WAVE 2A)
- [ ] Energy / battery-aware strip order
- [ ] Wet-slope cost (scenario `weather.wet` → extra slow corridor)
- [x] Geofence-aware `in_yard` already exists; planner treats the polygon as blocked margin (WAVE 2A)
- [ ] Multi-yard coverage sequence (paddock then suburban)

## Learning

- [x] Offline behaviour cloning from terrain-policy demos / episode logs (WAVE 3A)
- [x] RL scaffold on the Gymnasium env (REINFORCE / random-search; optional SB3) (WAVE 3A)
- [x] Imitation of `TerrainPolicy` as a baseline, not a claimed SOTA (WAVE 3A)
- [ ] Sim-to-real protocol: same ICD obs keys, no gym renderer on-box
- [ ] Curriculum: flat → suburban → wet_slope → night_dawn

## World library

- [x] Scenario DSL + 6 authored yards beyond `steep_yard` (WAVE 1A)
- [ ] Seasonal variants (long grass, leaf clutter) as scenario overlays
- [ ] Real-yard import (survey polygon → geofence + drain polylines)
- [x] Moving-animal density presets (`world.movers.density`: sparse / default / busy) (WAVE 2A)
- [ ] Narrow-gate / fence-line scenarios
- [ ] Property-scale (40 m+) yards at coarser resolution

## Orin runtime

- [ ] GStreamer / NVMM capture adapter that fills `obs["cameras"]`
- [ ] TensorRT (or similar) behind `Detector` / `TerrainObserver`
- [ ] Complementary filter / small EKF behind `ComplementaryPoseFilter`
- [ ] Watchdog process: stop wheels if vision or IMU stalls
- [ ] YAML extrinsics for the real 4–6 cam rig (same `CameraSpec`)
- [ ] No desktop OpenCV GUI / no gym renderer on-box (see README)

## Safety / ops

- [x] Scorecard gates (tip / drain counts) + farm thresholds (WAVE 1A)
- [x] E-stop contract in the ICD (software latch; hardware is still field) (WAVE 3A)
- [x] Black-box log: IMU, GPS valid bit, advice, wheel commands (WAVE 1B record + WAVE 3A incident)
- [x] Geofence violation → `stop` (already OOB terminate; pre-touch slow) (WAVE 2A)
- [x] Trimmer / living / tip telemetry JSON (headless) (WAVE 3A)
- [ ] Farm flake budget / quarantine a scenario instead of silent skip

## Data tooling

- [x] Dataset exporter (PNG + JSON sidecars + COCO-like index) (WAVE 1A)
- [x] Episode scorecards + JSON report (WAVE 1A)
- [x] Overnight farm + dry-run (WAVE 1A)
- [x] BEV debugger composites in demo output (WAVE 1A)
- [x] Train stub from export (`scripts/train_terrain_seg.py` / `python -m jims_mower.perception.train`) — WAVE 2B
- [x] Replay a recorded episode without re-simulating (WAVE 1B offline + WAVE 3A incident scrubber)
- [x] Label / hazard overlay on the incident viewer (WAVE 3A)
- [ ] Dataset versioning (schema field already `jims_mower.dataset.v1`)

## Design studies

- [ ] Camera count (4 vs 6) vs drain-lip recall on frozen seeds
- [ ] ToF vs RGB-only lips (ablate `stamp_tof_corners`)
- [ ] Front pitch (−12° vs −22°) vs channel visibility
- [ ] `drain_clearance_m` vs coverage % (scorecard, not mAP)
- [ ] Trimmer offset / radius vs leftover strips
- [ ] Heuristic vs oracle coverage gap (already sketched in README; keep honest)

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
```

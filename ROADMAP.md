# ROADMAP — no-hardware backlog

Structured checklist for work that does **not** require the physical mower.
Hardware bring-up (wiring, JetPack flash, field calibration) lives elsewhere.
There are no claimed mAP / FPS numbers on this list.

WAVE **1A** (this PR) is the foundation package: docs, scenario DSL, dataset
export, scorecards, a lightweight farm, and a BEV debugger. Later waves stay
unchecked until they land.

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

- [ ] Replace `classify_terrain_rgb` with a learned drain / lip / bank / grass head (train on exporter labels)
- [ ] Temporal consistency on hazard stamps (track lips across frames)
- [ ] Multi-camera NMS / association for `Detector` (one object, many views)
- [ ] Person / animal / toy categories beyond `MockDetector` blobs
- [ ] Hand-signal classifier behind `HandSignalCurriculum` (optional)
- [ ] Grass coverage net behind `GrassObserver` (replace color heuristic)
- [ ] Domain-randomised lighting / wet / dawn from scenario flags
- [ ] Uncertainty maps on `TerrainEstimate` (no fake confidence scores)

## Mapping

- [ ] Persistent BEV occupancy (detections + ToF, not god-view)
- [ ] Height-map fusion from RGB back-projection + downward ToF
- [ ] Geofence as a first-class map layer (inflate, visualize, plan)
- [ ] Multi-session yard memory (same scenario, new seed)
- [ ] Loop-closure *stub* only — do not drop a full SLAM stack in-repo
- [ ] Semantic layers: grass / non-grass / drain / bank / static

## Planning

- [ ] Online completeness: resume uncut cells after a person forces a stop
- [ ] Dynamic people: temporary block + replan (not just collision terminate)
- [ ] Energy / battery-aware strip order
- [ ] Wet-slope cost (scenario `weather.wet` → extra slow corridor)
- [ ] Geofence-aware `in_yard` already exists; planner should treat the polygon as blocked margin
- [ ] Multi-yard coverage sequence (paddock then suburban)

## Learning

- [ ] Offline behaviour cloning from [`export.py`](src/jims_mower/export.py) wheel logs
- [ ] Offline RL / replay on frozen seeds (scorecard as reward log)
- [ ] Imitation of `TerrainPolicy` as a baseline, not a claimed SOTA
- [ ] Sim-to-real protocol: same ICD obs keys, no gym renderer on-box
- [ ] Curriculum: flat → suburban → wet_slope → night_dawn

## World library

- [x] Scenario DSL + 6 authored yards beyond `steep_yard` (WAVE 1A)
- [ ] Seasonal variants (long grass, leaf clutter) as scenario overlays
- [ ] Real-yard import (survey polygon → geofence + drain polylines)
- [ ] Moving-animal density presets
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
- [ ] E-stop contract in the ICD (software + hardware)
- [ ] Black-box log: IMU, GPS valid bit, advice, wheel commands
- [ ] Geofence violation → `stop` (already OOB terminate; add pre-touch slow)
- [ ] Trimmer interlock telemetry dashboard (headless JSON is enough)
- [ ] Farm flake budget / quarantine a scenario instead of silent skip

## Data tooling

- [x] Dataset exporter (PNG + JSON sidecars + COCO-like index) (WAVE 1A)
- [x] Episode scorecards + JSON report (WAVE 1A)
- [x] Overnight farm + dry-run (WAVE 1A)
- [x] BEV debugger composites in demo output (WAVE 1A)
- [ ] Replay a `frames/*.json` folder without re-simulating
- [ ] Label review: overlay hazard PNG on top-down
- [ ] Dataset versioning (schema field already `jims_mower.dataset.v1`)

## Design studies

- [ ] Camera count (4 vs 6) vs drain-lip recall on frozen seeds
- [ ] ToF vs RGB-only lips (ablate `stamp_tof_corners`)
- [ ] Front pitch (−12° vs −22°) vs channel visibility
- [ ] `drain_clearance_m` vs coverage % (scorecard, not mAP)
- [ ] Trimmer offset / radius vs leftover strips
- [ ] Heuristic vs oracle coverage gap (already sketched in README; keep honest)

## Verify WAVE 1A

```bash
python -m pip install -e ".[dev]"
pytest
python -m jims_mower.export --steps 8 --seed 7 --cameras 4 --out dataset_out
python -m jims_mower.farm --dry-run --out farm_out
python -m jims_mower.demo --config suburban --steps 12 --out demo_out
# expect demo_out/bev_final.png and demo_out/step_000/bev.png
```

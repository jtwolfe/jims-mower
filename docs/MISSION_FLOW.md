# Mission flow: calibrate → explore → review → mow

No claimed mAP / FPS. This is the onboard control contract for a full
yard job, not a short `--policy terrain` clip.

## Why the old demo looked empty

`--policy terrain` builds a boustrophedon on the **current** observer
rasters and starts cutting immediately. A 160-step run at `dt=0.1` is
about 16 seconds and ~8 m of travel. Two to three percent coverage is
what that budget can paint — it is not a mapping failure.

Unknown cells used to stay cheap and traversable. The planner therefore
drew a “full yard” path across space the cameras had never seen.

## Phases

| Phase | Trimmer | What happens |
| --- | --- | --- |
| `calibrate_boundary` | off | Guided perimeter lap. Trail is smoothed into a keep-in polygon + home. |
| `explore` | off | Frontier exploration (known-free cells that touch unknown). Transit only through known-safe cells. Cameras / ToF / body disk grow the observed mask. |
| `review` (`MAP READY`) | off | Closure, completeness, confidence, disconnected regions. Freeze `YardProfile` + map snapshot. |
| `mow` | on (if advice ok/slow) | **Global** coverage over every reachable mowable cell on the frozen map. Local A* only detours; dropped connectors are reported as unreachable. |
| `return_home` | off | Known-safe path to the calibrate home. |
| `complete` / `fault` / `safe` | off | Hold. Existing ESTOP / limp / retrieve still apply. |

CLI:

```bash
# Live owner session (preferred): wall-clock sim + streaming viewer.
# Observed terrain grows from step 0 — do not wait for a finished folder.
jims-mower-live --config acre_yard_demo --speed 5
jims-mower-live --config acre_yard --speed 1
jims-mower-live --config acre_yard --speed 5 --phase-budget 0.4
jims-mower-live --fast --speed max --steps 40 --prepare-only --out live_tiny

# Post-hoc scrub of a finished episode (old path):
jims-mower-mission-demo --config golf_rough --out mission_out
jims-mower-mission-demo --config acre_yard --out mission_acre
jims-mower-mission-demo --fast --out mission_fast
jims-mower-viewer --episode mission_out
```

`--fast` loads `mission_tiny` (6×5 m) and short phase budgets so CI can
reach `mow` / `complete`. The full acre / golf run is longer and is
**not** a benchmark.

`--speed` on `jims-mower-live` is a wall-clock multiplier (`1`, `2`,
`5`, or `max`). The sim sleeps so each step lands near `dt / speed`
(env `dt` is 0.10 s). `max` is unpaced for CI.

## Acre live pacing (`acre_yard_demo`)

`acre_yard` is a ~226 m fence. At `dt=0.1` that is many minutes of
calibrate even at `--speed 5`, and a laptop often cannot hold 5×
because each acre step is expensive. **`acre_yard_demo` is a live demo
profile**, not a smaller world:

* Same 70×58 m layout, pond, sheds, paths, drains, and **true-height
  physics**.
* Slightly tighter authored keep-in (still covers the pond and sheds).
* Calibrate **confirms** the fence (`mission.calibrate_confirm_m: 28`)
  then closes on the authored keep-in. It does **not** require a full
  acre lap before EXPLORE. That is documented demo pacing, not a
  physics lie.
* Faster calibrate cruise / stride. Live also downsamples acre cameras
  to 48×36 so `--speed 5` can keep up.
* **Demo map-ready** at `explore_complete: 0.42` (or no remaining
  frontier at `0.28`) and `max_explore_steps: 1400`. `acre_yard` stays
  at `0.72` / full fence lap. That is documented demo pacing so MAP
  READY → MOW can happen in minutes, not an hour.
* `--phase-budget 0.4` scales the phase caps on any yard the same way.

Jamie **phone** command (owns the live job):

```bash
jims-mower-owner --live
# open http://127.0.0.1:8766/
```

Same session: `jims-mower-app --live --config acre_yard_demo --port 8766`.
Pair the BT stub, then Start / Pause / ESTOP / speed / MAP READY →
Start mow. Fog + observed preview is in the phone chrome; deep-link
`/viewer` is the desktop World Viewer on the same process.

Jamie **desktop** command (viewer chrome only):

```bash
jims-mower-live --config acre_yard_demo --speed 5
```

Open the viewer: session is **idle** (yard unknown) until **Start**.
Expect **CALIBRATE → EXPLORE → MAP READY → MOW** in a few minutes
wall-clock at `--speed 5`. MAP READY holds ~2 s (or **Start mow**).
Full acre mow to completion is still a manual `acre_yard` run, not CI.

Owner bar: Start / Pause / Resume, speed `1× 2× 5× max`, **Start mow**
/ Re-explore at MAP READY, ESTOP. Phone adds radio-path chips (BT teach
/ Wi-Fi map / LoRa sparse, simulated) and stuck vs dead-motor SOS
injection. Copy reads like a product (“Calibrating boundary…”,
“Exploring unknown yard…”, “Map ready — start mow?”, “Mowing…”).
Home / done writes `session_summary.json` (map %, planned/reachable,
cut %, skips, duration).

## Unknown-space semantics

`ObservedMap` keeps explicit `observed` / `explored` / `free` / `hazard`
/ `structure` / `elevation` / `confidence`.

* Unknown is **not** mowable and **not** safe transit.
* Frontiers are known-free cells adjacent to unknown, inside keep-in.
* Observer rasters are copied only onto cells this robot has stamped
  (camera ground hits + body/ToF disk). Authored/god-view structure in
  `obs["structure"]` is not a control input outside that mask.
* Map-ready when `observed` fraction ≥ `mission.explore_complete`, or
  there are no frontiers and the fraction is at least
  `mission.explore_no_frontier`.

## Coverage plan

`plan_coverage` now reports:

* `planned_mowable_cells`
* `reachable_mowable_cells`
* `unreachable_mowable_cells`
* `planned_coverage_fraction`
* `skipped_segments`

A disconnected island the A* connector cannot reach is counted, not
deleted. Strip heading uses the mowable principal axis, rotated toward
the contour when a mean grade is present.

IMU tip-stop on a ridge during `mow` reverses, pivot-reverses, then
skips a short waypoint cluster and local-replans. Ridge IMU is treated
as `slow` for the limp machine so the job does not park. Abort-to-home
needs 48 skips **and** 80 mow steps — leftover: long ridges can still
chew strips; a contour-following skip is follow-up.

Structures (path / building / bunker / bed / green) stay no-mow or
blocked exactly as in `docs/TERRAIN_MAPS.md`. Physics and evaluation
still use the true height field; control does not.

## Fog of war + observed terrain (owner map)

Unknown cells are **not** mowable and **not** safe. The owner view must
match that: a finished god-view `yard.glb` is the true height field
used by physics, not what the robot has learned.

**Learning terrain** in the live viewer means a **partial observed
elevation mesh** that grows with `ObservedMap`. Camera / body stamps
lift cells into a 3D surface (grade, sheds as low boxes, ponds as
slightly sunken cyan water). Unknown stays a dark pad + fog veil —
not painted holes on a finished mesh. Physics still uses the true
field; owner view and `MissionPolicy` use observed elevation only.

Live / owner mode **hides the true height-field mesh** by default.
Toggle **true elev (god-view debug)** to reveal the physics mesh —
evaluation only, same idea as **observer vs true elev**. The `yard.glb`
on disk is still the true field so debug / replay can show it.

Acre rasters coarsen to ≤96 px (2D) / ≤48 (observed mesh). Mesh
rebuilds every `--observed-mesh-stride` steps (default 8). Cameras are
JPEG-throttled (~2.5 Hz). Full acre explore→mow need not finish in CI.

## Viewer

### Live (owner session)

`jims-mower-live` starts `MissionPolicy` and the World Viewer together.
Open `http://127.0.0.1:8765/` immediately — the job is **idle** until
Start. Stdlib HTTP + SSE (`GET /api/live`) pushes pose, phase, owner
copy, map %, cut %, frontiers, plan (after freeze), and URLs for the
latest fog / observed / **observed mesh** / coverage / camera frames.
`POST /api/live/control` is the owner bar (start / pause / resume /
speed / start_mow / reexplore / estop / hold / yard / pair / inject).
The phone app (`jims-mower-app --live`) proxies the same endpoints —
do not fork `MissionPolicy`. History is also flushed into `live_out/`
so you can scrub later. You do not need a finished episode folder to
watch the robot.

### Scrub (finished episode)

`mission.json` plus `maps/snap/*.png` (stride-limited). The World Viewer
shows a phase bar (`CALIBRATE` … `DONE`), a growing observed overlay,
frontier markers, the explore route, then the frozen global mow plan,
and map / reachable / unreachable / cut percentages.

True elevation remains available under **true elev (god-view debug)** /
**observer vs true elev** — those toggles are evaluation, not the
controller’s map.

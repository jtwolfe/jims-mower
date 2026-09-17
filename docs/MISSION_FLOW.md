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
# Default speed is 1×. 2 / 5 / max stay on the owner bar.
jims-mower-live --config acre_yard_demo
jims-mower-live --config acre_yard
jims-mower-live --config acre_yard --speed 5 --phase-budget 0.4
jims-mower-live --fast --speed max --steps 40 --prepare-only --out live_tiny
# Local full-acre explore→mow (not CI):
jims-mower-live --config acre_yard --speed max --prepare-only --steps 20000 --out live_acre_full

# Post-hoc scrub of a finished episode (old path):
jims-mower-mission-demo --config golf_rough --out mission_out
jims-mower-mission-demo --config acre_yard --out mission_acre
jims-mower-mission-demo --fast --out mission_fast

# Laptop practice of the FIELD_TEST scorecard (not a field test):
jims-mower-field-dryrun --out artifacts/field_dryrun/scorecard.yaml
jims-mower-viewer --episode mission_out
```

`--fast` loads `mission_tiny` (6×5 m) and short phase budgets so CI can
reach `mow` / `complete`. The full acre / golf run is longer and is
**not** a benchmark.

`--speed` on `jims-mower-live` is a wall-clock multiplier (`1`, `2`,
`5`, or `max`). The sim sleeps so each step lands near `dt / speed`
(env `dt` is 0.10 s). `max` is unpaced: physics steps as fast as the
CPU can, and live map / observed-mesh / PNG / disk flush is coarsened
so owner-view encode does not drop acre to ~2 Hz.

## Acre live pacing (`acre_yard_demo`)

`acre_yard` is a ~226 m fence. At `dt=0.1` that is many minutes of
calibrate even at `--speed 5`, and a laptop often cannot hold 5×
because each acre step is expensive. **`acre_yard_demo` is a live demo
profile**, not a smaller world and not a partial-coverage gate:

* Same 70×58 m layout, pond, sheds, paths, drains, and **true-height
  physics**.
* Slightly tighter authored keep-in (still covers the pond and sheds).
* Calibrate **confirms** the fence (`mission.calibrate_confirm_m: 28`)
  then closes on the authored keep-in. It does **not** require a full
  acre lap before EXPLORE. That is documented demo pacing, not a
  physics lie.
* Faster calibrate cruise / stride. Live also downsamples acre cameras
  to 48×36 so `--speed 5` can keep up.
* **Owner map-ready is the full taught fence** on both `acre_yard_demo`
  and `acre_yard`: `explore_complete: 1.0` (or no remaining frontier at
  `0.90`) and `max_explore_steps: 4000`. A step cap is not MAP READY
  while reachable frontiers remain. There is no 30% / 420-step demo
  early exit and no owner **Full explore** switch.
* **Mow** plans the full keep-in (`mow_complete_frac: 0.0`). Owner cut %
  is that job fraction (not world-grass %). `cover_radius_m: 0.70`
  paints the coarse raster so strips show; the physical trimmer stays
  0.16 m. IMU tip-stop skips a cluster and keeps mowing.
* Cold start and **Reset** land at **1×**. Speed buttons stay for 2 / 5
  / max. Reset keeps the taught fence and clears tip latch, learned
  blockages, explore/mow progress, and Hold-safe.
* `--phase-budget 0.4` scales the phase caps on any yard the same way.
  CI proves explore→mow on a taught pocket / `mission_tiny`, not the
  whole acre. Local acre long-run:
  `jims-mower-live --config acre_yard --speed max --prepare-only --steps 20000`.

## First-run teach vs demo confirm fence

| Path | Fence | After Start |
| --- | --- | --- |
| **First-run** (`--live`, Teach → Save) | Owner-taught `YardProfile` (`jims_mower.yard.v1`) — scribble / short-trail fences are repaired to the authored yard-scale ring | Skip authored calibrate. Explore fog → MAP READY → mow. |
| **Demo** (`acre_yard_demo`, Start only) | Authored keep-in, `calibrate_confirm_m: 28` | Short confirm lap, then the same explore → MAP READY → mow. |
| **Full acre** (`--config acre_yard`) | Authored ~226 m fence, full lap | Same phases; long. Not CI. |

Jamie **phone** command (owns the live job):

```bash
jims-mower-owner --live
# Teach boundary → Save yard → Start job
jims-mower-owner --live --first-run
# open http://127.0.0.1:8766/
```

Same session: `jims-mower-app --live --config acre_yard_demo --port 8766`.
Pair the BT stub, teach (or skip), then Start / Pause / ESTOP / speed /
MAP READY → Start mow. The phone live map is map-first: a large **Mapping yard** (explore /
calibrate) vs **Mowing** chip, live trail, explore/frontier or coverage
plan overlay, and Map% / Cut% that swap emphasis with the phase. That
path overlay is on the 2D preview — you do not need the 3D fog viewer
to see where the robot has been or where it plans to go. Deep-link
`/viewer` is still the desktop World Viewer on the same process
(explore+frontiers+trail default on while mapping; plan+trail+coverage
default on while mowing). No mAP / RF claims.

`acre_yard` vs `acre_yard_demo`: same ~1-acre physics world (70×58 m,
pond, sheds, paths) and the same full-fence explore/mow gates. Use
**`acre_yard_demo`** for the live phone / laptop loop (short confirm or
a taught profile). Use **`acre_yard`** when you want a full fence lap
before EXPLORE. Neither is a coverage benchmark. CI uses `--fast`
(`mission_tiny`) plus a taught pocket — no full-acre mow.

Jamie **desktop** command (viewer chrome only):

```bash
jims-mower-live --config acre_yard_demo
```

Open the viewer: session is **idle** (yard unknown) until **Start**.
Default speed is **1×**. Expect **CALIBRATE → EXPLORE → MAP READY →
MOW → HOME → DONE** against the full taught fence (map target 100%).
MAP READY holds ~2 s (or **Start mow**). Cut % should rise while
mowing; the phone session card shows map / planned / cut / skips /
duration, then idle with the yard still loaded. Full acre wall-clock
to completion is a manual `acre_yard --speed max --prepare-only
--steps 20000` run, not CI.

Owner bar: Start / Pause / Resume, speed `1× 2× 5× max` (cold start
and Reset = 1×), first-class **Explore / Mow / Return** (manual phase
overrides — do not wait for auto MAP READY), and **Reset** (fresh job,
fence kept). There is no Full explore toggle. ESTOP stays.
Phone adds radio-path chips (BT teach / Wi-Fi map / LoRa sparse,
simulated), stuck / dead-motor SOS, and **low-SOC inject** (dock →
charge → resume leftover plan). Copy reads like a product
(“Calibrating boundary…”, “Seeking frontier · map 51% of target 100%
· step 200/4000”, “Map ready — start mow?”, “Mowing…”, “Low battery —
returning to charge”, “Charging…”, “Resuming mow”).
`explore_reason` is always on the live snapshot while mapping.
MAP READY is not “Hold — safe” (that is SafeState / ESTOP). A review
with 0 mowable cells stays on “Fence too small — re-teach the keep-in.”
`session_summary.json` tracks the live phase (not a leftover calibrate
card from Teach → Save). After MAP READY the phone draws **area types**
(grass / path / sand / building / water / drain / beds / keep-out) plus
the automatic **mowable mask**, learned **Blocked / no-go** cells, and
coverage-plan polyline.

## Unknown-space semantics

Learned invisible / dynamic no-go (no-progress, tip, collision, slip)
is a separate `blockage` raster — see
[`NAV_BLOCKAGES.md`](NAV_BLOCKAGES.md).

`ObservedMap` keeps explicit `observed` / `explored` / `free` / `hazard`
/ `structure` / `elevation` / `confidence` / `elevation_set` / `blockage`.

* Unknown is **not** mowable and **not** safe transit.
* Learned **blockage** is known no-go (lethal). It is not fog and not
  a drain/shed class — the phone paints it as “Blocked / no-go learned.”
* Frontiers are known-free cells adjacent to unknown, inside keep-in.
  A frontier A* can reach is not “yard reachable.” Failed progress
  stamps a blockage, skips that lip, and counts it unreachable.
* **Cameras** grow the seen mask (ground-plane hits). That is occupancy,
  not height.
* **IMU pitch/roll** is local chassis attitude / tip. It is a slow prior
  for a neighborhood around the robot, not a global hinge.
* **Elevation** is fused from wheel/pose `z` and locked neighbours on
  first stamp in the neighborhood, then frozen. Already mapped cells
  do not leap, and the growing edge does not take a new IMU plane.
* If the rig has a true 6–12 cm forward stereo pair, gym synthetic
  stereo stamps metric local elev in the 0.8–4 m band onto unlocked
  cells. Default look-around cams are not a pair (no-op). This is not
  COLMAP.
* MAP READY copies a `YardSnapshot` and `lock_observed()` on the live
  map so later observer / stereo stamps do not flop frozen cells.
* Authored/god-view structure in `obs["structure"]` is not a control
  input outside the observed mask.
* Map-ready when `observed` fraction ≥ `mission.explore_complete`, or
  there are no frontiers and the fraction is at least
  `mission.explore_no_frontier`. A step cap is not MAP READY while
  reachable frontiers remain. After a blockage stamp the job keeps
  seeking **other** frontiers; it does not spin on the same lip.

## Coverage plan

`plan_coverage` now reports:

* `planned_mowable_cells`
* `reachable_mowable_cells`
* `unreachable_mowable_cells`
* `planned_coverage_fraction`
* `skipped_segments`

A disconnected island the A* connector cannot reach because of a real
obstacle (drain / pond / shed) is counted, not deleted. Fog islands —
observed patches cut off only by unknown — are dropped from the planned
set so a taught acre rectangle does not report hundreds of fake
unreachable cells. Strip heading uses the mowable principal axis,
rotated toward the contour when a mean grade is present.

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
elevation mesh** that grows with `ObservedMap`. Camera stamps lift the
fog (seen mask) — the orange/green/gray *overlay* footprint is not the
3D hinge. Body / pose `z` plus locked neighbours lift cells into a 3D
surface once they have a height sample (grade, sheds as low boxes,
ponds as slightly sunken cyan water). The mesh must not flop when the
IMU tips: old cells keep their first height, and the growing edge
inherits neighbours instead of a new attitude plane. The vertex grid
stays fixed (triangles grow). Unknown stays a dark pad + fog veil.
Physics still uses the true field; owner view and `MissionPolicy` use
observed elevation only.

Live / owner mode **hides the true height-field mesh** by default.
Toggle **true elev (god-view debug)** to reveal the physics mesh —
evaluation only, same idea as **observer vs true elev**. The `yard.glb`
on disk is still the true field so debug / replay can show it.

Acre rasters coarsen to ≤96 px (2D). Observed mesh stays ≤48 on a
**fixed** vertex grid (high speed only skips rebuilds; it does not
swap 32↔48). Mesh rebuilds every `--observed-mesh-stride` steps
(default 8). Cameras are JPEG-throttled (~2.5 Hz). Full acre
explore→mow need not finish in CI.

## Viewer

### Live (owner session)

`jims-mower-live` starts `MissionPolicy` and the World Viewer together.
Open `http://127.0.0.1:8765/` immediately — the job is **idle** until
Start. Stdlib HTTP + SSE (`GET /api/live`) pushes pose, phase, owner
copy, map %, cut %, frontiers, plan (after freeze), a compact
`path_overlay` (downsampled trail / plan / frontiers + pose + mode
banner), and URLs for the latest fog / observed / **observed mesh** /
coverage / camera frames.
`POST /api/live/control` is the owner bar (start / pause / resume /
speed / explore / mow / return / reset / start_mow / reexplore /
estop / hold / yard / pair / inject). `inject kind=low_soc` drops gym
SOC so the mid-job dock-and-resume path can be tested.
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

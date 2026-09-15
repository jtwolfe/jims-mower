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
jims-mower-mission-demo --config golf_rough --out mission_out
jims-mower-mission-demo --config acre_yard --out mission_acre
jims-mower-mission-demo --fast --out mission_fast
jims-mower-viewer --episode mission_out
```

`--fast` loads `mission_tiny` (6×5 m) and short phase budgets so CI can
reach `mow` / `complete`. The full golf run is longer and is **not** a
benchmark.

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

Structures (path / building / bunker / bed / green) stay no-mow or
blocked exactly as in `docs/TERRAIN_MAPS.md`. Physics and evaluation
still use the true height field; control does not.

## Viewer

`mission.json` plus `maps/snap/*.png` (stride-limited). The World Viewer
shows a phase bar (`CALIBRATE` … `DONE`), a growing observed overlay,
frontier markers, the explore route, then the frozen global mow plan,
and map / reachable / unreachable / cut percentages.

True elevation remains available under **observer vs true elev** — that
toggle is evaluation, not the controller’s map.

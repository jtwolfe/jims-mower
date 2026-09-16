# WAVE UX-A — World Viewer + Teach Boundary

Local, sim-only UI. **No claimed mAP / FPS.** The mesh is a decimated
height field, not a photogrammetry product.

## World Viewer

After a demo or record run:

```bash
# Live owner session (observed terrain + fog). Jamie demo command:
jims-mower-live --config acre_yard_demo --speed 5
jims-mower-live --config acre_yard --speed 5
# open http://127.0.0.1:8765/  — animates while the sim runs

# Finished-episode scrub (post-hoc). Default --steps is 160 (not a 12-step
# smoke). Camera PiP folders land every 10 steps (--cam-stride).
jims-mower-demo --out demo_out
jims-mower-demo --steps 160 --cam-stride 10 --out demo_out
jims-mower-demo --config gradient_yard --steps 160 --out demo_gradient
jims-mower-viewer --episode demo_out
# open http://127.0.0.1:8765/
```

**Live vs scrub.** `jims-mower-demo --steps 80` + the viewer is a
scrubber of a pre-recorded run. The mesh is the full true height field
from step 0, so the yard looks already known. `jims-mower-live` is the
owner loop: the same `MissionPolicy` paced to wall-clock (`--speed`
`1`/`2`/`5`/`max`), SSE (`GET /api/live`) pushing pose / phase / map,
a **growing observed elevation mesh**, and a **fog veil** over unknown
cells. That is what “learning terrain” looks like — 3D surface where
cameras have stamped, fog everywhere else. Physics still uses the true
height field; the owner map and the controller do not. Toggle **true
elev (god-view debug)** to see the finished mesh. Acre streams are
coarsened (≤96 px rasters, ≤48 observed mesh, JPEG cameras) so the
browser does not OOM.

`acre_yard_demo` is a **live demo profile**: same ~1 acre features and
physics as `acre_yard`, slightly tighter keep-in, short calibrate
confirmation (`calibrate_confirm_m: 28`), and a **demo**
`explore_complete: 0.42` so `--speed 5` can reach **MAP READY then
MOW** in a practical window. `acre_yard` stays at `0.72` and a full
fence lap. It is not a coverage benchmark. See
[`MISSION_FLOW.md`](MISSION_FLOW.md). The viewer owner bar starts,
pauses, sets speed, holds MAP READY for a 2 s beat (or **Start mow**),
and ESTOPs locally.

The demo (and `jims-mower-record`) write a viewer bundle next to the
usual PNGs:

| File | What it is |
| --- | --- |
| `viewer.json` | manifest (`jims_mower.viewer.v1`) |
| `yard.glb` / `yard.obj` / `yard.json` | low-poly terrain |
| `maps/coverage.png` (+ `.npy`) | cut / uncut / non-grass |
| `maps/hazard.png` | observer/oracle hazard classes |
| `maps/occupancy.png` | detection occupancy |
| `poses.json` | scrubbable pose trail |
| `step_XXX/cam_*.png` | camera PiP frames |

The server is stdlib `http.server`. Static HTML/JS lives in
[`src/jims_mower/viewer_static/`](../src/jims_mower/viewer_static/) and
loads three.js from a CDN. Toggles: coverage paint, hazard, occupancy,
plan overlay, pose, geofence. Health / radio chips are **placeholders**.

Headless / CI (no listen):

```bash
jims-mower-viewer --episode demo_out --prepare-only
```

Rebuild the mesh only:

```bash
jims-mower-mesh --out yard.glb --config suburban --seed 7
jims-mower-mesh --out yard_grade.glb --config gradient_yard --seed 7
jims-mower-mesh --episode demo_out --out demo_out/yard.glb
```

The mesh stores true metres (`z_scale=1`). The viewer applies a 4× Y
lift so a ~5% property grade and 10–20 cm drains read on a 12 m yard.
Plan overlays use the same world XY and the height-field `z` (or
`maps/elevation.json`) so the path sits on the mesh, not a flat z=0
plane. Toggle **observer vs true elev** for the heuristic error overlay.

## Terrain (yard-scale gradient)

Default / suburban yards are a **gentle planar slope across the property**,
not a flat pad with a few local banks. `world.terrain.base_gradient`:

| Key | Role |
| --- | --- |
| `slope_rad` | Planar grade (radians). Optional `slope_pct` (percent) overrides it |
| `yaw_rad` | Direction of ascent |
| `undulation_m` | Long-wavelength sine + weak quadratic dish |

Drains and banks are carved **on top** of that tilt. `steep_yard` uses a
stronger grade. `gradient_yard` is a showcase: ascent along +x, one drain
running along y so it crosses the slope. The curriculum `flat` scenario
zeros the gradient. Attitude / tip / drain checks still sample the height
field, so pitch and roll are nonzero on a pure grade.

The heuristic observer recovers that grade from IMU pitch/roll + pose
(`elevation_prior`). See [`TERRAIN_MAPS.md`](TERRAIN_MAPS.md) for the
observer-vs-physics failure mode, path/building layers, and golf yards
(`golf_rough`, `golf_fairway_snip`, `acre_yard`).

## Teach Boundary

Drive a perimeter in sim, smooth the pose trail to a keep-in polygon,
edit it in the viewer, save a `YardProfile`.

```bash
# dedicated CLI
jims-mower-teach --steps 80 --out teach_out --cameras 4

# or via the demo
jims-mower-demo --policy teach --steps 80 --out teach_out

jims-mower-viewer --episode teach_out
```

In the viewer:

1. Drag yellow keep-in vertices.
2. **Add keep-out** for flower beds / play equipment, then drag those corners.
3. **Save profile** → `teach_out/profile.json` (`jims_mower.yard.v1`).

Load the taught fence back into the gym:

```bash
jims-mower-demo --profile teach_out/profile.json --steps 20 --out demo_taught
# or treat the JSON as a scenario
jims-mower-demo --config teach_out/profile.json --steps 20 --out demo_taught
```

`YardProfile` carries keep-in / keep-out, home pose, mesh filename, and
the raw trail. Physics still uses the height field; the profile is the
geofence + spawn, not a new renderer.

A short `--steps` run may not finish the rectangle. The keep-in then
falls back to the planned perimeter waypoints so you still get an
editable fence. Use more steps (or a smaller yard) to record a full loop.

## What this is not

- Not an owner phone app. WAVE UX-C (`jims-mower-app`, [`UX_C.md`](UX_C.md))
  is the local phone shell; it **reuses** this viewer at `/viewer` and the
  same `YardProfile` / `mesh_to_payload` — no second three.js stack.
- Not SLAM, not a measured mesh quality score.
- Not a live radio / BMS dashboard — those chips stay empty until hardware.

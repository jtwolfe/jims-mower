# WAVE UX-A — World Viewer + Teach Boundary

Local, sim-only UI. **No claimed mAP / FPS.** The mesh is a decimated
height field, not a photogrammetry product.

## World Viewer

After a demo or record run:

```bash
jims-mower-demo --steps 40 --out demo_out
jims-mower-viewer --episode demo_out
# open http://127.0.0.1:8765/
```

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
jims-mower-mesh --episode demo_out --out demo_out/yard.glb
```

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

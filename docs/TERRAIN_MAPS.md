# Observer maps vs physics, paths, and golf yards

No claimed mAP / FPS. This note is the routing / mapping contract.

## Observer-vs-physics failure mode

Physics (`HeightField`) is the true yard: a property-scale planar grade
plus drains, banks, undulation, and authored structures. Wheels sit on
that surface. The coverage planner does **not** see it.

`HeuristicTerrainObserver` (demo default) builds the maps in `obs`:

| Layer | What the robot is allowed to use |
| --- | --- |
| `elevation` / `slope` / `hazard` | RGB + ToF + **local** IMU / pose. Optional gym stereo stamp if the rig is a true 6–12 cm pair. No god-view DEM. Not COLMAP. |
| `elevation_prior` | Slow plane from pose `z` + a short XY fit, with IMU pitch/roll as a *weak* attitude prior. Costmap floor only. |
| `structure` | Authored path / building / bunker / garden plus a colour stub. |

### Sensor model (cameras / IMU / height)

A real mower does **not** rebuild a global tilted plane of the whole yard
from instantaneous chassis tip.

1. **Cameras** stamp where the ground was seen (footprint / ground-plane
   hits) → `ObservedMap.observed`. That is occupancy of known cells,
   not a height field.
2. **Local height at the robot**: wheel / pose `z` (and short-range ToF)
   is a local sample. The first neighborhood is `pose.z`. After that,
   new cells inherit locked neighbours (85%) plus a little `pose.z`
   (15%) so a climb creeps. They do **not** take the current IMU plane.
3. **IMU tilt (pitch/roll, not yaw)** is the local surface normal under
   the chassis. Use it for tip risk (`imu_advice`). It is **not** a
   license to re-orient the mapped sheet — globally *or* as a local
   flop of the bright growing patch.
4. **Already-observed heights stay put.** First stamp sets
   `elevation_set`. Old cells must not leap when the robot tips on a
   ridge. The growing edge must not reshape from a new attitude either.
5. **Near-field stereo** (true 6–12 cm pair, or gym synthetic stereo)
   may overwrite unlocked cells with metric local elev in the 0.8–4 m
   band. MAP READY `lock_observed()` then freezes the mow map. Not
   COLMAP. Default gym look-arounds are not a pair.

`elevation_prior` may still be a yard-scale raster so the costmap can
floor a flattened CV slope. The **owner / control** mesh is
`ObservedMap.elevation` on cells with `elevation_set` — a growing
surface, not a sheet hinged to the IMU. Physics still uses the true
height field.

**What went wrong on `gradient_yard` (first pass):** the heuristic used
to stamp only a local disk of `pose.z` and treat brown-ish pixels as
drain lips. On a ~0.14 rad grade the true field is about ±0.85 m, but
the observer map sat near `[-0.14, 0]` (MAE ≈ 0.43 m) and painted
thousands of false lips. The costmap then blocked or detoured as if the
yard were flat and pitted.

**What went wrong after that (flopping sheet / local patch):**
`paint_planar_grade` wrote the current IMU plane into **every**
elevation cell each step, and `ObservedMap.ingest_observer` copied
`obs["elevation"]` onto **all** seen cells. Live owner view on acre
often read as a **local** flop — the bright orange/green patch
reshaping and its edge expanding on each `mesh_seq` — more than a
single rigid yard hinge. 4× viewer relief plus a full mesh replace
(and a 32↔48 `mesh_side` swap, 696→1920 verts) made that rewrite
obvious. The broad surface could still look planar when orbited.

**Fix:**

1. Recover a *slow* plane from seated pose `z` + a short XY fit (IMU
   attitude is blended weakly, never “whichever tilt is larger”). Keep
   that raster as `elevation_prior`. Paint height only in a local
   neighborhood; freeze cells once sampled.
2. Gate isolated lip stamps: a lip must sit next to a channel (or a ToF
   drop). Tighten the brown heuristic so shaded grass / dirt is not a ditch.
3. Optional `planner.blend_elevation_prior`: the costmap floors observer
   slope with the low-frequency prior so a flattened CV map cannot fight
   physics.
4. Viewer: plan waypoints carry world `z` from the same height field as
   the mesh; `relief_scale` is applied to both. The observed-mesh
   **vertex grid stays fixed** for the session (grow triangles, freeze
   interior Y, do not swap 32↔48 mid-run). Fog / observed overlays stay
   flat (footprint colour ≠ mesh hinge). Toggle **observer vs true
   elev** for the error overlay.

Oracle maps still copy the height field (training / eval only).

A true forward stereo pair (see `configs/orin/extrinsics_stereo.yaml`)
can stamp metric local elev onto observed, **unlocked** cells. MAP READY
locks those cells so the frozen mow map does not jitter. Default gym
look-arounds are not a pair. Learned monocular depth is a future prior,
not the live metric source.

## Paths and human structures

First-class layers, not a second geofence:

| Class | YAML | Coverage | Costmap |
| --- | --- | --- | --- |
| `path_paved` | `paths:` polyline + `width_m` | no-mow | heavy cost (`planner.path_cost`) |
| `building` | `buildings:` polygon | no-mow | blocked |
| `bunker` | `bunkers:` `x,y,radius_m,depth_m` | no-mow | blocked (sand bowl) |
| `garden_bed` | `garden_beds:` polygon | no-mow | blocked |
| `green` | `greens:` polygon | no-mow / no-trimmer | blocked |
| `pond` | `ponds:` polygon + `depth_m` | no-mow | blocked + geofence keep-out |

Rasters: `obs["structure"]` / `info["structure"]` (see `STRUCTURE_NAMES`).
Semantic extras: `path_paved`, `building`, `bunker`, `garden_bed`.

The heuristic CV stub paints grey ribbons (`PATH_RGB`) and sand
(`BUNKER_RGB`) when those colours hit a camera. Authored YAML still wins
for planning; the stub is the onboard stand-in.

Fence / keep-out polygons stay on `geofence`. Use that for a property
line; use `paths` / `buildings` for hard surfaces inside the yard.

`ponds:` is a **water keep-out**, not a hydro simulator. The polygon is
painted as standing water (`TERRAIN_POND`), merged into geofence
keep-out, and excluded from coverage. Pair it with `n_puddles` and
authored drains for leftover wet spots. There is no flow or water-level
physics.

## Golf scenarios

```bash
jims-mower-demo --config golf_rough --steps 80 --out demo_golf
jims-mower-demo --config golf_fairway_snip --out demo_fairway
jims-mower-mission-demo --config golf_rough --out mission_out
jims-mower-viewer --episode mission_out
jims-mower-mesh --out golf.glb --config golf_rough --seed 3
```

`golf_rough` is multi-scale undulation + swales, a cart-path polyline, a
sand bunker bowl, a shed polygon, and a garden bed. `golf_fairway_snip`
is a smaller clip with a gentler roll, a path, a bunker, and a no-mow
green.

Layouts `golf_rough` / `golf_fairway` / `acre_yard` turn off random
drains/banks so the authored hard areas stay readable.

## Acre yard

Suburban/rural **~1-acre** block for explore/mow. Footprint **70×58 m**
(4060 m² ≈ **1.003 acre**). Cells are **0.50 m** → 140×116 = **16,240**
cells (same “stay tractable” idea as `property_scale` at 48×40 m @
0.40 m). Not a coverage benchmark.

| Layer | What is authored |
| --- | --- |
| Trees | Naturalistic clusters (NW / east / south) plus lone trees |
| Bushes | `garden_beds` along the west fence, back fence, front edge, and one island |
| Sand | Three `bunkers` (play bowl + two smaller traps) |
| Rough | `layout: acre_yard` — multi-scale undulation, swales, mild grade, plus drains and a bank |
| Paths | Paved ribbons: gate → lawn → shed, plus west and outbuilding spurs |
| Sheds | Main shed (NE) and a small outbuilding (SE) |
| Water | `ponds:` keep-out (NW) + two drains + `n_puddles: 3` leftover wet spots |

Live owner view paints ponds as saturated cyan (slightly sunken) and
sheds as tan low boxes on the **observed** mesh once cameras stamp
those cells. That is a readability cue, not photoreal water.

```bash
jims-mower-demo --config acre_yard --steps 80 --out demo_acre
# Live owner demo (same acre features; short calibrate confirm → EXPLORE):
jims-mower-live --config acre_yard_demo --speed 5
jims-mower-live --config acre_yard --speed 5
jims-mower-mission-demo --config acre_yard --out mission_acre
jims-mower-viewer --episode mission_acre
jims-mower-mesh --out acre.glb --config acre_yard --seed 3 --stride 4
```

Do **not** expect CI or a short demo to mow the whole acre. Strip
spacing is 1.20 m and mission step caps are raised so explore/mow can
move, not so the job finishes. Mesh export should use `--stride 3` or
`4` (default stride 2 is still loadable, just heavier).

`acre_yard_demo` is a **live demo profile**: same 70×58 m physics and
authored pond/sheds, slightly tighter keep-in, `calibrate_confirm_m: 28`,
and `explore_complete: 0.30` / `max_explore_steps: 420` so `--speed 5`
and `--speed max` can reach **MAP READY then MOW** in minutes.
`acre_yard` stays at `0.72` / 4000 and a full fence lap.
A first-run teach (`jims-mower-owner --live` → Teach → Save) replaces
that authored confirm with a `YardProfile` keep-in; the acre physics
world does not shrink. The live viewer shows a growing **observed**
elevation mesh (unknown stays fog). Physics still uses the true field.

### DEM hook (optional, not used in CI)

`world.terrain.dem_path` loads a vendored `.npy` height patch, resamples
it to the yard grid, mean-centers it, and **adds** it to the procedural
field. CI never downloads a public DEM.

A 32×32 synthetic fixture ships at
`src/jims_mower/data/dem/tiny_patch.npy` (`bundled_dem_path()`). Golf
scenarios stay procedural by default — point `dem_path` at that file
only when you want the hook exercised.

```yaml
world:
  terrain:
    dem_path: src/jims_mower/data/dem/tiny_patch.npy   # 2-D float32
    multi_scale_amp_m: 0.08
    swale_amp_m: 0.06
```

Public sources if you author a larger crop offline (do not fetch in CI):

| Source | Notes |
| --- | --- |
| [ELVIS](https://elevation.fsdf.org.au/) | Best for real AU yards; LiDAR where available (Geoscience Australia). |
| [GA LiDAR 5 m DEM](https://ecat.ga.gov.au/geonetwork/eng/api/records/22be4b55-2465-4320-e053-10a3070a5236) | Via ELVIS / GA catalogue. |
| [OpenTopography Copernicus GLO-30](https://portal.opentopography.org/raster?opentopoID=OTSDEM.032021.4326.3) | ~30 m — coarse for a 50–200 m yard snip. |

OSM golf tags (`leisure=golf_course`, `golf=fairway|green|bunker`) are
layout masks, not elevation. Crop / resample offline to a 2-D `.npy`
and vendor it; `apply_dem_npy` only adds height.

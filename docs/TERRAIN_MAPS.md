# Observer maps vs physics, paths, and golf yards

No claimed mAP / FPS. This note is the routing / mapping contract.

## Observer-vs-physics failure mode

Physics (`HeightField`) is the true yard: a property-scale planar grade
plus drains, banks, undulation, and authored structures. Wheels sit on
that surface. The coverage planner does **not** see it.

`HeuristicTerrainObserver` (demo default) builds the maps in `obs`:

| Layer | What the robot is allowed to use |
| --- | --- |
| `elevation` / `slope` / `hazard` | RGB + ToF + IMU / pose. No god-view DEM. |
| `elevation_prior` | Rolling yard-scale plane from IMU pitch/roll + pose (and a short XY fit). |
| `structure` | Authored path / building / bunker / garden plus a colour stub. |

**What went wrong on `gradient_yard`:** the heuristic used to stamp only a
local disk of `pose.z` and treat brown-ish pixels as drain lips. On a
~0.14 rad grade the true field is about ±0.85 m, but the observer map sat
near `[-0.14, 0]` (MAE ≈ 0.43 m) and painted thousands of false lips.
The costmap then blocked or detoured as if the yard were flat and pitted.
The World Viewer mesh used **true** elevation while the plan overlay sat
on z ≈ 0, so routing looked divorced from the sloping mesh.

**Fix:**

1. Recover a yard-scale plane from IMU + seated pose (optional rolling
   least-squares as the robot moves). Paint that prior into elevation /
   slope every step. Drain drops are residuals on the plane, not a flat
   `pose.z`.
2. Gate isolated lip stamps: a lip must sit next to a channel (or a ToF
   drop). Tighten the brown heuristic so shaded grass / dirt is not a ditch.
3. Optional `planner.blend_elevation_prior`: the costmap floors observer
   slope with the low-frequency prior so a flattened CV map cannot fight
   physics.
4. Viewer: plan waypoints carry world `z` from the same height field as
   the mesh; `relief_scale` is applied to both. Toggle **observer vs true
   elev** for the error overlay.

Oracle maps still copy the height field (training / eval only).

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

```bash
jims-mower-demo --config acre_yard --steps 80 --out demo_acre
jims-mower-live --config acre_yard --speed 5
jims-mower-mission-demo --config acre_yard --out mission_acre
jims-mower-viewer --episode mission_acre
jims-mower-mesh --out acre.glb --config acre_yard --seed 3 --stride 4
```

Do **not** expect CI or a short demo to mow the whole acre. Strip
spacing is 1.20 m and mission step caps are raised so explore/mow can
move, not so the job finishes. Mesh export should use `--stride 3` or
`4` (default stride 2 is still loadable, just heavier).

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

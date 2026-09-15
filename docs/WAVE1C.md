# WAVE 1C — Scenario library, domain randomisation, weather, grass stub

Parallel wave. Do **not** put this material in `ROADMAP.md` (WAVE 1A owns that file).

This wave expands the gym’s **yards** and the **numpy geometric renderer**. No claimed mAP / FPS / sim-to-real scores.

## Run a scenario

```bash
jims-mower-demo --config configs/scenarios/suburban.yaml --out demo_suburban
jims-mower-demo --config configs/scenarios/playground.yaml --policy scripted --out demo_play
```

`load_config("configs/scenarios/<name>.yaml")` is the same path the env and demo already use. Names:

| YAML | Layout / weather |
| --- | --- |
| `suburban.yaml` | Backyard furniture, hose, mild drain, mild yard grade |
| `gradient_yard.yaml` | Whole-yard planar slope + a drain crossing it |
| `rural_paddock.yaml` | Larger sparse paddock |
| `playground.yaml` | Many toys |
| `orchard.yaml` | Tree rows |
| `terrace.yaml` | Parallel retaining-wall banks |
| `kerb_gutter.yaml` | Edge gutter + kerb |
| `wet_swale.yaml` | Swale after rain, puddles, wet specular |
| `night_porch.yaml` | Night pack + porch lights |

## World features

- **Layouts** (`world.layout`): `random`, `suburban`, `paddock`, `playground`, `orchard`, `terrace`, `kerb_gutter`, `swale`.
- **Rain puddles** (`world.n_puddles`): shallow circular depressions. Temporary weather hazards — they shade as water, mark the hazard raster as *steep/caution*, and ask the chassis to *slow*. They do **not** terminate as a drain-drop.
- **Hose / extension cord** (`world.n_hoses`, `world.n_cords`): soft ground clutter. The body can drive over them. If the trimmer disk overlaps one while spinning, `info["cutter_risk"]` is set (string-line / cable cut).

## Domain randomisation (renderer)

`domain_randomization` flags, all optional, seeded from the episode RNG (plus `seed` if set):

| Flag | Effect |
| --- | --- |
| `lighting` | Jitter the sun / moon direction and ambient |
| `colour_jitter` | Per-channel RGB scale |
| `shadow_blobs` | Dark ellipses on the ground |
| `motion_blur` | Horizontal box filter (off by default) |
| `camera_dirt` | Specks on the lens |
| `vignette` | Darken frame corners |
| `wet_specular` | Highlight on wet grass |

The renderer stays **numpy-only**. With DR off and `weather.pack: clear` the frames match the pre-1C geometric look.

## Weather / time-of-day packs

`weather.pack`: `clear` | `dawn` | `dusk` | `night` | `rain`.

- Dawn / dusk shift colour temperature (warm).
- Night darkens the sky and ground; `porch_lights: true` adds warm point lights (also used by `night_porch.yaml`).
- Rain enables a wet-ground hint and pairs with puddles in `wet_swale.yaml`.

## Grass growth / continue tomorrow

`world.grass.enabled: true` — on the **next** `reset()`, a fraction (`regenerate_frac`) of previously cut cells grow back instead of wiping the yard.

`world.grass.persist_path` (or `reset(options={"load_grass": path, "save_grass": path})`) writes/reads a `.npz` of the cut / grass masks so a later process can continue the same yard.

Off by default so existing tests still see a fresh lawn every episode.

## Tests

- Scenario YAML load + a couple of `reset()` checks (`tests/test_scenarios.py`).
- DR flags change saved frames; same seed is stable (`tests/test_renderer.py`).
- Grass save/load + regenerate (`tests/test_maps.py`).
- Soft hose / cutter-risk (`tests/test_safety.py`).

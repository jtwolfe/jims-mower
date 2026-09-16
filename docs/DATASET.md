# Dataset harness + label protocol (CV-8)

Build-order **§8**. Record frames into `jims_mower.dataset.v1` and label
them. **No trained production head. No published mAP / IoU.**

The WAVE 1A exporter already writes the folder layout. This document is
the **rig protocol**: what to capture, how to split, how to paint
drain / lip / bank / grass on real CSI. Fake CSI is OK in CI to prove
the pipeline; those frames are tinted blocks, not photos.

Schema helper: `jims_mower.dataset.validate_dataset_meta`.
Layout: `jims-mower-export` → `LAYOUT.md` inside the folder.

---

## Record

```bash
# Gym renderer (oracle labels) — existing path
jims-mower-export --steps 20 --seed 7 --out dataset_out

# Fake CSI / FakeGst → same schema (CI / bench without JetPack)
jims-mower-export --config configs/orin/bench.yaml --adapter fake_csi \
  --steps 8 --val-frac 0.2 --out dataset_out

# Prefer the stereo EXAMPLE names on the Orin path
jims-mower-export --config configs/orin/extrinsics_stereo.yaml \
  --adapter fake_csi --steps 8 --out dataset_out
```

On the wired rig, keep `--adapter fake_csi` until Gst is up, then
`runtime.cameras.adapter: gst` (or `--adapter gst`) so
`obs["cameras"]` is the CSI path at the ICD size
(`runtime.capture.downsample_rgb`). Downsample **before** you label.

`meta.json` records:

| Field | Meaning |
| --- | --- |
| `schema` | must be `jims_mower.dataset.v1` |
| `source` / `adapter` | `renderer` / `fake_csi` / `fake_gst` / `gst` |
| `split` | train / val frame indices (see below) |
| `map_claim` / `fps_claim` | always `null` |

`split.json` is a copy of `meta.split` for humans.

---

## Train / val split

Deterministic **last fraction** of frames is val
(`jims_mower.dataset.assign_frame_split`, default `val_frac=0.20`).

- Frames `0 .. n-n_val-1` → train
- Frames `n-n_val .. n-1` → val
- `jims-mower-train-terrain` uses `meta.split.train_frames` when present

Do not tune a published score on val. The split exists so the stub
does not train on every frame.

---

## Label protocol (real CSI)

Oracle rasters from the gym (`labels/*_hazard.png`, grass, elevation)
are **sim**. On the rig, replace or overlay them.

### Classes (hazard PNG, uint8)

| Value | Name | Paint |
| --- | --- | --- |
| 0 | free / grass | mowable turf |
| 1 | steep / bank | pitched soil or retaining face |
| 2 | drain lip | concrete / dirt edge of a channel |
| 3 | channel | the open drain itself |

Same integers as the gym oracle. Grass PNG stays `0` uncut, `1` cut,
`255` non-grass when you have a coverage strip; otherwise leave 255
and say so in `meta.notes`.

### How to label

1. Export a bag (`images/{frame}_{camera}.png`).
2. In any editor / CVAT / LabelMe, paint the **hazard** raster at the
   **map** resolution in `meta.map_shape` (not the camera size) — or
   paint per-camera masks and project later. Keep the same filenames.
3. Label **drain lips** as 2 and the **channel** as 3. Do not mark
   dry grass as a lip.
4. One person, one day, one lawn is a start. Version the folder
   (`meta.measured_at`, a git tag). Write the yard name in notes.
5. Val frames must be labelled with the same book — do not leave them
   blank and then quote an IoU.

### What not to do

- Do not publish mAP / IoU until a **locked** held-out set exists
  (build-order §9).
- Do not treat FakeCsi tinted frames as terrain.
- Do not train a production TRT head in this step.

---

## Train stub

```bash
jims-mower-train-terrain --dataset dataset_out --out terrain_mlp.npz
```

Numpy colour+position stub. Still not a field head.

---

## Honesty

CV-8 acceptance: **N frames from the rig, labelled, versioned,
train/val documented.** CI proves FakeCsi → `jims_mower.dataset.v1`
with a split. Real CSI + human labels remain a follow-up after
JetPack and the §7 measured YAML.

# WAVE 2B — perception pipeline (sim-only)

Numpy-first stubs. **No claimed mAP / IoU / FPS / sim-to-real scores.**

## Train a terrain stub from an export

The WAVE 1A exporter already writes oracle hazard rasters plus synced RGB.
WAVE 2B samples ground-plane pixels (colour + image UV + world XY) and fits a
tiny **softmax logistic** or **one-hidden-layer MLP** in numpy.

```bash
# Tiny labelled dump (oracle labels) then train. Torch is not required.
python -m jims_mower.export --steps 8 --seed 7 --cameras 4 --out dataset_out
python -m jims_mower.perception.train --dataset dataset_out --out terrain_mlp.npz --hidden 8

# Same thing via the scripts/ wrapper (exports if --dataset is omitted):
python scripts/train_terrain_seg.py --steps 6 --out terrain_mlp.npz --export-out dataset_out

# Optional ONNX (sim_only; not field-ready; requires pip install -e ".[onnx]"):
jims-mower-train-terrain --dataset dataset_out --out artifacts/terrain_mlp.npz \
  --onnx artifacts/terrain_seg.onnx
jims-mower-export-trt --onnx artifacts/terrain_seg.onnx --dry-run

# Optional torch extra (never used in CI):
#   pip install -e ".[torch]"
#   python scripts/train_terrain_seg.py --dataset dataset_out --backend torch --out terrain_mlp.npz
```

Load the weights:

```python
from jims_mower.env import MowerEnv
from jims_mower.perception import LearnedTerrainObserver

env = MowerEnv(
    terrain_observer=LearnedTerrainObserver("terrain_mlp.npz"),
)
# or: perception.terrain_mode: learned  and  perception.weights_path: terrain_mlp.npz
# or: perception.terrain_mode: onnx     and  perception.onnx_path: artifacts/terrain_seg.onnx
```

`LearnedTerrainObserver` follows the same `estimate(...)` contract as
`BlindTerrainObserver`. It **must ignore** `context.terrain`. Demo / farm
accept `--terrain-observer learned`.

The trainer reports **train loss and class counts only**. Do not treat those
as a published metric.

## Domain randomisation for training exports

Renderer DR is off by default so geometric tests stay stable. For a training
dump, turn it on so colour+position features see lighting / dirt / vignette
jitter (still the same oracle labels):

```bash
python -m jims_mower.export --steps 12 --seed 3 --cameras 4 \
  --domain-rand --out dataset_dr
```

`--domain-rand` sets `domain_randomization.enabled` plus lighting, colour
jitter, shadow blobs, camera dirt, and vignette. You can also author it in
YAML (WAVE 1C flags):

```yaml
domain_randomization:
  enabled: true
  lighting: true
  colour_jitter: true
  shadow_blobs: true
  camera_dirt: true
  vignette: true
  wet_specular: false   # pair with weather.pack: rain / wet_swale
```

`meta.json` records the DR snapshot. Pair with scenario weather packs
(`night_dawn`, `wet_slope`, `wet_swale`) when you want tint + wet ground as
well as the DR knobs. Labels stay **oracle** either way.

## Multi-camera BEV fuse

`jims_mower.perception.fuse` back-projects each camera’s class+confidence
onto the flat-yard plane and keeps the highest-confidence class per cell
(range-weighted). Heuristic and learned observers share this merge.

## Temporal filters

- **Hazard hysteresis** — a cell must accumulate evidence before it appears;
  unseen cells decay and drop below a forget threshold.
- **Tracklets** — greedy XY association for `person` / `dog` detections.
  Lives on `info["tracklets"]`. Not MOT / mAP.

## Verify (CPU, no torch)

```bash
python -m pip install -e ".[dev]"
pytest
python scripts/train_terrain_seg.py --steps 4 --cameras 4 --epochs 8 \
  --out /tmp/terrain_mlp.npz --export-out /tmp/jims_export
```

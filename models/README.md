# Model artifacts

Large weights stay **out of git**. Train locally:

```bash
jims-mower-export --adapter fake_csi --steps 6 --out artifacts/dataset
jims-mower-train-terrain --dataset artifacts/dataset \
  --out artifacts/terrain_mlp.npz --onnx artifacts/terrain_seg.onnx
```

Sim-trained ONNX / npz files are a **dev artifact** (`sim_only`). They are
**not field-ready**. `iou_claim` / `map_claim` / `fps_claim` stay null
until a held-out **real** label set exists — see
[`docs/DATASET.md`](../docs/DATASET.md).

Heuristic remains the live mow default unless you point
`perception.terrain_mode` at `onnx` / `trt` **and** a measured engine is
actually configured.

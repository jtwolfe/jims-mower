# Stereo + extrinsics calibration bench

Build-order **§7** (S2R-2 / S2R-3). Software procedure + gym acceptance
so a human on the rig can measure and commit *measured* YAML.

This is **not** a field measurement. We cannot tape a checkerboard on an
Orin in CI. Do not invent baseline centimetres, disparity residuals,
FPS, or mAP.

Linked from [`SIM_TO_REAL.md`](SIM_TO_REAL.md),
[`PRODUCT_TO_HARDWARE.md`](PRODUCT_TO_HARDWARE.md),
[`HARDWARE_DESIGN.md`](HARDWARE_DESIGN.md) §7.

---

## What you are measuring

A **forward stereo pair** on a ~50 cm zero-turn. Body frame: **x
forward, y left, z up**.

| Quantity | Band | Why |
| --- | --- | --- |
| Baseline \(B\) | **6–12 cm** (example YAML uses 8 cm) | Depth resolution \(\delta Z \approx (Z^2 / fB)\,\delta d\). Wider than ~12 cm on this body is a look-around, not this pair. |
| Near-field range | **0.8–4 m** | Tip / lip / obstacle band. Gym synthetic stereo lives here. |
| Pitch | ~20° down (example −22°) | Drain lips in the lower image third. |

Default gym `front_left` / `front_right` (40° yaw, ~40 cm) are **not**
a stereo pair. `find_stereo_pair` rejects them.

---

## EXAMPLE vs MEASURED

| File | Header / flag | Meaning |
| --- | --- | --- |
| [`configs/orin/extrinsics_stereo.yaml`](../configs/orin/extrinsics_stereo.yaml) | `# EXAMPLE` and `calibration.measured: false` | Documented example poses. **Keep it that way.** |
| [`configs/orin/extrinsics_stereo_measured.template.yaml`](../configs/orin/extrinsics_stereo_measured.template.yaml) | `# MEASURED TEMPLATE` and `calibration.measured: true` + `template: true` | Copy-this scaffold. Placeholder numbers. |
| `configs/orin/extrinsics_stereo_measured.yaml` | `# MEASURED` and `calibration.measured: true`, `template: false` | **Commit only after a real bench.** Gitignored until then. |

Also set:

```yaml
calibration:
  measured: true          # false on the EXAMPLE file
  template: false         # true only on the template
  measured_at: "2026-09-16"
  tape_baseline_cm: 8.1   # what the tape said, not a guess
  notes: "optical centres, warm, 2 m board"
```

`jims-mower-calibrate` refuses to treat `extrinsics_stereo.yaml` as
measured.

---

## Procedure on the rig

Tools: steel tape or calipers, a printed checkerboard (or a painted
drain-lip fixture), a flat floor, the Orin + CSI pair powered.

### 1. Tape the baseline

1. Mark each camera’s **optical centre** (module datasheet, or the
   centre of the lens barrel if that is all you have — write which).
2. Measure **left → right** along the stereo bar. That distance is
   \(B\). Convert to metres for YAML `y` (body +y is left):

   ```
   stereo_left.y  = +B/2
   stereo_right.y = −B/2
   ```

   Example at 8 cm: `y: ±0.04`.
3. Measure **forward** (`x`) and **up** (`z`) from the body origin
   (geometric centre, IMU if it sits there) to the same optical
   centres. Shared `x` / `z` / yaw / pitch for the pair.
4. Confirm \(B\) is in **6–12 cm**. If it is not, move the mounts —
   do not “fix” `find_stereo_pair`.

### 2. Checkerboard / drain-lip fixture

1. Print a checkerboard (or use a rigid board with known squares).
   A painted **drain-lip** of known height works for the map check.
2. Place the target **in front** of the pair, square to the stereo
   axis, at a taped distance in the **0.8–4 m** band.
   Suggested stations: **1.5 m, 2.5 m, 3.5 m** (and 0.8 / 4.0 m if
   you have the space).
3. Tape is **horizontal, stereo-midpoint → board**. Write the lever
   (front bumper vs optical midpoint) in `calibration.notes`.
4. Capture a left/right pair at each station. Keep the chassis still.

### 3. Write the YAML

1. Copy the EXAMPLE file:

   ```bash
   cp configs/orin/extrinsics_stereo.yaml \
      configs/orin/extrinsics_stereo_measured.yaml
   ```

   or start from the template (same placeholder poses).
2. Replace every `x, y, z, yaw_deg, pitch_deg` with **measured**
   numbers. Do not leave the example ±4 cm if the tape said otherwise.
3. Set the `calibration:` block as above (`measured: true`,
   `template: false`, `tape_baseline_cm`, date).
4. Header comment: `# MEASURED` — not `# EXAMPLE`.

### 4. Verify disparity vs tape (0.8–4 m)

On the box, after CSI is live (build-order §5 field line):

1. At each taped station, read horizontal disparity on the
   checkerboard (or a high-contrast lip edge) near the image centre.
2. Convert: \(Z = f B / d\) (same units as
   `jims_mower.perception.stereo.range_from_disparity`).
3. Compare \(Z\) to the **ray range** (tape / \(\cos\) pitch if the
   optical axis is pitched). Gym identity tolerance is **2 cm** on
   ideal stereo. On hardware, write the residual you actually saw —
   do not invent one here.
4. Fail the bench if a station in-band is wildly off (wrong baseline
   sign, swapped left/right, pitch not shared).

Gym stand-in (no checkerboard in CI):

```bash
jims-mower-calibrate --fixture
# or
pytest tests/test_calibration.py tests/test_stereo.py
```

Ideal disparity from known geometry must match tape; a known lip
must land in the right `ObservedMap` cells. That is **not** a
published matcher / mAP.

### 5. Commit

Commit `configs/orin/extrinsics_stereo_measured.yaml` (and the
packaged copy under `src/jims_mower/data/orin/` if you ship it)
**only** after the tape check. Leave the EXAMPLE file untouched.

---

## CLI

```bash
# Checklist + validate the EXAMPLE YAML (baseline 6–12 cm, not measured)
jims-mower-calibrate

# Same for a file you are editing
jims-mower-calibrate --yaml configs/orin/extrinsics_stereo_measured.yaml

# Gym lip + tape identity (synthetic stereo)
jims-mower-calibrate --fixture --json

# Template flags
jims-mower-calibrate --template
```

Look-around YAML (`extrinsics_6cam.yaml`) is **rejected** — it is not
a pair.

---

## Gym acceptance (what CI proves)

| Check | Tolerance | Not claimed |
| --- | --- | --- |
| EXAMPLE baseline in 6–12 cm | exact 8 cm in the example file | field tape |
| EXAMPLE `calibration.measured` is false | — | — |
| Template has `measured: true` + `template: true` | — | committed measurement |
| Ideal \(d = fB/Z\) vs tape at 1.5 / 2 / 2.5 / 3.5 m | 2 cm | matcher residual |
| Lip pad at 2 m → correct ObservedMap cells | 1 cell @ 0.25 m | field kerb mAP |
| Reconstructed lip ground-\(x\) vs tape-to-near-face | 20 cm (≤ 1 cell) | hardware residual |

`fps_claim` / `map_claim` stay `null`.

---

## After fab (follow-ups)

1. Flash JetPack; confirm CSI names `stereo_left` / `stereo_right`.
2. Run this procedure; write `extrinsics_stereo_measured.yaml`.
3. `jims-mower-calibrate --yaml …` must PASS with `measured: true`.
4. Point the live / bench overlay at the measured file.
5. Only then start build-order §9 (ONNX / TRT heads) on real frames.

Physical measure-and-commit is **still required**. This PR only makes
that commit possible without guessing.

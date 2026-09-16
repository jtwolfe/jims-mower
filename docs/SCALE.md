# Wheel / trimmer scale (S2R-4)

How to turn a **1.0 command** into measured metres per second and
trimmer RPM. The gym uses `robot.max_wheel_speed_mps × drive.scale`.
Default `scale: 1.0` and `measured: false`. **Do not invent Kv.**

Linked from [`PRODUCT_TO_HARDWARE.md`](PRODUCT_TO_HARDWARE.md),
[`CALIBRATION.md`](CALIBRATION.md), [`SIM_TO_REAL.md`](SIM_TO_REAL.md).

---

## YAML

```yaml
robot:
  max_wheel_speed_mps: 1.2   # gym / ICD full-scale; not a motor Kv
  drive:
    scale: 1.0
    measured: false
    measured_at: ""
    notes: ""
  trimmer:
    scale: 1.0
    measured: false
    measured_at: ""
    notes: ""
```

`measured: true` only after you write the bench numbers below. The
repo ships `measured: false`. There is no silent Kv in this file.

Gym: a `[1.0, 1.0, 0]` command translates at
`max_wheel_speed_mps * drive.scale`. A `scale` of `0.5` halves the
distance in the same `dt`. Trimmer `scale` is stored for the tach
procedure — the gym does not invent RPM.

---

## Drive: 1.0 command → m/s

Tools: tape or marked floor, stopwatch (or a known `dt` logger),
level pad, dummy load **or** the chassis with trimmer **off**.

1. Confirm hardware ESTOP is reset and the watchdog is happy.
2. Mark a start line. Command **both wheels 1.0**, trimmer 0, for a
   timed interval \(T\) (e.g. 2.0 s) on the flat.
3. Tape the distance \(D\). Ground speed \(v = D / T\).
4. Write `drive.scale = v / max_wheel_speed_mps` (the ICD full-scale
   already in YAML, usually 1.2). Example: you measured 0.60 m/s →
   `scale: 0.50` if `max_wheel_speed_mps` is 1.2.
5. Set `drive.measured: true`, `measured_at`, and a note (surface,
   pack voltage, tyre). Repeat left-only / right-only if they disagree
   — do not average away a dead motor.
6. Do **not** copy a motor Kv from a datasheet into this field.

---

## Trimmer: 1.0 command → RPM

Tools: optical / contact tachometer, string head **unloaded** (no
grass), eye protection.

1. Command trimmer 1.0 with wheels 0, chassis chocked, ESTOP ready.
2. Read RPM at the head after it settles (not the inrush).
3. Write `trimmer.scale` so that `command * scale` maps onto the
   RPM you will use later (often `measured_rpm / target_rpm`). If you
   have not chosen a target RPM yet, leave `scale: 1.0` and put the
   raw RPM in `notes`.
4. Set `trimmer.measured: true` only after that reading. **No
   invented RPM.**

---

## Honesty

- `measured: false` is the default. Bring-up reports SKIP, not a
  fake m/s.
- Do not claim acre runtime from wheel scale.
- Field tape on turf will differ from a bench pad — measure again
  after fab, do not copy the gym 1.2.

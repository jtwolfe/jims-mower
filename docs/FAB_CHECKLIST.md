# Fab checklist + BOM freeze (§19)

Build-order **§19**. Step-by-step from
[`HARDWARE_DESIGN.md`](HARDWARE_DESIGN.md). This is **not** a pretend
chassis and **not** a shopping cart with SKUs.

Structured freeze: [`configs/hardware/bom.yaml`](../configs/hardware/bom.yaml)
(`jims_mower.bom.v1`). Every number is **assumption**, **class**, or
**target** until a hang / bench marks it **measured**.

Linked from [`PRODUCT_TO_HARDWARE.md`](PRODUCT_TO_HARDWARE.md) §19,
[`HARDWARE_DESIGN.md`](HARDWARE_DESIGN.md) §6 / §9 / §10,
[`ESTOP.md`](ESTOP.md), [`CALIBRATION.md`](CALIBRATION.md),
[`PACK_THERMAL.md`](PACK_THERMAL.md).

---

## Before you cut metal

1. Read HARDWARE_DESIGN §1. The 22 kg / 0.18 m CG figures are
   **assumptions**. If you already know they are wrong, revise §1
   *before* the first water-jet / saw cut.
2. Confirm gym envelope you are matching: body 0.50 m, track 0.40 m,
   wheelbase 0.40 m, `collision_radius_m` 0.28 m. Growing past ~0.55 m
   means retuning those knobs.
3. Freeze the BOM class list (below). Do not pick fake SKUs in git.

---

## Fab steps

Do these in order. Tick in the shop copy; do not tick them here.

### 1. Chassis cut

- Cut the belly / tub to the 500 mm cube class (3 mm Al or HDPE — your
  call; both are **uncertain**).
- Ground clearance **70–90 mm**. Gym `wheel_drop_m` 80 mm is the
  drain-fail height — do not sit lower without changing that.
- Skirts above grass. **No under-deck blade.** Front +x is the trimmer.

### 2. Wheel track + wheelbase

- Wheel centreline track **400 mm**. Wheelbase **400 mm**.
- Wheel OD ~200 mm (\(r_w = 0.10\) m) is an **assumption** until you
  tape the tyre you bought.
- Hubs: 24 V class, ≥7 N·m peak, encoder. Uncertain SKU.

### 3. Hang-measure mass and CG

- Weigh the **wet** article (chassis + hubs + trimmer + Orin + pack).
- Hang from two known points (or a CG jig). Measure \(h_\mathrm{cg}\)
  above the wheel-patch plane.
- Write \(m\) and \(h_\mathrm{cg}\) into HARDWARE_DESIGN §1 and change
  **Status** from assumption → **measured**.
- Then revise tip-angle math (next section). Do **not** raise software
  `tip_roll_rad` / `tip_pitch_rad` to “match” the new static tip.

### 4. Stereo baseline mount

- Forward pair, optical-centre baseline **6–12 cm**, shared yaw/pitch.
- Names: `stereo_left` / `stereo_right` plus side / rear mono
  (`front`, `rear`, `left`, `right`).
- Default gym `front_left` / `front_right` (40° / ~40 cm) are **not**
  a stereo pair. Do not fab six look-arounds and call that depth.
- After the bar is rigid, follow [`CALIBRATION.md`](CALIBRATION.md)
  and commit `extrinsics_stereo_measured.yaml`. EXAMPLE YAML stays
  unmeasured.

### 5. ESTOP paddle wire

- NC paddle in the enable / coil path. Python GPIO may **monitor**.
  It must not be the only path.
- Traction + trimmer rails die. Orin / hotel stay up (Orin fuse
  **does not** open with the paddle).
- Reset: [`ESTOP.md`](ESTOP.md). Software Start does not restore a
  latched paddle.
- Field line (not claimed): hit the paddle while a dummy load spins.

### 6. Fuse map

Class-scale from HARDWARE_DESIGN §10 (resize after stall current):

| Fuse | Class | Opens on paddle? |
| --- | --- | --- |
| Pack | 40 A | no (upstream) |
| Orin | 5 A | **no** |
| Hub L / R | 15 A | rail downstream dead |
| Trimmer | 15 A | rail downstream dead |

Mark every fuse on the lid.

### 7. Orin mount

- Orin Nano 8 GB + carrier. Confirm barrel voltage. **Do not** feed
  8S LFP raw into a 19 V barrel.
- Duct or finned lid. Closed cube in sun will run hot
  (**assumption**). Thermal trip stays a stub until
  [`PACK_THERMAL.md`](PACK_THERMAL.md).
- No gym renderer on-box.

### 8. Camera names

ICD keys (do not rename): `cameras`, `imu`, `gps`, `tof`.
Camera dict keys must match `CameraSpec` names after Gst
(`stereo_left` / `stereo_right` + mono, or the gym look-around names
if that is what you actually mounted — but then you do not have a
pair).

ToF order: FL, FR, RL, RR. IMU near CG, body \(x\) forward, \(z\) up.

---

## After hang-measure — revise tip-angle math

HARDWARE_DESIGN §2 uses

\[
\tan\alpha_\mathrm{roll} = \frac{t/2}{h_\mathrm{cg}},\qquad
\tan\alpha_\mathrm{pitch} = \frac{b/2}{h_\mathrm{cg}}.
\]

With the **assumed** \(t = b = 0.40\) m and \(h_\mathrm{cg} = 0.18\) m
that is \(\alpha \approx 48°\). After you measure \(m\) and
\(h_\mathrm{cg}\):

1. Recompute \(\alpha_\mathrm{roll}\) and \(\alpha_\mathrm{pitch}\).
2. Recompute grade / yaw torque in §3 (\(F = mg\sin\theta\),
   \(T_\mathrm{yaw} \approx \mu mg\, t/2\)).
3. Leave gym software trips at `tip_roll_rad = 0.40` (~23°) and
   `tip_pitch_rad = 0.45` (~26°) unless the **machine** needs a
   *lower* trip. Never raise them to the static geometric tip.
4. Point reviewers at HARDWARE_DESIGN §2 and §11 (“Do not fix
   tip/drain physics to match a tall CG”).

---

## BOM freeze (part classes)

Same rows as HARDWARE_DESIGN §9, with the columns this row asked for.
YAML: [`configs/hardware/bom.yaml`](../configs/hardware/bom.yaml).

| qty | class | buy/make | verify-before-spin | open questions | numbers |
| --- | --- | --- | --- | --- | --- |
| 1 | Chassis / belly, 500 mm tub | make | hang-measure mass + CG | thickness vs 22 kg | **assumption** |
| 2 | Drive hub + encoder, 24 V ≥7 N·m | buy | stall current; resize fuses | wheel OD / gearbox | **class** |
| 1 | Trimmer spindle 24–36 V + string | buy | electrical cut, not PWM 0 | clutch vs current limit | **class** |
| 2 or 4 | Wheels ~8″ turf | buy | tape \(r_w\); track 400 mm | foam vs pneumatic | **assumption** |
| 1 | LFP pack + BMS, 24 V 50 Ah class | buy | [`PACK_THERMAL.md`](PACK_THERMAL.md) | CG low, between wheels | **class** |
| 1 | Charger 24 V 10 A class | buy | log `charge_time_h` | dock vs barrel | **class** |
| 1 | Orin Nano 8 GB + carrier | buy | barrel voltage; separate fuse | carrier variant | **class** |
| 4–6 | CSI cams; **one 6–12 cm stereo pair** + mono | buy | tape baseline; camera names | module SKU | **class** |
| 1 | IMU 6-axis I2C | buy | rest ≈ `(0,0,9.81,0,0,0)` | address vs 0x68 comment | **class** |
| 1 | UART GNSS + valid bit | buy | lever arm; surveyed peg | RTK later | **class** |
| 0/2/4 | ToF VL53-class downward | buy | board-under-wheel | count | **class** |
| 1 | ESTOP paddle NC 10 A+ | buy | dummy-load spin | GPIO sense vs coil | **class** |
| 1–2 | Contactor / FET traction + trimmer | buy | hotel stays up | one vs two | **class** |
| see §10 | Fuses (class-scale amps) | buy | resize after stall | blade vs poly | **class** |
| 1 | BT + optional Wi-Fi + LoRa | buy | do not claim RF range | which stick | **class** |
| 1 | Bump / skirt | make | no under-deck blade | material | **target** |

No prices. No SKUs. `bom.measured` stays **false** until hang / stall
/ pack bench rewrite the rows.

---

## Honesty

- Checklist ready. Chassis **not** fabbed in this repo.
- Mass, CG, tip angles, stall amps: **unmeasured**.
- Next physical work after fab: pack bench (§18), then the
  residential-acre scorecard ([`FIELD_TEST.md`](FIELD_TEST.md)).

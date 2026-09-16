# Hardware design — construction math

Target: a **~50 × 50 × 50 cm** zero-turn residential mower, Jetson Orin
Nano class compute, **forward stereo pair + side/rear mono** (4–6 RGB
cameras), **front whipper-snipper**,
meant to be **1-acre capable**. This note does the sizing math so fab
is not a guess. It is **not** a measured field pack, not a must-buy
cart, and not a SIL claim.

**No claimed field runtime** until row 18 of
[`PRODUCT_TO_HARDWARE.md`](PRODUCT_TO_HARDWARE.md) is measured.

ICD keys that the wiring must feed (do not rename): `cameras`, `imu`,
`gps`, `tof`, `pose`, action `[left, right, trimmer]`. See
[`../ICD.md`](../ICD.md) and [`runtime_contract.md`](runtime_contract.md).

Gym geometry already matches the envelope
(`RobotConfig` length/width/height 0.50 m, track 0.40 m, wheelbase
0.40 m, `max_wheel_speed_mps` 1.2, trimmer offset 0.32 m / radius
0.16 m). Those are **sim defaults**, not certified hardware.

---

## 1. Assumptions (mark every number)

| Symbol | Assumed value | Why / source | Status |
| --- | --- | --- | --- |
| \(m\) | **22 kg** wet | Chassis + 2×hub + trimmer + Orin + 24 V pack. Could be 18–28 kg. | **assumption** |
| \(h_\mathrm{cg}\) | **0.18 m** | Slightly below mid-height; battery low. Measure on a hang. | **assumption** |
| \(L, W, H\) | **0.50 m** | Envelope. | target |
| \(t\) track | **0.40 m** | Gym `track_m`. Wheel centreline. | matches gym |
| \(b\) wheelbase | **0.40 m** | Gym `wheelbase_m`. | matches gym |
| \(r_w\) wheel radius | **0.10 m** | ~8″ OD pneumatic or foam. | **assumption** |
| \(\eta\) drivetrain | **0.70** | Hub + gearbox. | **assumption** |
| \(\mu\) grass | **0.45** | Dry turf skid. Wet is lower. | **assumption** |
| \(\theta_\mathrm{grade}\) | **15°** continuous, **20°** stall | Steeper than most lawns; gym `steep_slope_rad` 0.30 ≈ 17°. | design |
| \(v\) | **1.2 m/s** max | Gym `max_wheel_speed_mps`. Cruise ~0.8 m/s. | matches gym |
| \(w_\mathrm{cut}\) | **0.32 m** | Trimmer offset ≈ effective strip. | matches gym |
| \(P_\mathrm{trim,cont}\) | **180 W** | String in grass, not peak stall. | **assumption** |
| \(P_\mathrm{orin}\) | **10 W** mean / **15 W** TDP | Orin Nano 8 GB class, not a bench wattmeter. | class |
| \(P_\mathrm{drive,cruise}\) | **60 W** both hubs | See §3. | derived + assumed η |
| \(A\) | **4047 m²** | 1 acre. | definition |
| \(g\) | **9.81 m/s²** | ICD `GRAVITY_MPS2`. | constant |

If a later hang-test says \(m=26\) kg or \(h_\mathrm{cg}=0.22\) m,
**recompute tip and torque** before cutting metal.

---

## 2. Mass, CG, tip angles

Static tip when the gravity vector through the CG leaves the support
polygon (four wheel patches, approximate as a \(t \times b\) rectangle).

**Roll (about \(x\), downhill to the side):**

\[
\tan\alpha_\mathrm{roll} = \frac{t/2}{h_\mathrm{cg}}
= \frac{0.20}{0.18} = 1.111
\quad\Rightarrow\quad
\alpha_\mathrm{roll} \approx 48^\circ
\]

**Pitch (about \(y\), nose up/down):**

\[
\tan\alpha_\mathrm{pitch} = \frac{b/2}{h_\mathrm{cg}}
= \frac{0.20}{0.18} = 1.111
\quad\Rightarrow\quad
\alpha_\mathrm{pitch} \approx 48^\circ
\]

That is the **static** geometric tip. It is **not** the software trip.
Gym `tip_roll_rad = 0.40` ≈ **23°**, `tip_pitch_rad = 0.45` ≈ **26°**.
Those fire earlier so the controller can reverse / skip. Do not raise
the software trip to 48° to “match physics.”

**CG shift from a 1.2 m/s² accel** (hard launch):

\[
\Delta x = \frac{a_x\, h_\mathrm{cg}}{g} \approx \frac{1.2\times 0.18}{9.81} \approx 0.022\,\mathrm{m}
\]

Effective roll/pitch margins shrink a few degrees. Wet grass + a
channel lip is the real tip case — the gym already punishes a wheel in
a drain (`wheel_drop_m = 0.08`).

**Ground pressure** (4 patches, 20 cm² each — **assumption**):

\[
p \approx \frac{22\times 9.81}{4\times 0.002} \approx 27\,\mathrm{kPa}
\]

Fine on turf; too high on wet soil if you use skinny wheels.

---

## 3. Drive motors

Zero-turn: equal-and-opposite wheel speeds (ICD action). Size for
**grade + spin-in-place**, then check continuous power.

### 3.1 Grade (both wheels driving)

\[
F_\mathrm{grade} = m g \sin\theta
\]

| Grade | \(\theta\) | \(F_\mathrm{grade}\) | Per wheel \(F\) | Wheel torque \(T=F r_w\) | Motor shaft \(T/\eta\) |
| --- | --- | --- | --- | --- | --- |
| 15° | 0.262 rad | \(22\times9.81\times\sin 15° \approx 55.8\,\mathrm{N}\) | 27.9 N | 2.79 N·m | **4.0 N·m** |
| 20° stall | 0.349 rad | 73.8 N | 36.9 N | 3.69 N·m | **5.3 N·m** |

Add rolling resistance \(F_{rr} = C_{rr} m g\) with \(C_{rr}\approx 0.08\)
on turf (**assumption**): +17 N total → +0.9 N·m/wheel at the motor.

**Design hold:** ≥ **6 N·m** per hub at the shaft, 30 s stall, 15°
continuous.

### 3.2 Zero-turn yaw

Skid-steer yaw against friction, both wheels opposing:

\[
T_\mathrm{yaw,ground} \approx \mu\, m g\, \frac{t}{2}
\approx 0.45 \times 22 \times 9.81 \times 0.20
\approx 19.4\,\mathrm{N\cdot m}
\]

Per wheel tangential force \(F = T_\mathrm{yaw}/t \approx 48.5\,\mathrm{N}\),
wheel torque \(F r_w \approx 4.85\,\mathrm{N\cdot m}\), shaft
\(4.85/0.70 \approx **6.9 N·m**\).

**Zero-turn, not grade, sizes the peak.** Wet \(\mu\) is lower (easier
to spin, worse to climb).

### 3.3 Continuous power at cruise

Resistive force on the flat ≈ \(F_{rr}\) + aero (ignore aero at 1 m/s):
~17 N. Plus 25% for strip turns (**assumption**):

\[
P_\mathrm{mech} = F v \approx 21\,\mathrm{N}\times 0.8\,\mathrm{m/s} \approx 17\,\mathrm{W}
\quad\Rightarrow\quad
P_\mathrm{elec} \approx 17 / 0.70 \approx 24\,\mathrm{W}
\]

Gym `BatteryConfig.drive_w = 25` is in this ballpark **on the flat**.
On a 15° grade at 0.5 m/s: \(P_\mathrm{mech} = 55.8\times 0.5 \approx 28\,\mathrm{W}\)
mech → ~40 W elec. **Budget 60 W average** for mixed residential grade
so the pack math is not optimistic.

Wheel rpm at 1.2 m/s: \(\omega = v/r_w = 12\,\mathrm{rad/s} \approx 115\,\mathrm{rpm}\).
Hub-motor / mid-drive class: 24–36 V, ~100–150 rpm wheel, peak 7–10 N·m.

**Part class (not a SKU):** two sealed hub or gearmotors, 24 V class,
≥7 N·m peak, ≥100 W each peak, IP54+, encoder. **Uncertain SKU** — pick
after measuring wheel OD and gearbox.

---

## 4. Trimmer motor

Front offset 0.32 m, radius 0.16 m, hub height 0.12 m (gym). This is a
**string head**, not a deck.

Handheld 18–36 V trimmers advertise 200–600 W **peak**. Continuous in
ordinary grass is much lower. **Assumption:** 180 W continuous, 400 W
for 2 s in thick clumps, duty ~60% while painting (turns + living
interlock).

\[
P_\mathrm{trim,avg} \approx 0.60 \times 180 = 108\,\mathrm{W}
\]

Interlock must cut **electrical** power, not just the ICD request bit
(PLN-4 / RT-6).

**Part class:** 24–36 V brushless spindle, 8–12 k string, clutch or
software current limit. **Uncertain SKU.**

---

## 5. Battery — hours, acre, charge

### 5.1 Time to cover 1 acre (geometry, not a runtime claim)

\[
s \approx \frac{A}{w_\mathrm{cut}} = \frac{4047}{0.32} \approx 12\,650\,\mathrm{m}
\]

At 0.80 m/s **effective** (includes turns — **assumption**):

\[
t_\mathrm{mow} \approx 12\,650 / 0.80 / 3600 \approx 4.4\,\mathrm{h}
\]

Add explore / dock / skip (~25% — **assumption**) → **~5.5 h** of
wall-clock work for a clean acre. Overlap, keep-outs, and living stops
make this longer. **This is a duty-cycle estimate, not a field number.**

### 5.2 Energy

| Load | Mean W | Notes |
| --- | --- | --- |
| Drive | 60 | §3.3 mixed grade |
| Trimmer | 108 | 60% of 180 W |
| Orin + cameras + radios | 18 | 10 W SoC + 8 W sensors/link **assumption** |
| Hotel (BMS, 5 V) | 4 | **assumption** |
| **Total** | **~190 W** | while mowing |

\[
E \approx 190\,\mathrm{W} \times 5.5\,\mathrm{h} \approx 1045\,\mathrm{Wh}
\]

Add 20% reserve to limp-home at `min_soc` 0.25 → **~1.3 kWh** nameplate
if you want one acre on one charge **in this model**.

Gym `BatteryConfig.capacity_wh = 50` is a **thermal/SOC stub** so short
tests can limp. It is **not** this pack.

### 5.3 Chemistry / voltage / Ah

Prefer **LiFePO4** for abuse and outdoor temperature (**assumption**:
residential, not a race pack).

| Pack | V × Ah | Wh | Notes |
| --- | --- | --- | --- |
| 24 V 40 Ah LFP | 25.6 V × 40 Ah | 1024 Wh | Tight vs 1.3 kWh model |
| 24 V 50 Ah LFP | 25.6 V × 50 Ah | 1280 Wh | Matches the model reserve |
| 36 V 30 Ah LFP | 38.4 V × 30 Ah | 1152 Wh | Less current, heavier harness |

**Recommendation for fab planning:** 24 V (8S LFP) **50 Ah class**,
~1.3 kWh, ~8–10 kg. **Uncertain SKU.** Fit inside the 50 cm cube with
the CG **low and between the wheels**.

Charge: 24 V 10 A (~250 W) charger → 20–100% ≈ \(0.8 \times 1280 / 250
\approx 4.1\,\mathrm{h}\). 20 A if you want ~2 h; watch thermal.

`schedule.min_soc` default 0.25 is a **job gate**, not a BMS cutoff.
Hardware cutoff should sit near gym `stop_soc` 0.05 **after** you
measure the real empty.

---

## 6. Wheels, track, clearance, chassis

```
          500 mm
     ┌──────────────┐
     │   cameras    │  z ≈ 380 mm (gym extrinsics)
   L │   Orin+heatsink
   0 │   batt (low) │
   0 │  ●────────●  │  track 400 mm
     │   hubs       │
     │      ○ trimmer (front, +x)
     └──────────────┘
```

| Item | Value | Notes |
| --- | --- | --- |
| Wheel OD | ~200 mm | \(r_w=0.10\) m |
| Track | 400 mm | Must match planner `track_m` or retune |
| Wheelbase | 400 mm | |
| Ground clearance | **70–90 mm** | Gym `wheel_drop_m` 80 mm is the drain-fail height |
| Body | 500 mm cube class | Skirts above grass; no under-deck blade |
| Trimmer | Front +x, offset 0.32 m | Keep the hub inside the 1.5 m living radius logic |
| Dock / home | Rear or side contacts | `YardProfile.home` is a pose, not a pinout |

Do not grow the body past ~0.55 m without changing `collision_radius_m`
(0.28 m) and the costmap inflation.

---

## 7. Cameras, IMU, GNSS, ToF

**Prefer a calibrated forward stereo pair** (baseline **~6–12 cm** on
the 50 cm body) plus side / rear monoculars. Do **not** fab six
independent look-around monoculars and call that a depth rig. Grass is
low-texture; live control is near-field disparity in the 0.8–4 m band,
not COLMAP / full-yard SfM.

Preferred example: [`configs/orin/extrinsics_stereo.yaml`](../configs/orin/extrinsics_stereo.yaml)
— **example poses, not calibrated**:

| Name | Body \(x,y,z\) m | yaw / pitch | FOV | Role |
| --- | --- | --- | --- | --- |
| stereo_left / stereo_right | 0.24, ±0.04, 0.38 | 0° / −22° | 70° | **Metric pair**, \(B \approx 8\) cm |
| front | 0.25, 0, 0.38 | 0° / −22° | 70° | Fill / teach |
| rear | −0.25, 0, 0.38 | 180° / −12° | 70° | Mono |
| left / right | 0, ±0.25, 0.38 | ±90° / −12° | 70° | Mono |

Gym look-around example (`configs/orin/extrinsics_6cam.yaml`) keeps the
old 40° / ~40 cm `front_left` / `front_right`. Those are **not** a
stereo pair — `find_stereo_pair` rejects them. Studies that compare
4/5/6 vs **tip/drain rates in sim** still use that look-around; they
are not mAP.

Depth resolution \(\delta Z \approx (Z^2 / fB)\,\delta d\) gets worse
with range. Size ObservedMap cells from that in the 0.8–4 m band.

Mount pitch is a **lip visibility** choice (see `jims-mower-study
--kind pitch`). After fab, replace the YAML with **measured**
extrinsics and a taped baseline (PRODUCT_TO_HARDWARE S2R-2).

| Sensor | Placement | ICD |
| --- | --- | --- |
| IMU | Near CG, body frame \(x\) forward, \(z\) up | `imu` length-6; rest ≈ `(0,0,9.81,0,0,0)` |
| GNSS | Roof, clear sky, lever arm noted | `gps` `(x,y,z,valid)` — yard frame on-box after origin |
| ToF ×4 | FL, FR, RL, RR, downward | `tof` metres; unused stay 0; `sensors.tof.count` ∈ {0,2,4} |

**Part classes (uncertain SKUs):** IMX219/IMX477-class CSI modules;
BMI088 / ICM-42688-P class IMU (`drivers.py` comments 0x68);
VL53L1X class ToF (comment 0x29); any UART GNSS with a valid bit.

---

## 8. Compute thermal / power

Orin Nano 8 GB, JetPack 6 class ([`JETSON.md`](JETSON.md)).

| Mode | Assumed SoC W | Board budget |
| --- | --- | --- |
| Idle / dock | 5–8 | Fan or heat-spreader to chassis |
| Explore / mow + TRT | 10–15 TDP | **Do not** run the gym renderer |
| Thermal trip | gym `t_hot_c` 75 / `t_crit_c` 85 | Stub until measured |

Closed 50 cm cube in Australian sun will exceed 35 °C ambient
(**assumption**). Give the Orin a duct or a finned lid. `OrinBudget`
`heat_c_per_w = 0.35` is a first-order RC, not a CFD.

Power at the barrel: 19–24 V input typical for Nano carrier boards —
**confirm the carrier**. Do not feed 8S LFP raw into a 19 V barrel
without a regulator. Fuse the Orin separately from the hubs (see §10).

---

## 9. BOM — part **classes**

Not a must-buy list. If the SKU is uncertain, it says so.

| Class | Qty | Example class | Uncertain SKU? | Feeds ICD / role |
| --- | --- | --- | --- | --- |
| Chassis / belly | 1 | 3 mm Al or HDPE tub, 500 mm | yes | mass / CG |
| Drive hub + encoder | 2 | 24 V, ≥7 N·m peak | yes | action `[0]`, `[1]` |
| Trimmer spindle | 1 | 24–36 V BLDC + string head | yes | action `[2]` |
| Wheels | 2 or 4 | 8″, turf tread | yes | \(r_w\) |
| LFP pack + BMS | 1 | 24 V 50 Ah class | yes | `battery_soc` later |
| Charger | 1 | 24 V 10 A class | yes | dock |
| Orin Nano 8 GB + carrier | 1 | JetPack 6 | carrier variant yes | compute |
| CSI cameras | 4–6 | IMX219-class, 70°; **one matched stereo pair** (6–12 cm) + mono | yes | `cameras` |
| IMU | 1 | 6-axis I2C | yes | `imu` |
| GNSS | 1 | UART, 1 Hz+ | yes | `gps` |
| ToF | 0/2/4 | VL53-class downward | yes | `tof` |
| ESTOP paddle | 1 | NC, 10 A+ | yes | kills rails |
| Contactor / FET | 1–2 | Traction + trimmer | yes | RT-6 |
| Fuses | see §10 | blade or poly | — | |
| Radios | BT + optional Wi-Fi + LoRa | USB/UART | yes | not measured |
| Bump / skirt | 1 | non-structural | yes | |

---

## 10. Wiring / fuse / ESTOP sketch

Software ESTOP (ICD `SafeState`) **zeros commands**. Hardware ESTOP must
still drop power if Python is wedged.

```
   PACK+ ── PACK FUSE (e.g. 40 A) ──┬── DC/DC 24→19/5 V ── Orin FUSE (5 A) ── Orin
                                    ├── HUB L FUSE (15 A) ── FET/driver ── left hub
                                    ├── HUB R FUSE (15 A) ── FET/driver ── right hub
   PACK− ───────────────────────────┴── TRIM FUSE (15 A) ── FET ── trimmer
                                              ▲
   ESTOP paddle (NC) ── cuts FET enables AND/OR a contactor on PACK+
   after the pack fuse. Python GPIO may monitor, must not be the only path.

   I2C: IMU 0x68 class, ToF 0x29 class (docs only — probe the real parts)
   UART: GNSS
   CSI: stereo pair + side/rear mono (4–6) → carrier
```

Amp numbers are **class-scale** from §3–5 (24 V, 60 W drive ≈ 2.5 A
cruise, stall much higher; 180 W trimmer ≈ 7.5 A). **Resize fuses after
you measure stall current.** Mark every fuse on the lid.

Living interlock and watchdog zero the **command**. ESTOP paddle zeros
the **rail**.

---

## 11. What must match the gym after fab

| Gym / ICD | Hardware |
| --- | --- |
| Body \(x\) forward, \(y\) left, \(z\) up | Same IMU / camera frame |
| Action \([-1,1]\), \([-1,1]\), \([0,1]\) | Scale to real max m/s after a tape test (S2R-4) |
| `trimmer` on if \(>0.5\) **and** interlock | Electrical cut, not only PWM 0 |
| Camera names `stereo_left` / `stereo_right` + mono, or gym `front`, … | Same dict keys after Gst |
| ToF order FL, FR, RL, RR | Same array |
| `YardProfile` metres | Survey origin + GNSS lever arm |

Do not “fix” tip/drain physics to match a tall CG. Change \(h_\mathrm{cg}\)
in this document and retune **software** trips.

---

## 12. Honest leftovers

- Runtime hours and acre-per-charge: **unmeasured**.
- Motor SKUs, camera modules, pack brand: **not picked**.
- RF range, TRT FPS, detector mAP: **out of scope** (and forbidden as
  invented numbers).
- Next *physical* work: PRODUCT_TO_HARDWARE build-order §3 (hardware
  ESTOP), then capture, then stereo calibration. Gym CV terrain
  (stereo + seg stub + frozen elev) is software build-order §2.

# Pack + thermal bench (RT-5 / HD-batt)

Build-order **§18**. Procedure + config hooks only. This repo has **no
lab bench**. Do **not** invent watt-hours, charge hours, board °C, or
acre runtime.

Linked from [`PRODUCT_TO_HARDWARE.md`](PRODUCT_TO_HARDWARE.md) (RT-5,
§18), [`HARDWARE_DESIGN.md`](HARDWARE_DESIGN.md) §5 / §8,
[`SCHEDULE.md`](SCHEDULE.md).

---

## What you are measuring

| Quantity | Config key | How |
| --- | --- | --- |
| Pack energy | `runtime.battery.capacity_wh` | Discharge Wh (coulomb / energy count) |
| Charge time | `runtime.battery.charge_time_h` | Empty → charger terminate, hours |
| Board °C under load | `runtime.thermal.board_load_c` | Peak Orin / carrier °C on a logged load |
| Provenance | `runtime.battery.measured` | `true` only after the three numbers above (or Wh + hours; °C may follow) |

Gym `capacity_wh: 50` is a **thermal/SOC stub** so short tests can limp.
[`HARDWARE_DESIGN.md`](HARDWARE_DESIGN.md) §5 has a **model** (~1.3 kWh
class) for fab planning. **Neither is a bench number.** Do not copy
either into a MEASURED file.

`acre_runtime_h` stays **null**. Cover-time geometry in HARDWARE_DESIGN
§5.1 is a duty-cycle estimate, not a field claim.

---

## STUB vs TEMPLATE vs MEASURED

| File | Flags | Meaning |
| --- | --- | --- |
| [`configs/default.yaml`](../configs/default.yaml) | `measured: false`, `capacity_wh: 50` | Gym stub. Keep it. |
| [`configs/orin/pack_measured.template.yaml`](../configs/orin/pack_measured.template.yaml) | `measured: true`, `template: true`, Wh / hours **null** | Copy-this scaffold. |
| `configs/orin/pack_measured.yaml` | `measured: true`, `template: false`, filled Wh / hours / date | **Commit only after a real bench.** Gitignored until then. |

`load_config` **refuses** `measured: true` on the silent 50 Wh default
(no `charge_time_h`, no `measured_at`). A real 50 Wh bench is allowed
only when `notes` say the Wh came off the bench.

---

## Procedure on the bench

Tools: a current shunt or clamp + logger (or BMS coulomb counter), a
voltmeter or the same logger on pack voltage, a wall clock, a dummy
load or the hubs + trimmer on blocks, a thermocouple or Orin zone temp.

### 1. Capacity (Wh)

1. Charge to **BMS full / charger terminate**. Write the terminate
   condition in `notes` (voltage, current tail, or BMS flag).
2. Rest until surface charge dies (your call — write the rest time).
3. Discharge at a **known** load (resistive dummy, or both hubs +
   trimmer on blocks at a stated command). Log \(I(t)\) and \(V(t)\).
4. Stop at the **hardware empty** you will use as `stop_soc` (gym
   default 0.05 is a placeholder). Do not drain a LiFePO4 pack to a
   destructive voltage.
5. Integrate:

   ```
   Ah = ∫ I dt
   Wh = ∫ V · I dt
   ```

   Prefer Wh. If you only have Ah, write `Wh ≈ V_nom × Ah` and say
   which \(V_nom\) you used.
6. Write that Wh into `runtime.battery.capacity_wh`. **Not** the
   HARDWARE_DESIGN model. **Not** 50 unless that is what you measured.

### 2. Charge time (h)

1. From the empty you just used, connect the charger you will dock
   with.
2. Wall-clock until charger terminate / BMS full.
3. Write hours into `runtime.battery.charge_time_h`.

### 3. Board °C under load

1. Close the lid (or the sun-load you actually expect).
2. Run a representative compute + drive load (explore/mow class — **no
   gym renderer** on the Orin).
3. Log peak board / SoC zone °C. Write `runtime.thermal.board_load_c`.
4. Only then set `runtime.thermal.measured: true`. Until that log
   exists, leave `measured: false` and `board_load_c: null`.
5. Gym `t_hot_c` 75 / `t_crit_c` 85 stay **stub trips** until you
   retune them from this log. Do not invent a TDP.

### 4. Commit the YAML

```yaml
runtime:
  battery:
    measured: true
    template: false
    capacity_wh: null     # ← your Wh
    charge_time_h: null   # ← your hours
    measured_at: "YYYY-MM-DD"
    notes: "shunt + dummy load, terminate at …"
  thermal:
    measured: false       # true only after board_load_c
    board_load_c: null
```

Overlay the file on the yard config, or merge those keys into the Orin
runtime YAML. `OrinBudget` drains SOC as
`(power_w × dt / 3600) / capacity_wh`. The schedule SOC gate still
compares `battery.soc` to `schedule.min_soc`; remaining energy is
`soc × capacity_wh` when status is asked — **not** acre hours.

---

## Honesty

- Procedure ready. Numbers **null** until a human benches the pack.
- `not_a_power_trace: true` stays on `/status` — OrinBudget is still an
  RC, not a BMS.
- No acre-per-charge claim in README or this file.

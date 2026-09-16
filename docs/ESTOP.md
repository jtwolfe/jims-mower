# Hardware ESTOP + power kill

Software ESTOP (`SafeStateMachine`) **zeros commands**. Hardware ESTOP
must still drop **traction and trimmer rails** if Python is wedged.

This note is the **sim model + wiring design** for PRODUCT_TO_HARDWARE
build-order **§3** (RT-6, HD-wire). It is **not** a claimed SIL rating
and **not** a passed bench paddle test. The physical acceptance line
is still: hit a real paddle while a dummy load spins; rails go dead
without Orin / Python.

Construction math and the fuse sketch live in
[`HARDWARE_DESIGN.md`](HARDWARE_DESIGN.md) §10. ICD software latch:
[`../ICD.md`](../ICD.md). Product checklist:
[`PRODUCT_TO_HARDWARE.md`](PRODUCT_TO_HARDWARE.md).

---

## What opens when the paddle is hit

The paddle is a **normally-closed (NC)** contact in the enable / coil
path. Hitting it **opens** that path. Python GPIO may **monitor** the
same contact. It must **not** be the only path.

```
                    ┌─ Orin / Python (monitor only) ─┐
                    │  SafeStateMachine zeros cmds   │
                    └──────────────┬─────────────────┘
                                   │ GPIO sense (optional)
   PACK+ ── PACK FUSE (40 A class)─┴── ESTOP CONTACTOR / FET ENABLE
                                              │
                    paddle (NC) ── opens enable / coil
                                              │
              ┌───────────────┬───────────────┴───────────────┐
              │               │                               │
         HUB L FUSE      HUB R FUSE                    TRIM FUSE
           15 A            15 A                          15 A
              │               │                               │
           FET/driver      FET/driver                       FET
              │               │                               │
          left hub        right hub                       trimmer
```

| Rail | What opens | Independent of Orin? |
| --- | --- | --- |
| Left hub | Contactor / FET enable after pack fuse | **yes** |
| Right hub | same enable (or a second FET) | **yes** |
| Trimmer | same enable (or a dedicated trim FET) | **yes** |
| Orin + cameras | **stays up** (own fuse / DC-DC) | so the owner app can still say “rails dead” |

Hotel / compute stay alive on purpose: a dark Orin after ESTOP is worse
for retrieve. The paddle kills **motion**, not the brain.

Amp numbers are **class-scale** (24 V, 60 W drive cruise, 180 W trimmer).
**Resize after you measure stall current.** Mark every fuse on the lid.

---

## Fuse map (class-scale, not measured)

| Fuse | Class | Protects |
| --- | --- | --- |
| Pack | 40 A | Whole PACK+ after the battery |
| Orin | 5 A | DC/DC 24→19/5 V → carrier |
| Hub L / Hub R | 15 A each | Drive FETs / hubs |
| Trimmer | 15 A | Spindle FET |

See [`HARDWARE_DESIGN.md`](HARDWARE_DESIGN.md) §10. Do not treat these
as a BOM SKU.

---

## Independence from Orin / Python

```
policy / SafeStateMachine  →  command  [left, right, trimmer]
sensor watchdog            →  may zero the command
FaultBus / radio           →  may zero or limp the command
HardwareEstop (this)       →  LAST rail filter: latched → 0, 0, off
physics / FETs             →  only see the rail command
```

A latched paddle zeros wheels and the trimmer **underneath** the
controller. `SafeStateMachine.clear()`, owner **Start**, and hand-signal
`go` restore the **software** latch only. They must **not** pick the
contactors back up.

Gym object: `jims_mower.hardware_estop.HardwareEstop` on `MowerEnv`.
`info["hw_estop"]` / live `estop_kind` distinguish **software** vs
**hardware**.

---

## Reset procedure (field + sim)

**Field (when the paddle exists):**

1. Confirm the yard is clear. Do not stand in front of the trimmer.
2. Release the paddle (twist-to-release or lift — follow the part).
3. Press the chassis **RESET** (or re-enable the contactor coil).
4. If software ESTOP was also latched, **Start** on the phone.
5. Run `jims-mower-selftest` (or a short unloaded spin) before a job.

**Sim (no physical paddle):**

| Action | What it does |
| --- | --- |
| `env.hit_hw_estop()` / live `hw_estop` / inject `paddle` | Latch. Rails dead. |
| `SafeStateMachine.clear()` / owner Start / `clear` | Software latch only. Rails stay dead if HW is latched. |
| `env.reset_hw_estop()` / live `hw_reset` / inject `paddle` with `mode: reset` | Documented hardware reset. |
| `env.reset()` | New gym episode (sim power-cycle). **Not** the field reset. |

---

## Self-test hook (gym)

`jims-mower-selftest` includes `hw_estop_rails`: command a dummy load,
hit the sim paddle, assert pose does not move and the trimmer is off.
That is **not** a hardware ATE.

```bash
jims-mower-selftest
pytest tests/test_hardware_estop.py tests/test_safe_state.py
```

---

## Honest leftovers

- Bench paddle + dummy-load spin: **not done**. Needs the wired
  contactor / FET and a real NC paddle.
- Fuse values: **class-scale**, not measured stall.
- SIL / cert: **do not claim** (SAF-5).

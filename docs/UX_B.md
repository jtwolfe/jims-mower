# WAVE UX-B — faults + radio sim

Parallel to UX-A (owner / retrieve presentation). Notes live here so
[`ROADMAP.md`](../ROADMAP.md) stays a short append and does not fight
WAVE 4 checkbox edits. **No claimed mAP / FPS / RF range / SIL rating.**

## FaultBus

[`jims_mower.faults.FaultBus`](../src/jims_mower/faults.py) is the single
injector for gym + runtime:

| Kind | What it does |
| --- | --- |
| `motor_left` / `motor_right` | Kill a drive motor mid-episode. Modes: `cmd_ignored`, `encoder_stuck`, `open_circuit`. Latches **`FAULT_IMMOBILISED`**. |
| `trimmer_jam` | Head request ignored |
| `cam_blind` | Black RGB frames (all cameras, or a name list) |
| `imu_freeze` | Hold the last IMU sample |
| `gnss_dropout` | Force `obs["gps"][3] = 0` (unifies the existing `gps.dropout_prob` bit) |
| `stuck` | Advisory only — wheels still work |
| `hw_estop` / `paddle` | Hardware paddle latch (`HardwareEstop`). Not FaultBus. Rails dead until `hw_reset`. |

A **dead drive motor** zeros *both* wheels and the trimmer so a live
wheel cannot drag the chassis across a drain. That is **not** the WAVE
2A stuck path.

**Stuck** (lip / channel / `code=STUCK`) still runs reverse → pivot →
call-for-help. **`FAULT_IMMOBILISED`** holds, sets `retrieve: true`, and
skips recovery drive.

Schedule from YAML (`faults.enabled` + `faults.inject`) or at runtime:

```python
env.reset(seed=0)
env.fault_bus.inject("motor_left", mode="open_circuit")
# or reset(options={"inject_fault": {"kind": "motor_left", "at_step": 2}})
```

Off by default. Existing episodes are unchanged.

## SOS / retrieve hooks

`info["fault"]` is always present:

```json
{
  "schema": "jims_mower.fault.v1",
  "code": "FAULT_IMMOBILISED",
  "component": "drive_left",
  "pose": {"x": 1.2, "y": 3.4, "theta": 0.1, "z": 0.0, "pitch": 0.0, "roll": 0.0},
  "retrieve": true
}
```

`jims-mower-telemetry` adds `sos` + `sos_steps`. `jims-mower-incident`
paints a red SOS banner when `retrieve` is set. Episode JSONL keeps
`fault` in the slim info.

## Self-test

`jims-mower-selftest` (software only):

- Unloaded equal-wheel spin — pose must move
- Dead-motor hold — no drag, SOS latched
- IMU still at rest (specific force near *g*, quiet gyro)
- Camera frame entropy — live frames have entropy; `cam_blind` is black

`not_a_benchmark: true`. Not a hardware ATE.

## Radio simulator

[`jims_mower.radio.RadioSim`](../src/jims_mower/radio.py) — **no RF
hardware**. Three stub bearers:

| Channel | Preference | Class-scale stub |
| --- | --- | --- |
| `wifi` | 1st | ~20 Mbps, ~40 m, low drop |
| `bt` | 2nd | ~1 Mbps, ~12 m |
| `lora` | 3rd | ~5 kbps, ~2 km |

Command / heartbeat tries Wi-Fi, then BT, then LoRa. Out of range or a
drawn drop fails that bearer. `radio.on_loss`:

- `stop_beacon` — zero wheels + trimmer, `RADIO_LOSS` with `retrieve: true`
- `limp_home` — scale wheels by `planner.safe_state.limp_scale`, no retrieve

`radio.enabled` defaults false.

## Honesty

Bandwidth / range / drop figures are **class-scale stubs** for failover
tests. They are not a link budget. Owner status never exposes them —
`rf_claim` is always null. No mAP / FPS.

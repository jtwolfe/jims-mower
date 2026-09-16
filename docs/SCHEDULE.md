# Weekly schedule engine

`YardProfile.schedule` is no longer a stored stub. The engine in
[`src/jims_mower/schedule.py`](../src/jims_mower/schedule.py) evaluates the
same `jims_mower.yard.v1` document and **arms** or **stops** an owner job.

This is not a cloud calendar, not push notifications, and not a weather
API. Rain is an explicit flag (`status.weather.rain` / env `weather.wet`).

## Why this was first

ROADMAP / UX-C listed schedule as done because the JSON fields existed.
Nothing started a job. Computer vision is still `MockDetector` + numpy
stubs — putting a dataset harness first would ship another incomplete
head. A working weekly window is the first **owner-visible** gap that
can be finished as one vertical slice (build → test → review → revise)
without inventing mAP.

See [`PRODUCT_TO_HARDWARE.md`](PRODUCT_TO_HARDWARE.md) item 4 and build
order §1.

## Document fields

| Field | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Health-page toggle. Off → idle, owner Start only |
| `days` | `[]` | `mon`…`sun` |
| `start_local` | `09:00` | 24h `HH:MM` in `timezone` |
| `duration_min` | `60` | Auto-stop only for jobs **this engine armed** |
| `timezone` | `Australia/Brisbane` | IANA name (user locale). `local` still accepted. 09:00 means 09:00 in this zone, not UTC-by-accident. |
| `min_soc` | `0.25` | Skip if `battery.soc` is below this. SOC is a fraction of `runtime.battery.capacity_wh` (50 Wh gym stub unless `measured: true` — [`PACK_THERMAL.md`](PACK_THERMAL.md)). Not an acre-runtime claim. |
| `skip_rain` | `true` | Skip when the rain / wet flag is set |
| `arm_window_min` | `15` | Minutes after start still eligible |
| `note` | engine blurb | Old `stub — not a scheduler` is rewritten on load |

## Clock

- **Wall clock** in `jims-mower-app` / `--live` (and a 1 s serve ticker).
- **Frozen / sim clock** via `backend.set_clock(FrozenClock(...))` for
  tests. Advance the same object to expire `duration_min`.
  `2026-09-13 23:00 UTC` is Monday `09:00` in `Australia/Brisbane`
  (UTC+10, no DST). `2026-09-14 09:00 UTC` is `19:00` Brisbane — not
  the 09:00 window.

`GET /status` and `tick()` both poll. Each weekly window is consumed at
most once (arm **or** skip). `already_running` does **not** consume, so
an owner Stop inside the window can still let the engine arm.

## Gates (skip reasons)

`soc_low` · `rain` · `fault` · `estop` · `disabled` · `no_days` ·
`already_running` · `duration` (stop, not skip).

A skip during the arm window does not retry until the next scheduled day.

## Status block

`GET /status` includes `schedule` (`jims_mower.schedule.v1`): `enabled`,
`next_run` / `next_run_local`, `action` (`idle` / `arm` / `skip` /
`hold` / `stop`), `reason`, gates. Phone **Health** shows next run and
an enable toggle that `PUT /yard`s the same document.

## Tests

```bash
pytest tests/test_schedule.py tests/test_yard_profile.py tests/test_app_api.py
```

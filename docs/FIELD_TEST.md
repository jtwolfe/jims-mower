# Field-test scorecard (residential acre)

Build-order **§20**. Template for a taught residential acre. Score
**tips / drain entries / leftover uncut / ESTOP pulls** — not mAP,
IoU, or FPS.

This repo has **not** run that acre. `field_run: false` on the
template. Do not invent counts.

**Practice on a laptop before the real acre:**

```bash
jims-mower-field-dryrun --out artifacts/field_dryrun/scorecard.yaml
```

That command walks bring-up, a short `mission_tiny` teach → explore →
MAP READY → mow → return-home (capped steps are OK), appearance living
interlock, gym red-bias hand-signal stop, tip inject, rain/SOC schedule
skips, day-2 new-process restore, and a sim HW ESTOP paddle. It writes
the same scorecard schema with `domain: gym_dryrun` and
`field_ready: false`. **Honest: not a field test.** No mAP / FPS / acre
runtime. Optional `--scenario acre_yard_demo` is the same gym walk on
the acre fixture — still not the lawn.

Artifact schema: `jims_mower.field_scorecard.v1`
([`configs/field/scorecard.template.yaml`](../configs/field/scorecard.template.yaml)).

Linked from [`PRODUCT_TO_HARDWARE.md`](PRODUCT_TO_HARDWARE.md) §20,
[`MISSION_FLOW.md`](MISSION_FLOW.md), [`ESTOP.md`](ESTOP.md),
[`SURVEY_ORIGIN.md`](SURVEY_ORIGIN.md), [`SCHEDULE.md`](SCHEDULE.md),
[`PACK_THERMAL.md`](PACK_THERMAL.md).

---

## Scorecard (print / copy)

Copy the YAML template, or tick this list on paper and type it later.

### Preflight

| Check | Pass? | Notes |
| --- | --- | --- |
| `jims-mower-selftest` on the **wired** robot | | gym streams are not this line |
| Hardware ESTOP paddle + dummy load; rails die; Start does not restore | | [`ESTOP.md`](ESTOP.md) |
| SOC from the fuel gauge (fraction 0–1) | | gym stub 50 Wh is not this |
| Rain flag (`weather.wet` / `status.weather.rain`) readable | | no radar required |

### Teach + origin

| Check | Pass? | Notes |
| --- | --- | --- |
| Teach keep-in (and keep-outs / home) | | phone `#/map` |
| Surveyed origin / peg | | [`SURVEY_ORIGIN.md`](SURVEY_ORIGIN.md). Tape-stop still required. |

### Mission

| Phase | Pass? | Notes |
| --- | --- | --- |
| Explore | | fog grows; unknown ≠ safe |
| MAP READY | | lock holds; mow on the frozen map |
| Mow | | coverage planner, not god-view |
| Return-home | | dock / home pose |

### Interlocks and skips

| Check | Pass? | Notes |
| --- | --- | --- |
| Living interlock (person / dog in `safety_radius_m`) | | trimmer request refused in one cycle |
| Tip / ramp recovery | | reverse → pivot → help; no drain entry |
| Rain skip | | window consumed as skip |
| SOC skip | | below `schedule.min_soc` |
| Day-2 multi-session resume | | cold load yesterday's uncut; no reteach |

### Score (integers / area — not detector scores)

| Metric | Value |
| --- | --- |
| Tips | |
| Drain entries | |
| Leftover uncut cells | |
| Leftover uncut area (m²) | |
| ESTOP pulls | |

Leave `acre_runtime_h` **null** unless this run finished **and** the
pack is `measured: true` ([`PACK_THERMAL.md`](PACK_THERMAL.md)). Even
then, write wall-clock hours only if you timed them — do not back-solve
from HARDWARE_DESIGN §5.

`map_claim` / `iou_claim` / `fps_claim` stay **null**.

---

## YAML artifact

```yaml
schema: jims_mower.field_scorecard.v1
field_run: false          # true only after this acre was run
field_ready: false
domain: ""                # gym_dryrun from jims-mower-field-dryrun — not a field test
pack_measured: false
preflight: {self_test: null, hw_estop_paddle: null, soc: null, rain_flag: null}
mission: {teach_boundary: null, surveyed_origin: null, explore: null,
          map_ready: null, mow: null, return_home: null}
checks: {living_interlock: null, tip_ramp_recovery: null,
         rain_skip: null, soc_skip: null, day2_resume: null}
score: {tips: null, drain_entries: null, leftover_uncut_cells: null,
        leftover_uncut_m2: null, estop_pulls: null}
acre_runtime_h: null
map_claim: null
iou_claim: null
fps_claim: null
```

Load / validate:

```python
from jims_mower.field_scorecard import empty_scorecard, load_scorecard

card = load_scorecard()          # template; field_run is false
blank = empty_scorecard()        # same keys, in memory
```

A filled run may be written to `configs/field/scorecard.yaml` (gitignored)
or kept with the yard bag. The loader **refuses** non-null mAP / IoU /
FPS claims and refuses `acre_runtime_h` unless `field_run` and
`pack_measured` are both true.

---

## Honesty

- Scorecard **ready**. Field **not** run.
- `jims-mower-field-dryrun` may write gym leftover / injected-tip /
  sim-ESTOP numbers. Those are **gym_dryrun**. They are not the acre.
- No leftover-uncut, tip, or ESTOP numbers from a **field** run in this repo.
- After the first acre, revise gym constants only if the **machine**
  needs it. Do not “fix” physics to match a tall CG
  ([`FAB_CHECKLIST.md`](FAB_CHECKLIST.md)).

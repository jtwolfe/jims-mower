# Terrain recovery: grade-aware blockage, trail retrace, fused state

Honest model for Jamie. This is the plan that beats the live stuck
baseline (`tests/fixtures/phase0_baseline.md`). No lidar stack. No
rigid-body engine. Software tip stays below static \(\alpha\).

See also [`NAV_BLOCKAGES.md`](NAV_BLOCKAGES.md), [`TIP_HILL.md`](TIP_HILL.md),
[`MISSION_FLOW.md`](MISSION_FLOW.md).

## Phase 0 — live stuck (beat these)

Owner `:8766` `acre_yard_demo` on 2026-09-17, post PR #51 (70×70×40,
belly CG, static-α latch, full-fence cut≥0.95, 1×, thorough Reset):

| Field | Stuck baseline |
| --- | --- |
| map | **~40%** of taught keep-in |
| step | **8001 / 4000** (full-explore still seeking) |
| n_blockages | **3263** |
| blocked_cells | **1367** |
| unreachable / blocked_frontiers | **3263** |
| frontiers | 40 |
| owner_copy | *Tip risk — reversing* |
| tilt_kind | tip |
| pitch / roll | ~0.02 / ~0.30 (climbable, not static α) |
| recovery | one-step reverse nudge → skip → replan → stamp the face |

Trail retrace was missing. Climbable grade was painted as learned no-go.

Full snap contract (trimmed): `tests/fixtures/phase0_baseline_snap.json`.

## Beat-criteria (Phase 4 gates)

Nothing is “resolved” until these pass on a **real gym** (not mocked).
CI uses a taught `mission_tiny` pocket. Full acre is a local long-run.

1. Explore map ≥ **~0.99** of keep-in (tiny taught pocket OK for CI).
2. Blockage events **bounded** vs baseline — must not sit past
   `max_explore_steps` with ~40 frontiers forever / thousands of
   blockages at ~40% map. CI: `n_blockages` << 3263 (tiny job **< 80**;
   a job still exploring below 50% map must stay **< 40** events).
3. Climbable hill: **contour / climb or retrace** — not a mass blockage
   stamp on the face.
4. Software tip reverse works. Past static α → sticky SOS until Reset.
5. Mow cut ≥ **~0.95** of planned, then `return_home` / `complete`.
6. Reset: clear tip / blockages / progress, keep fence, Idle Ready @ **1×**.
7. CI tests **fail** if thrash regresses (blockage explosion or map
   stuck ~40%).
8. Local acre smoke is documented below. CI still proves pocket e2e.

`tests/test_phase4_validation.py` writes a summary JSON
(`map_pct`, `n_blockages`, `cut_pct`, `phases`, `terrain_state`) and
asserts these gates.

## What changed

### Grade-aware blockage

`stamp_blockage` only for:

* body collision
* static-α / env tip-over
* drain / lip stop
* hard structure (building / bunker / bed / green)
* repeated zero-travel with **non-grade** advice

Climbable grade (`KIND_GRADE`, look-ahead climbable, attitude inside
`max_climb_slope_rad`) does **not** mint a learned no-go. A cooldown
and minimum stamp separation stop one stall from minting thousands of
events.

### Trail retrace

Explore (and mow) drop breadcrumbs. On explore stall, software
tip-risk, or a blocked frontier the robot **retraces N metres** along
that trail (reverse path), then replans. Mow keeps the existing reverse
/ skip-cluster recovery — following breadcrumbs on a 6×5 / 70 cm body
walked the chassis out of bounds. Immobilise / SOS does not reverse
into a past-static tip. Drain-edge stop is stamped and retraced, not
demoted to grade-reroute.

### Fused terrain state

One owner state: `ok | contour | tip_reverse | retrace | blocked_nogo | immobilised`.

One owner line. Snapshot must not report `tilt_kind=ok` with tip-risk
copy. Tip-risk with a near-level seated attitude must carry a
look-ahead reason.

## Local acre smoke (Jamie — not CI)

```bash
# Owner phone on :8766 (same as the Phase 0 capture)
jims-mower-owner --live --config acre_yard_demo --port 8766
# or the desktop live viewer
jims-mower-live --config acre_yard_demo

# Headless full-acre explore→mow (not CI)
jims-mower-live --config acre_yard --speed max --prepare-only --steps 20000 --out live_acre_full
```

CI pocket:

```bash
pytest tests/test_phase4_validation.py tests/test_full_fence_job.py -q
jims-mower-live --fast --speed max --steps 40 --prepare-only --out live_tiny
```

Default speed stays **1×**. No Full-explore toggle. Software tip is
not raised to static α.

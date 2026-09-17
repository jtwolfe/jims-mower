# Phase 4 gates (sign-off)

See [`TERRAIN_RECOVERY.md`](TERRAIN_RECOVERY.md) for the full plan and
the Phase 0 stuck numbers. This page is the checklist the PR body must
fill honestly.

| # | Gate | CI proof | Pass? |
| --- | --- | --- | --- |
| 1 | Explore map ≥ ~0.99 of keep-in | `test_phase4_validation_summary` / `test_taught_tiny_explore_mow_near_complete` | CI |
| 2 | Blockage events bounded (no 3263 @ 40% map) | `test_phase4_validation_summary`, `test_phase4_thrash_does_not_explode` | CI |
| 3 | Climbable hill: contour / retrace, not mass stamp | `test_gentle_hill_does_not_accumulate_blockage` | CI |
| 4 | Software tip reverse; static α → sticky SOS | `test_tip_immobilise` + phase4 tip check | CI |
| 5 | Mow cut ≥ ~0.95 then home / complete | same as #1 (#50 bars) | CI |
| 6 | Reset: clear tip/blockages/progress, keep fence, Idle Ready @ 1× | `test_live_reset_lands_idle_ready_at_1x` | CI |
| 7 | Thrash-regression tests fail on blockage explosion / 40% stuck | `test_phase4_thrash_does_not_explode` | CI |
| 8 | Local acre smoke documented | `docs/TERRAIN_RECOVERY.md` | docs |

Acre wall-clock is **not** CI. Do not claim the live acre is fixed
from the pocket run alone.

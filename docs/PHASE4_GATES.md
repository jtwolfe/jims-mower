# Phase 4 gates (sign-off)

See [`TERRAIN_RECOVERY.md`](TERRAIN_RECOVERY.md) for the full plan and
the Phase 0 stuck numbers. This page is the checklist the PR body must
fill honestly.

| # | Gate | CI proof | Pass? |
| --- | --- | --- | --- |
| 1 | Explore map ≥ ~0.99 of keep-in | `test_phase4_validation_summary` / `test_taught_tiny_explore_mow_near_complete` | **passed** (pocket map 1.00) |
| 2 | Blockage events bounded (no 3263 @ 40% map) | `test_phase4_validation_summary`, `test_phase4_thrash_does_not_explode` | **passed** (pocket `n_blockages=2`) |
| 3 | Climbable hill: contour / retrace, not mass stamp | `test_gentle_hill_does_not_accumulate_blockage` | **passed** (hill `n_blockages=0`) |
| 4 | Software tip reverse; static α → sticky SOS | `test_tip_immobilise` + phase4 tip check | **passed** |
| 5 | Mow cut ≥ ~0.95 then home / complete | same as #1 (#50 bars) | **passed** (cut 0.982 → complete) |
| 6 | Reset: clear tip/blockages/progress, keep fence, Idle Ready @ 1× | `test_live_reset_lands_idle_ready_at_1x` | **passed** |
| 7 | Thrash-regression tests fail on blockage explosion / 40% stuck | `test_phase4_thrash_does_not_explode` | **passed** |
| 8 | Local acre smoke documented | `docs/TERRAIN_RECOVERY.md` | **passed** (docs) |

Local full pytest is green. Acre wall-clock is **not** CI. Do **not**
claim the live `:8766` acre is fixed from the pocket run alone.
Numbers: `tests/fixtures/phase4_after.json`.

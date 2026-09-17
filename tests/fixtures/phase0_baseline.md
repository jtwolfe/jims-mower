# Phase 0 baseline (live :8766 acre_yard_demo)

Captured 2026-09-17 from owner `:8766` `acre_yard_demo` (post PR #51).
Trimmed snap: `phase0_baseline_snap.json`. Plan + gates:
[`docs/TERRAIN_RECOVERY.md`](../../docs/TERRAIN_RECOVERY.md),
[`docs/PHASE4_GATES.md`](../../docs/PHASE4_GATES.md).

Beat these before sign-off. Not resolved until Phase 4 gates pass.

```json
{
  "captured_at": "2026-09-17T12:27:45",
  "phase": "explore",
  "job_state": "running",
  "owner_copy": "Tip risk \u2014 reversing",
  "tilt_kind": "tip",
  "mode_banner": {
    "label": "Mapping yard",
    "tone": "map",
    "kind": "mapping",
    "hold": null,
    "phase": "explore",
    "job_state": "running"
  },
  "map_pct": 0.3993620247869058,
  "cut_pct": 0.0,
  "step": 8002,
  "speed_label": "1",
  "n_blockages": 3263,
  "blocked_cells": 1367,
  "blocked_frontiers": 3263,
  "n_frontiers": 40,
  "explore_reason": {
    "code": "tip_recovery",
    "label": "Tip risk \u2014 reversing \u00b7 map 40% of target 100% \u00b7 step 8001/4000 \u00b7 40 frontiers \u00b7 3263 unreachable",
    "map_pct": 0.3993620247869058,
    "target_pct": 1.0,
    "phase_step": 8001,
    "max_steps": 4000,
    "n_frontiers": 40,
    "n_waypoints": 15,
    "unreachable_frontiers": 3263,
    "skipped_frontiers": 51,
    "blocked_cells": 1367,
    "blocked_frontiers": 3263,
    "n_blockages": 3263,
    "full_explore": true
  },
  "pitch": 0.018748806757428522,
  "roll": 0.2984350992095766,
  "pose_xy": [
    35.483804455273116,
    18.75676524378783
  ],
  "skips": 0,
  "estop": false,
  "faults": []
}
```

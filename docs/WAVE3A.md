# WAVE 3A — Learning + ops tooling

Builds on latest `main` after WAVE 2A + WAVE 2B (perception pipeline).
No claimed mAP / FPS / imitation scores / RL returns. torch is the
WAVE 2B `[torch]` extra; Stable-Baselines3 is the WAVE 3A `[rl]` extra.
Neither is required for CI.

## Behaviour cloning

`jims-mower-bc collect` runs `TerrainPolicy` and writes compact
`(features, actions)` arrays — not raw camera pixels. `train` fits a
16-unit tanh MLP in numpy. `jims-mower-demo --policy bc` loads
`bc_weights.npz` (or `--bc-weights`) when the file exists and falls
back to the terrain planner otherwise.

`--log-bc DIR` on the demo also dumps pairs from whatever policy is
running.

## RL scaffold

Gymnasium `jims_mower/Mower-v0` is unchanged. `jims-mower-rl` runs a
**1-episode** CPU smoke:

| `--algo` | What it is |
| --- | --- |
| `reinforce` | Discrete REINFORCE over 5 wheel primitives |
| `random-search` | One random linear candidate |
| `sb3` | Optional PPO (`pip install -e ".[rl]"`); skipped if missing |

Every action is passed through `mask_action`: drain lip / channel in
the forward cone zeros *forward* wheels and the trimmer. Reverse and
pivots stay allowed.

## Safety / ops

- `SafeStateMachine` (RUN / LIMP / ESTOP / SAFE) wraps controller
  commands. `info["estop"]` latches zeros until `clear()`.
- Hardware paddle is a separate rail latch (`HardwareEstop`,
  `info["hw_estop"]`). `clear()` does not restore rails.
- `jims-mower-incident` writes a camera + hazard + advice scrubber
  (HTML slider + PNGs) from a `jims-mower-record` directory.
- `jims-mower-telemetry` emits coverage %, tip rate, drain entries,
  and living near-miss counts. `not_a_benchmark: true`.

## Owner UX stub

`jims-mower-owner` writes a phone-sized HTML map (geofence polygons +
plan polyline + pose). It is a mock overlay, not a shipping app.
WAVE UX-C (`jims-mower-app`, [`UX_C.md`](UX_C.md)) is the live local
shell + YardProfile API.

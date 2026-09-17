# Navigation blockages: unknown, free, and learned no-go

Honest model for Jamie. No claimed SLAM / mAP / RF. This is the onboard
map the owner phone already shows — extended so full explore can finish
a yard instead of thrashing at ~20%.

See also [`MISSION_FLOW.md`](MISSION_FLOW.md) (phases) and
[`UX.md`](UX.md) / [`UX_C.md`](UX_C.md) (owner overlay).

## 1. Simulator bug vs inadequate blockage management?

**Inadequate dynamic / invisible-obstacle management — not a physics lie.**

Gym physics still uses the true height field and body-circle collisions.
Control does **not**. `ObservedMap` only trusts cells the cameras / body
disk / ToF have stamped. That split is intentional (unknown is not
safe). The live symptom — Full explore ON, `map_pct≈0.21` of target
`0.80`, `40` frontiers, `unreachable_frontiers: 0`, still commanding
motion — is what that split does when a frontier is A*-reachable in
known-free but the chassis cannot actually enter the cells that would
grow the map.

A* succeeding to the nearest free/unknown lip is not “reachable yard.”
It is “reachable last known-free cell.” If that lip sits on a shed
face, a fence the RGB never classified, a swale lip, or a path
junction the robot cannot cross, the planner reports `0` unreachable
and retries the same cluster forever. Full explore then ignores
`max_explore_steps` while frontiers remain. That matches the `:8766`
trace (`step 8001/4000`).

The gym can also **block without painting an obstacle disk**:

| Physics can stop you | Painted on the owner map today? |
| --- | --- |
| Authored shed / bed / pond height (`sit_on_terrain`, tip / pond-stop) | Only after those cells are *observed* and ingest copies structure |
| Drain channel / lip | Only after observer hazard lands on seen cells |
| Tree / furniture body collision | Terminates the gym episode; live job ends — not the 21% thrash |
| Geofence keep-in / keep-out | Costmap + fence advice, not a “blocked” raster |
| Gentle swale / grade (acre undulation) | IMU tip-stop / slow; no learned no-go until this change |
| Camera ground-plane miss (wall, not dirt) | Seen mask does not grow; structure stays fog |

So: not a broken integrator. The robot was never taught to **mark**
“I commanded motion and did not progress.”

## 2. How did we manage invisible blockages before?

We did not have a distinct layer.

* **Unknown** (`~observed`) — lethal for A* (`unknown_blocked`).
* **Known-free** — observed, not hard structure, not a drain channel.
* **Semantic hazard** — drain / steep from the gated observer.
* **Hard structure** — building / bunker / garden / green, only on
  cells we have seen (authored god-view is ignored in fog).
* **Occupancy** — detections + short ToF, decaying; used on the *mow*
  costmap, not the explore costmap.
* **Tip recovery** — reverse / look-around on IMU `stop`. Did **not**
  stamp the map or invalidate the frontier.

Failed progress (pose barely moves, repeated tip, same 8-waypoint
explore path) left `unreachable_frontiers` at 0 because A* still
found a path to the nearest lip.

## 3. How we map them now

`ObservedMap.blockage` is a learned **no-go** raster, distinct from
fog, drain, and sheds:

* A no-progress watchdog (explore or mow) watches distance to the
  current waypoint, pose travel, map % growth, collision / tip, and
  “commanded forward but almost no travel” (wheel-slip proxy).
* After `mission.blockage_no_progress_steps` (default 16) it stamps a
  disk (`stamp_blockage`) **ahead of the chassis** — never under the
  robot — as observed + blockage + high occupancy.
* Those cells paint **Blocked / no-go learned** (`#c44c7a`) on the
  phone area overlay and enter `area_legend`.
* Persist with the mission bundle (`omap_blockage` / `blockage` in
  the npz). Cold-load keeps yesterday’s no-go.

Unknown stays unknown until we stamp. A learned blockage is
**known-blocked**: fog lifts, the cell is not free, A* will not
transit it.

## 4. How we deal with them

1. **Stamp** the local disk (lethal / high cost).
2. **Invalidate** that frontier (and a small cell cluster) so
   `plan_explore` will not retry it.
3. **Count** skipped + A*-failed + leftover untried frontiers as
   `unreachable_frontiers` (no more honest `0` while 40 lips remain).
4. **Recover** — reverse, pivot (same idea as mow tip-recovery), then
   replan. Prefer a frontier *away* from the last stamp.
5. After `blockage_replan_after` (default 6) blocked frontiers with
   stale map growth, owner copy / `explore_reason` reads
   **“Blocked — remapping around obstacle”**. Manual **Return** stays
   available. Full explore does **not** declare MAP READY at ~20%
   just because the step cap elapsed.

Demo early-exit (`explore_complete: 0.30` / short cap) is unchanged
when Full explore is off.

## 5. Owner line

`explore_reason.label` includes map % of target, step budget,
frontier count, and a non-zero unreachable count when we have
skipped or failed lips. Codes:

| code | Owner words |
| --- | --- |
| `seeking_frontier` | Seeking frontier · … |
| `blockage_stamped` / `remapping` | Blocked — remapping around obstacle · … |
| `frontier_skipped` | Frontier unreachable — skipping · … |
| `tip_recovery` | Tip recovery · … |
| `path_blocked` | Path blocked — looking around · … |

## What this is not

Not a new SLAM stack. Not a claim that every real fence will be
seen. Not an invented acre-runtime / RF / mAP number. Physics still
owns the true height field; the owner map and `MissionPolicy` own
observed + learned no-go.

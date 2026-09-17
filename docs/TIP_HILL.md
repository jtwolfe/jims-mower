# Tip-stop vs climbable grade

Honest model for Jamie. Camera tilt on a hill is **not** the same as a
tip. This note is the contract the gym and owner copy use. No invented
mAP / RF / acre runtime.

See also [`NAV_BLOCKAGES.md`](NAV_BLOCKAGES.md) (learned no-go) and
[`MISSION_FLOW.md`](MISSION_FLOW.md) (phases).

## 1. Two different trips

| Layer | What it measures | Trip | What the robot should do |
| --- | --- | --- | --- |
| **IMU tip-stop** | Chassis roll / pitch (accel + fused pose) | Near `robot.tip_roll_rad` / `tip_pitch_rad` (defaults 0.40 / 0.45 ≈ 23° / 26°) | **Tip risk — reversing.** Reverse, pivot, skip a short cluster. Do **not** keep reverse-looping. |
| **Climbable grade** | Observer / prior slope vs `planner.max_climb_slope_rad` (default 0.32 ≈ 18°; acre demo 0.34) | Below the climb cap, or between climb and the tip-safe margin | **Steep grade — contouring.** Slow on the face, A* prefers a contour. Climb if the path is still under the cap. |
| **Physics tip-over** | True height-field sit | Full tip thresholds | Episode / job terminate. Software trips stay **below** the static chassis tip (do not raise them to “match” fab). |

IMU `stop` used to fire at `imu_stop_frac * tip` (0.85 × 0.40 = **0.34 rad**).
That is the same number as acre-demo `max_climb_slope_rad`. A climbable
face looked like a tip, the camera was tilted, and the robot thrashed
reverse / Hold-safe. Grade-aware classify + a short median / hold filter
fixes that without moving the physics tip.

## 2. When we contour vs climb vs mark no-go

1. **Climb** — slope and chassis attitude ≤ `max_climb_slope_rad`. Slow
   corridor if the cell is labeled steep. Explore may walk it.
2. **Contour** — slope (or attitude) above the climb cap but **below**
   the tip-safe margin (`tip_lethal_frac * min(tip_roll, tip_pitch)`,
   default 0.95 × 0.40 ≈ 0.38 rad). Costmap / explore A* pay a high
   contour cost. Owner line: *Steep grade — contouring*. Not a learned
   blockage.
3. **Tip-stop** — filtered attitude at/above the tip-safe margin, or a
   true physics tip. Owner line: *Tip risk — reversing*. Explore stamps
   a learned no-go only on this path (same #46 blockage disk), not on a
   climbable hill.
4. **Lethal / no-go** — slope ≥ tip-safe margin, drain lip / channel,
   hard structure, or a **learned** blockage from #46. Fence stays.

Ridge / bank / drain-lip edges: a sharp **roll** on a bank is tip-risk.
A drain lip is still `reroute` from physics (do not drop a wheel). A
gentle swale stays grade / slow.

## 3. Filter

A single IMU spike on a gentle hill must not reverse. `TipHoldFilter`
takes a short median window (`imu_tilt_window`, default 5) and requires
`imu_stop_hold_steps` (default 3) consecutive tip classifications before
emitting `stop`. Attitude at the **full** tip trip still fires
immediately.

## 4. Reset vs Re-teach

Owner **Reset** (`POST /api/live/control` `cmd=reset`):

* stops the job
* clears tip cool-down, explore spin, software hold / safe, transient
  recovery
* returns to idle, ready for **Explore**
* **keeps** the taught fence
* **keeps** learned blockages unless `clear_blockages=true`

**Re-teach** is a different button. Reset does not drop the keep-in.

## What this is not

Not a retune of gym physics tip constants. Not a claim that every real
bank is safe. Not a new SLAM stack.

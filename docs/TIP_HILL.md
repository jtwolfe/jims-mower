# Tip-stop vs climbable grade

Honest model for Jamie. Camera tilt on a hill is **not** the same as a
tip. This note is the contract the gym and owner copy use. No invented
mAP / RF / acre runtime.

See also [`CHASSIS_PHYSICS.md`](CHASSIS_PHYSICS.md) (kinematic sit,
acre mesh, speed, sensors), [`NAV_BLOCKAGES.md`](NAV_BLOCKAGES.md)
(learned no-go) and [`MISSION_FLOW.md`](MISSION_FLOW.md) (phases).

## 1. Two different trips

| Layer | What it measures | Trip | What the robot should do |
| --- | --- | --- | --- |
| **IMU tip-stop** | Chassis roll / pitch (accel + fused pose) | Past `max_climb_slope_rad`, or at `robot.tip_roll_rad` / `tip_pitch_rad` (defaults 0.55 / 0.55 ≈ 31.5°, about half of static \(\alpha\)) | **Tip risk — reversing.** Reverse, pivot, skip a short cluster. Do **not** keep reverse-looping. |
| **Climbable grade** | Observer / prior slope vs `planner.max_climb_slope_rad` (default 0.32 ≈ 18°; acre demo 0.34) | Chassis ≤ climb cap; mapped cells between climb and the tip-safe margin | **Steep grade — contouring.** Slow on the face, A* prefers a contour. Climb if the path is still under the cap. |
| **Physics tip-over** | True height-field sit vs static \(\alpha(t,b,h_\mathrm{cg})\) | \(\alpha = \mathrm{atan}((t/2)/h_\mathrm{cg})\) ≈ **63°** when \(t=b=0.55\) m and assumed \(h_\mathrm{cg}=0.14\) m | **Latch tipped / immobilised.** Owner SOS, `tilt_kind=tip`, not Idle Ready. Gym episode may still terminate; live/mission must surface the latch until Reset / retrieve. Software trips stay **below** this static \(\alpha\) (do not raise them to 63°). |

IMU `stop` used to fire at `imu_stop_frac * tip` (0.85 × 0.40 = **0.34 rad**).
That was the same number as acre-demo `max_climb_slope_rad`. A climbable
face looked like a tip, the camera was tilted, and the robot thrashed
reverse / Hold-safe. Grade-aware classify + a short median / hold filter
fixes that. Software tip is now ~0.55 rad; static \(\alpha\) is ~1.10 rad.
Do not collapse those two numbers.

## 2. When we contour vs climb vs mark no-go

1. **Climb** — slope and chassis attitude ≤ `max_climb_slope_rad`. Slow
   corridor if the cell is labeled steep. Explore may walk it.
2. **Contour (planner)** — *mapped / observed* slope above the climb
   cap but **below** the tip-safe margin (`tip_lethal_frac *
   min(tip_roll, tip_pitch)`, default 0.95 × 0.55 ≈ 0.52 rad). Costmap
   / explore A* pay a high contour cost. Incomplete elevation must not
   invent a blocked wall. Owner line: *Steep grade — contouring*. Not a
   learned blockage.
3. **Tip-stop (chassis IMU)** — attitude **past** `max_climb_slope_rad`
   but still **under** the physics tip. This is urgent: a ridge can jump
   ~0.10 rad in one physics step, so we do not wait for `imu_stop_frac`.
   Owner line: *Tip risk — reversing*. Explore stamps a learned no-go
   only on this path (same #46 blockage disk), not on a climbable hill.
   Sit look-ahead that would **exceed tip** is a hard stop on the
   physics command (do not drive *into* that face). During `mow` that
   is **contour / reroute**, not a waypoint-cluster skip — skipping
   here left a tiny keep-in at ~50% cut. A single non-urgent IMU spike
   still holds (see Filter).
4. **Past-tip / immobilise** — seated `|roll|` or `|pitch|` ≥
   static \(\alpha(t,b,h_\mathrm{cg})\), or env `tipover`. Latch.
   Owner line: *SOS — immobilised. Retrieve the mower.* `tilt_kind=tip`.
   Job is not Ready. Survives idle, explore stall, and gym terminate
   until explicit **Reset / retrieve**. Snapshot must not report
   `tilt_kind=ok` while the pose is past the software trip, and must
   not report Ready while past static \(\alpha\).
5. **Lethal / no-go** — slope ≥ tip-safe margin, drain lip / channel,
   hard structure, or a **learned** blockage from #46. Fence stays.

Ridge / bank / drain-lip edges: a sharp **roll** on a bank is tip-risk.
A drain lip is still `reroute` from physics (do not drop a wheel). A
gentle swale stays grade / slow.

## 3. Filter

A single IMU spike on a gentle hill must not reverse. `TipHoldFilter`
takes a short median window (`imu_tilt_window`, default 5) and requires
`imu_stop_hold_steps` (default 3) consecutive *non-urgent* tip
classifications before emitting `stop`. **Urgent** samples (past the
climb cap, or attitude at the full physics tip) fire immediately — a
real tip after level driving must not look like a spike.

## 4. Reset vs Re-teach

Owner **Reset** (`POST /api/live/control` `cmd=reset`):

* stops the job
* clears tip cool-down, explore spin, software hold / safe, transient
  recovery
* clears the **tip latch** and reseats at home (retrieve). If that sit
  is still past tip, SOS re-arms — Reset does not paint Ready over a
  fallen-over chassis
* returns to idle, ready for **Explore** only when attitude is under tip
* **keeps** the taught fence
* **keeps** learned blockages unless `clear_blockages=true`

**Re-teach** is a different button. Reset does not drop the keep-in.

## 5. Jamie's four questions (honest)

1. **Real rolling mower, or ground-polygon tangent?** Kinematic sit.
   `integrate_pose` then `sit_on_terrain` (`atan2` of four contact
   heights). Immobilise uses static \(\alpha(t,b,h_\mathrm{cg})\), not
   a rolling inertia engine. See
   [`CHASSIS_PHYSICS.md`](CHASSIS_PHYSICS.md).
2. **Raise height-field resolution?** Yes on the acre. `acre_yard` /
   `acre_yard_demo` are **0.25 m** (was 0.50 m — one cell across the
   chassis). Golf / default stay as-is.
3. **Forward ground lidar, or cameras enough?** Cameras + IMU + corner
   ToF + existing stereo first. A short sit-probe (`grade_look_ahead_m`)
   uses that height so we slow / contour *before* seated pitch crosses
   climb → tip. Optional forward ranger later. No lidar stack in this
   change.
4. **Decrease top speed while moving around?** Yes. Explore cap
   `explore_cruise: 0.45` (≈ 0.54 m/s at `max_wheel_speed_mps: 1.2`).
   Further slow on `KIND_GRADE` / steep via `grade_speed_factor`.

Owner lines stay **Steep grade — contouring** vs **Tip risk — reversing**
vs **SOS — immobilised** once past static \(\alpha\). Software trips
stay at about half of static. We did not raise them to 63° to “pass”
a hill. Still kinematic sit plus a static-α latch.

## What this is not

Not a claim that every real bank is safe. Not a new SLAM stack. Not a
rolling rigid-body engine and not a lidar product. Not a license to
raise software tips to static \(\alpha\).

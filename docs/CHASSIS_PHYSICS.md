# Chassis physics (honest)

What the gym actually integrates when Jamie sees the mower “tilt
over” on a hill. No invented mAP / RF / field numbers. Owner copy stays
in [`TIP_HILL.md`](TIP_HILL.md): *Steep grade — contouring* vs
*Tip risk — reversing*.

## Are we simulating a real rolling mower?

**No.** Motion is planar differential-drive (`integrate_pose`), then
`sit_on_terrain` samples front / rear / left / right heights and sets
pitch and roll with `atan2`. That is **kinematic seating on a height
field** — a ground-polygon tangent — not rigid-body rolling, not
inertia, not CG tip moments.

Software tip checks compare that seated (and IMU) pitch / roll to
`tip_roll_rad` / `tip_pitch_rad` / `max_climb_slope_rad`. Physics
tip-over is the same sit crossing the tip thresholds. We did **not**
retune those fab constants in this change.

Full CG tip-moment physics is **future work**. This repo does not ship
a rolling rigid-body engine.

## Should we increase ground-polygon / height-field resolution?

**Yes, on the acre.** Default yards stay at `world.resolution_m: 0.10`.
`acre_yard` / `acre_yard_demo` were `0.50` m — one cell across a
~0.50 m chassis / ~0.40 m wheelbase — so drain lips and berm faces
became one-cell cliffs and pitch jumped a step. Those scenarios are
now **0.25 m** (70×58 m → 280×232 = 64,960 cells). Fine enough that
front and rear contacts are not the same cell; coarse enough that a
live `--speed 5` demo is still usable. Not `0.10` on the acre (that
would be ~406k cells). Golf / default stay as-is.

## Need a forward ground lidar, or are cameras enough?

**Cameras + IMU + existing corner ToF + stereo look-ahead first.**
The gym already has a multi-cam rig, a 6–12 cm stereo pair (0.8–4 m
band), four downward ToF corners, and IMU. This change probes that
height (physics field and observer / stereo+ToF raster) a short
distance ahead (`planner.grade_look_ahead_m`, default 1.10 m) with the
same sit model so the controller can slow or contour *before* seated
pitch crosses climb → tip.

A dedicated forward ground ranger is **optional later**, not a v1 hard
dependency. We did not add a lidar stack.

## Decrease top speed while moving around?

**Yes — especially explore, steep, and unknown.** `max_wheel_speed_mps:
1.2` with `dt: 0.10` is a 0.12 m step at a 1.0 wheel command. Demo
explore used to command `explore_cruise: 0.98` (~1.2 m/s). Defaults are
now `explore_cruise: 0.45` (≈ 0.54 m/s at 1.2 m/s wheel max).
`KIND_GRADE` / look-ahead steep multiplies by `grade_speed_factor`
(0.50). Other `slow` advice still uses `slow_speed_factor` (0.35).
CI `--fast` settings may still raise cruise so tests finish.

## What this is not

Not a claim that every real bank is safe. Not a new SLAM or lidar
product. Not a change to owner phone/desktop chrome from #47.

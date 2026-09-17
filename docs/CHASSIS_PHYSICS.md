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
`tip_roll_rad` / `tip_pitch_rad` / `max_climb_slope_rad`. Those are
**earlier** than the geometric static tip
\(\alpha = \mathrm{atan}((t/2)/h_\mathrm{cg})\) (and the same in
pitch with \(b\)). Defaults: software ≈ **0.55 rad (~31.5°)**,
static \(\alpha \approx 1.10\) rad (**63°**) when
\(t = b = 0.55\) m and assumed \(h_\mathrm{cg} = 0.14\) m.

True tip-over / immobilise is the sit crossing **static \(\alpha\)**,
not the software trip. `sit_on_terrain` will report |roll| / |pitch|
past the software trip on a steep face that is still under \(\alpha\);
that is **Tip risk — reversing**, not SOS. Once seated attitude (or
env `tipover`) crosses static \(\alpha\) we **latch** tipped /
immobilised: wheels hold, owner SOS, `tilt_kind=tip`. Idle / explore
stall / gym terminate must not clear that to Ready. Only owner
**Reset / retrieve** (reseat at home) clears the latch — and if the
retrieve pose still sits past static \(\alpha\), it re-arms.

That latch is a **minimal CG / static-tip latch** (support polygon
\(t \times b\), assumed belly \(h_\mathrm{cg}\)). Not a rolling
rigid-body moment engine. Full inertia physics is still **future
work**. Never raise software trips to \(\alpha\) to hide a tall CG.

## Should we increase ground-polygon / height-field resolution?

**Yes, on the acre.** Default yards stay at `world.resolution_m: 0.10`.
`acre_yard` / `acre_yard_demo` were `0.50` m — one cell across a
~0.70 m chassis / ~0.55 m wheelbase — so drain lips and berm faces
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
pitch crosses climb → tip. A predicted sit **past the physics tip**
is a **hard stop** (do not keep commanding forward). Climbable faces
still contour. The probe starts ~0.06 m ahead of the hub so a bank
inside the old collision-radius blind zone cannot roll the chassis in one step.

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
product. Software trips stay **below** static \(\alpha\). Past-static
is immobilise / SOS, not Idle Ready.

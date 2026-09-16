# Surveyed origin + day-2 session (MAP-5 / MAP-4)

Build-order **§14**. Software model only. This repo has **no WGS84 field
survey**. Do not treat example lat/lon (none shipped) or gym metres as
a taped peg.

Linked from [`PRODUCT_TO_HARDWARE.md`](PRODUCT_TO_HARDWARE.md),
[`CALIBRATION.md`](CALIBRATION.md).

---

## What is stored

`YardProfile.origin` (`jims_mower.yard.v1`):

| Field | Meaning |
| --- | --- |
| `lat_deg` / `lon_deg` / `alt_m` | Optional WGS84 label of the **peg** (dock / survey nail). Null until a human RTKs it. |
| `e_m` / `n_m` / `u_m` | Where that peg sits in the gym **world** frame. Default `(0, 0, 0)` = SW corner. |
| `frame` | `local_enu` |
| `surveyed` | `true` only after a real tape/RTK. Requires lat/lon. |

**Keep-in / keep-out / home** are metres **east / north of the peg**.
World XY used by physics and the planner is `origin + local ENU`.

When origin is the default `(0, 0, 0)`, local ENU **is** today's gym
world frame. Existing profiles keep working.

ICD `gps` stays `(x, y, z, valid)`. Those metres are **ENU relative to
the peg**. `valid=0` is dropout — fence advice falls back to chassis
pose. Fusion converts ENU → world before blending.

---

## How to set a surveyed origin on the rig

Tools: RTK rover (or a total station), a steel tape, a nail or painted
peg at the dock.

1. **Plant the peg** at the dock / home pose. This is the ENU origin,
   not the lot corner unless you want it to be.
2. **Record WGS84** of the peg (RTK fix). Write `lat_deg`, `lon_deg`,
   `alt_m` on the YardProfile. Set `surveyed: true` only when those
   numbers came off the receiver, not a map click.
3. **Teach keep-in in metres** from that peg (walk / tape east-north).
   The phone teach trail is already in gym metres; if the peg is the
   gym SW corner, leave `e_m=n_m=0`.
4. **If the peg is not world `(0,0)`**, set `e_m` / `n_m` to the peg's
   gym coordinates so keep-in local + origin = world.
5. **Tape-stop acceptance** (not claimed here): drive at the fence
   with a valid GNSS fix. The body must stop **before** the tape, not
   3 m past. Real RTK + a surveyed peg are required for that line.

There is **no** WGS84↔ENU converter in this repo that you should trust
on a lawn. Do not invent an Earth-radius shortcut and call it a survey.

---

## Multi-session (MAP-4)

`jims-mower-mission` / `save_mission` writes:

| File | Contents |
| --- | --- |
| `session.npz` | cut / uncut grass, pose, **ObservedMap fog**, elev, lock |
| `session.json` | meta (phase, uncut cells, origin) |
| `session.yard.json` | YardProfile (keep-in, home, origin) |

A **new process** loads that bundle (`reset(options={"load_mission": …})`
+ `MissionPolicy.restore_session`). Uncut cells stay planned. Fog stays
dark. No reteach.

Live owner pause / close also writes `live_out/session.npz`.

This is a cold file load, not “the same sim object is still in RAM.”

---

## Honesty

- No acre runtime, mAP, FPS, or RF range.
- Gym GNSS is noisy metres + a `valid` bit, not a u-blox log.
- Field IoU / matcher / tape still later (build-order §11–§13 leftovers).

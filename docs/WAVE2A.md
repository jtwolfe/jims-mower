# WAVE 2A — Dynamic world + behaviour

Builds on latest `main` after WAVE 1A + 1C + 1B (EKF / record-replay).
WAVE 2A keeps the EKF default pose filter and adds movers / geofences /
recovery / hand-signal hooks / mission resume.

No claimed mAP / FPS / sim-to-real scores. Planner rasters stay small
enough for an Orin Nano-class in-process loop. Do not run the gym
renderer on-box.

## Moving people / animals

`step_movers` still random-walks by default. Authored obstacles (and
`world.movers.default_mode`) can use:

| Mode | Behaviour |
| --- | --- |
| `wander` | heading jitter (legacy) |
| `patrol` | bounce along waypoints |
| `loop` | cycle waypoints |
| `line` | walk to the last point and stop |

Occupancy is rebuilt from detections every step. `info["living_advice"]`
extends the trimmer interlock: **slow** / **reroute** / **stop**. The
terrain policy replans when a detection sits on the path.

`world.movers.density`: `sparse` | `default` | `busy`.

## Geofences

Scenario YAML keep-in / keep-out polygons (GPS frame = world XY metres):

```yaml
geofence:
  keep_in: [[0.6, 0.6], [11.4, 0.6], [11.4, 11.4], [0.6, 11.4]]
  keep_out:
    - [[8.2, 8.2], [10.4, 8.2], [10.4, 10.4], [8.2, 10.4]]
```

Legacy `geofence: [[x, y], ...]` is still a keep-in. The costmap blocks
cells outside keep-in and inside keep-out (plus `planner.geofence_inflate_m`).
`in_yard` terminates on a true violation; `geofence_advice` slows before
touch.

Demo: `jims-mower-demo --config geofence_movers`.

## Recovery

Repeated `stop` / `reroute` from tip risk or a wheel-in-channel lip:

1. Reverse a few steps
2. Zero-turn pivot
3. After `planner.max_recoveries`, hold and set `help` (call-for-help)

Living-thing stops do **not** start this machine (do not reverse into a person).

## Hand-signal hooks

When `curriculum.hand_signals` is on, `obs["hand_signal"]` overrides wheels:

| Signal | Override |
| --- | --- |
| stop | zero wheels + trimmer |
| go | clear help, resume coverage |
| follow | track nearest person `world_xy` (mock/oracle is enough) |
| back | reverse |

This is a policy hook, not a trained classifier. With
`detector_backend: appearance`, a gym red-bias on the person crop can
fill `hand_signal` (KIND_RGB person → `stop`). That is **not** a field
gesture model and has **no** confusion matrix.

## Multi-session resume

Grass persist already writes the yard mask. WAVE 2A adds pose:

```bash
jims-mower-demo --save-mission /tmp/mission.npz --out demo_save
jims-mower-mission inspect --in /tmp/mission.npz
jims-mower-mission resume --in /tmp/mission.npz --out demo_resume
```

Schema: `jims_mower.mission.v1` (cut, grass, pose, scenario, seed).

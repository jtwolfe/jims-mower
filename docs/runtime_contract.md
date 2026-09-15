# Runtime contract

Versioned messages shared by the gym, the logger, and (later) an Orin
Nano-class runtime. Python types and JSON Schema dicts live in
`jims_mower.contract`. Current **`version` is `"1"`**. Bump it when a
required field changes.

Raster payloads (cameras, maps) are stored as numpy arrays in episode
`frames/*.npz`. The JSON objects below are the metadata / interchange
shape.

## CameraFrame

| Field | Type | Notes |
| --- | --- | --- |
| `version` | string | `"1"` |
| `name` | string | `front`, `rear`, … |
| `width`, `height` | int | pixels |
| `encoding` | string | default `rgb8` |
| `stamp_s` | number | host time, seconds |
| `frame_id` | string | default `camera` |

Gym: `obs["cameras"][name]` is uint8 RGB `(H, W, 3)`.

## ImuSample

| Field | Type | Notes |
| --- | --- | --- |
| `version` | string | `"1"` |
| `accel_mps2` | `[ax, ay, az]` | body specific force; level rest ≈ `[0, 0, 9.81]` |
| `gyro_radps` | `[gx, gy, gz]` | body rad/s |
| `stamp_s` | number | |
| `frame_id` | string | default `imu` |

Gym: `obs["imu"]` is length-6 `[ax, ay, az, gx, gy, gz]`.

## GpsFix

| Field | Type | Notes |
| --- | --- | --- |
| `version` | string | `"1"` |
| `x`, `y`, `z` | number | metres, world / yard frame |
| `valid` | number | `1` fix, `0` dropout |
| `stamp_s` | number | |
| `frame_id` | string | default `gps` |

Gym: `obs["gps"]` is `[x, y, z, valid]`. `z` is optional on the robot
(`planner.ekf.use_gps_z`).

## TofArray

| Field | Type | Notes |
| --- | --- | --- |
| `version` | string | `"1"` |
| `ranges_m` | `[FL, FR, RL, RR]` | downward metres |
| `stamp_s` | number | |
| `frame_id` | string | default `tof` |

Gym: `obs["tof"]` length 4.

## DetectionSet

| Field | Type | Notes |
| --- | --- | --- |
| `version` | string | `"1"` |
| `detections` | array | each: `label`, `camera`, `bbox [u,v,w,h]`, `confidence`, optional `world_xy`, `hand_signal`, `category`, `depth_m` |
| `stamp_s` | number | |

Gym: padded `obs["detections"]` plus `info["detections"]` dicts.

## TerrainMaps

| Field | Type | Notes |
| --- | --- | --- |
| `version` | string | `"1"` |
| `elevation` / `slope` / `hazard` / `confidence` | string | raster names in the npz (same `H×W` as coverage) |
| `resolution_m`, `width_m`, `height_m` | number | |
| `source` | string | `oracle` / `heuristic` / `blind` |
| `stamp_s` | number | |

Hazard labels: `0` free, `1` steep, `2` drain lip, `3` channel.
`confidence` is `0–1` per cell.

## Plan

| Field | Type | Notes |
| --- | --- | --- |
| `version` | string | `"1"` |
| `waypoints` | `[{x,y}, …]` or `[[x,y], …]` | world metres |
| `index` | int | current waypoint |
| `n_segments` | int | |
| `advice` | string | `ok` / `slow` / `reroute` / `stop` |
| `stamp_s` | number | |

## WheelCommand

| Field | Type | Notes |
| --- | --- | --- |
| `version` | string | `"1"` |
| `left`, `right` | number | fraction of `max_wheel_speed_mps`, `[-1, 1]` |
| `trimmer` | number | request; interlock may refuse (`> 0.5` means on) |
| `stamp_s` | number | |

Gym action is `[left, right, trimmer]`.

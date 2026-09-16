# WAVE 3B — runtime / Orin packaging / design studies

Builds on latest `main` after WAVE 2A. This wave does **not** claim mAP,
FPS, or a measured Orin power trace. Default `pip install` stays
gymnasium + numpy + pyyaml + pillow.

## Fake drivers

`jims_mower.runtime.drivers` publish gym (or recorded) observations as
versioned contract messages onto an in-process bus:

| Driver | Pretends to be | Contract kind | Bus tag |
| --- | --- | --- | --- |
| `FakeImuDriver` | I2C 6-axis (`0x68`) | `ImuSample` | `i2c` |
| `FakeGnssDriver` | UART GNSS (`/dev/ttyUSB0`) | `GpsFix` | `uart` |
| `FakeCsiDriver` | CSI / NVMM (`nvarguscamerasrc`) | `CameraFrame` + `obs["cameras"]` | `csi` |
| `FakeTofDriver` | I2C ToF (`0x29`) | `TofArray` | `i2c` |

Queues are `InProcessBus` deques. Payloads pass `validate_payload`.
Pixel rasters for the gym / episode `npz` stay numpy. `FakeCsiDriver` /
`FakeGstAdapter` can also **fill** `obs["cameras"]` at the ICD contract
size (named frames, fresh stamps). The bus `CameraFrame` message is
still metadata (name, size, encoding, stamp) matching
[`runtime_contract.md`](runtime_contract.md).

These are **not** Linux `i2c-dev` or GStreamer drivers. On the Orin,
replace the publish path; keep the kinds.

## Bridge (no ZMQ / gRPC / ROS 2 in CI)

`MultiprocessBridge` shuttles obs → contract messages. Default is
in-process. `--process` / `use_process=True` starts one `spawn` child
so the gym process can stay separate from the driver process.

```bash
jims-mower-record --out /tmp/ep --steps 8 --cameras 4
jims-mower-bridge /tmp/ep --out /tmp/ep-bridge.json
```

Optional `[ros2]` extra is a **marker**. Node stubs live in
`jims_mower.runtime.ros2_stubs` and import-guard `rclpy`. CI does not
install a ROS 2 distro. Topic map: `/jims_mower/imu`, `/gps`,
`/camera`, `/tof`, `/cmd_vel`, …

## Battery / thermal stub

`OrinBudget` is a first-order SOC + thermal RC. Class-scale watts /
watt-hours, not board telemetry. Off by default (`runtime.enabled:
false`). When enabled, `info["budget_advice"]` is `ok` / `slow` /
`stop` and `TerrainPolicy` combines it with terrain / living / fence
advice (limp or hold).

`info` also carries `battery_soc`, `thermal_c`, `budget_reason`,
`not_a_power_trace: true`.

## Design studies

```bash
jims-mower-study --dry-run --out study_out
# laptop / overnight (not PR CI):
jims-mower-study --scenario steep_yard --steps 20 --out study_out
```

Sweeps camera count **4 / 5 / 6**, ToF **0 / 2 / 4**, IMU noise
multipliers (default 1× and 4×) on frozen seeds. Writes `report.md` +
`report.csv` with **tip_rate** and **drain_entry_rate** (fraction of
episodes with tip-over / wheel-in-channel). Not detector recall. No
FPS column.

ToF `count` 2 keeps FL/FR only; unused corners stay 0 so
`stamp_tof_corners` skips them.

## Jetson packaging

See [`JETSON.md`](JETSON.md), [`docker/Dockerfile.aarch64`](../docker/Dockerfile.aarch64),
and `jims-mower-export-trt --dry-run` (no weights shipped).

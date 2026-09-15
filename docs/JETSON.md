# Jetson Orin Nano packaging notes

This package is the **training / eval gym** plus a thin runtime *stub*.
It is not a flashed image and not a measured onboard stack.

Target class: **Orin Nano 8 GB, ARM64, JetPack 6**.

## Dockerfile.aarch64

[`docker/Dockerfile.aarch64`](../docker/Dockerfile.aarch64) is a
**notes-first** starting point. GitHub Actions CI is x86_64 and does
**not** build this file. Build it on an aarch64 host / the board:

```bash
# on the Jetson, after JetPack 6
docker build -f docker/Dockerfile.aarch64 -t jims-mower:orin .
```

The image installs the gym extras used for **record / replay / bridge**
only. It does not install a desktop OpenCV GUI, a ROS 2 distro, or a
trained TensorRT engine.

## What NOT to run on-box

| Do not run | Why |
| --- | --- |
| `jims_mower.renderer` / gym `render()` | Numpy pinhole views are for the laptop gym. Real cameras are CSI/USB. |
| `OracleTerrainObserver` | God-view height field. Training / eval only. |
| `jims-mower-farm` overnight matrix | Laptop / CI job. Burns idle time and writes scorecards, not a robot loop. |
| `jims-mower-study` full sweep | Same — design study on frozen seeds, not a field tool. |
| Desktop OpenCV HighGUI / Qt | No display on a headless mower. |
| Full VIO / SLAM stack in this repo | Keep fusion behind `EkfPoseFilter` / your EKF. |

On-box loop (your code, not shipped):

```
CSI / GStreamer / NVMM  →  CameraFrame   (`GstNvmmAdapter` stub; not in CI)
I2C IMU                 →  ImuSample
UART GNSS               →  GpsFix
I2C ToF                 →  TofArray
        ↓
your TerrainObserver / Detector  (TensorRT INT8 is the usual path)
        ↓
costmap + planner + controller in-process
```

Use `Fake*Driver` + `MultiprocessBridge` on a laptop to keep the same
message kinds. Swap the publish path on the board.

## TensorRT placeholder

```bash
jims-mower-export-trt --dry-run --out /tmp/trt.json
# or:
python scripts/export_tensorrt.py --dry-run
```

No ONNX is in this repository. Point `--onnx` at **your** drain / lip /
bank / grass head. The script prints a `trtexec` line and sets
`fps_claim: null` / `map_claim: null`. Do not paste invented FPS.

`TrtDetector` / `TrtTerrainObserver` (`perception.detector_backend: trt`)
are the same contract: they load an engine **if you provide one**, otherwise
they delegate to the mock / numpy heads. `SensorWatchdog` zeros wheels when
IMU or camera frames freeze (`runtime.watchdog.enabled`).

Example body-frame extrinsics: [`configs/orin/extrinsics_6cam.yaml`](../configs/orin/extrinsics_6cam.yaml).

## Optional ROS 2

```bash
pip install -e ".[ros2]"   # marker extra; does not pip-install rclpy
```

Install Humble / Jazzy from the distro on the Orin if you want nodes.
Stubs: `jims_mower.runtime.ros2_stubs`. CI uses the multiprocessing
bridge instead.

## Record / replay on the bench

Same episode directory the gym already writes:

```bash
jims-mower-record --out /tmp/ep --steps 20 --cameras 4
jims-mower-replay /tmp/ep --mode offline
jims-mower-bridge /tmp/ep
```

Downsample real cameras to the gym contract (default 80×60 in sim) before
the observer. Keep rasters small.

# docs

Wave notes and the on-box message contract live here.

| File | What it is |
| --- | --- |
| [`WAVE1B.md`](WAVE1B.md) | EKF pose filter, uncertainty costmap, record/replay, latency harness |
| [`WAVE1C.md`](WAVE1C.md) | Scenario library, domain randomisation, weather, grass stub |
| [`WAVE2A.md`](WAVE2A.md) | Movers, geofences, recovery, hand-signal hooks, mission resume |
| [`WAVE2B.md`](WAVE2B.md) | Numpy terrain stub, BEV fuse, temporal filters, train CLI |
| [`WAVE3A.md`](WAVE3A.md) | BC stub, RL scaffold, ESTOP/limp, incident/telemetry, owner overlay |
| [`WAVE3B.md`](WAVE3B.md) | Fake drivers, multiprocessing bridge, studies, Orin budget stub |
| [`JETSON.md`](JETSON.md) | What not to run on-box; Dockerfile.aarch64; TensorRT placeholder |
| [`runtime_contract.md`](runtime_contract.md) | Versioned CameraFrame / IMU / GPS / ToF / maps / Plan / WheelCommand / SafeState |

Repo-root [`ROADMAP.md`](../ROADMAP.md) and [`ICD.md`](../ICD.md) landed
with WAVE 1A. Later waves add sections; they do not invent mAP / FPS.

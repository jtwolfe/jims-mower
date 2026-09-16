# On-box loop (RT-7)

A flashed Orin can run the control loop **without** a laptop gym
renderer. The process is `jims-mower-onbox`. It must not import
`jims_mower.renderer`.

This is **not** SIL. Hardware ESTOP is still the paddle
([`ESTOP.md`](ESTOP.md)). Fake CSI / IMU adapters are the software
path until JetPack + real cameras. No FPS claim.

## What runs

| Piece | On-box |
| --- | --- |
| Config | `configs/orin/bench.yaml` (or your overlay) |
| Cameras | `FakeCsiDriver` / `FakeGstAdapter`, or `GstNvmmAdapter` (raises without Gst) |
| IMU | level-rest fake until a real driver |
| Watchdog | on; frozen stamps zero wheels |
| HW ESTOP | last rail filter (sim latch). Field paddle still required |
| Black box | JSONL path you pass (`--blackbox`) |

Hold command defaults to `0,0,0`. Policy / phone Start is a later
process. This unit keeps rails honest while you bring the box up.

## Install

```bash
# on the Orin, after JetPack + `pip install -e .`
sudo mkdir -p /etc/jims-mower /var/lib/jims-mower
sudo cp configs/orin/bench.yaml /etc/jims-mower/bench.yaml
# edit adapter / blackbox path for your image
sudo cp deploy/jims-mower.service /etc/systemd/system/jims-mower.service
# fix User= and ExecStart= paths for the venv
sudo systemctl daemon-reload
sudo systemctl enable --now jims-mower.service
```

First-boot checks (laptop or Orin, may use the gym for self-test):

```bash
jims-mower-bringup --config configs/orin/bench.yaml
jims-mower-onbox --config configs/orin/bench.yaml --steps 8 --blackbox /tmp/bb.jsonl
```

## Journal

```bash
journalctl -u jims-mower.service -f
```

The unit runs `--hold`. Ctrl-C / `systemctl stop` ends the process.
That is **not** a hardware ESTOP.

## ESTOP is still hardware

`HardwareEstop` in this process is the sim / rail-filter model. A
software crash, `systemctl stop`, or owner app Start **cannot** replace
the paddle. See [`ESTOP.md`](ESTOP.md) and
[`HARDWARE_DESIGN.md`](HARDWARE_DESIGN.md) §10.

## Clock / timezone (SCH-2)

Set the Orin zone before the first scheduled job. Default owner locale
is `Australia/Brisbane` (09:00 means 09:00 there, not UTC).

```bash
sudo timedatectl set-timezone Australia/Brisbane
# or export TZ=Australia/Brisbane for a one-off process
timedatectl status
```

`YardProfile.schedule.timezone` is that IANA string. `local` still
resolves to the host zone if you leave it.

## Pair before Start (UX-5)

The live owner app refuses Start until Bluetooth is **paired**
(`owner.require_pair: true`). Pairing is still simulated (no BlueZ).
Unpair / radio-lost **holds the job safe** — it does not ESTOP.
Re-pair before Start. See [`UX_C.md`](UX_C.md).

## Honesty

- No claimed onboard FPS / mAP.
- `GstNvmmAdapter` is not physical CSI until you wire JetPack.
- OTA A/B and BT/LoRa stay stubs. Radio is sim transports + `rf_claim: null`.
  Do not claim RF range.

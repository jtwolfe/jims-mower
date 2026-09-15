# WAVE UX-C — thin owner app + YardProfile API

Local phone-sized owner shell and a small HTTP JSON API. Not a store
listing, not a cloud account, not a claimed mapping / radio score.

The WAVE 3A `jims-mower-owner` overlay remains a **static HTML export**.
This wave is the live app: load / save a **YardProfile**, command a sim
or recorded episode, and walk unbox → first mow on a narrow layout.

## Buy → mow checklist

Matches the radio profile: **Bluetooth pair required**, **Wi-Fi optional**,
**LoRa long-range** for command / status once the mower leaves the porch.

1. **Unbox** — charge to a green ring, unfold the handle, confirm the
   hardware ESTOP paddle moves freely. Wheel the mower to the yard edge.
2. **Bluetooth pair** — phone stays next to the mower for first contact.
   The app will not skip this step. No account, no QR cloud claim.
3. **Wi-Fi (optional)** — skip unless you want OTA / map upload on the
   home AP. The mower does not need Wi-Fi to mow.
4. **LoRa long-range** — default command link after pairing. Walk to the
   far fence and confirm the status pill still reads a LoRa link. If it
   drops, move closer; do not “fix” it by forcing Wi-Fi.
5. **Place home** — park on the dock / start pose. The app stores
   `home` `{x, y, theta}` in the YardProfile. Return-to-home uses this.
6. **Teach keep-in** — walk or tap the yellow boundary (at least three
   vertices). Mark keep-outs (beds, pond, dog run) as red holes.
7. **First mow** — stay in the yard. Start from the map. Use **SOS /
   ESTOP** if anything feels wrong. Software ESTOP zeros wheels and the
   trimmer until you Start again (operator clear).

Hardware bring-up (flash, wiring, field RF) is still out of this repo.

## YardProfile (`jims_mower.yard_profile.v1`)

JSON document. Load / save / validate in
[`src/jims_mower/yard_profile.py`](../src/jims_mower/yard_profile.py).

| Field | Meaning |
| --- | --- |
| `home` | Dock / return pose `{x, y, theta}` metres / rad |
| `keep_in` | Allowed work polygon (`[x, y]` vertices) |
| `keep_out` | List of no-go polygons |
| `mesh_path` | Relative path to a mesh JSON (no `..`, not absolute) |
| `radio` | `bluetooth`, `wifi.enabled` / `ssid`, `lora.enabled` / `channel`, `primary` |
| `schedule` | Stub only: `days`, `start_local` (`HH:MM`), `duration_min` |
| `width_m` / `height_m` / `resolution_m` | Local metre frame |

Example: [`configs/yards/example_profile.json`](../configs/yards/example_profile.json).
A WAVE 4 survey JSON can be lifted with `yard_profile_from_survey`.

## Local app API

`jims-mower-app` serves stdlib `ThreadingHTTPServer` (no Starlette).
JSON in / JSON out. Same origin as the static shell.

| Method | Path | Body / query |
| --- | --- | --- |
| `GET` | `/status` | pose, battery, state machine, radio link, faults |
| `GET` | `/yard` | current YardProfile |
| `PUT` | `/yard` | full YardProfile (validated) |
| `POST` | `/command` | `{cmd, reason?}` — `start` / `stop` / `return` / `estop` / `teach` |
| `GET` | `/map/mesh` | coarse occupancy mesh + `ux_a_href` |
| `GET` | `/map/coverage` | downsampled cut/uncut raster |
| `GET` | `/events` | SSE `data: <status>`; `?n=2` bounds the stream for tests |
| `GET` | `/` | phone shell |
| `GET` | `/viewer` | UX-A mesh viewer if present, else `#/map` |

`POST /command` `start` after ESTOP is the operator clear.

## Thin app shell

Static HTML / CSS / JS in [`src/jims_mower/app/static/`](../src/jims_mower/app/static/).
Narrow phone chrome (~390 px). Hash routes:

- `#/onboard/unbox` → `pair` → `home` → `teach` → `mow`
- `#/map` — 2D SVG yard (keep-in / keep-out / pose / coverage)
- `#/health` — battery, thermal, radios, hours, schedule stub
- `#/fault` — ESTOP / SOS

`viewer.js` draws the 2D map only. **Do not add three.js here.**

## Coordinate with UX-A

If a mesh viewer lands later, drop it at `app/static/ux_a/index.html`
and reuse that stack. The shell already deep-links:

- `GET /viewer`
- map page `HEAD /static/ux_a/index.html`
- `/map/mesh` → `ux_a_href`

See [`src/jims_mower/app/static/ux_a/README.md`](../src/jims_mower/app/static/ux_a/README.md).

## CLI

```bash
# Live demo env (default)
jims-mower-app --config geofence_movers --port 8765

# Recorded episode
jims-mower-record --out /tmp/jm-ep --steps 8 --cameras 4
jims-mower-app --episode /tmp/jm-ep

# YardProfile on disk (PUT persists)
jims-mower-app --backend memory --yard configs/yards/example_profile.json
```

Open `http://127.0.0.1:8765/`. `--backend memory` is a kinematic stub for
UI / API tests; default is the gym demo env.

## Tests

```bash
pytest tests/test_yard_profile.py tests/test_app_api.py
jims-mower-app --help
```

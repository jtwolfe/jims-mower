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

## YardProfile (`jims_mower.yard.v1`)

Same document as WAVE UX-A (`jims_mower.yard.v1` in
[`src/jims_mower/profile.py`](../src/jims_mower/profile.py)). UX-C adds
optional **radio** and **schedule** fields. Load / save / validate reject
bad schema, short polygons, and parent-traversing `mesh` paths.

| Field | Meaning |
| --- | --- |
| `home` | Dock / return pose `{x, y, theta}` metres / rad |
| `keep_in` | Allowed work polygon (`[x, y]` vertices) |
| `keep_out` | List of no-go polygons |
| `mesh` | Relative mesh path (`yard.glb` / JSON). Alias: `mesh_path` |
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
| `GET` | `/viewer` | UX-A World Viewer (`viewer_static` + CDN three.js) |
| `GET` | `/api/manifest` | UX-A viewer.json-shaped live bundle |
| `GET` / `POST` | `/api/profile` | same YardProfile as `/yard` |
| `GET` | `/api/live` | live SSE (`jims_mower.live.v1`) when `--live` |
| `GET` | `/api/live/snapshot` | latest live frame |
| `POST` | `/api/live/control` | same owner bar as `jims-mower-live` |

`--live` `/status` also carries `backend: live`, `robot`
(idle / pairing / live / fault), `owner_copy`, `radio_path` chips,
map/cut %, `session_summary`, and fog / observed URLs.

`POST /command` `start` after ESTOP is the operator clear. On `--live`
it forwards to `LiveSession.control` (`start` / `pause` / `estop` /
`start_mow` / `inject` / `pair`).

## Thin app shell

Static HTML / CSS / JS in [`src/jims_mower/app/static/`](../src/jims_mower/app/static/).
Narrow phone chrome (~390 px). Hash routes:

- `#/onboard/unbox` → `pair` → `home` → `teach` → `mow`
- `#/map` — live job (status / radios / Start / Pause / ESTOP / fog) when `--live`; else 2D SVG yard
- `#/health` — battery, thermal, radio-path chips, hours, schedule stub
- `#/fault` — ESTOP / SOS vs stuck recovery

`viewer.js` is a 2D SVG fallback for the phone chrome only. The three.js
World Viewer is **UX-A** (`viewer_static/` + `mesh_to_payload`). The
owner app serves those same assets at `/viewer` and `/data/yard.json`.
Do not add a second WebGL stack.

## Reuse UX-A + UX-B

- `GET /viewer` — rewritten `viewer_static/index.html` (CDN three.js)
- `GET /api/manifest`, `GET|POST /api/profile`, `GET /data/yard.json`
- `GET /map/mesh` — `jims_mower.mesh.v1` payload (`ux_a_href: /viewer`)
- Map page link: “Open UX-A mesh viewer”
- `GET /status` surfaces UX-B `info["fault"]` (SOS `retrieve`) and RadioSim
  (`radio.sim`) when the sim / episode backend has them. YardProfile radio
  prefs stay the owner preference document.

## CLI

```bash
# Owner phone + live job (one command). Open http://127.0.0.1:8766/
jims-mower-owner --live
jims-mower-app --live --config acre_yard_demo --port 8766

# Same live session contract as the desktop viewer:
# POST /api/live/control  GET /api/live  (no second MissionPolicy)
# Desktop-only chrome still: jims-mower-live --config acre_yard_demo --speed 5

# Recorded episode / kinematic stub
jims-mower-record --out /tmp/jm-ep --steps 8 --cameras 4
jims-mower-app --episode /tmp/jm-ep
jims-mower-app --backend memory --yard configs/yards/example_profile.json --port 8765
```

`--live` wraps `LiveSession` in-process (default yard `acre_yard_demo`,
port **8766**, speed 5×). Full `acre_yard` stays available
(`--config acre_yard`). `--fast` is `mission_tiny` for CI.

Open `http://127.0.0.1:8766/`. Pair the BT stub, Start job, watch fog,
MAP READY → Start mow, Pause / ESTOP. Inject stuck vs dead-motor SOS
from the job or SOS tab.

## Tests

```bash
pytest tests/test_yard_profile.py tests/test_app_api.py tests/test_app_live.py tests/test_live.py
jims-mower-app --live --help
jims-mower-owner --help
```

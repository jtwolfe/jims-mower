# WAVE UX-C — thin owner app + YardProfile API

Local phone-sized owner shell and a small HTTP JSON API. Not a store
listing, not a cloud account, not a claimed mapping / radio score.

The WAVE 3A `jims-mower-owner` overlay remains a **static HTML export**.
This wave is the live app: load / save a **YardProfile**, command a sim
or recorded episode, and walk unbox → first mow on a narrow layout.

## Buy → mow checklist

Matches the radio profile: **Bluetooth pair required** before Start,
**Wi-Fi optional**, **LoRa far-fence sim** for Pause / ESTOP /
Start-if-paired after BT is lost. No metre range claimed.

1. **Unbox** — charge to a green ring, unfold the handle, confirm the
   hardware ESTOP paddle moves freely. Wheel the mower to the yard edge.
2. **Bluetooth pair** — explicit Pair request (optional gym PIN `2468`).
   States: unpaired → pairing → paired / failed / lost. Live owner
   `require_pair: true`. Start is refused until `paired`. Unpair /
   radio-lost **holds safe** (not ESTOP). No BlueZ. No account, no QR.
3. **Wi-Fi (optional)** — skip unless you want OTA / map upload on the
   home AP. The mower does not need Wi-Fi to mow.
4. **LoRa far-fence sim** — when BT is lost, Pause / ESTOP /
   Start-if-paired can still ride the LoRa sim channel. Owner status
   shows the active transport and `rf_claim: null`. Do not invent metres.
5. **Place home** — park on the dock / start pose. The app stores
   `home` `{x, y, theta}` in the YardProfile. Return-to-home uses this.
6. **Teach keep-in** — walk or tap the yellow boundary (at least three
   vertices). Mark keep-outs (beds, pond, dog run) as red holes.
7. **First mow** — stay in the yard. Start from the map. Use **SOS /
   ESTOP** if anything feels wrong. Software ESTOP zeros **commands**
   until you Start again (operator clear). The hardware paddle (when
   wired) drops **rails**; Start does not restore them — reset the
   paddle. Gym: live `hw_estop` / `hw_reset`. See [`ESTOP.md`](ESTOP.md).

Hardware bring-up (flash, wiring, field RF) is still out of this repo.

## YardProfile (`jims_mower.yard.v1`)

Same document as WAVE UX-A (`jims_mower.yard.v1` in
[`src/jims_mower/profile.py`](../src/jims_mower/profile.py)). UX-C adds
optional **radio** and **schedule** fields. Load / save / validate reject
bad schema, short polygons, and parent-traversing `mesh` paths.

| Field | Meaning |
| --- | --- |
| `home` | Dock / return pose `{x, y, theta}` metres east/north of `origin` |
| `keep_in` | Allowed work polygon (`[e, n]` metres from `origin`) |
| `keep_out` | List of no-go polygons |
| `origin` | Surveyed/local-ENU peg (`e_m/n_m/u_m`, optional lat/lon). Default gym SW corner. **Not** a WGS84 field survey unless `surveyed: true`. See [`SURVEY_ORIGIN.md`](SURVEY_ORIGIN.md). |
| `mesh` | Relative mesh path (`yard.glb` / JSON). Alias: `mesh_path` |
| `radio` | `bluetooth`, `wifi.enabled` / `ssid`, `lora.enabled` / `channel`, `primary`. Owner overlay adds `rf_claim: null` — never RSSI / metres. |
| `pairing` | State machine (`unpaired` / `pairing` / `paired` / `failed` / `lost`). Persist on the profile. |
| `schedule` | Weekly window: `enabled`, `days`, `start_local` (`HH:MM`), `duration_min`, `timezone` (default `Australia/Brisbane`), `min_soc`, `skip_rain`, `arm_window_min`. Evaluated by `ScheduleEngine` — arms / skips / duration-stops a job. Not a cloud calendar. |
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
| `GET` | `/yards` | multi-yard list + active name (UX-3) |
| `POST` | `/yards/select` | `{name}` — switch profile; keep-in/home replace (no fence bleed) |
| `GET` | `/notifications` | in-app skip/finish list (not SMS; UX-2) |
| `GET` | `/ota` | SAF-4 documented no-op (`available: false`) |
| `POST` | `/command` | `{cmd, reason?, pin?}` — `start` / `stop` / `return` / `estop` / `teach` / `pair` / `unpair` / `explore` / `mow` |
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
map/cut/planned %, `session_summary`, fog / observed / coverage / areas
URLs, `explore_reason` (seeking frontier / path blocked / tip recovery /
map % of target / step cap), `full_explore`, `charge_state`, and a
`path_overlay` (`trail` / `plan` / `frontiers` / `pose` / `target` /
`phase`) plus `mode_banner` so the phone can tell **Mapping** from
**Mowing** without the 3D viewer. `state.mission` follows **phase**
(explore / review / teach — never `running` → mowing). Overlay arrays
are downsampled for SSE. Keep-out holes draw red on the phone fence.
Area-type legend: grass / mow-this / path / sand / building / water /
drain / beds / keep-out / blocked (no-go learned) / fog. See
[`NAV_BLOCKAGES.md`](NAV_BLOCKAGES.md). No mAP / RF range claimed.

`POST /command` `start` after ESTOP is the operator clear. On `--live`
it forwards to `LiveSession.control` (`start` / `pause` / `estop` /
`explore` / `mow` / `return` / `full_explore` / `start_mow` / `inject` /
`pair` / `teach` / `save_yard` / `load_yard`). Manual Explore / Mow /
Return do not wait for auto phase transitions. Low-SOC inject docks,
charges in gym, and resumes the leftover uncut plan.

## Thin app shell

Static HTML / CSS / JS in [`src/jims_mower/app/static/`](../src/jims_mower/app/static/).
Narrow phone chrome (~390 px). Hash routes:

- `#/onboard/unbox` → `pair` → `home` → `teach` → `mow`
- `#/map` — live job: large **Mapping yard** / **Mowing** mode chip,
  `explore_reason` line, then a **Manual phases** card with always-visible
  **Explore / Mow / Return home** (enabled from `can_*`, disabled reason
  shown), a **Full explore** On/Off toggle, then the map with area-type
  raster, a labeled **Area types** legend (grass / mow-this / path /
  sand / building / water / drain / beds / keep-out / blocked / fog)
  plus the Path Fog/Mapped/Trail/Plan/Cut row. **Reset** sits with the
  manual phases (stop job, clear tip cool-down, keep the fence; learned
  blockages stay unless `clear_blockages`). **Low battery** inject is on Map
  and Health (`{cmd: inject, kind: low_soc}`). Pair still required
  before Start. Else 2D SVG yard. Desktop widens this same chrome and
  embeds `/viewer?embed=1` as the map pane — one owner shell, not a
  second product. Layer toggles live under **Advanced**.
- `#/health` — battery, thermal, radio-path chips, hours, schedule enable toggle + next run, Low battery inject when `--live`
- `#/fault` — ESTOP / SOS vs stuck recovery

`viewer.js` is a 2D SVG fallback for the phone chrome only. The three.js
World Viewer is **UX-A** (`viewer_static/` + `mesh_to_payload`). The
owner app serves those same assets at `/viewer` and embeds them on
desktop (`/viewer?embed=1`). Do not add a second WebGL stack.

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

## First-run vs demo confirm fence

Jamie's command:

```bash
jims-mower-owner --live
# or jump the onboarding hash: jims-mower-owner --live --first-run
# open http://127.0.0.1:8766/
```

**First-run (real setup):** Unbox → Pair BT stub → **Teach boundary**
(drive the perimeter in sim, or tap/edit keep-in vertices) → **Save
yard**. That writes `live_out/profile.json` (`jims_mower.yard.v1`) — the
same UX-A `YardProfile` as `jims-mower-teach`. Idle then shows that yard
selected. **Start job** uses the taught keep-in / home as the geofence
and **skips** authored `calibrate_confirm_m`. A short Teach drive is not
a yard fence — Save rejects or repairs scribble keep-ins (min area /
bbox vs the world; authored acre rectangle is the editable starter).
Explore fog → MAP READY → mow. MAP READY copy is “Map ready — start
mow?”, never “Hold — safe.” 0 mowable cells offers Re-teach.

**Demo confirm fence (no teach):** Start on `acre_yard_demo` still
confirms the authored keep-in after ~28 m of trail, then explores. That
is documented demo pacing, not how an owner teaches a yard.

Reload a saved document with **Load saved yard** or
`jims-mower-owner --live --yard live_out/profile.json`. Physics stays on
the acre-scale (or `--fast` tiny) world; the profile is fence + home,
not a second mesh.

Open `http://127.0.0.1:8766/`. Pair the BT stub, teach (or skip to the
demo fence), Start job, watch fog, MAP READY → Start mow, Pause / ESTOP.
Inject stuck vs dead-motor SOS from the job or SOS tab.

## Tests

```bash
pytest tests/test_yard_profile.py tests/test_app_api.py tests/test_app_live.py tests/test_first_run.py tests/test_live.py tests/test_path_overlay.py tests/test_schedule.py tests/test_owner_phases.py
jims-mower-app --live --help
jims-mower-owner --live --first-run --help
```

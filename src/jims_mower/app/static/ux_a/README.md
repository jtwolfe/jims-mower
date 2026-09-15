# UX-A mesh viewer hook

This folder is intentionally empty of a WebGL stack.

When WAVE UX-A lands a three.js (or similar) mesh viewer, drop it here as
`index.html` plus shared assets. The owner app already deep-links:

- `GET /viewer` → `/static/ux_a/index.html` if that file exists, else `#/map`
- Map page probes `HEAD /static/ux_a/index.html` and shows an “Open UX-A mesh viewer” link
- `GET /map/mesh` includes `ux_a_href`

Do **not** add a second three.js bundle in UX-C. Reuse the UX-A viewer.

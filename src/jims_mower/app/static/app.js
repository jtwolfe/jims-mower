(() => {
  const ONBOARD = ["unbox", "pair", "home", "teach", "mow"];
  const KEY = "jims_mower_onboarded";

  const state = {
    status: null,
    yard: null,
    coverage: null,
    live: null,
    pairing: { bt: true, wifi: false, lora: true },
    teachPts: [],
  };

  const $ = (sel) => document.querySelector(sel);
  const screen = () => $("#screen");

  function needsFirstRun() {
    const st = state.status || {};
    return !!(st.first_run && !st.taught);
  }

  function route() {
    const hash = location.hash || "";
    if (hash.startsWith("#/onboard/")) return hash.slice(2);
    if (hash === "#/map" || hash === "#/live") return "map";
    if (hash === "#/health") return "health";
    if (hash === "#/fault") return "fault";
    if (needsFirstRun()) return "onboard/unbox";
    if (localStorage.getItem(KEY)) return "map";
    return "onboard/unbox";
  }

  function go(name) {
    location.hash = `#/${name}`;
  }

  function isLive() {
    const st = state.status || {};
    return st.backend === "live" || st.live === true || !!state.live;
  }

  async function api(path, opts) {
    const res = await fetch(path, Object.assign({ headers: { "Content-Type": "application/json" } }, opts || {}));
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || res.statusText);
    return data;
  }

  function setRobotPill(status) {
    const pill = $("#robot-pill");
    if (!pill) return;
    const robot = (status && status.robot) || ((status && status.state && status.state.mission) || "idle");
    pill.textContent = robot;
    pill.className = ["idle", "pairing", "live", "fault"].includes(robot) ? robot : "idle";
  }

  async function refresh() {
    const [status, yard, coverage] = await Promise.all([
      api("/status"),
      api("/yard"),
      api("/map/coverage"),
    ]);
    state.status = status;
    state.yard = yard;
    state.coverage = coverage;
    const soc = Math.round(100 * Number((status.battery || {}).soc || 0));
    $("#batt-pill").textContent = `${soc}%`;
    setRobotPill(status);
    return status;
  }

  async function command(cmd, reason) {
    if (isLive() && cmd !== "pair") {
      await liveControl(cmd === "stop" ? "pause" : cmd, reason ? { reason } : {});
      return;
    }
    await api("/command", { method: "POST", body: JSON.stringify({ cmd, reason }) });
    await refresh();
    render();
  }

  async function liveControl(cmd, extra) {
    const body = Object.assign({ cmd }, extra || {});
    const data = await api("/api/live/control", { method: "POST", body: JSON.stringify(body) });
    state.live = data;
    if (data && data.ok === false) throw new Error(data.error || "live control failed");
    await refresh();
    render();
    return data;
  }

  async function saveYard(patch) {
    const next = Object.assign({}, state.yard, patch);
    state.yard = await api("/yard", { method: "PUT", body: JSON.stringify(next) });
    render();
  }

  function clock() {
    const d = new Date();
    $("#clock").textContent = `${d.getHours()}:${String(d.getMinutes()).padStart(2, "0")}`;
  }

  function nav(active) {
    const bar = $("#tabbar");
    const onboard = route().startsWith("onboard/");
    bar.hidden = onboard;
    bar.querySelectorAll("button").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.href === `#/${active}`);
    });
  }

  function onboardFrame(step, title, body, primary, onPrimary, extra) {
    const idx = ONBOARD.indexOf(step);
    const dots = ONBOARD.map((_, i) => `<i class="${i <= idx ? "on" : ""}"></i>`).join("");
    return `
      <div class="dots">${dots}</div>
      <h1>${title}</h1>
      ${body}
      <button class="btn primary" id="next">${primary}</button>
      ${extra || ""}`;
  }

  function renderOnboard(step) {
    if (step === "unbox") {
      screen().innerHTML = onboardFrame(
        "unbox",
        "Unbox",
        `<div class="hero">Charge · unfold · roll out</div>
         <p class="lead">Charge to a green ring, unfold the handle, and wheel the mower to the yard edge. Keep the ESTOP paddle clear.</p>`,
        "I've unboxed it",
        null
      );
      $("#next").onclick = () => go("onboard/pair");
      return;
    }
    if (step === "pair") {
      screen().innerHTML = onboardFrame(
        "pair",
        "Pair radios",
        `<p class="lead">Bluetooth is required for first contact. Wi-Fi is optional. LoRa is the long-range command link. Stub only — no RF hardware.</p>
         <div class="card radio-list">
           <label>Bluetooth <span class="tag">required</span></label>
           <label>Wi-Fi <span class="tag opt">optional</span></label>
           <label>LoRa <span class="tag">long-range</span></label>
         </div>
         <p class="sub">SSID is only needed if you turn Wi-Fi on later in Health.</p>`,
        "Pair over Bluetooth",
        null
      );
      $("#next").onclick = async () => {
        if (!state.yard) await refresh();
        const radio = Object.assign({}, (state.yard && state.yard.radio) || {}, {
          bluetooth: true,
          primary: "lora",
          lora: { enabled: true, channel: ((state.yard && state.yard.radio && state.yard.radio.lora) || {}).channel || 1 },
          wifi: (state.yard && state.yard.radio && state.yard.radio.wifi) || { enabled: false, ssid: "" },
        });
        if (state.yard) await saveYard({ radio });
        try { await command("pair"); } catch (_err) { /* memory backend has no pair */ }
        go("onboard/home");
      };
      return;
    }
    if (step === "home") {
      const pose = (state.status && state.status.pose) || { x: 2, y: 2, theta: 0 };
      screen().innerHTML = onboardFrame(
        "home",
        "Place home",
        `<p class="lead">Park on the dock / start pose. We store this as the return point.</p>
         <div class="card"><strong>Current pose</strong>${pose.x.toFixed(1)} m, ${pose.y.toFixed(1)} m</div>
         <svg id="yard-svg" viewBox="0 0 16 12"></svg>`,
        "Use this pose as home",
        null
      );
      if (window.JimsViewer && state.yard) {
        window.JimsViewer.drawYard($("#yard-svg"), state);
      }
      $("#next").onclick = async () => {
        await saveYard({ home: pose });
        go("onboard/teach");
      };
      return;
    }
    if (step === "teach") {
      screen().innerHTML = onboardFrame(
        "teach",
        "Teach the yard",
        `<p class="lead">Drive the perimeter in sim, or tap the map to edit keep-in vertices. This writes a YardProfile — the same fence UX-A teach uses.</p>
         <svg id="yard-svg" viewBox="0 0 16 12"></svg>
         <button class="btn ghost" id="teach-cmd">Drive perimeter</button>
         <button class="btn ghost" id="load-yard" ${!(state.status && state.status.yard_saved) ? "hidden" : ""}>Load saved yard</button>`,
        "Save yard",
        null
      );
      const svg = $("#yard-svg");
      if (window.JimsViewer && state.yard) window.JimsViewer.drawYard(svg, state);
      svg.onclick = (ev) => {
        const box = svg.getBoundingClientRect();
        const w = Number(state.yard.width_m);
        const h = Number(state.yard.height_m);
        const x = ((ev.clientX - box.left) / box.width) * w;
        const y = (1 - (ev.clientY - box.top) / box.height) * h;
        state.teachPts.push([x, y]);
        if (state.teachPts.length >= 3) {
          state.yard.keep_in = state.teachPts.slice();
          window.JimsViewer.drawYard(svg, state);
        }
      };
      $("#teach-cmd").onclick = () => (isLive() ? liveControl("teach") : command("teach"));
      const loadBtn = $("#load-yard");
      if (loadBtn) loadBtn.onclick = async () => {
        if (isLive()) await liveControl("load_yard");
        go("onboard/mow");
      };
      $("#next").onclick = async () => {
        if (isLive()) {
          const extra = state.teachPts.length >= 3 ? { keep_in: state.teachPts } : {};
          await liveControl("save_yard", extra);
        } else if (state.teachPts.length >= 3) {
          await saveYard({ keep_in: state.teachPts });
        }
        go("onboard/mow");
      };
      return;
    }
    screen().innerHTML = onboardFrame(
      "mow",
      "First mow",
      `<p class="lead">Stay in the yard for the first run. Start, watch the pose, and hit SOS if anything feels wrong.</p>
       <div class="card"><strong>Radios</strong>BT paired · LoRa long-range · Wi-Fi optional</div>`,
      "Start first mow",
      null,
      `<button class="btn ghost" id="skip">Skip to layout</button>`
    );
    $("#next").onclick = async () => {
      localStorage.setItem(KEY, "1");
      go("map");
    };
    $("#skip").onclick = () => {
      localStorage.setItem(KEY, "1");
      go("map");
    };
  }

  function radioChipsHtml(path) {
    const chips = (path && path.chips) || [
      { id: "bt", label: "BT teach", active: true },
      { id: "wifi", label: "Wi-Fi map", active: false },
      { id: "lora", label: "LoRa sparse", active: false },
    ];
    return `<div class="radio-chips" aria-label="radio path">${chips
      .map((c) => `<span class="${c.active ? "on" : ""}">${c.label}</span>`)
      .join("")}</div>`;
  }

  function sessionCardHtml(st, live) {
    const card = (live && live.session_summary) || st.session_summary || {};
    const done = !!(st.done || (live && (live.done || live.phase === "complete" || live.phase === "return_home")));
    if (!done || !card || !card.schema) return "";
    const reach = card.reachable || 0;
    const unreach = card.unreachable || 0;
    const denom = reach + unreach;
    const reachPct = denom ? (100 * reach) / denom : 0;
    return `<div class="session-card" id="session-card"><strong>Session</strong>
map ${((card.map_pct || 0) * 100).toFixed(1)}% · planned ${((card.planned_pct || 0) * 100).toFixed(1)}%
reachable ${reachPct.toFixed(1)}% · cut ${((card.cut_pct || 0) * 100).toFixed(1)}%
skips ${card.skips || 0} · ${(card.duration_s || 0).toFixed(1)}s sim${card.wall_s ? ` · ${card.wall_s.toFixed(1)}s wall` : ""}</div>`;
  }

  function renderPairingStub() {
    screen().innerHTML = `
      <h1>Pair Bluetooth</h1>
      <p class="lead">Hold the phone next to the mower for first contact. Stub only — no real RF.</p>
      <div class="card radio-list">
        <label>Bluetooth <span class="tag">required</span></label>
        <label>Wi-Fi <span class="tag opt">optional</span></label>
        <label>LoRa <span class="tag">long-range after pair</span></label>
      </div>
      <button class="btn primary" id="pair-bt">Pair over Bluetooth</button>`;
    $("#pair-bt").onclick = () => command("pair");
  }

  function fenceOverlayHtml(st, live) {
    const yard = state.yard || {};
    const keep = (live && live.keep_in) || st.keep_in || yard.keep_in || [];
    if (!keep.length) return "";
    const w = Number(yard.width_m || live.width_m || 16);
    const h = Number(yard.height_m || live.height_m || 12);
    const pts = keep.map((p) => `${Number(p[0] != null ? p[0] : p.x)},${Number(p[1] != null ? p[1] : p.y)}`).join(" ");
    return `<svg class="fence-overlay" id="fence-svg" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-hidden="true">
      <g transform="translate(0 ${h}) scale(1 -1)">
        <polygon points="${pts}" fill="none" stroke="#ffcc33" stroke-width="0.18"/>
      </g>
    </svg>`;
  }

  function renderLiveJob() {
    const st = state.status || {};
    const live = state.live || {};
    const paired = st.paired === true || live.paired === true;
    if (!paired) {
      renderPairingStub();
      return;
    }
    const job = live.job_state || (st.state || {}).job_state || "idle";
    const taught = !!(st.taught || live.taught);
    const copy = st.owner_copy || live.owner_copy || "Yard unknown — start a job when ready.";
    const mapPct = Number(st.map_pct != null ? st.map_pct : 100 * (live.map_pct || 0));
    const cutPct = Number(st.cut_pct != null ? st.cut_pct : 100 * (live.cut_pct || 0));
    const speed = String(st.speed_label || live.speed_label || "5");
    const canMow = !!(st.can_start_mow || live.can_start_mow);
    const fog = live.fog_url || st.fog_url || "/api/live/fog.png";
    const observed = live.observed_url || st.observed_url || "/api/live/observed.png";
    const path = st.radio_path || live.radio_path || {};
    const yardName = (state.yard && state.yard.name) || st.yard || live.yard || "yard";
    const saved = !!(st.yard_saved || live.yard_saved);
    const teaching = job === "teach";
    screen().innerHTML = `
      <h1>Live job</h1>
      ${radioChipsHtml(path)}
      <p class="owner-copy" id="owner-copy">${copy}</p>
      <p class="sub" id="yard-chip">${yardName}${taught ? " · taught fence" : " · authored demo fence until you teach"}</p>
      <div class="live-preview">
        <img class="obs" id="obs-img" alt="observed terrain" src="${observed}"/>
        <img class="fog" id="fog-img" alt="fog of war" src="${fog}"/>
        ${fenceOverlayHtml(st, live)}
      </div>
      <div class="row">
        <div class="chip">Map<b id="map-pct">${mapPct.toFixed(1)}%</b></div>
        <div class="chip">Cut<b id="cut-pct">${cutPct.toFixed(1)}%</b></div>
      </div>
      <div class="speed-row" id="speed-row">
        <button type="button" data-speed="1">1×</button>
        <button type="button" data-speed="2">2×</button>
        <button type="button" data-speed="5">5×</button>
        <button type="button" data-speed="max">max</button>
      </div>
      <div class="row">
        <button class="btn ghost" id="teach-cmd" ${job === "running" ? "disabled" : ""}>Teach boundary</button>
        <button class="btn ghost" id="save-yard">${teaching ? "Save yard" : "Save yard"}</button>
      </div>
      <button class="btn ghost" id="load-yard" ${saved ? "" : "hidden"}>Load saved yard</button>
      <div class="row">
        <button class="btn primary" id="start" ${job === "running" || job === "teach" ? "disabled" : ""}>Start job</button>
        <button class="btn ghost" id="pause" ${job !== "running" && job !== "teach" ? "disabled" : ""}>Pause</button>
      </div>
      <button class="btn ghost" id="resume" ${job !== "paused" && job !== "hold" ? "disabled" : ""}>Resume</button>
      ${canMow ? `<button class="btn warn" id="start-mow">Start mow</button>` : ""}
      <button class="btn danger" id="estop">ESTOP</button>
      ${sessionCardHtml(st, live)}
      <div class="row">
        <button class="btn ghost" id="inj-stuck">Inject stuck</button>
        <button class="btn ghost" id="inj-sos">Inject SOS</button>
      </div>
      <p style="margin-top:10px"><a class="linkish" href="/viewer">Open live fog viewer</a></p>`;
    document.querySelectorAll("#speed-row button").forEach((btn) => {
      btn.classList.toggle("on", btn.dataset.speed === speed);
      btn.onclick = () => liveControl("speed", { speed: btn.dataset.speed });
    });
    $("#teach-cmd").onclick = () => liveControl("teach");
    $("#save-yard").onclick = () => liveControl("save_yard");
    const loadBtn = $("#load-yard");
    if (loadBtn) loadBtn.onclick = () => liveControl("load_yard");
    $("#start").onclick = () => liveControl("start");
    $("#pause").onclick = () => liveControl("pause");
    $("#resume").onclick = () => liveControl("resume");
    const mow = $("#start-mow");
    if (mow) mow.onclick = () => liveControl("start_mow");
    $("#estop").onclick = () => liveControl("estop");
    $("#inj-stuck").onclick = () => liveControl("inject", { kind: "stuck" });
    $("#inj-sos").onclick = async () => {
      await liveControl("inject", { kind: "sos" });
      go("fault");
    };
  }

  function renderMap() {
    if (isLive()) {
      renderLiveJob();
      return;
    }
    const st = state.status || {};
    const mission = ((st.state || {}).mission) || "idle";
    screen().innerHTML = `
      <h1>Yard</h1>
      <p class="sub">${(state.yard && state.yard.name) || "yard"} · ${mission}</p>
      <svg id="yard-svg" viewBox="0 0 16 12"></svg>
      <div class="row" style="margin-top:12px">
        <div class="chip">Coverage<b>${Number(st.coverage_pct || 0).toFixed(1)}%</b></div>
        <div class="chip">Pose<b>${Number((st.pose || {}).x || 0).toFixed(1)}, ${Number((st.pose || {}).y || 0).toFixed(1)}</b></div>
      </div>
      <div class="row">
        <button class="btn primary" id="start">Start</button>
        <button class="btn ghost" id="stop">Stop</button>
      </div>
      <button class="btn ghost" id="ret">Return home</button>
      <p style="margin-top:10px"><a class="linkish" id="ux-a" href="/viewer">Open UX-A mesh viewer</a></p>`;
    if (window.JimsViewer && state.yard) window.JimsViewer.drawYard($("#yard-svg"), state);
    $("#start").onclick = () => command("start");
    $("#stop").onclick = () => command("stop");
    $("#ret").onclick = () => command("return");
  }

  function renderHealth() {
    const st = state.status || {};
    const bat = st.battery || {};
    const radio = st.radio || {};
    const sch = (state.yard && state.yard.schedule) || {};
    const path = st.radio_path || (state.live && state.live.radio_path) || {};
    screen().innerHTML = `
      <h1>Health</h1>
      <p class="lead">Maintenance and radio prefs. Schedule is a stub.</p>
      ${radioChipsHtml(path)}
      <div class="row">
        <div class="chip">Battery<b>${Math.round(100 * Number(bat.soc || 0))}%</b></div>
        <div class="chip">Thermal<b>${Number(bat.temp_c || 0).toFixed(0)}°C</b></div>
      </div>
      <div class="card">
        <strong>Radio link</strong>
        ${radio.link || "none"} · RSSI ${radio.rssi || "—"}
        <div class="sub">BT ${radio.bluetooth && radio.bluetooth.paired ? "paired" : "no"} ·
          Wi-Fi ${radio.wifi && radio.wifi.enabled ? "on" : "optional / off"} ·
          LoRa ${radio.lora && radio.lora.enabled ? "long-range" : "off"}
          ${(path && path.simulated) ? " · simulated path" : ""}</div>
      </div>
      <div class="card">
        <strong>Hours mowed</strong>${Number(st.hours_mowed || 0).toFixed(2)}
        <div class="sub">String-head check every 8 hours (stub).</div>
      </div>
      <div class="card">
        <strong>Schedule stub</strong>
        ${(sch.days || []).join(", ") || "no days"} @ ${sch.start_local || "—"}
        <div class="sub">${sch.note || ""}</div>
      </div>
      <button class="btn ghost" id="redo">Replay onboarding</button>`;
    $("#redo").onclick = () => {
      localStorage.removeItem(KEY);
      go("onboard/unbox");
    };
  }

  function renderFault() {
    const st = state.status || {};
    const live = state.live || {};
    const faults = st.faults || live.faults || [];
    const sos = faults.some((f) => f.retrieve || f.code === "FAULT_IMMOBILISED");
    const stuck = faults.some((f) => f.code === "STUCK");
    const banner = sos
      ? `<div class="fault-banner"><strong>SOS — immobilised</strong><span>Dead motor. Retrieve the mower. Wheels and trimmer are held.</span></div>`
      : stuck
        ? `<div class="fault-banner"><strong>Stuck</strong><span>Recovery reverse / pivot / help still runs. Not a retrieve.</span></div>`
        : "";
    const list = faults.length
      ? faults.map((f) => `<div class="fault-banner"><strong>${f.code}</strong><span>${f.detail || ""}</span></div>`).join("")
      : `<div class="card"><strong>All clear</strong>No latched owner faults.</div>`;
    screen().innerHTML = `
      <h1>SOS</h1>
      ${banner}
      ${list}
      <p class="owner-copy">${st.owner_copy || live.owner_copy || ""}</p>
      <button class="big-sos" id="estop">ESTOP</button>
      <p class="sub" style="text-align:center">Software latch · zeros wheels and trimmer until you Start again</p>
      <button class="btn ghost" id="resume">Resume (start)</button>
      ${isLive() ? `<div class="row">
        <button class="btn ghost" id="inj-stuck">Inject stuck</button>
        <button class="btn ghost" id="inj-sos">Inject dead-motor SOS</button>
      </div>` : ""}`;
    $("#estop").onclick = () => command("estop", "sos");
    $("#resume").onclick = () => command("start");
    if (isLive()) {
      $("#inj-stuck").onclick = () => liveControl("inject", { kind: "stuck" });
      $("#inj-sos").onclick = () => liveControl("inject", { kind: "sos" });
    }
  }

  function render() {
    const r = route();
    nav(r.startsWith("onboard/") ? "" : r);
    if (r.startsWith("onboard/")) renderOnboard(r.split("/")[1] || "unbox");
    else if (r === "health") renderHealth();
    else if (r === "fault") renderFault();
    else renderMap();
  }

  function patchLiveChrome(frame) {
    if (!frame || frame.live === false) return;
    state.live = frame;
    const copy = $("#owner-copy");
    if (copy && frame.owner_copy) copy.textContent = frame.owner_copy;
    const mapEl = $("#map-pct");
    if (mapEl && frame.map_pct != null) mapEl.textContent = `${(100 * Number(frame.map_pct)).toFixed(1)}%`;
    const cutEl = $("#cut-pct");
    if (cutEl && frame.cut_pct != null) cutEl.textContent = `${(100 * Number(frame.cut_pct)).toFixed(1)}%`;
    const fog = $("#fog-img");
    if (fog && frame.fog_url) fog.src = frame.fog_url;
    const obs = $("#obs-img");
    if (obs && frame.observed_url) obs.src = frame.observed_url;
    if (state.status) {
      state.status.owner_copy = frame.owner_copy;
      state.status.can_start_mow = frame.can_start_mow;
      state.status.robot = state.status.robot;
    }
  }

  $("#tabbar").addEventListener("click", (ev) => {
    const btn = ev.target.closest("button");
    if (btn && btn.dataset.href) location.hash = btn.dataset.href;
  });

  window.addEventListener("hashchange", render);
  clock();
  setInterval(clock, 15000);

  refresh()
    .catch(() => null)
    .finally(() => {
      if (!location.hash) {
        location.hash = needsFirstRun() || !localStorage.getItem(KEY) ? "#/onboard/unbox" : "#/map";
      }
      render();
    });

  if (window.EventSource) {
    const src = new EventSource("/events");
    src.onmessage = (ev) => {
      try {
        state.status = JSON.parse(ev.data);
        const soc = Math.round(100 * Number((state.status.battery || {}).soc || 0));
        $("#batt-pill").textContent = `${soc}%`;
        setRobotPill(state.status);
        const r = route();
        if (r === "map" && $("#yard-svg") && window.JimsViewer && state.yard && !isLive()) {
          window.JimsViewer.drawYard($("#yard-svg"), state);
        }
      } catch (_err) {
        /* ignore parse errors */
      }
    };
    const live = new EventSource("/api/live");
    live.onmessage = (ev) => {
      try {
        const frame = JSON.parse(ev.data);
        if (frame && frame.live) {
          patchLiveChrome(frame);
          setRobotPill(state.status || {});
        }
      } catch (_err) {
        /* ignore */
      }
    };
  }
})();

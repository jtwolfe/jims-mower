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
    const live = !!(status && (status.live || status.backend === "live"));
    const mode = (status && status.mode_banner) || {};
    const kind = mode.kind || "";
    if (live && robot === "live" && kind && kind !== "idle") {
      const short = { mapping: "mapping", mowing: "mowing", teach: "teach", done: "done" }[kind] || kind;
      pill.textContent = short;
      pill.className = ["mapping", "mowing", "teach", "done"].includes(kind) ? kind : "live";
      return;
    }
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

  function pairingState(st) {
    const p = (st && st.pairing) || {};
    if (p.state) return p.state;
    return st && st.paired ? "paired" : "unpaired";
  }

  function isPaired(st) {
    return pairingState(st) === "paired";
  }

  async function command(cmd, reason, extra) {
    const body = Object.assign({ cmd, reason: reason || "" }, extra || {});
    if (isLive()) {
      const mapped = cmd === "stop" ? "pause" : cmd;
      const liveExtra = Object.assign({}, extra || {});
      if (reason) liveExtra.reason = reason;
      await liveControl(mapped, liveExtra);
      return;
    }
    await api("/command", { method: "POST", body: JSON.stringify(body) });
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
        `<p class="lead">Bluetooth pair is required before Start. Wi-Fi is optional. LoRa is the far-fence sim command link after pair — no metre range claimed. No BlueZ.</p>
         <div class="card radio-list">
           <label>Bluetooth <span class="tag">required</span></label>
           <label>Wi-Fi <span class="tag opt">optional</span></label>
           <label>LoRa <span class="tag">far-fence sim</span></label>
         </div>
         <p class="sub">Gym PIN 2468 if prompted. SSID is only needed if you turn Wi-Fi on later in Health.</p>
         <input class="pin" id="pair-pin" inputmode="numeric" maxlength="8" placeholder="PIN (2468)" value="2468"/>`,
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
        const pinEl = $("#pair-pin");
        const pin = pinEl ? pinEl.value : "2468";
        try { await command("pair", "", { pin }); } catch (_err) { /* show next even if already paired */ }
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
      const yw = Number((state.yard && state.yard.width_m) || 16);
      const yh = Number((state.yard && state.yard.height_m) || 12);
      if (!state.teachPts.length && state.yard && (state.yard.keep_in || []).length >= 3) {
        state.teachPts = state.yard.keep_in.map((p) => [Number(p[0]), Number(p[1])]);
      }
      screen().innerHTML = onboardFrame(
        "teach",
        "Teach the yard",
        `<p class="lead">Drive the perimeter, or tap to edit the starter keep-in (authored yard rectangle). A short drive is not a fence — Save keeps or repairs a yard-scale ring.</p>
         <svg id="yard-svg" viewBox="0 0 ${yw} ${yh}"></svg>
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
        try {
          if (isLive()) {
            const extra = state.teachPts.length >= 3 ? { keep_in: state.teachPts } : {};
            await liveControl("save_yard", extra);
          } else if (state.teachPts.length >= 3) {
            await saveYard({ keep_in: state.teachPts });
          }
          go("onboard/mow");
        } catch (err) {
          const lead = document.querySelector(".lead");
          if (lead) lead.textContent = err.message || "Fence too small — edit the starter rectangle.";
        }
      };
      return;
    }
    screen().innerHTML = onboardFrame(
      "mow",
      "First mow",
      `<p class="lead">Stay in the yard for the first run. Start, watch the pose, and hit SOS if anything feels wrong.</p>
       <div class="card"><strong>Radios</strong>Pair before Start · LoRa far-fence sim · no RF range</div>`,
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
    const done = !!(
      st.done ||
      (live && (live.done || live.phase === "complete" || live.phase === "return_home")) ||
      card.phase === "complete"
    );
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
    const st = state.status || {};
    const live = state.live || {};
    const pair = pairingState(st) || pairingState(live);
    const copy = {
      unpaired: "Pair Bluetooth before Start. Sim pair — no BlueZ.",
      pairing: "Pairing…",
      failed: "Pairing failed — gym PIN is 2468.",
      lost: "Radio lost — hold safe. Pair again to Start.",
    }[pair] || "Pair Bluetooth before Start. Sim pair — no BlueZ.";
    screen().innerHTML = `
      <h1>Pair Bluetooth</h1>
      <p class="lead">${copy}</p>
      <p class="sub">State: ${pair} · rf_claim: none · no metre range</p>
      <div class="card radio-list">
        <label>Bluetooth <span class="tag">required</span></label>
        <label>Wi-Fi <span class="tag opt">optional</span></label>
        <label>LoRa <span class="tag">far-fence sim after pair</span></label>
      </div>
      <input class="pin" id="pair-pin" inputmode="numeric" maxlength="8" placeholder="PIN (2468)" value="2468"/>
      <button class="btn primary" id="pair-bt">Pair over Bluetooth</button>`;
    $("#pair-bt").onclick = () => {
      const pinEl = $("#pair-pin");
      command("pair", "", { pin: pinEl ? pinEl.value : "2468" }).catch((err) => {
        const lead = document.querySelector(".lead");
        if (lead) lead.textContent = err.message || "Pairing failed";
      });
    };
  }

  const MODE_COPY = {
    idle: { label: "Ready", tone: "idle", kind: "idle" },
    teach: { label: "Teaching boundary", tone: "teach", kind: "mapping" },
    calibrate_boundary: { label: "Teaching boundary", tone: "teach", kind: "mapping" },
    explore: { label: "Mapping yard", tone: "map", kind: "mapping" },
    review: { label: "Map ready — review", tone: "review", kind: "mapping" },
    mow: { label: "Mowing", tone: "mow", kind: "mowing" },
    return_home: { label: "Returning home", tone: "home", kind: "mowing" },
    complete: { label: "Done", tone: "done", kind: "done" },
    safe: { label: "Hold — safe", tone: "idle", kind: "idle" },
  };

  function asXYList(raw) {
    return (raw || []).map((p) => (Array.isArray(p) ? p : [p.x, p.y])).filter((p) => p[0] != null && p[1] != null);
  }

  function overlayFrom(st, live) {
    const built = (live && live.path_overlay) || (st && st.path_overlay) || {};
    const livePlan = asXYList((live && live.plan) || (st && st.plan));
    const liveExplore = asXYList((live && live.explore) || (st && st.explore));
    const liveFrontiers = asXYList((live && live.frontiers) || (st && st.frontiers));
    const idx = built.waypoint_index != null ? built.waypoint_index : (live && live.waypoint_index != null ? live.waypoint_index : (st && st.waypoint_index));
    const plan = (built.plan && built.plan.length ? built.plan : livePlan) || [];
    const explore = (built.explore && built.explore.length ? built.explore : liveExplore) || [];
    const active = plan.length ? plan : explore;
    let target = built.target;
    if (!target && active.length && idx != null) {
      const i = Math.max(0, Math.min(active.length - 1, Number(idx)));
      target = active[i];
    }
    return Object.assign({}, built, {
      trail: built.trail || [],
      plan,
      explore,
      frontiers: (built.frontiers && built.frontiers.length ? built.frontiers : liveFrontiers) || [],
      pose: built.pose || (live && live.pose) || (st && st.pose),
      target,
      waypoint_index: idx,
      path_remaining: built.path_remaining != null ? built.path_remaining : (live && live.path_remaining != null ? live.path_remaining : st && st.path_remaining),
    });
  }

  function modeFrom(st, live) {
    const overlay = overlayFrom(st, live);
    const raw = (live && live.mode_banner) || (st && st.mode_banner) || overlay.mode || {};
    const phase = raw.phase || overlay.phase || (live && live.phase) || ((st && st.state) || {}).phase || "idle";
    const job = raw.job_state || (live && live.job_state) || ((st && st.state) || {}).job_state || "idle";
    if (job === "idle" && phase !== "complete" && phase !== "teach") {
      return { label: "Ready", tone: "idle", kind: "idle", hold: null, phase, job_state: job };
    }
    const fallback = MODE_COPY[phase] || MODE_COPY.idle;
    return {
      label: raw.label || fallback.label,
      tone: raw.tone || fallback.tone,
      kind: raw.kind || fallback.kind,
      hold: raw.hold || null,
      phase,
      job_state: job,
    };
  }

  function mapBoxSize(st, live) {
    const yard = state.yard || {};
    return {
      w: Number(yard.width_m || live.width_m || st.width_m || 16),
      h: Number(yard.height_m || live.height_m || st.height_m || 12),
      keep: (live && live.keep_in) || st.keep_in || yard.keep_in || [],
      keepOut: (live && live.keep_out) || st.keep_out || yard.keep_out || [],
    };
  }

  function paintLiveOverlay(st, live) {
    const svg = $("#path-overlay");
    if (!svg || !window.JimsViewer || !window.JimsViewer.drawPathOverlay) return;
    const box = mapBoxSize(st, live);
    window.JimsViewer.drawPathOverlay(svg, {
      width: box.w,
      height: box.h,
      keepIn: box.keep,
      keepOut: box.keepOut,
      overlay: overlayFrom(st, live),
      pose: (live && live.pose) || (st && st.pose),
    });
  }

  function modeBannerHtml(st, live) {
    const mode = modeFrom(st, live);
    const tone = mode.tone || "idle";
    const label = mode.label || "Ready";
    const hold = mode.hold ? `<span class="mode-hold" id="mode-hold">${mode.hold}</span>` : `<span class="mode-hold" id="mode-hold" hidden></span>`;
    return `<div class="mode-banner tone-${tone}" id="mode-banner" data-kind="${mode.kind || "idle"}">
      <span class="mode-label" id="mode-label">${label}</span>
      ${hold}
    </div>`;
  }

  function progressRowHtml(st, live) {
    const overlay = overlayFrom(st, live);
    const mode = modeFrom(st, live);
    const kind = overlay.progress_kind || mode.kind || "idle";
    const mapping = kind === "mapping";
    const mowing = kind === "mowing";
    const mapPct = Number(st.map_pct != null ? st.map_pct : 100 * (live.map_pct || 0));
    const cutPct = Number(st.cut_pct != null ? st.cut_pct : 100 * (live.cut_pct || 0));
    const remaining = overlay.path_remaining != null ? overlay.path_remaining : live.path_remaining;
    const planned = Number(st.planned_pct != null ? st.planned_pct : 100 * (live.planned_pct || 0));
    const cutLabel = mapping ? "Cut (idle)" : "Cut";
    const idx = overlay.waypoint_index;
    let pathLabel = "—";
    if (remaining != null && remaining !== "") pathLabel = `${Number(remaining)} left`;
    if (idx != null && Number(overlay.n_waypoints || live.n_waypoints || 0) > 0) {
      const n = Number(overlay.n_waypoints || live.n_waypoints || 0);
      pathLabel = `${Number(idx) + 1}/${n}`;
    }
    if (mowing && planned > 0) pathLabel = `${pathLabel} · ${planned.toFixed(0)}%`;
    return `<div class="row progress-row" id="progress-row" data-kind="${kind}">
      <div class="chip ${mapping ? "emphasis" : "secondary"}" id="map-chip"><span class="chip-kicker">Map</span><b id="map-pct">${mapPct.toFixed(1)}%</b></div>
      <div class="chip ${mowing ? "emphasis" : "muted"}" id="cut-chip"><span class="chip-kicker" id="cut-kicker">${cutLabel}</span><b id="cut-pct">${cutPct.toFixed(1)}%</b></div>
      <div class="chip" id="path-chip"><span class="chip-kicker">Path</span><b id="path-remaining">${pathLabel}</b></div>
    </div>`;
  }

  function areaLegendRows(st, live) {
    const rows = (st && st.area_legend) || (live && live.area_legend) || [];
    if (rows.length) return rows;
    return [
      { id: "grass", label: "Grass", color: "#2e8c3a" },
      { id: "mowable", label: "Mow this", color: "#7de66e" },
      { id: "path", label: "Path", color: "#80807a" },
      { id: "sand", label: "Sand", color: "#d2b478" },
      { id: "building", label: "Building", color: "#b08a56" },
      { id: "water", label: "Water", color: "#1ca4d6" },
      { id: "drain", label: "Drain", color: "#c46024" },
      { id: "beds", label: "Beds", color: "#58763c" },
      { id: "keepout", label: "Keep-out", color: "#c82828" },
      { id: "fog", label: "Fog", color: "#1c1e22" },
    ];
  }

  function legendHtml(st, live) {
    const areas = areaLegendRows(st, live)
      .map((row) => `<span><i class="swatch area" style="background:${row.color}"></i>${row.label}</span>`)
      .join("");
    return `<div class="map-legend" id="map-legend" aria-label="map legend">
      <span><i class="swatch fog"></i>Fog</span>
      <span><i class="swatch mapped"></i>Mapped</span>
      <span><i class="swatch trail"></i>Trail</span>
      <span><i class="swatch plan"></i>Plan</span>
      <span><i class="swatch cut"></i>Cut</span>
      ${areas}
    </div>`;
  }

  function fenceOverlayHtml(st, live) {
    const box = mapBoxSize(st, live);
    return `<svg class="path-overlay" id="path-overlay" viewBox="0 0 ${box.w} ${box.h}" preserveAspectRatio="none" aria-hidden="true"></svg>`;
  }

  function renderLiveJob() {
    const st = state.status || {};
    const live = state.live || {};
    const paired = isPaired(st) || isPaired(live);
    if (!paired) {
      renderPairingStub();
      return;
    }
    const job = live.job_state || (st.state || {}).job_state || "idle";
    const taught = !!(st.taught || live.taught);
    const copy = st.owner_copy || live.owner_copy || "Yard unknown — start a job when ready.";
    const speed = String(st.speed_label || live.speed_label || "5");
    const needsReteach = !!(st.needs_reteach || live.needs_reteach || st.fence_unusable || live.fence_unusable);
    const canMow = !needsReteach;
    const reason = (st.explore_reason || live.explore_reason || {});
    const reasonLine = reason.label || "";
    const fullExplore = !!(st.full_explore || live.full_explore);
    const fog = live.fog_url || st.fog_url || "/api/live/fog.png";
    const observed = live.observed_url || st.observed_url || "/api/live/observed.png";
    const coverage = live.coverage_url || st.coverage_url || "/api/live/coverage.png";
    const areas = live.areas_url || st.areas_url || "/api/live/areas.png";
    const path = st.radio_path || live.radio_path || {};
    const yardName = (state.yard && state.yard.name) || st.yard || live.yard || "yard";
    const saved = !!(st.yard_saved || live.yard_saved);
    const teaching = job === "teach";
    const kind = (overlayFrom(st, live).progress_kind || modeFrom(st, live).kind || "idle");
    screen().innerHTML = `
      <h1>Live job</h1>
      ${radioChipsHtml(path)}
      ${modeBannerHtml(st, live)}
      <p class="owner-copy secondary" id="owner-copy">${copy}</p>
      <p class="sub explore-reason" id="explore-reason" ${reasonLine ? "" : "hidden"}>${reasonLine}</p>
      <p class="sub" id="yard-chip">${yardName}${taught ? " · taught fence" : " · authored demo fence until you teach"}</p>
      <div class="live-preview kind-${kind}" id="live-preview">
        <img class="obs" id="obs-img" alt="observed terrain" src="${observed}"/>
        <img class="cut" id="cut-img" alt="cut coverage" src="${coverage}"/>
        <img class="areas" id="areas-img" alt="area types and mowable mask" src="${areas}"/>
        <img class="fog" id="fog-img" alt="fog of war" src="${fog}"/>
        ${fenceOverlayHtml(st, live)}
      </div>
      ${legendHtml(st, live)}
      ${progressRowHtml(st, live)}
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
        <button class="btn primary" id="start" ${job === "running" || job === "teach" || needsReteach ? "disabled" : ""}>Start job</button>
        <button class="btn ghost" id="pause" ${job !== "running" && job !== "teach" ? "disabled" : ""}>Pause</button>
      </div>
      <button class="btn ghost" id="resume" ${job !== "paused" && job !== "hold" ? "disabled" : ""}>Resume</button>
      <div class="row phase-row" id="phase-row">
        <button class="btn ghost" id="cmd-explore">Explore</button>
        <button class="btn warn" id="cmd-mow" ${canMow ? "" : "disabled"}>Mow</button>
        <button class="btn ghost" id="cmd-return">Return</button>
      </div>
      <button class="btn ghost ${fullExplore ? "on" : ""}" id="full-explore" aria-pressed="${fullExplore ? "true" : "false"}">${fullExplore ? "Full explore on" : "Full explore off"}</button>
      ${needsReteach ? `<button class="btn warn" id="reteach-cmd">Re-teach fence</button>` : ""}
      ${canMow && (st.can_start_mow || live.can_start_mow) ? `<button class="btn warn" id="start-mow">Start mow</button>` : ""}
      <button class="btn danger" id="estop">ESTOP</button>
      ${sessionCardHtml(st, live)}
      <div class="row">
        <button class="btn ghost" id="inj-stuck">Inject stuck</button>
        <button class="btn ghost" id="inj-sos">Inject SOS</button>
        <button class="btn ghost" id="inj-radio">Inject radio lost</button>
        <button class="btn ghost" id="inj-soc">Inject low SOC</button>
        <button class="btn ghost" id="unpair">Unpair</button>
      </div>
      <p style="margin-top:10px"><a class="linkish" href="/viewer">Open live fog viewer</a></p>`;
    document.querySelectorAll("#speed-row button").forEach((btn) => {
      btn.classList.toggle("on", btn.dataset.speed === speed);
      btn.onclick = () => liveControl("speed", { speed: btn.dataset.speed });
    });
    $("#teach-cmd").onclick = () => liveControl("teach");
    $("#save-yard").onclick = () => liveControl("save_yard");
    const reteach = $("#reteach-cmd");
    if (reteach) reteach.onclick = () => liveControl("teach");
    const loadBtn = $("#load-yard");
    if (loadBtn) loadBtn.onclick = () => liveControl("load_yard");
    $("#start").onclick = () => liveControl("start");
    $("#pause").onclick = () => liveControl("pause");
    $("#resume").onclick = () => liveControl("resume");
    const mow = $("#start-mow");
    if (mow) mow.onclick = () => liveControl("mow");
    const exploreBtn = $("#cmd-explore");
    if (exploreBtn) {
      exploreBtn.onclick = () => {
        const fullEl = $("#full-explore");
        liveControl("explore", { full: !!(fullEl && fullEl.classList.contains("on")) });
      };
    }
    const mowBtn = $("#cmd-mow");
    if (mowBtn) mowBtn.onclick = () => liveControl("mow");
    const returnBtn = $("#cmd-return");
    if (returnBtn) returnBtn.onclick = () => liveControl("return");
    const fullBtn = $("#full-explore");
    if (fullBtn) {
      fullBtn.onclick = () => liveControl("full_explore", { enabled: !fullBtn.classList.contains("on") });
    }
    $("#estop").onclick = () => liveControl("estop");
    $("#inj-stuck").onclick = () => liveControl("inject", { kind: "stuck" });
    const injSoc = $("#inj-soc");
    if (injSoc) injSoc.onclick = () => liveControl("inject", { kind: "low_soc", soc: 0.12 });
    $("#inj-sos").onclick = async () => {
      await liveControl("inject", { kind: "sos" });
      go("fault");
    };
    const injRadio = $("#inj-radio");
    if (injRadio) injRadio.onclick = () => liveControl("inject", { kind: "radio_lost" });
    const unpairBtn = $("#unpair");
    if (unpairBtn) unpairBtn.onclick = () => liveControl("unpair");
    paintLiveOverlay(st, live);
  }

  function renderMap() {
    if (isLive()) {
      renderLiveJob();
      return;
    }
    const st = state.status || {};
    if (st.require_pair && !isPaired(st)) {
      renderPairingStub();
      return;
    }
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

  function scheduleCardHtml(st, yard) {
    const live = st.schedule || {};
    const sch = Object.assign({}, (yard && yard.schedule) || {}, live);
    const enabled = !!sch.enabled;
    const next = sch.next_run_local || sch.next_run || "—";
    const reason = sch.reason || sch.skip_reason;
    const action = sch.action || "idle";
    const rain = !!(st.weather && (st.weather.rain || st.weather.wet));
    const skipLine = reason
      ? `${action === "skip" ? "Skipped" : action}: ${reason}`
      : enabled
        ? "Armed — will start in the next window if SOC / rain / fault gates pass."
        : "Disabled — jobs only start from the map.";
    return `<div class="card" id="schedule-card">
        <div class="toggle-row">
          <strong>Schedule</strong>
          <button type="button" class="toggle ${enabled ? "on" : ""}" id="sched-toggle" aria-pressed="${enabled ? "true" : "false"}" aria-label="Enable schedule">${enabled ? "On" : "Off"}</button>
        </div>
        <div class="sub" id="sched-next">Next run: ${next}</div>
        <div class="sub">${(sch.days || []).join(", ") || "no days"} @ ${sch.start_local || "—"} · ${sch.timezone || "Australia/Brisbane"}</div>
        <div class="sub">SOC gate ${(Number(sch.min_soc != null ? sch.min_soc : 0.25) * 100).toFixed(0)}% · rain skip ${sch.skip_rain === false ? "off" : "on"}${rain ? " · raining now" : ""}</div>
        <div class="sub" id="sched-reason">${skipLine}</div>
      </div>`;
  }

  function renderHealth() {
    const st = state.status || {};
    const bat = st.battery || {};
    const radio = st.radio || {};
    const path = st.radio_path || (state.live && state.live.radio_path) || {};
    screen().innerHTML = `
      <h1>Health</h1>
      <p class="lead">Maintenance, radios, and the weekly job window.</p>
      ${radioChipsHtml(path)}
      <div class="row">
        <div class="chip">Battery<b>${Math.round(100 * Number(bat.soc || 0))}%</b></div>
        <div class="chip">Thermal<b>${Number(bat.temp_c || 0).toFixed(0)}°C</b></div>
      </div>
      <div class="card">
        <strong>Radio link</strong>
        ${radio.transport || radio.link || "none"} · rf_claim ${radio.rf_claim == null ? "none" : radio.rf_claim}
        <div class="sub">BT ${pairingState(st)} ·
          Wi-Fi ${radio.wifi && radio.wifi.enabled ? "on" : "optional / off"} ·
          LoRa ${radio.lora && radio.lora.enabled ? "far-fence sim" : "off"}
          ${(path && path.simulated) ? " · simulated path" : ""} · no metre range</div>
      </div>
      <div class="card">
        <strong>Hours mowed</strong>${Number(st.hours_mowed || 0).toFixed(2)}
        <div class="sub">String-head check every 8 hours (stub).</div>
      </div>
      ${scheduleCardHtml(st, state.yard)}
      <button class="btn ghost" id="redo">Replay onboarding</button>`;
    const toggle = $("#sched-toggle");
    if (toggle) {
      toggle.onclick = async () => {
        const yard = state.yard || {};
        const sch = Object.assign({}, yard.schedule || {}, { enabled: !toggle.classList.contains("on") });
        try {
          await saveYard({ schedule: sch });
          await refresh();
          render();
        } catch (err) {
          const reason = $("#sched-reason");
          if (reason) reason.textContent = err.message || "Could not save schedule";
        }
      };
    }
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
    const hwEstop = faults.some((f) => f.code === "HW_ESTOP") || live.hw_estop;
    const swEstop = !!(st.state && st.state.mission === "estop") || live.estop;
    const banner = hwEstop
      ? `<div class="fault-banner"><strong>Hardware E-STOP</strong><span>Paddle latched. Traction and trimmer rails are dead. Software Start does not restore them — reset the paddle.</span></div>`
      : sos
      ? `<div class="fault-banner"><strong>SOS — immobilised</strong><span>Dead motor. Retrieve the mower. Wheels and trimmer are held.</span></div>`
      : stuck
        ? `<div class="fault-banner"><strong>Stuck</strong><span>Recovery reverse / pivot / help still runs. Not a retrieve.</span></div>`
        : swEstop
          ? `<div class="fault-banner"><strong>Software E-STOP</strong><span>Owner latch. Zeros wheel and trimmer commands until you Start again. Not the hardware paddle.</span></div>`
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
      <p class="sub" style="text-align:center">Software latch · zeros commands until Start. Hardware paddle is a separate rail kill.</p>
      <button class="btn ghost" id="resume">Resume (start)</button>
      ${isLive() ? `<div class="row">
        <button class="btn ghost" id="inj-stuck">Inject stuck</button>
        <button class="btn ghost" id="inj-sos">Inject dead-motor SOS</button>
        <button class="btn danger" id="inj-paddle">Hit paddle (sim)</button>
        <button class="btn ghost" id="hw-reset">Reset paddle</button>
      </div>` : ""}`;
    $("#estop").onclick = () => command("estop", "sos");
    $("#resume").onclick = () => command("start");
    if (isLive()) {
      $("#inj-stuck").onclick = () => liveControl("inject", { kind: "stuck" });
      $("#inj-sos").onclick = () => liveControl("inject", { kind: "sos" });
      $("#inj-paddle").onclick = () => liveControl("hw_estop");
      $("#hw-reset").onclick = () => liveControl("hw_reset");
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
    const prev = state.live || {};
    state.live = frame;
    const phaseChanged = prev.phase !== frame.phase || prev.job_state !== frame.job_state;
    const buttonsChanged =
      !!prev.can_start_mow !== !!frame.can_start_mow ||
      !!prev.needs_reteach !== !!frame.needs_reteach ||
      !!prev.full_explore !== !!frame.full_explore ||
      String(prev.charge_state || "") !== String(frame.charge_state || "");
    if (state.status) {
      state.status.cut_pct = 100 * Number(frame.cut_pct || 0);
      state.status.map_pct = 100 * Number(frame.map_pct || 0);
      state.status.done = !!(frame.done || frame.phase === "complete");
      state.status.owner_copy = frame.owner_copy;
      state.status.can_start_mow = frame.can_start_mow;
      state.status.explore_reason = frame.explore_reason;
      state.status.full_explore = frame.full_explore;
      state.status.charge_state = frame.charge_state;
      state.status.areas_url = frame.areas_url || state.status.areas_url;
      state.status.area_legend = frame.area_legend || state.status.area_legend;
      state.status.needs_reteach = frame.needs_reteach;
      state.status.path_overlay = frame.path_overlay;
      state.status.mode_banner = frame.mode_banner;
      state.status.planned_pct = 100 * Number(frame.planned_pct || 0);
      state.status.waypoint_index = frame.waypoint_index;
      state.status.coverage_url = frame.coverage_url || state.status.coverage_url;
      state.status.frontiers = frame.frontiers;
      state.status.explore = frame.explore;
      state.status.plan = frame.plan;
      if (frame.session_summary) state.status.session_summary = frame.session_summary;
      if (state.status.state) {
        state.status.state.phase = frame.phase;
        state.status.state.job_state = frame.job_state;
        state.status.state.mission = frame.mission || (frame.path_overlay && frame.path_overlay.mission) || state.status.state.mission;
      }
    }
    if (route() === "map" && (phaseChanged || buttonsChanged || (frame.done && !$("#session-card")))) {
      render();
      return;
    }
    const copy = $("#owner-copy");
    if (copy && frame.owner_copy) copy.textContent = frame.owner_copy;
    const reasonEl = $("#explore-reason");
    if (reasonEl) {
      const label = (frame.explore_reason && frame.explore_reason.label) || "";
      reasonEl.textContent = label;
      reasonEl.hidden = !label;
    }
    const mode = frame.mode_banner || (frame.path_overlay && frame.path_overlay.mode) || {};
    const banner = $("#mode-banner");
    if (banner) {
      banner.className = `mode-banner tone-${mode.tone || "idle"}`;
      banner.dataset.kind = mode.kind || "idle";
    }
    const label = $("#mode-label");
    if (label && mode.label) label.textContent = mode.label;
    const hold = $("#mode-hold");
    if (hold) {
      if (mode.hold) {
        hold.hidden = false;
        hold.textContent = mode.hold;
      } else {
        hold.hidden = true;
        hold.textContent = "";
      }
    }
    const preview = $("#live-preview");
    const kind = (frame.path_overlay && frame.path_overlay.progress_kind) || mode.kind || "idle";
    if (preview) preview.className = `live-preview kind-${kind}`;
    const row = $("#progress-row");
    if (row) row.dataset.kind = kind;
    const mapChip = $("#map-chip");
    if (mapChip) mapChip.className = `chip ${kind === "mapping" ? "emphasis" : "secondary"}`;
    const cutChip = $("#cut-chip");
    if (cutChip) cutChip.className = `chip ${kind === "mowing" ? "emphasis" : "muted"}`;
    const cutKicker = $("#cut-kicker");
    if (cutKicker) cutKicker.textContent = kind === "mapping" ? "Cut (idle)" : "Cut";
    const mapEl = $("#map-pct");
    if (mapEl && frame.map_pct != null) mapEl.textContent = `${(100 * Number(frame.map_pct)).toFixed(1)}%`;
    const cutEl = $("#cut-pct");
    if (cutEl && frame.cut_pct != null) cutEl.textContent = `${(100 * Number(frame.cut_pct)).toFixed(1)}%`;
    const pathEl = $("#path-remaining");
    if (pathEl) {
      const overlay = overlayFrom(state.status || {}, frame);
      const remaining = overlay.path_remaining;
      const idx = overlay.waypoint_index;
      const n = Number(overlay.n_waypoints || frame.n_waypoints || 0);
      const planned = Number(state.status && state.status.planned_pct != null ? state.status.planned_pct : 100 * (frame.planned_pct || 0));
      let pathLabel = remaining != null ? `${Number(remaining)} left` : "—";
      if (idx != null && n > 0) pathLabel = `${Number(idx) + 1}/${n}`;
      if (kind === "mowing" && planned > 0) pathLabel = `${pathLabel} · ${planned.toFixed(0)}%`;
      pathEl.textContent = pathLabel;
    }
    const fog = $("#fog-img");
    if (fog && frame.fog_url) fog.src = frame.fog_url;
    const obs = $("#obs-img");
    if (obs && frame.observed_url) obs.src = frame.observed_url;
    const cutImg = $("#cut-img");
    if (cutImg && frame.coverage_url) cutImg.src = frame.coverage_url;
    const areasImg = $("#areas-img");
    if (areasImg && frame.areas_url) areasImg.src = frame.areas_url;
    const fullBtn = $("#full-explore");
    if (fullBtn && frame.full_explore != null) {
      fullBtn.classList.toggle("on", !!frame.full_explore);
      fullBtn.setAttribute("aria-pressed", frame.full_explore ? "true" : "false");
      fullBtn.textContent = frame.full_explore ? "Full explore on" : "Full explore off";
    }
    paintLiveOverlay(state.status || {}, frame);
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
        if (r === "health") {
          const next = $("#sched-next");
          const sch = state.status.schedule || {};
          if (next) next.textContent = `Next run: ${sch.next_run_local || sch.next_run || "—"}`;
          const reason = $("#sched-reason");
          if (reason && (sch.reason || sch.skip_reason)) {
            reason.textContent = `${sch.action || "skip"}: ${sch.reason || sch.skip_reason}`;
          }
          const toggle = $("#sched-toggle");
          if (toggle && sch.enabled != null) {
            toggle.classList.toggle("on", !!sch.enabled);
            toggle.setAttribute("aria-pressed", sch.enabled ? "true" : "false");
            toggle.textContent = sch.enabled ? "On" : "Off";
          }
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

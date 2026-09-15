(() => {
  const ONBOARD = ["unbox", "pair", "home", "teach", "mow"];
  const KEY = "jims_mower_onboarded";

  const state = {
    status: null,
    yard: null,
    coverage: null,
    pairing: { bt: true, wifi: false, lora: true },
    teachPts: [],
  };

  const $ = (sel) => document.querySelector(sel);
  const screen = () => $("#screen");

  function route() {
    const hash = location.hash || "";
    if (hash.startsWith("#/onboard/")) return hash.slice(2);
    if (hash === "#/map") return "map";
    if (hash === "#/health") return "health";
    if (hash === "#/fault") return "fault";
    if (localStorage.getItem(KEY)) return "map";
    return "onboard/unbox";
  }

  function go(name) {
    location.hash = `#/${name}`;
  }

  async function api(path, opts) {
    const res = await fetch(path, Object.assign({ headers: { "Content-Type": "application/json" } }, opts || {}));
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || res.statusText);
    return data;
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
    return status;
  }

  async function command(cmd, reason) {
    await api("/command", { method: "POST", body: JSON.stringify({ cmd, reason }) });
    await refresh();
    render();
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
        `<p class="lead">Bluetooth is required for first contact. Wi-Fi is optional. LoRa is the long-range command link.</p>
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
        const radio = Object.assign({}, state.yard.radio, {
          bluetooth: true,
          primary: "lora",
          lora: { enabled: true, channel: (state.yard.radio.lora || {}).channel || 1 },
          wifi: state.yard.radio.wifi || { enabled: false, ssid: "" },
        });
        await saveYard({ radio });
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
        `<p class="lead">Keep-in is the yellow fence. Keep-outs stay red. Tap the map to add a keep-in vertex, or keep the loaded polygon.</p>
         <svg id="yard-svg" viewBox="0 0 16 12"></svg>
         <button class="btn ghost" id="teach-cmd">Enter teach mode</button>`,
        "Save boundary",
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
      $("#teach-cmd").onclick = () => command("teach");
      $("#next").onclick = async () => {
        if (state.teachPts.length >= 3) await saveYard({ keep_in: state.teachPts });
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
      `<button class="btn ghost" id="skip">Skip to map</button>`
    );
    $("#next").onclick = async () => {
      localStorage.setItem(KEY, "1");
      await command("start");
      go("map");
    };
    $("#skip").onclick = () => {
      localStorage.setItem(KEY, "1");
      go("map");
    };
  }

  function renderMap() {
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
      <p style="margin-top:10px"><a class="linkish" id="ux-a" href="/#/map">2D map (UX-A mesh viewer not installed)</a></p>`;
    if (window.JimsViewer && state.yard) window.JimsViewer.drawYard($("#yard-svg"), state);
    $("#start").onclick = () => command("start");
    $("#stop").onclick = () => command("stop");
    $("#ret").onclick = () => command("return");
    if (window.JimsViewer) {
      window.JimsViewer.uxAAvailable().then((ok) => {
        const a = $("#ux-a");
        if (ok && a) {
          a.href = window.JimsViewer.uxAHref;
          a.textContent = "Open UX-A mesh viewer";
        }
      });
    }
  }

  function renderHealth() {
    const st = state.status || {};
    const bat = st.battery || {};
    const radio = st.radio || {};
    const sch = (state.yard && state.yard.schedule) || {};
    screen().innerHTML = `
      <h1>Health</h1>
      <p class="lead">Maintenance and radio prefs. Schedule is a stub.</p>
      <div class="row">
        <div class="chip">Battery<b>${Math.round(100 * Number(bat.soc || 0))}%</b></div>
        <div class="chip">Thermal<b>${Number(bat.temp_c || 0).toFixed(0)}°C</b></div>
      </div>
      <div class="card">
        <strong>Radio link</strong>
        ${radio.link || "none"} · RSSI ${radio.rssi || "—"}
        <div class="sub">BT ${radio.bluetooth && radio.bluetooth.paired ? "paired" : "no"} ·
          Wi-Fi ${radio.wifi && radio.wifi.enabled ? "on" : "optional / off"} ·
          LoRa ${radio.lora && radio.lora.enabled ? "long-range" : "off"}</div>
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
    const faults = st.faults || [];
    const list = faults.length
      ? faults.map((f) => `<div class="fault-banner"><strong>${f.code}</strong><span>${f.detail || ""}</span></div>`).join("")
      : `<div class="card"><strong>All clear</strong>No latched owner faults.</div>`;
    screen().innerHTML = `
      <h1>SOS</h1>
      ${list}
      <button class="big-sos" id="estop">ESTOP</button>
      <p class="sub" style="text-align:center">Software latch · zeros wheels and trimmer until you Start again</p>
      <button class="btn ghost" id="resume">Resume (start)</button>`;
    $("#estop").onclick = () => command("estop", "sos");
    $("#resume").onclick = () => command("start");
  }

  function render() {
    const r = route();
    nav(r.startsWith("onboard/") ? "" : r);
    if (r.startsWith("onboard/")) renderOnboard(r.split("/")[1] || "unbox");
    else if (r === "health") renderHealth();
    else if (r === "fault") renderFault();
    else renderMap();
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
        location.hash = localStorage.getItem(KEY) ? "#/map" : "#/onboard/unbox";
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
        const r = route();
        if (r === "map" && $("#yard-svg") && window.JimsViewer && state.yard) {
          window.JimsViewer.drawYard($("#yard-svg"), state);
        }
      } catch (_err) {
        /* ignore parse errors */
      }
    };
  }
})();

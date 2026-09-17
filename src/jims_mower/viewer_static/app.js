import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const $ = (id) => document.getElementById(id);

const embedMode = (() => {
  const q = new URLSearchParams(location.search);
  return q.get("embed") === "1" || q.get("chrome") === "embed";
})();
if (embedMode) {
  document.documentElement.classList.add("embed");
  document.body.classList.add("embed");
}

const state = {
  manifest: null,
  profile: null,
  mission: null,
  poses: [],
  plan: [],
  explore: [],
  trail: [],
  index: 0,
  playing: false,
  width: 12,
  height: 12,
  resolution: 0.1,
  relief: 4,
  elevation: null,
  meshGroup: null,
  overlays: {},
  fenceGroup: null,
  poseMarker: null,
  targetMarker: null,
  planLine: null,
  exploreLine: null,
  frontierGroup: null,
  trailLine: null,
  handles: [],
  drag: null,
  live: false,
  followLive: true,
  lastMapSeq: -1,
  lastCamSeq: -1,
  lastStep: -1,
  liveSource: null,
  unknownPad: null,
  observedTerrain: null,
  lastMeshSeq: -1,
  lastDefaultPhase: null,
  hasTarget: false,
};

const renderer = new THREE.WebGLRenderer({ canvas: $("view"), antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setClearColor(0x0c0d10, 1);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 200);
camera.position.set(8, 14, 16);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(6, 0, 6);
controls.enableDamping = true;
scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const sun = new THREE.DirectionalLight(0xfff4d6, 0.9);
sun.position.set(8, 18, 4);
scene.add(sun);
scene.add(new THREE.HemisphereLight(0xb8d0ff, 0x3a2a18, 0.35));
scene.add(new THREE.GridHelper(40, 40, 0x2a2e26, 0x1a1d18));

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

function resize() {
  const el = $("view");
  const w = el.clientWidth || el.parentElement.clientWidth;
  const h = el.clientHeight || el.parentElement.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);

function sampleElev(x, y) {
  const grid = state.elevation;
  if (!grid || !grid.values || !grid.cols) return 0;
  const res = grid.resolution_m || state.resolution || 0.1;
  const col = Math.max(0, Math.min(grid.cols - 1, Math.floor(x / res)));
  const row = Math.max(0, Math.min(grid.rows - 1, Math.floor(y / res)));
  const v = grid.values[row * grid.cols + col];
  return Number.isFinite(v) ? v : 0;
}

function worldToScene(x, y, z = null) {
  const elev = z == null ? sampleElev(x, y) : z;
  const lift = 0.06 + elev * (state.relief || 4);
  return new THREE.Vector3(x, lift, y);
}

function lineFromXY(points, color, closed = false) {
  const pts = points.map((pt) => {
    const x = pt[0];
    const y = pt[1];
    const z = pt.length > 2 && pt[2] != null ? pt[2] : sampleElev(x, y);
    return worldToScene(x, y, z);
  });
  if (closed && pts.length > 2) pts.push(pts[0].clone());
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  return new THREE.Line(geo, new THREE.LineBasicMaterial({ color, linewidth: 2 }));
}

function applyMeshPayload(payload) {
  const pos = new Float32Array(payload.positions);
  const nrm = new Float32Array(payload.normals || []);
  const col = new Float32Array(payload.colors || []);
  const uv = new Float32Array(payload.uvs || []);
  const idx = new Uint32Array(payload.indices);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  if (nrm.length) geo.setAttribute("normal", new THREE.BufferAttribute(nrm, 3));
  else geo.computeVertexNormals();
  if (col.length) geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  if (uv.length) geo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  geo.setIndex(new THREE.BufferAttribute(idx, 1));
  const mat = new THREE.MeshLambertMaterial({
    vertexColors: col.length > 0,
    color: col.length ? 0xffffff : 0x3d8c4a,
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.name = "terrain";
  return mesh;
}

async function loadTerrain(manifest) {
  const group = new THREE.Group();
  group.name = "yard";
  let loaded = false;
  if (manifest.mesh_json) {
    try {
      const payload = await fetch(`/data/${manifest.mesh_json}`).then((r) => r.json());
      const trueMesh = applyMeshPayload(payload);
      trueMesh.userData.kind = "true";
      group.add(trueMesh);
      loaded = true;
    } catch (err) {
      console.warn("mesh json failed", err);
    }
  }
  if (!loaded && manifest.mesh) {
    try {
      const gltf = await new GLTFLoader().loadAsync(`/data/${manifest.mesh}`);
      gltf.scene.userData.kind = "true";
      gltf.scene.traverse((obj) => {
        if (obj.isMesh) obj.userData.kind = "true";
      });
      group.add(gltf.scene);
      loaded = true;
    } catch (err) {
      console.warn("glb failed", err);
    }
  }
  if (!loaded) {
    const fallback = new THREE.Mesh(
      new THREE.PlaneGeometry(manifest.width_m || 12, manifest.height_m || 12),
      new THREE.MeshLambertMaterial({ color: 0x2e8c3a, side: THREE.DoubleSide })
    );
    fallback.rotation.x = -Math.PI / 2;
    fallback.position.set((manifest.width_m || 12) / 2, 0, (manifest.height_m || 12) / 2);
    group.add(fallback);
  }

  const maps = manifest.maps || {};
  const loader = new THREE.TextureLoader();
  const mkOverlay = async (key, url, color, opacity = 0.42) => {
    if (!url) return null;
    try {
      const tex = await loader.loadAsync(`/data/${url}`);
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.flipY = true;
      const w = manifest.width_m || 12;
      const h = manifest.height_m || 12;
      const plane = new THREE.Mesh(
        new THREE.PlaneGeometry(w, h),
        new THREE.MeshBasicMaterial({
          map: tex,
          transparent: true,
          opacity,
          depthWrite: false,
          color,
        })
      );
      plane.rotation.x = -Math.PI / 2;
      plane.position.set(w / 2, key === "fog" ? 0.18 : 0.12, h / 2);
      plane.visible = false;
      group.add(plane);
      state.overlays[key] = plane;
      return plane;
    } catch (err) {
      console.warn("overlay", key, err);
      return null;
    }
  };
  await mkOverlay("coverage", maps.coverage, 0xffffff);
  await mkOverlay("hazard", maps.hazard, 0xffffff);
  await mkOverlay("occupancy", maps.occupancy, 0xffffff);
  await mkOverlay("error", maps.elevation_error, 0xffffff);
  await mkOverlay("observed", maps.observed, 0xffffff);
  await mkOverlay("fog", maps.fog, 0xffffff, 1.0);
  return group;
}

function applyPhaseLayerDefaults(phase) {
  if (!phase || phase === state.lastDefaultPhase) return;
  state.lastDefaultPhase = phase;
  const mapping = phase === "explore" || phase === "calibrate_boundary" || phase === "teach";
  const mowing = phase === "mow" || phase === "return_home" || phase === "charging" || phase === "complete";
  if (mapping) {
    if ($("tog-explore")) $("tog-explore").checked = true;
    if ($("tog-trail")) $("tog-trail").checked = true;
    if ($("tog-plan")) $("tog-plan").checked = false;
    if ($("tog-coverage")) $("tog-coverage").checked = false;
  } else if (mowing) {
    if ($("tog-plan")) $("tog-plan").checked = true;
    if ($("tog-trail")) $("tog-trail").checked = true;
    if ($("tog-coverage")) $("tog-coverage").checked = true;
    if ($("tog-explore")) $("tog-explore").checked = false;
  } else if (phase === "review") {
    if ($("tog-plan")) $("tog-plan").checked = true;
    if ($("tog-trail")) $("tog-trail").checked = true;
    if ($("tog-explore")) $("tog-explore").checked = false;
  }
  setOverlayVis();
}

function setOverlayVis() {
  if (state.overlays.coverage) state.overlays.coverage.visible = $("tog-coverage").checked;
  if (state.overlays.hazard) state.overlays.hazard.visible = $("tog-hazard").checked;
  if (state.overlays.occupancy) state.overlays.occupancy.visible = $("tog-occupancy").checked;
  if (state.overlays.error) state.overlays.error.visible = $("tog-error") && $("tog-error").checked;
  if (state.overlays.observed) {
    state.overlays.observed.visible = $("tog-observed") ? $("tog-observed").checked : false;
  }
  const god = $("tog-god") && $("tog-god").checked;
  const fogOn = $("tog-fog") && $("tog-fog").checked && !god;
  if (state.overlays.fog) state.overlays.fog.visible = !!fogOn;
  setOwnerMeshVis(fogOn);
  if (state.planLine) state.planLine.visible = $("tog-plan").checked;
  if (state.exploreLine) {
    state.exploreLine.visible = $("tog-explore") ? $("tog-explore").checked : false;
  }
  if (state.frontierGroup) {
    state.frontierGroup.visible = $("tog-explore") ? $("tog-explore").checked : true;
  }
  if (state.poseMarker) state.poseMarker.visible = $("tog-pose").checked;
  if (state.targetMarker) state.targetMarker.visible = $("tog-pose").checked && !!state.hasTarget;
  if (state.fenceGroup) state.fenceGroup.visible = $("tog-fence").checked;
  if (state.trailLine) state.trailLine.visible = $("tog-trail").checked;
}

function rebuildFence() {
  if (state.fenceGroup) scene.remove(state.fenceGroup);
  const g = new THREE.Group();
  const keepIn = (state.profile && state.profile.keep_in) || [];
  if (keepIn.length >= 2) g.add(lineFromXY(keepIn, 0xffcc33, true));
  for (const hole of (state.profile && state.profile.keep_out) || []) {
    if (hole.length >= 2) g.add(lineFromXY(hole, 0xc82828, true));
  }
  state.handles.forEach((h) => scene.remove(h));
  state.handles = [];
  keepIn.forEach((pt, i) => {
    const sph = new THREE.Mesh(
      new THREE.SphereGeometry(0.12, 12, 12),
      new THREE.MeshLambertMaterial({ color: 0xffee88 })
    );
    sph.position.copy(worldToScene(pt[0], pt[1], 0.16));
    sph.userData = { kind: "keep_in", index: i };
    g.add(sph);
    state.handles.push(sph);
  });
  ((state.profile && state.profile.keep_out) || []).forEach((hole, hi) => {
    hole.forEach((pt, i) => {
      const sph = new THREE.Mesh(
        new THREE.SphereGeometry(0.10, 10, 10),
        new THREE.MeshLambertMaterial({ color: 0xff6655 })
      );
      sph.position.copy(worldToScene(pt[0], pt[1], 0.16));
      sph.userData = { kind: "keep_out", hole: hi, index: i };
      g.add(sph);
      state.handles.push(sph);
    });
  });
  scene.add(g);
  state.fenceGroup = g;
  setOverlayVis();
}

function poseAt(i) {
  if (!state.poses.length) return { x: 1, y: 1, theta: 0, z: 0 };
  return state.poses[Math.max(0, Math.min(i, state.poses.length - 1))];
}

function phaseAt(i) {
  const p = poseAt(i);
  if (p.phase) return p.phase;
  const ranges = (state.mission && state.mission.phase_ranges) || [];
  for (const row of ranges) {
    const start = row.start || 0;
    const end = row.end == null ? Infinity : row.end;
    if (i >= start && i <= end) return row.phase;
  }
  return "";
}

function nearestSnapshot(step) {
  const snaps = (state.mission && state.mission.snapshots) || [];
  if (!snaps.length) return null;
  return snaps.reduce((best, cur) =>
    Math.abs((cur.step || 0) - step) < Math.abs((best.step || 0) - step) ? cur : best
  );
}

function setPhaseBar(phase) {
  document.querySelectorAll(".phase-seg").forEach((el) => {
    el.classList.toggle("active", el.dataset.phase === phase);
  });
  const chip = $("phase-chip");
  if (chip) chip.textContent = `phase ${phase || "—"}`;
}

function setMissionMetrics(i) {
  const el = $("mission-metrics");
  if (!el) return;
  const snap = nearestSnapshot(i);
  const metrics = (state.mission && state.mission.metrics) || {};
  const pose = poseAt(i);
  const completion = pose.map_completion != null
    ? pose.map_completion
    : (snap && snap.map_completion) || metrics.map_completion || 0;
  const cut = pose.actual_coverage_fraction != null
    ? pose.actual_coverage_fraction
    : (snap && snap.actual_coverage_fraction) || metrics.actual_coverage_fraction || 0;
  const planned = metrics.planned_coverage_fraction || 0;
  const reach = metrics.reachable_mowable_cells || 0;
  const unreach = metrics.unreachable_mowable_cells || 0;
  const denom = reach + unreach;
  const reachPct = denom ? (100 * reach / denom) : 0;
  const unreachPct = denom ? (100 * unreach / denom) : 0;
  el.textContent =
    `map ${(100 * completion).toFixed(1)}%\n` +
    `reachable mowable ${reachPct.toFixed(1)}%  unreachable ${unreachPct.toFixed(1)}%\n` +
    `planned ${(100 * planned).toFixed(1)}%  cut ${(100 * cut).toFixed(1)}%`;
}

function updateObservedOverlay(i) {
  const plane = state.overlays.observed;
  const snap = nearestSnapshot(i);
  if (!plane || !snap || !snap.file) return;
  const loader = new THREE.TextureLoader();
  loader.load(`/data/${snap.file}`, (tex) => {
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.flipY = true;
    if (plane.material.map) plane.material.map.dispose();
    plane.material.map = tex;
    plane.material.needsUpdate = true;
  });
}

function updatePose(i) {
  state.index = i;
  const p = poseAt(i);
  if (state.poseMarker) {
    state.poseMarker.position.copy(worldToScene(p.x, p.y, p.z || 0));
    state.poseMarker.rotation.y = -(p.theta || 0);
  }
  $("scrub").value = String(i);
  const phase = phaseAt(i);
  $("scrub-label").textContent = `step ${i} / ${Math.max(0, state.poses.length - 1)} · ${phase || ""}`;
  setPhaseBar(phase);
  setMissionMetrics(i);
  updateObservedOverlay(i);
  const showMow = phase === "mow" || phase === "return_home" || phase === "charging" || phase === "complete" || phase === "review";
  if (state.planLine) state.planLine.visible = $("tog-plan").checked && showMow;
  if (state.exploreLine) {
    const exploreOn = $("tog-explore") ? $("tog-explore").checked : true;
    state.exploreLine.visible = exploreOn && (phase === "explore" || phase === "calibrate_boundary");
  }
  updatePip(i);
}

function nearestCamStep(step) {
  const steps = (state.manifest && state.manifest.camera_steps) || [];
  if (!steps.length) return null;
  return steps.reduce((best, cur) =>
    Math.abs(cur.step - step) < Math.abs(best.step - step) ? cur : best
  );
}

function updatePip(step) {
  const rec = nearestCamStep(step);
  const select = $("cam-select");
  const name = select.value;
  if (!rec || !name) {
    $("pip-meta").textContent = "no frames";
    return;
  }
  $("pip-img").src = `/data/${rec.dir}/cam_${name}.png`;
  $("pip-meta").textContent = `${rec.dir} · ${name}`;
}

function makePoseMarker() {
  const g = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.ConeGeometry(0.16, 0.42, 8),
    new THREE.MeshLambertMaterial({ color: 0x222222 })
  );
  body.rotation.x = Math.PI / 2;
  g.add(body);
  scene.add(g);
  return g;
}

function makeTargetMarker() {
  const mesh = new THREE.Mesh(
    new THREE.RingGeometry(0.12, 0.2, 16),
    new THREE.MeshBasicMaterial({ color: 0xffee88, side: THREE.DoubleSide })
  );
  mesh.rotation.x = -Math.PI / 2;
  mesh.visible = false;
  scene.add(mesh);
  return mesh;
}

function placeTargetMarker(target) {
  if (!state.targetMarker) state.targetMarker = makeTargetMarker();
  const x = target == null ? null : (Array.isArray(target) ? target[0] : target.x);
  const y = target == null ? null : (Array.isArray(target) ? target[1] : target.y);
  state.hasTarget = x != null && y != null;
  if (!state.hasTarget) {
    state.targetMarker.visible = false;
    return;
  }
  state.targetMarker.position.copy(worldToScene(x, y, 0.14));
  state.targetMarker.visible = $("tog-pose") ? $("tog-pose").checked : true;
}

function pointerFromEvent(ev) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
}

function groundHit(ev) {
  pointerFromEvent(ev);
  raycaster.setFromCamera(pointer, camera);
  const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  const hit = new THREE.Vector3();
  raycaster.ray.intersectPlane(plane, hit);
  return hit;
}

renderer.domElement.addEventListener("pointerdown", (ev) => {
  pointerFromEvent(ev);
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(state.handles, false);
  if (hits.length) {
    state.drag = hits[0].object;
    controls.enabled = false;
  }
});
renderer.domElement.addEventListener("pointermove", (ev) => {
  if (!state.drag || !state.profile) return;
  const hit = groundHit(ev);
  if (!hit) return;
  const x = Math.max(0, Math.min(state.width, hit.x));
  const y = Math.max(0, Math.min(state.height, hit.z));
  const data = state.drag.userData;
  if (data.kind === "keep_in") {
    state.profile.keep_in[data.index] = [x, y];
  } else if (data.kind === "keep_out") {
    state.profile.keep_out[data.hole][data.index] = [x, y];
  }
  rebuildFence();
  const again = state.handles.find(
    (h) =>
      h.userData.kind === data.kind &&
      h.userData.index === data.index &&
      h.userData.hole === data.hole
  );
  state.drag = again || null;
});
window.addEventListener("pointerup", () => {
  state.drag = null;
  controls.enabled = true;
});

$("btn-add-keepout").addEventListener("click", () => {
  if (!state.profile) state.profile = { keep_in: [], keep_out: [], home: { x: 1, y: 1, theta: 0 } };
  const cx = state.width * 0.5;
  const cy = state.height * 0.5;
  const r = 0.7;
  state.profile.keep_out = state.profile.keep_out || [];
  state.profile.keep_out.push([
    [cx - r, cy - r],
    [cx + r, cy - r],
    [cx + r, cy + r],
    [cx - r, cy + r],
  ]);
  rebuildFence();
  $("save-status").textContent = "keep-out added — drag corners, then save";
});

$("btn-save").addEventListener("click", async () => {
  if (!state.profile) return;
  state.profile.schema = state.profile.schema || "jims_mower.yard.v1";
  state.profile.not_a_benchmark = true;
  try {
    const res = await fetch("/api/profile", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state.profile),
    });
    const body = await res.json();
    $("save-status").textContent = `saved ${body.path || "profile.json"}`;
  } catch (err) {
    const blob = new Blob([JSON.stringify(state.profile, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "profile.json";
    a.click();
    $("save-status").textContent = "downloaded profile.json (server save failed)";
  }
});

["tog-coverage", "tog-hazard", "tog-occupancy", "tog-observed", "tog-fog", "tog-god", "tog-plan", "tog-explore", "tog-pose", "tog-fence", "tog-trail", "tog-error"].forEach(
  (id) => $(id) && $(id).addEventListener("change", setOverlayVis)
);

$("scrub").addEventListener("input", (ev) => updatePose(Number(ev.target.value)));
$("cam-select").addEventListener("change", () => updatePip(state.index));
$("btn-play").addEventListener("click", () => {
  if (state.live) {
    state.followLive = !state.followLive;
    $("btn-play").textContent = state.followLive ? "Follow" : "Paused";
    return;
  }
  state.playing = !state.playing;
  $("btn-play").textContent = state.playing ? "Pause" : "Play";
});

function tick() {
  requestAnimationFrame(tick);
  if (state.playing && state.poses.length && !state.followLive) {
    const next = state.index + 1;
    if (next >= state.poses.length) {
      state.playing = false;
      $("btn-play").textContent = "Play";
    } else {
      updatePose(next);
    }
  }
  controls.update();
  renderer.render(scene, camera);
}

function ensureUnknownPad() {
  if (state.unknownPad) return state.unknownPad;
  const w = state.width || 12;
  const h = state.height || 12;
  const pad = new THREE.Mesh(
    new THREE.PlaneGeometry(w, h),
    new THREE.MeshLambertMaterial({ color: 0x12141a, side: THREE.DoubleSide })
  );
  pad.rotation.x = -Math.PI / 2;
  pad.position.set(w / 2, 0.01, h / 2);
  pad.name = "unknownPad";
  scene.add(pad);
  state.unknownPad = pad;
  return pad;
}

function setOwnerMeshVis(fogOn) {
  // Owner mode: growing observed elevation + fog. God-view shows the
  // true physics mesh. Control never reads the unfogged field.
  const hideTrue = !!fogOn;
  ensureUnknownPad();
  if (state.unknownPad) state.unknownPad.visible = hideTrue;
  if (state.observedTerrain) state.observedTerrain.visible = hideTrue;
  if (state.meshGroup) {
    const overlaySet = new Set(Object.values(state.overlays).filter(Boolean));
    state.meshGroup.traverse((obj) => {
      if (!obj.isMesh) return;
      if (overlaySet.has(obj)) return;
      if (obj === state.observedTerrain) return;
      if (obj.userData && obj.userData.kind === "observed") return;
      obj.visible = !hideTrue;
    });
  }
  if (state.overlays.observed && state.overlays.observed.material) {
    state.overlays.observed.material.opacity = hideTrue ? 0.28 : 0.42;
  }
}

function disposeMesh(mesh) {
  if (!mesh) return;
  if (mesh.geometry) mesh.geometry.dispose();
  if (mesh.material) mesh.material.dispose();
}

function applyObservedMesh(payload) {
  if (!payload || !payload.positions || (payload.vertex_count || 0) < 3) return;
  if ((payload.indices || []).length < 3) return;
  // Overlay fog / observed planes stay flat (texture on z≈0). This mesh
  // carries real metres of learned height. Same viewer relief as the
  // true-field group — not a second IMU hinge.
  const pos = new Float32Array(payload.positions);
  const idx = new Uint32Array(payload.indices);
  const existing = state.observedTerrain;
  const sameVerts =
    existing &&
    existing.geometry &&
    existing.geometry.getAttribute("position") &&
    existing.geometry.getAttribute("position").count === pos.length / 3;
  if (sameVerts) {
    const geo = existing.geometry;
    const prev = geo.getAttribute("position").array;
    const locked = existing.userData.lockedVerts;
    if (locked && locked.size) {
      // Interior Y stays put. New verts (not yet in a quad) take payload
      // height so the patch grows without a local re-tilt of old cells.
      for (const vi of locked) {
        const y = vi * 3 + 1;
        pos[y] = prev[y];
      }
    }
    const nextLocked = new Set(locked || []);
    for (let i = 0; i < idx.length; i += 1) nextLocked.add(idx[i]);
    existing.userData.lockedVerts = nextLocked;
    geo.getAttribute("position").array.set(pos);
    geo.getAttribute("position").needsUpdate = true;
    if (payload.normals && geo.getAttribute("normal") && payload.normals.length) {
      geo.getAttribute("normal").array.set(new Float32Array(payload.normals));
      geo.getAttribute("normal").needsUpdate = true;
    }
    if (payload.colors && geo.getAttribute("color") && payload.colors.length) {
      geo.getAttribute("color").array.set(new Float32Array(payload.colors));
      geo.getAttribute("color").needsUpdate = true;
    }
    geo.setIndex(new THREE.BufferAttribute(idx, 1));
    geo.computeBoundingSphere();
  } else {
    const mesh = applyMeshPayload(payload);
    mesh.name = "observedTerrain";
    mesh.userData.kind = "observed";
    mesh.userData.lockedVerts = new Set(idx);
    if (state.observedTerrain) {
      if (state.meshGroup) state.meshGroup.remove(state.observedTerrain);
      else scene.remove(state.observedTerrain);
      disposeMesh(state.observedTerrain);
    }
    state.observedTerrain = mesh;
    if (state.meshGroup) state.meshGroup.add(mesh);
    else scene.add(mesh);
  }
  const chip = $("mesh-chip");
  if (chip && payload.vertex_count) {
    chip.textContent = `observed ${payload.triangle_count || 0} t · ${payload.vertex_count} v`;
  }
  setOverlayVis();
}

function ensureOverlayPlane(key, opacity) {
  if (state.overlays[key]) return state.overlays[key];
  if (!state.meshGroup) return null;
  const w = state.width || 12;
  const h = state.height || 12;
  const plane = new THREE.Mesh(
    new THREE.PlaneGeometry(w, h),
    new THREE.MeshBasicMaterial({
      transparent: true,
      opacity: opacity == null ? 0.42 : opacity,
      depthWrite: false,
      color: 0xffffff,
    })
  );
  plane.rotation.x = -Math.PI / 2;
  plane.position.set(w / 2, key === "fog" ? 0.18 : 0.12, h / 2);
  state.meshGroup.add(plane);
  state.overlays[key] = plane;
  return plane;
}

function loadOverlayUrl(key, url, opacity) {
  if (!url) return;
  const plane = ensureOverlayPlane(key, opacity);
  if (!plane) return;
  const loader = new THREE.TextureLoader();
  loader.load(url, (tex) => {
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.flipY = true;
    if (plane.material.map) plane.material.map.dispose();
    plane.material.map = tex;
    plane.material.needsUpdate = true;
    setOverlayVis();
  });
}

function replaceLine(key, points, color) {
  if (state[key]) {
    scene.remove(state[key]);
    state[key].geometry.dispose();
  }
  if (!points || points.length < 2) {
    state[key] = null;
    return;
  }
  const line = lineFromXY(points, color, false);
  scene.add(line);
  state[key] = line;
}

function replaceFrontiers(pts) {
  if (state.frontierGroup) {
    scene.remove(state.frontierGroup);
    state.frontierGroup.traverse((child) => {
      if (child.geometry) child.geometry.dispose();
    });
  }
  if (!pts || !pts.length) {
    state.frontierGroup = null;
    return;
  }
  const g = new THREE.Group();
  pts.forEach((pt) => {
    const x = Array.isArray(pt) ? pt[0] : pt.x;
    const y = Array.isArray(pt) ? pt[1] : pt.y;
    const sph = new THREE.Mesh(
      new THREE.SphereGeometry(0.08, 8, 8),
      new THREE.MeshLambertMaterial({ color: 0x42c4dc })
    );
    sph.position.copy(worldToScene(x, y, 0.12));
    g.add(sph);
  });
  scene.add(g);
  state.frontierGroup = g;
  setOverlayVis();
}

async function postControl(cmd, extra) {
  const body = Object.assign({ cmd }, extra || {});
  const res = await fetch("/api/live/control", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const json = await res.json();
  if (json && json.live !== false) applyLiveFrame(json);
  return json;
}

function setOwnerBar(frame) {
  const bar = $("owner-bar");
  if (!bar) return;
  if (embedMode) {
    bar.hidden = true;
    return;
  }
  bar.hidden = false;
  const copy = $("owner-copy");
  if (copy) copy.textContent = frame.owner_copy || "Yard unknown — start a job when ready.";
  const yard = $("owner-yard");
  if (yard && frame.yard) yard.value = frame.yard;
  document.querySelectorAll(".speed-btn").forEach((btn) => {
    const want = frame.speed_label != null ? String(frame.speed_label) : "1";
    btn.classList.toggle("active", btn.dataset.speed === want);
  });
  const idle = frame.job_state === "idle";
  const start = $("btn-job-start");
  if (start) start.disabled = frame.job_state === "running";
  const pause = $("btn-job-pause");
  if (pause) pause.disabled = frame.job_state !== "running";
  const resume = $("btn-job-resume");
  if (resume) resume.disabled = frame.job_state !== "paused" && frame.job_state !== "hold";
  const startMow = $("btn-start-mow");
  if (startMow) startMow.disabled = !!frame.needs_reteach;
  const reexplore = $("btn-reexplore");
  if (reexplore) reexplore.hidden = !frame.can_reexplore;
  const fullBtn = $("btn-full-explore");
  if (fullBtn) fullBtn.classList.toggle("active", !!frame.full_explore);
  const reason = frame.explore_reason || {};
  if (reason.label) $("save-status").textContent = reason.label;
  setPhaseBar(idle ? "" : (frame.phase || ""));
  applyPhaseLayerDefaults(idle ? "" : (frame.phase || ""));
  const chip = $("phase-chip");
  if (chip) chip.textContent = idle ? "IDLE" : `LIVE ${frame.phase_label || frame.phase || "—"}`;
  const summary = $("session-summary");
  if (summary && frame.session_summary && (frame.done || frame.phase === "complete" || frame.phase === "return_home")) {
    const s = frame.session_summary;
    const reach = s.reachable || 0;
    const unreach = s.unreachable || 0;
    const denom = reach + unreach;
    const reachPct = denom ? (100 * reach) / denom : 0;
    summary.hidden = false;
    summary.textContent =
      `session  map ${(100 * (s.map_pct || 0)).toFixed(1)}%\n` +
      `planned ${(100 * (s.planned_pct || 0)).toFixed(1)}%  reachable ${reachPct.toFixed(1)}%\n` +
      `cut ${(100 * (s.cut_pct || 0)).toFixed(1)}%  skips ${s.skips || 0}\n` +
      `duration ${(s.duration_s || 0).toFixed(1)}s sim` +
      (s.wall_s ? ` · ${s.wall_s.toFixed(1)}s wall` : "");
  }
}

function bindOwnerBar() {
  const start = $("btn-job-start");
  if (start) {
    start.addEventListener("click", () => {
      const yard = $("owner-yard") && $("owner-yard").value;
      postControl("start", { yard }).catch((err) => {
        $("save-status").textContent = String(err);
      });
    });
  }
  const pause = $("btn-job-pause");
  if (pause) pause.addEventListener("click", () => postControl("pause").catch(() => {}));
  const resume = $("btn-job-resume");
  if (resume) resume.addEventListener("click", () => postControl("resume").catch(() => {}));
  document.querySelectorAll(".speed-btn").forEach((btn) => {
    btn.addEventListener("click", () => postControl("speed", { speed: btn.dataset.speed }).catch(() => {}));
  });
  const startMow = $("btn-start-mow");
  if (startMow) startMow.addEventListener("click", () => postControl("mow").catch(() => {}));
  const explore = $("btn-explore");
  if (explore) explore.addEventListener("click", () => postControl("explore").catch(() => {}));
  const ret = $("btn-return");
  if (ret) ret.addEventListener("click", () => postControl("return").catch(() => {}));
  const fullBtn = $("btn-full-explore");
  if (fullBtn) {
    fullBtn.addEventListener("click", () => {
      postControl("full_explore", { enabled: !fullBtn.classList.contains("active") }).catch(() => {});
    });
  }
  const reexplore = $("btn-reexplore");
  if (reexplore) reexplore.addEventListener("click", () => postControl("reexplore").catch(() => {}));
  const reset = $("btn-reset");
  if (reset) reset.addEventListener("click", () => postControl("reset").catch(() => {}));
  const estop = $("btn-estop");
  if (estop) estop.addEventListener("click", () => postControl("estop").catch(() => {}));
  const yard = $("owner-yard");
  if (yard) {
    yard.addEventListener("change", () => {
      postControl("yard", { yard: yard.value }).catch((err) => {
        $("save-status").textContent = String(err);
      });
    });
  }
}

function applyLiveFrame(frame) {
  if (!frame || frame.live === false) return;
  setOwnerBar(frame);
  if (frame.step != null && frame.step !== state.lastStep && frame.pose) {
    const row = Object.assign({}, frame.pose, {
      phase: frame.phase,
      phase_label: frame.phase_label,
      map_completion: frame.map_pct,
      actual_coverage_fraction: frame.cut_pct,
      step: frame.step,
    });
    state.poses.push(row);
    state.lastStep = frame.step;
    $("scrub").max = String(Math.max(0, state.poses.length - 1));
  }
  if (state.followLive && frame.pose && state.poseMarker) {
    state.poseMarker.position.copy(worldToScene(frame.pose.x, frame.pose.y, frame.pose.z || 0));
    state.poseMarker.rotation.y = -(frame.pose.theta || 0);
    const idle = frame.job_state === "idle";
    const phase = idle ? "" : (frame.phase || "");
    const phaseLabel = idle ? "idle" : (frame.phase_label || phase);
    $("scrub").value = String(Math.max(0, state.poses.length - 1));
    $("scrub-label").textContent = `LIVE step ${frame.step || 0} · ${phaseLabel}`;
    const el = $("mission-metrics");
    if (el) {
      const reach = frame.reachable || 0;
      const unreach = frame.unreachable || 0;
      const denom = reach + unreach;
      const reachPct = denom ? (100 * reach) / denom : 0;
      const unreachPct = denom ? (100 * unreach) / denom : 0;
      el.textContent =
        `map ${(100 * (frame.map_pct || 0)).toFixed(1)}%\n` +
        `reachable mowable ${reachPct.toFixed(1)}%  unreachable ${unreachPct.toFixed(1)}%\n` +
        `planned ${(100 * (frame.planned_pct || 0)).toFixed(1)}%  cut ${(100 * (frame.cut_pct || 0)).toFixed(1)}%\n` +
        `skips ${frame.skips || 0}`;
    }
    const showMow = phase === "mow" || phase === "return_home" || phase === "charging" || phase === "complete" || phase === "review";
    if (state.planLine) state.planLine.visible = $("tog-plan").checked && showMow;
    applyPhaseLayerDefaults(phase);
  }
  const overlay = frame.path_overlay || {};
  if (overlay.trail && overlay.trail.length >= 2) {
    replaceLine("trailLine", overlay.trail, 0xaa88ff);
    if (state.trailLine) state.trailLine.visible = $("tog-trail") ? $("tog-trail").checked : true;
  }
  placeTargetMarker(overlay.target);
  if (frame.mesh_seq != null && frame.mesh_seq !== state.lastMeshSeq && frame.observed_mesh_url) {
    fetch(frame.observed_mesh_url)
      .then((r) => r.json())
      .then((payload) => applyObservedMesh(payload))
      .catch((err) => console.warn("observed mesh", err));
    state.lastMeshSeq = frame.mesh_seq;
  }
  if (frame.map_seq != null && frame.map_seq !== state.lastMapSeq) {
    loadOverlayUrl("observed", frame.observed_url, 0.28);
    loadOverlayUrl("fog", frame.fog_url, 1.0);
    if (frame.coverage_url) loadOverlayUrl("coverage", frame.coverage_url, 0.55);
    state.lastMapSeq = frame.map_seq;
    const overlayPts = frame.path_overlay || {};
    replaceFrontiers(overlayPts.frontiers && overlayPts.frontiers.length ? overlayPts.frontiers : (frame.frontiers || []));
    const explorePts = (overlayPts.explore && overlayPts.explore.length ? overlayPts.explore : (frame.explore || [])).map((p) => (Array.isArray(p) ? p : [p.x, p.y]));
    replaceLine("exploreLine", explorePts, 0xf0a030);
    const planSrc = overlayPts.plan && overlayPts.plan.length >= 2 ? overlayPts.plan : (frame.plan || []);
    if (planSrc.length >= 2) {
      replaceLine("planLine", planSrc.map((p) => (Array.isArray(p) ? p : [p.x, p.y])), 0x2ad4e6);
    }
    setOverlayVis();
  }
  if (frame.cam_seq != null && frame.cam_seq !== state.lastCamSeq) {
    const select = $("cam-select");
    const name = (select && select.value) || (frame.cameras && frame.cameras[0]) || "front";
    $("pip-img").src = `/api/live/cam/${name}?v=${frame.cam_seq}`;
    $("pip-meta").textContent = `live · ${name}`;
    state.lastCamSeq = frame.cam_seq;
  }
  if (frame.keep_in && frame.keep_in.length >= 3) {
    if (!state.profile) state.profile = { keep_in: [], keep_out: [], home: { x: 1, y: 1, theta: 0 } };
    state.profile.keep_in = frame.keep_in;
    if (frame.keep_out) state.profile.keep_out = frame.keep_out;
    rebuildFence();
  }
}

function startLive() {
  state.live = true;
  state.followLive = true;
  const chip = $("live-chip");
  if (chip) {
    chip.hidden = false;
    chip.classList.add("live");
    chip.textContent = "LIVE";
  }
  if ($("tog-fog")) $("tog-fog").checked = true;
  if ($("tog-god")) $("tog-god").checked = false;
  if ($("tog-observed")) $("tog-observed").checked = false;
  if ($("tog-error")) $("tog-error").checked = false;
  if ($("btn-play")) $("btn-play").textContent = "Follow";
  bindOwnerBar();
  setOverlayVis();
  if (typeof EventSource === "undefined") {
    $("save-status").textContent = "EventSource missing — poll /api/live/snapshot";
    return;
  }
  const src = new EventSource("/api/live");
  state.liveSource = src;
  src.onmessage = (ev) => {
    try {
      applyLiveFrame(JSON.parse(ev.data));
    } catch (err) {
      console.warn("live frame", err);
    }
  };
  src.onerror = () => {
    $("save-status").textContent = "live stream reconnecting…";
  };
}

async function boot() {
  resize();
  const manifest = await fetch("/api/manifest").then((r) => r.json());
  state.manifest = manifest;
  state.width = manifest.width_m || 12;
  state.height = manifest.height_m || 12;
  state.resolution = manifest.resolution_m || 0.1;
  state.relief = manifest.relief_scale || 4;
  if (manifest.maps && manifest.maps.elevation) {
    try {
      state.elevation = await fetch(`/data/${manifest.maps.elevation}`).then((r) => r.json());
    } catch (err) {
      console.warn("elevation json failed", err);
    }
  }
  const live = !!manifest.live;
  $("subtitle").textContent = live
    ? `LIVE ${manifest.policy || "mission"} · observed terrain + fog · ${state.width.toFixed(1)}×${state.height.toFixed(1)} m · no mAP/FPS`
    : `${manifest.policy || "sim"} · ${state.width.toFixed(1)}×${state.height.toFixed(1)} m · no mAP/FPS`;
  $("mesh-chip").textContent = `mesh ${manifest.vertex_count || 0} v / ${manifest.triangle_count || 0} t`;
  controls.target.set(state.width / 2, 0, state.height / 2);
  camera.position.set(state.width * 0.15, Math.max(state.width, state.height) * 0.9, state.height * 1.15);

  state.meshGroup = await loadTerrain(manifest);
  // Viewer-only Y lift so a ~5% yard grade and 10–20 cm drains read on a
  // 12 m pad. Applies to true + observed meshes equally. Fog / observed
  // *overlays* are flat planes (holes, not a hinged sheet).
  state.meshGroup.scale.y = state.relief || 4;
  scene.add(state.meshGroup);
  if (manifest.maps && manifest.maps.observed_mesh) {
    try {
      const payload = await fetch(`/data/${manifest.maps.observed_mesh}`).then((r) => r.json());
      applyObservedMesh(payload);
    } catch (err) {
      console.warn("observed mesh boot", err);
    }
  }

  if (manifest.poses) {
    const pack = await fetch(`/data/${manifest.poses}`).then((r) => r.json());
    state.poses = pack.poses || [];
  }
  if (manifest.mission) {
    try {
      state.mission = await fetch(`/data/${manifest.mission}`).then((r) => r.json());
    } catch (err) {
      console.warn("mission json failed", err);
    }
  }
  if (manifest.plan) {
    const plan = await fetch(`/data/${manifest.plan}`).then((r) => r.json());
    state.plan = (plan.waypoints || []).map((p) => [p.x, p.y, p.z]);
    if (state.plan.length >= 2) {
      state.planLine = lineFromXY(state.plan, 0x2ad4e6, false);
      scene.add(state.planLine);
    }
    const explore = plan.explore || (state.mission && state.mission.explore_route) || [];
    state.explore = explore.map((p) => [p.x, p.y, p.z]);
    if (state.explore.length >= 2) {
      state.exploreLine = lineFromXY(state.explore, 0xf0a030, false);
      scene.add(state.exploreLine);
    }
  }
  if (state.mission && state.mission.frontiers && state.mission.frontiers.length) {
    const g = new THREE.Group();
    state.mission.frontiers.forEach((pt) => {
      const sph = new THREE.Mesh(
        new THREE.SphereGeometry(0.08, 8, 8),
        new THREE.MeshLambertMaterial({ color: 0x42c4dc })
      );
      sph.position.copy(worldToScene(pt.x, pt.y, 0.12));
      g.add(sph);
    });
    scene.add(g);
    state.frontierGroup = g;
  }
  if (manifest.profile) {
    state.profile = await fetch(`/data/${manifest.profile}`).then((r) => r.json());
    state.trail = state.profile.trail || [];
  } else {
    state.profile = {
      schema: "jims_mower.yard.v1",
      name: "taught",
      width_m: state.width,
      height_m: state.height,
      keep_in: [
        [0.8, 0.8],
        [state.width - 0.8, 0.8],
        [state.width - 0.8, state.height - 0.8],
        [0.8, state.height - 0.8],
      ],
      keep_out: [],
      home: { x: 1, y: 1, theta: 0 },
      mesh: "yard.glb",
      not_a_benchmark: true,
    };
  }
  if (state.trail.length >= 2) {
    state.trailLine = lineFromXY(state.trail, 0xaa88ff, false);
    state.trailLine.visible = false;
    scene.add(state.trailLine);
  }
  rebuildFence();
  state.poseMarker = makePoseMarker();
  state.targetMarker = makeTargetMarker();

  const cams = manifest.cameras || [];
  const select = $("cam-select");
  select.innerHTML = "";
  cams.forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    select.appendChild(opt);
  });
  $("scrub").max = String(Math.max(0, state.poses.length - 1));
  $("tog-coverage").checked = false;
  if (live) {
    startLive();
  } else if ($("tog-fog") && manifest.fog) {
    $("tog-fog").checked = true;
  }
  setOverlayVis();
  if (state.poses.length) updatePose(0);
  else if (state.poseMarker) updatePose(0);
  tick();
}

boot().catch((err) => {
  $("save-status").textContent = String(err);
  console.error(err);
});

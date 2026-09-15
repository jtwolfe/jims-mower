import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const $ = (id) => document.getElementById(id);

const state = {
  manifest: null,
  profile: null,
  poses: [],
  plan: [],
  trail: [],
  index: 0,
  playing: false,
  width: 12,
  height: 12,
  meshGroup: null,
  overlays: {},
  fenceGroup: null,
  poseMarker: null,
  planLine: null,
  trailLine: null,
  handles: [],
  drag: null,
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

function worldToScene(x, y, z = 0.04) {
  return new THREE.Vector3(x, z, y);
}

function lineFromXY(points, color, closed = false) {
  const pts = points.map(([x, y]) => worldToScene(x, y, 0.08));
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
      group.add(applyMeshPayload(payload));
      loaded = true;
    } catch (err) {
      console.warn("mesh json failed", err);
    }
  }
  if (!loaded && manifest.mesh) {
    try {
      const gltf = await new GLTFLoader().loadAsync(`/data/${manifest.mesh}`);
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
  const mkOverlay = async (key, url, color) => {
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
          opacity: 0.42,
          depthWrite: false,
          color,
        })
      );
      plane.rotation.x = -Math.PI / 2;
      plane.position.set(w / 2, 0.12, h / 2);
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
  return group;
}

function setOverlayVis() {
  if (state.overlays.coverage) state.overlays.coverage.visible = $("tog-coverage").checked;
  if (state.overlays.hazard) state.overlays.hazard.visible = $("tog-hazard").checked;
  if (state.overlays.occupancy) state.overlays.occupancy.visible = $("tog-occupancy").checked;
  if (state.planLine) state.planLine.visible = $("tog-plan").checked;
  if (state.poseMarker) state.poseMarker.visible = $("tog-pose").checked;
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

function updatePose(i) {
  state.index = i;
  const p = poseAt(i);
  if (state.poseMarker) {
    state.poseMarker.position.copy(worldToScene(p.x, p.y, 0.18 + (p.z || 0)));
    state.poseMarker.rotation.y = -(p.theta || 0);
  }
  $("scrub").value = String(i);
  $("scrub-label").textContent = `step ${i} / ${Math.max(0, state.poses.length - 1)}`;
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

["tog-coverage", "tog-hazard", "tog-occupancy", "tog-plan", "tog-pose", "tog-fence", "tog-trail"].forEach(
  (id) => $(id).addEventListener("change", setOverlayVis)
);

$("scrub").addEventListener("input", (ev) => updatePose(Number(ev.target.value)));
$("cam-select").addEventListener("change", () => updatePip(state.index));
$("btn-play").addEventListener("click", () => {
  state.playing = !state.playing;
  $("btn-play").textContent = state.playing ? "Pause" : "Play";
});

function tick() {
  requestAnimationFrame(tick);
  if (state.playing && state.poses.length) {
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

async function boot() {
  resize();
  const manifest = await fetch("/api/manifest").then((r) => r.json());
  state.manifest = manifest;
  state.width = manifest.width_m || 12;
  state.height = manifest.height_m || 12;
  $("subtitle").textContent = `${manifest.policy || "sim"} · ${state.width.toFixed(1)}×${state.height.toFixed(1)} m · no mAP/FPS`;
  $("mesh-chip").textContent = `mesh ${manifest.vertex_count || 0} v / ${manifest.triangle_count || 0} t`;
  controls.target.set(state.width / 2, 0, state.height / 2);
  camera.position.set(state.width * 0.15, Math.max(state.width, state.height) * 0.9, state.height * 1.15);

  state.meshGroup = await loadTerrain(manifest);
  // Viewer-only lift so 10–20 cm drains/banks read on a 12 m yard.
  state.meshGroup.scale.y = 4;
  scene.add(state.meshGroup);

  if (manifest.poses) {
    const pack = await fetch(`/data/${manifest.poses}`).then((r) => r.json());
    state.poses = pack.poses || [];
  }
  if (manifest.plan) {
    const plan = await fetch(`/data/${manifest.plan}`).then((r) => r.json());
    state.plan = (plan.waypoints || []).map((p) => [p.x, p.y]);
    if (state.plan.length >= 2) {
      state.planLine = lineFromXY(state.plan, 0x2ad4e6, false);
      scene.add(state.planLine);
    }
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
  setOverlayVis();
  updatePose(0);
  tick();
}

boot().catch((err) => {
  $("save-status").textContent = String(err);
  console.error(err);
});

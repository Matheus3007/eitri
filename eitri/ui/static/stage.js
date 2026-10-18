// The 3D viewport (Stage), the telemetry timeline (Timeline) and SimView, which puts them together with the HUD
// (title, colorbar, probe, view tools) and the dock (transport, readouts, field/colormap/deform/range/fps).
// Used by the Simulation tab, the Part tab and both Compare panes.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { $, h, esc, fmt, api, icon, Select, Combo } from "./ui.js";

// ---------- colormaps ----------
const stops = (s) => (t) => {
  for (let i = 1; i < s.length; i++) if (t <= s[i][0]) {
    const [t0, a] = s[i - 1], [t1, b] = s[i], u = (t - t0) / (t1 - t0 || 1);
    return [a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1]), a[2] + u * (b[2] - a[2])];
  }
  return s[s.length - 1][1];
};
export const COLORMAPS = {
  forge: stops([[0, [0.09, 0.1, 0.13]], [0.3, [0.36, 0.06, 0.07]], [0.58, [0.86, 0.22, 0.07]], [0.8, [1, 0.56, 0.18]], [0.93, [1, 0.82, 0.5]], [1, [1, 0.97, 0.9]]]),
  turbo: (t) => [
    0.13572138 + t * (4.6153926 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943)))),
    0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604)))),
    0.1066733 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))))],
  inferno: stops([[0, [0.001, 0, 0.014]], [0.25, [0.258, 0.039, 0.406]], [0.5, [0.735, 0.216, 0.33]], [0.75, [0.978, 0.557, 0.035]], [1, [0.988, 0.998, 0.645]]]),
  viridis: stops([[0, [0.267, 0.005, 0.329]], [0.25, [0.229, 0.322, 0.546]], [0.5, [0.128, 0.567, 0.551]], [0.75, [0.369, 0.789, 0.383]], [1, [0.993, 0.906, 0.144]]]),
  steel: stops([[0, [0.06, 0.07, 0.08]], [0.5, [0.33, 0.35, 0.39]], [1, [0.86, 0.88, 0.91]]]),
};
const OVER = [0.21, 0.88, 0.82];  // above the range: the live cyan
const UNYIELDED = [0.36, 0.38, 0.42];  // plastic strain: material still elastic, plain steel
const lin = (c) => { c = Math.min(1, Math.max(0, c)); return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); };
export const gradientCss = (name, dir = "to top") =>
  `linear-gradient(${dir}, ${Array.from({ length: 11 }, (_, i) => { const c = COLORMAPS[name](i / 10).map((x) => Math.round(Math.min(1, Math.max(0, x)) * 255)); return `rgb(${c}) ${i * 10}%`; }).join(",")})`;

const DEFAULTS = {  // per field: colormap and range
  "plastic strain": { cmap: "forge", max: 0.15 }, "von Mises": { cmap: "forge" }, "of yield": { cmap: "turbo", max: 1 },
  speed: { cmap: "viridis" }, density: { cmap: "steel", max: 1 }, part: { cmap: "viridis" },
};

// ---------- Stage ----------
export class Stage {
  constructor(host) {
    this.host = host;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.renderer.localClippingEnabled = true;
    host.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    const pm = new THREE.PMREMGenerator(this.renderer);
    this.env = pm.fromScene(new RoomEnvironment(), 0.04).texture;
    this.camera = new THREE.PerspectiveCamera(30, 1, 0.1, 10000);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true; this.controls.dampingFactor = 0.12;
    this.scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x1a1410, 0.9));
    const key = new THREE.DirectionalLight(0xffffff, 1.6); key.position.set(2, -3, 4); this.scene.add(key);
    const rim = new THREE.DirectionalLight(0xff8a50, 0.6); rim.position.set(-3, 2, -1); this.scene.add(rim);
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(host);
    this.alive = true;
    const loop = () => {
      if (!this.alive) return;
      requestAnimationFrame(loop);
      this.onTick?.();
      this.controls.update();
      if (this.autoRotate && this.root) this.root.rotation.z += 0.0025;
      this.renderer.render(this.scene, this.camera);
    };
    loop();
  }
  resize() {
    const w = this.host.clientWidth, hgt = this.host.clientHeight;
    if (!w || !hgt) return;
    this.renderer.setSize(w, hgt);
    this.camera.aspect = w / hgt; this.camera.updateProjectionMatrix();
  }
  clear() {
    if (this.root) {
      this.scene.remove(this.root);
      this.root.traverse((o) => { o.geometry?.dispose(); (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => m?.dispose()); });
      this.root = null;
    }
    this.scene.environment = null;
  }
  dispose() { this.alive = false; this.ro.disconnect(); this.clear(); this.renderer.dispose(); this.renderer.forceContextLoss(); this.renderer.domElement.remove(); }
  setRoot(obj, fitTarget = null) { this.clear(); this.root = obj; this.fitTarget = fitTarget; this.scene.add(obj); }
  fit(view = this.view || "iso") {
    this.view = view;
    if (!this.root) return;
    const box = new THREE.Box3().setFromObject(this.fitTarget || this.root);  // the part, not its floor grid
    const sph = box.getBoundingSphere(new THREE.Sphere());
    const c = sph.center, r = sph.radius || 1;
    const d = r / Math.sin(THREE.MathUtils.degToRad(this.camera.fov / 2)) * 1.06 * (this.fitScale || 1);
    const dirs = { iso: [0.55, -0.95, 0.75], top: [0, 0, 1], front: [0, -1, 0.0001], side: [1, 0, 0.0001] };
    const v = new THREE.Vector3(...dirs[view]).normalize();
    this.camera.up.set(0, view === "top" ? 1 : 0, view === "top" ? 0 : 1);
    this.camera.position.copy(c).addScaledVector(v, d);
    this.camera.near = r / 100; this.camera.far = r * 100; this.camera.updateProjectionMatrix();
    this.controls.target.copy(c);
    this.controls.update();
  }
  pick(ev, objs) {
    const r = this.renderer.domElement.getBoundingClientRect();
    const p = new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
    const rc = new THREE.Raycaster();
    rc.setFromCamera(p, this.camera);
    return rc.intersectObjects(objs, false)[0] || null;
  }
}

async function data(dir, part) {  // never a ".bin" URL or a download content type (IDM, see server.py)
  return (await fetch(`/api/data?dir=${encodeURIComponent(dir)}&part=${part}`)).arrayBuffer();
}

// ---------- Timeline ----------
export class Timeline {
  constructor(host, onSeek) {
    this.wrap = h(`<div class="tl-wrap"><canvas></canvas><div class="tl-tip glass"></div></div>`);
    host.appendChild(this.wrap);
    this.cv = this.wrap.querySelector("canvas");
    this.tip = this.wrap.querySelector(".tl-tip");
    this.onSeek = onSeek;
    this.k = 0; this.hover = null;
    new ResizeObserver(() => this.draw()).observe(this.wrap);
    const seek = (e) => { const k = this.frameAt(e); if (k !== null) this.onSeek(k); };
    this.cv.onpointerdown = (e) => { this.drag = true; this.cv.setPointerCapture(e.pointerId); seek(e); };
    this.cv.onpointermove = (e) => { this.hover = this.xToT(e); if (this.drag) seek(e); this.draw(); this.showTip(e); };
    this.cv.onpointerup = () => { this.drag = false; };
    this.cv.onpointerleave = () => { this.hover = null; this.tip.style.display = "none"; this.draw(); };
  }
  set({ times, steps, unit, stepUnit, curve, frameCurve }) {
    Object.assign(this, { times, steps, unit, stepUnit, curve, frameCurve });
    this.draw();
  }
  setCurve(curve) { this.curve = curve; this.draw(); }
  setFrame(k) { this.k = k; this.draw(); }
  get pad() { return { l: 54, r: 16, t: 10, b: 22 }; }
  xToT(e) {
    const r = this.cv.getBoundingClientRect(), P = this.pad;
    if (!this.times?.length) return null;
    const t0 = this.times[0], t1 = this.times[this.times.length - 1];
    return t0 + Math.min(1, Math.max(0, (e.clientX - r.left - P.l) / (r.width - P.l - P.r))) * (t1 - t0);
  }
  frameAt(e) {
    const t = this.xToT(e);
    if (t === null) return null;
    let best = 0;
    for (let i = 1; i < this.times.length; i++) if (Math.abs(this.times[i] - t) < Math.abs(this.times[best] - t)) best = i;
    return best;
  }
  sample(c, t) {
    if (!c) return null;
    const T = c.t; let i = 1;
    while (i < T.length - 1 && T[i] < t) i++;
    const u = (t - T[i - 1]) / (T[i] - T[i - 1] || 1);
    return c.v[i - 1] + Math.min(1, Math.max(0, u)) * (c.v[i] - c.v[i - 1]);
  }
  showTip(e) {
    if (this.hover === null) return;
    const k = this.frameAt(e), c = this.curve;
    const v = c ? this.sample(c, this.hover) : null;
    this.tip.innerHTML = `${fmt(this.hover, 4)} ${esc(this.unit)} · frame ${k}${c ? ` · <b style="color:var(--accent-2)">${fmt(v, 4)} ${esc(c.unit)}</b>` : ""}`;
    this.tip.style.display = "block";
    const r = this.wrap.getBoundingClientRect(), x = e.clientX - r.left;
    this.tip.style.left = `${Math.min(r.width - this.tip.offsetWidth - 6, x + 12)}px`;
  }
  draw() {
    const cv = this.cv, dpr = Math.min(2, window.devicePixelRatio);
    const W = this.wrap.clientWidth, H = this.wrap.clientHeight;
    if (!W) return;
    cv.width = W * dpr; cv.height = H * dpr;
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    if (!this.times?.length) return;
    const P = this.pad, t0 = this.times[0], t1 = this.times[this.times.length - 1];
    const X = (t) => P.l + (t - t0) / (t1 - t0 || 1) * (W - P.l - P.r);
    const ih = H - P.t - P.b;
    g.font = "10px 'JetBrains Mono', monospace";
    // grid
    g.strokeStyle = "rgba(255,255,255,0.05)"; g.lineWidth = 1;
    for (let i = 0; i <= 4; i++) { const y = Math.round(P.t + ih * i / 4) + 0.5; g.beginPath(); g.moveTo(P.l, y); g.lineTo(W - P.r, y); g.stroke(); }
    // channel: the selected curve (area) and the per-frame field peak (line)
    const plot = (c, color, fill) => {
      if (!c || !c.t.length) return;
      let lo = Math.min(0, ...c.v), hi = Math.max(...c.v);
      if (!(hi > lo)) hi = lo + 1;
      const Y = (v) => P.t + ih - (v - lo) / (hi - lo) * ih;
      g.beginPath();
      c.t.forEach((t, i) => (i ? g.lineTo(X(t), Y(c.v[i])) : g.moveTo(X(t), Y(c.v[i]))));
      if (fill) {
        g.save(); g.lineTo(X(c.t[c.t.length - 1]), P.t + ih); g.lineTo(X(c.t[0]), P.t + ih); g.closePath();
        const gr = g.createLinearGradient(0, P.t, 0, P.t + ih); gr.addColorStop(0, "rgba(255,91,31,0.35)"); gr.addColorStop(1, "rgba(255,91,31,0.02)");
        g.fillStyle = gr; g.fill(); g.restore();
        g.beginPath(); c.t.forEach((t, i) => (i ? g.lineTo(X(t), Y(c.v[i])) : g.moveTo(X(t), Y(c.v[i]))));
      }
      g.strokeStyle = color; g.lineWidth = 1.5; g.stroke();
      return { lo, hi };
    };
    const rng = plot(this.curve, "#ff6a2b", true);
    const second = this.frameCurve && this.frameCurve !== this.curve;
    if (second) plot(this.frameCurve, "rgba(53,224,208,0.55)", false);
    // legend, top right
    g.textAlign = "right"; let lx = W - P.r - 6;
    for (const [c, col] of [[second ? this.frameCurve : null, "rgba(53,224,208,0.8)"], [this.curve, "#ff6a2b"]]) {
      if (!c) continue;
      const txt = `${c.name}${c.unit ? ` [${c.unit}]` : ""}`;
      g.fillStyle = "#b6bdc7"; g.fillText(txt, lx, P.t + 10);
      lx -= g.measureText(txt).width + 8; g.fillStyle = col; g.fillRect(lx - 12, P.t + 5, 12, 2); lx -= 26;
    }
    if (rng) {
      g.fillStyle = "#7a838f"; g.textAlign = "right";
      g.fillText(fmt(rng.hi, 3), P.l - 8, P.t + 8); g.fillText(fmt(rng.lo, 3), P.l - 8, P.t + ih);
      g.save(); g.translate(12, P.t + ih / 2); g.rotate(-Math.PI / 2); g.textAlign = "center"; g.fillStyle = "#4b525c";
      g.fillText(this.curve.unit, 0, 0); g.restore();
    }
    // frame ticks: every saved frame; labels on round times
    g.textAlign = "center";
    const n = this.times.length, every = Math.max(1, Math.ceil(n / Math.max(4, (W - P.l - P.r) / 70)));
    for (let i = 0; i < n; i++) {
      const x = Math.round(X(this.times[i])) + 0.5, major = i % every === 0 || i === n - 1;
      g.strokeStyle = i <= this.k ? "rgba(255,138,61,0.65)" : "rgba(255,255,255,0.16)";
      g.beginPath(); g.moveTo(x, H - P.b); g.lineTo(x, H - P.b + (major ? 6 : 3)); g.stroke();
      if (major && (i === n - 1 || X(this.times[n - 1]) - x > 40)) { g.fillStyle = "#7a838f"; g.fillText(fmt(this.times[i], 3), x, H - 4); }
    }
    // played region + playhead
    const px = Math.round(X(this.times[this.k])) + 0.5;
    g.fillStyle = "rgba(255,91,31,0.05)"; g.fillRect(P.l, P.t, px - P.l, ih);
    if (this.hover !== null) { const hx = Math.round(X(this.hover)) + 0.5; g.strokeStyle = "rgba(255,255,255,0.25)"; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(hx, P.t); g.lineTo(hx, H - P.b); g.stroke(); g.setLineDash([]); }
    g.strokeStyle = "#ff5b1f"; g.lineWidth = 1.5; g.beginPath(); g.moveTo(px, P.t - 4); g.lineTo(px, H - P.b); g.stroke();
    g.fillStyle = "#ff5b1f"; g.beginPath(); g.moveTo(px - 5, P.t - 6); g.lineTo(px + 5, P.t - 6); g.lineTo(px, P.t + 1); g.closePath(); g.fill();
  }
}

// ---------- SimView ----------
export class SimView {
  constructor(host, { compact = false, onFrame = null, onLoad = null } = {}) {
    this.compact = compact; this.onFrame = onFrame; this.onLoad = onLoad;
    this.el = h(`<div style="display:flex;flex-direction:column;flex:1;height:100%;min-height:0;min-width:0;position:relative">
      <div class="stage-host" style="flex:1;min-height:0">
        <div class="hud tl"></div><div class="hud tr"></div><div class="hud rc glass"></div><div class="hud bl"></div>
        <div class="probe glass"></div>
        <div class="loading"><div class="box glass"><span class="spinner"></span><span class="msg">Loading…</span></div></div>
      </div>
      <div class="dock" style="display:none"></div></div>`);
    host.replaceChildren(this.el);
    this.hostEl = this.el.querySelector(".stage-host");
    this.stage = new Stage(this.hostEl);
    this.hud = { tl: this.hostEl.querySelector(".hud.tl"), tr: this.hostEl.querySelector(".hud.tr"), rc: this.hostEl.querySelector(".hud.rc"), bl: this.hostEl.querySelector(".hud.bl") };
    this.probe = this.hostEl.querySelector(".probe");
    this.dock = this.el.querySelector(".dock");
    this.playing = false; this.loop = true; this.fps = 12; this.k = 0; this.scale = 1; this.vmax = null; this.wire = false;
    this.stage.onTick = () => this.tick();
    this.toolbar();
    this.hostEl.addEventListener("pointermove", (e) => this.onProbe(e));
    this.hostEl.addEventListener("pointerleave", () => { this.probe.style.display = "none"; this.highlight(-1); });
    this.keys = (e) => {
      if (!this.m || this.m.kind === "part" || !this.el.offsetParent || e.target.closest("input,textarea")) return;
      if (e.key === " ") { e.preventDefault(); this.toggle(); }
      if (e.key === "ArrowRight") this.seek(Math.min(this.k + 1, this.n - 1));
      if (e.key === "ArrowLeft") this.seek(Math.max(this.k - 1, 0));
    };
    if (!compact) document.addEventListener("keydown", this.keys);
  }
  dispose() { document.removeEventListener("keydown", this.keys); this.stage.dispose(); }
  loading(msg) { const l = this.hostEl.querySelector(".loading"); l.classList.toggle("on", !!msg); if (msg) l.querySelector(".msg").textContent = msg; }
  empty(msg) {
    this.stage.clear(); this.m = null; this.dock.style.display = "none";
    this.hud.tl.innerHTML = msg ? `<div class="glass hud-title"><div class="eyebrow">Viewport</div><div class="t">${esc(msg)}</div></div>` : "";
    this.hud.tr.innerHTML = ""; this.hud.bl.innerHTML = "";
  }
  toolbar() {
    const tools = [["iso", "Isometric", () => this.stage.fit("iso")], ["top", "Top", () => this.stage.fit("top")],
      ["front", "Front", () => this.stage.fit("front")], ["fit", "Fit", () => this.stage.fit()],
      ["edges", "Mesh / edges", () => this.toggleWire()], ["section", "Section cut", () => this.toggleSection()]];
    this.hud.rc.innerHTML = tools.map(([ic, t], i) => `<button class="toolbtn" data-i="${i}" title="${t}">${icon(ic)}</button>`).join("");
    for (const b of this.hud.rc.querySelectorAll("button")) b.onclick = () => { tools[+b.dataset.i][2](); this.syncTools(); };
  }
  syncTools() {
    const bs = this.hud.rc.querySelectorAll("button");
    bs[4].classList.toggle("on", this.wire); bs[5].classList.toggle("on", !!this.section);
    bs[5].style.display = this.m?.kind === "part" ? "" : "none";
  }

  // ----- animation bundles (FEA, explicit, topology) -----
  async loadAnim(dir, { keepView = false } = {}) {
    this.loading("Loading simulation…");
    try {
      const m = await api(`/api/timeline?dir=${encodeURIComponent(dir)}`);
      const [mesh, frames, vis] = await Promise.all([data(dir, "mesh"), data(dir, "frames"), m.has_vis ? data(dir, "vis") : null]);
      this.m = m; this.dir = dir; this.n = m.times.length;
      const n = m.n_nodes, nt = m.n_tris;
      this.base = new Float32Array(mesh, 0, n * 3);
      this.tris = new Uint32Array(mesh, n * 12, nt * 3);
      this.frames = new Float32Array(frames);
      this.vis = vis ? new Uint8Array(vis) : null;
      this.stride = (m.has_disp ? n * 3 : 0) + m.fields.length * n;
      const g = new THREE.BufferGeometry();
      this.pos = new Float32Array(n * 3); this.pos.set(this.base);
      this.col = new Float32Array(n * 3);
      g.setAttribute("position", new THREE.BufferAttribute(this.pos, 3));
      g.setAttribute("color", new THREE.BufferAttribute(this.col, 3));
      g.setIndex(new THREE.BufferAttribute(new Uint32Array(this.tris), 1));
      this.flat = m.kind !== "explicit";
      const mat = this.flat ? new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide })
        : new THREE.MeshStandardMaterial({ vertexColors: true, side: THREE.DoubleSide, roughness: 0.55, metalness: 0.15, envMapIntensity: 0.5 });
      this.mesh = new THREE.Mesh(g, mat);
      const root = new THREE.Group(); root.add(this.mesh);
      this.wireObj = new THREE.LineSegments(new THREE.WireframeGeometry(g), new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.22 }));
      this.wireObj.visible = this.wire; root.add(this.wireObj);
      this.stage.setRoot(root);
      this.stage.scene.environment = this.flat ? null : this.stage.env;
      this.section = null;
      if (!keepView) this.stage.fit(this.flat ? "top" : "iso");
      this.fieldIdx = Math.max(0, m.fields.findIndex((f) => f.name === m.default_field));
      this.scale = m.disp_scale_default || 1;
      this.applyFieldDefaults();
      this.buildDock();
      this.k = 0;
      this.show(0);
      this.renderTitle();
      this.syncTools();
      this.onLoad?.(this);
    } finally { this.loading(null); }
  }
  applyFieldDefaults() {
    const f = this.m.fields[this.fieldIdx], d = DEFAULTS[f.name] || {};
    this.cmap = d.cmap || "forge";
    this.vmax = null;
    this.autoMax = d.max !== undefined ? (f.name === "plastic strain" ? Math.min(f.max, d.max) : d.max) : f.max;
  }
  field() { return this.m.fields[this.fieldIdx]; }
  range() {
    const f = this.field();
    const lo = ["of yield", "density"].includes(f.name) ? 0 : Math.max(0, f.min);
    const hi = this.vmax ?? this.autoMax;
    return [lo, hi > lo ? hi : lo + 1];
  }
  value(k, i) {
    const m = this.m, n = m.n_nodes;
    return this.frames[k * this.stride + (m.has_disp ? n * 3 : 0) + this.fieldIdx * n + i];
  }
  show(k) {
    if (!this.m || this.m.kind === "part") return;
    const m = this.m, n = m.n_nodes;
    this.k = k;
    const off = k * this.stride;
    if (m.has_disp) for (let i = 0; i < n * 3; i++) this.pos[i] = this.base[i] + this.scale * this.frames[off + i];
    const foff = off + (m.has_disp ? n * 3 : 0) + this.fieldIdx * n;
    const [lo, hi] = this.range();
    const cm = COLORMAPS[this.cmap], elastic = this.field().name === "plastic strain" ? 0.002 : -Infinity;
    for (let i = 0; i < n; i++) {
      const v = this.frames[foff + i];
      let c;
      if (v < elastic) c = UNYIELDED;
      else { const t = (v - lo) / (hi - lo); c = t > 1.0001 ? OVER : cm(Math.max(0, t)); }
      // colormaps are defined in sRGB; vertex colors are linear (the renderer converts back on output)
      this.col[i * 3] = lin(c[0]); this.col[i * 3 + 1] = lin(c[1]); this.col[i * 3 + 2] = lin(c[2]);
    }
    const geo = this.mesh.geometry;
    if (this.vis) {
      const v = this.vis.subarray(k * m.n_tris, (k + 1) * m.n_tris), idx = [];
      for (let t = 0; t < m.n_tris; t++) if (v[t]) idx.push(this.tris[3 * t], this.tris[3 * t + 1], this.tris[3 * t + 2]);
      geo.setIndex(new THREE.BufferAttribute(new Uint32Array(idx), 1));
    }
    geo.attributes.position.needsUpdate = true;
    geo.attributes.color.needsUpdate = true;
    if (!this.flat) geo.computeVertexNormals();
    geo.computeBoundingSphere();
    if (this.wire && m.has_disp) { this.wireObj.geometry.dispose(); this.wireObj.geometry = new THREE.WireframeGeometry(geo); }
    this.renderColorbar();
    this.renderReadout();
    this.timeline?.setFrame(k);
    this.onFrame?.(k, this);
  }
  seek(k) { this.show(Math.max(0, Math.min(this.n - 1, k))); }
  seekTime(t) {
    if (!this.m || this.m.kind === "part") return;
    let best = 0;
    for (let i = 1; i < this.n; i++) if (Math.abs(this.m.times[i] - t) < Math.abs(this.m.times[best] - t)) best = i;
    if (best !== this.k) this.show(best);
  }
  toggle(on = !this.playing) {
    this.playing = on;
    if (on && this.k >= this.n - 1) this.show(0);
    const b = this.dock.querySelector(".play");
    if (b) b.innerHTML = icon(on ? "pause" : "play");
  }
  tick() {
    if (!this.playing || !this.m || this.m.kind === "part") return;
    const now = performance.now();
    if (now - (this.last || 0) < 1000 / this.fps) return;
    this.last = now;
    if (this.k >= this.n - 1) { if (this.loop) this.show(0); else this.toggle(false); return; }
    this.show(this.k + 1);
  }
  frameCurve() {
    const m = this.m, n = m.n_nodes, f = this.field();
    const v = [];
    for (let k = 0; k < this.n; k++) {
      let mx = -Infinity;
      for (let i = 0; i < n; i++) { const x = this.value(k, i); if (x > mx) mx = x; }
      v.push(mx);
    }
    return { name: `peak ${f.name}`, unit: f.unit === "-" ? "" : f.unit, t: m.times, v };
  }
  buildDock() {
    const m = this.m, d = this.dock;
    d.style.display = "";
    d.innerHTML = `<div class="dock-row">
        <div class="transport">
          <button class="toolbtn" data-a="first" title="First frame">${icon("first")}</button>
          <button class="toolbtn" data-a="prev" title="Previous frame (←)">${icon("prev")}</button>
          <button class="toolbtn play" data-a="play" title="Play / pause (space)">${icon("play")}</button>
          <button class="toolbtn" data-a="next" title="Next frame (→)">${icon("next")}</button>
          <button class="toolbtn" data-a="last" title="Last frame">${icon("last")}</button>
          <button class="toolbtn ${this.loop ? "on" : ""}" data-a="loop" title="Loop">${icon("loop")}</button>
        </div>
        <div class="readout"></div><div class="spacer"></div>
        <span data-c="trace"></span><span data-c="field"></span>${this.compact ? "" : `<span data-c="cmap"></span>`}
        ${m.has_disp ? `<span data-c="scale"></span>` : ""}<span data-c="max"></span>${this.compact ? "" : `<span data-c="fps"></span>`}
      </div>`;
    const act = { first: () => this.seek(0), prev: () => this.seek(this.k - 1), next: () => this.seek(this.k + 1), last: () => this.seek(this.n - 1),
      play: () => this.toggle(), loop: (b) => { this.loop = !this.loop; b.classList.toggle("on", this.loop); } };
    for (const b of d.querySelectorAll("[data-a]")) b.onclick = () => act[b.dataset.a](b);
    const slot = (c) => d.querySelector(`[data-c=${c}]`);
    new Select(slot("field"), { label: "Field", width: this.compact ? "190px" : "205px", value: this.fieldIdx,
      options: m.fields.map((f, i) => ({ value: i, label: f.name, sub: f.unit !== "-" ? f.unit : "" })),
      onChange: (v) => { this.fieldIdx = v; this.applyFieldDefaults(); this.maxCombo.set(null, false); this.maxCombo.setPresets(this.maxPresets()); this.cmapSel?.set(this.cmap, false); this.refreshTrace(); this.show(this.k); } });
    if (!this.compact) this.cmapSel = new Select(slot("cmap"), { label: "Map", width: "132px", value: this.cmap,
      options: Object.keys(COLORMAPS).map((k) => ({ value: k, label: k })), onChange: (v) => { this.cmap = v; this.show(this.k); } });
    if (m.has_disp) new Combo(slot("scale"), { label: "Deform ×", width: "138px", value: this.scale,
      presets: [0, 1, 2, 5, 10, 20, 50, 100, 200, 500], onChange: (v) => { this.scale = v; this.show(this.k); } });
    this.maxCombo = new Combo(slot("max"), { label: "Max", width: "132px", allowAuto: true, value: null, presets: this.maxPresets(),
      onChange: (v) => { this.vmax = v; this.show(this.k); } });
    if (!this.compact) new Combo(slot("fps"), { label: "FPS", width: "110px", value: this.fps, presets: [2, 5, 12, 24, 30, 60], digits: 3,
      onChange: (v) => { this.fps = Math.max(0.5, v || 12); } });
    const curves = m.curves || [];
    this.traceOpts = [{ value: "peak", label: "peak of field" }, ...curves.map((c, i) => ({ value: i, label: c.name, sub: c.unit }))];
    this.trace = curves.length ? 0 : "peak";
    this.traceSel = new Select(slot("trace"), { label: "Trace", width: this.compact ? "190px" : "205px", value: this.trace, options: this.traceOpts,
      onChange: (v) => { this.trace = v; this.refreshTrace(); } });
    const tlHost = h(`<div></div>`); d.appendChild(tlHost);
    this.timeline = new Timeline(tlHost, (k) => this.seek(k));
    if (this.compact) this.timeline.wrap.style.height = "64px";
    this.refreshTrace();
  }
  maxPresets() {
    const f = this.field(), top = f.max || 1;
    const nice = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 250, 500, 1000, 1500, 2000, 5000];
    return nice.filter((v) => v <= top * 1.6 && v >= top / 60).slice(-8);
  }
  refreshTrace() {
    const fc = this.frameCurve();
    const curve = this.trace === "peak" ? fc : this.m.curves[this.trace];
    this.timeline.set({ times: this.m.times, steps: this.m.steps, unit: this.m.time_unit, stepUnit: this.m.step_unit, curve, frameCurve: fc });
    this.timeline.setFrame(this.k);
  }
  renderReadout() {
    const r = this.dock.querySelector(".readout");
    if (!r) return;
    const m = this.m, st = m.steps;
    const stepTxt = st ? `${fmt(st[this.k])}<span class="of"> / ${fmt(st[st.length - 1])}</span>` : "–";
    r.innerHTML = `<div><span class="lbl">Frame</span>${String(this.k).padStart(String(this.n - 1).length, "0")}<span class="of"> / ${this.n - 1}</span></div>
      <div><span class="lbl">${esc(m.time_unit.includes("hit") ? "Hit force" : m.time_unit === "iteration" ? "Iteration" : "Time")}</span>${fmt(m.times[this.k], 4)} <span class="of">${esc(m.time_unit.replace(" hit force", ""))}</span></div>
      ${this.compact ? "" : `<div><span class="lbl">${esc(m.step_unit || "Step")}</span>${stepTxt}</div>`}`;
  }
  renderTitle() {
    const m = this.m;
    const kind = { explicit: "Explicit dynamics", fea: "Static FEA", topo: "Topology optimization", part: "STEP part" }[m.kind] || m.kind;
    const stats = m.kind === "part"
      ? [["Faces", m.faces.length], ["Edges", fmt(m.n_segments)], ["Triangles", fmt(m.n_tris)]]
      : [["Nodes", fmt(m.n_nodes)], ["Facets", fmt(m.n_tris)], ["Frames", this.n], ...(m.steps ? [[m.step_unit === "solver cycle" ? "Cycles" : "Steps", fmt(m.steps[m.steps.length - 1])]] : [])];
    this.hud.tl.innerHTML = `<div class="glass hud-title"><div class="eyebrow">${esc(kind)}</div><div class="t">${esc(this.title || m.title)}</div>
      <div class="stats">${stats.map(([k, v]) => `<div><span>${k}</span>${v}</div>`).join("")}</div></div>`;
  }
  renderColorbar() {
    const f = this.field(), [lo, hi] = this.range();
    const ticks = Array.from({ length: 6 }, (_, i) => lo + (hi - lo) * i / 5);
    this.hud.tr.innerHTML = `<div class="glass colorbar"><div class="head">Field<b>${esc(f.name)}${f.unit !== "-" ? ` <span class="muted">[${esc(f.unit)}]</span>` : ""}</b></div>
      <div class="body"><div class="bar" style="background:${gradientCss(this.cmap)}"></div>
      <div class="ticks">${ticks.map((t, i) => `<span style="top:${100 - i * 20}%">${fmt(t, 3)}</span>`).join("")}</div></div>
      <div class="note"><span><i style="background:var(--live)"></i>above max</span>${f.name === "plastic strain" ? `<span><i style="background:#5c6169"></i>elastic (not yielded)</span>` : ""}</div></div>`;
  }

  // ----- STEP part -----
  async loadPart(dir, { title = null, mass_g = null } = {}) {
    this.loading("Tessellating STEP…");
    try {
      const m = await api(`/files/${dir}/manifest.json`);
      const [mesh, edges, fid] = await Promise.all([data(dir, "mesh"), data(dir, "edges"), data(dir, "faceid")]);
      this.m = m; this.dir = dir; this.title = title; this.massG = mass_g;
      const n = m.n_nodes;
      const pos = new Float32Array(mesh, 0, n * 3), tri = new Uint32Array(mesh, n * 12, m.n_tris * 3);
      this.faceOf = new Uint32Array(fid);
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(pos.slice(), 3));
      this.col = new Float32Array(n * 3).fill(1);
      g.setAttribute("color", new THREE.BufferAttribute(this.col, 3));
      g.setIndex(new THREE.BufferAttribute(new Uint32Array(tri), 1));
      g.computeVertexNormals();
      this.partTris = tri;
      // machined Hardox: bright, slightly cool, satin -- reads as steel under the room environment
      const steel = new THREE.MeshStandardMaterial({ color: 0xe4e8ee, metalness: 0.78, roughness: 0.26, envMapIntensity: 1.35, vertexColors: true, side: THREE.FrontSide });
      this.mesh = new THREE.Mesh(g, steel);
      const cap = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: 0xff5b1f, side: THREE.BackSide }));  // the cut face
      cap.visible = false; this.cap = cap;
      const eg = new THREE.BufferGeometry();
      eg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(edges), 3));
      this.wireObj = new THREE.LineSegments(eg, new THREE.LineBasicMaterial({ color: 0x0b0d10, transparent: true, opacity: 0.7 }));
      this.wireObj.visible = this.wire = true;
      const root = new THREE.Group(); root.add(this.mesh, cap, this.wireObj);
      const [[x0, y0, z0], [x1, y1, z1]] = m.bbox_mm;
      const size = Math.max(x1 - x0, y1 - y0);
      const grid = new THREE.GridHelper(size * 2.2, 22, 0x343b45, 0x1a1e24);
      grid.rotation.x = Math.PI / 2; grid.position.set((x0 + x1) / 2, (y0 + y1) / 2, z0 - size * 0.04);
      grid.material.transparent = true; grid.material.opacity = 0.5; grid.material.depthWrite = false;
      root.add(grid);
      this.stage.setRoot(root, this.mesh);
      this.stage.scene.environment = this.stage.env;
      this.section = null;
      this.stage.fit("iso");
      this.dock.style.display = "none";
      this.renderTitle();
      this.hud.tr.innerHTML = `<div class="glass hud-title" style="min-width:200px"><div class="eyebrow">Envelope</div>
        <div class="stats" style="flex-direction:column;gap:6px">
          <div><span>X · Y · Z</span>${[x1 - x0, y1 - y0, z1 - z0].map((v) => fmt(v, 4)).join(" × ")} mm</div>
          ${mass_g ? `<div><span>Mass (exact, from the B-rep profile)</span>${fmt(mass_g, 4)} g</div>` : ""}
          <div><span>Volume (display)</span>${fmt(m.volume_mm3 / 1000, 4)} cm³</div></div></div>`;
      this.hud.bl.innerHTML = "";
      this.syncTools();
      this.onLoad?.(this);
    } finally { this.loading(null); }
  }
  highlight(face) {
    if (!this.m || this.m.kind !== "part" || face === this.hl) return;
    this.hl = face;
    this.col.fill(1);
    if (face >= 0) {
      const t = this.partTris;
      for (let i = 0; i < this.faceOf.length; i++) if (this.faceOf[i] === face) for (let j = 0; j < 3; j++) {
        const v = t[3 * i + j]; this.col[3 * v] = 1.0; this.col[3 * v + 1] = 0.5; this.col[3 * v + 2] = 0.3;
      }
    }
    this.mesh.geometry.attributes.color.needsUpdate = true;
  }
  toggleWire() { this.wire = !this.wire; if (this.wireObj) { this.wireObj.visible = this.wire; if (this.wire && this.m?.kind !== "part") this.show(this.k); } }
  toggleSection() {
    if (!this.m || this.m.kind !== "part") return;
    if (this.section) {
      this.section = null; this.mesh.material.clippingPlanes = []; this.wireObj.material.clippingPlanes = []; this.cap.visible = false;
      this.hud.bl.innerHTML = ""; return;
    }
    const [[, y0], [, y1]] = this.m.bbox_mm;
    const plane = new THREE.Plane(new THREE.Vector3(0, -1, 0), (y0 + y1) / 2);
    this.section = plane;
    this.mesh.material.clippingPlanes = [plane]; this.wireObj.material.clippingPlanes = [plane];
    this.cap.material.clippingPlanes = [plane]; this.cap.visible = true;
    this.hud.bl.innerHTML = `<div class="glass" style="padding:8px 10px;display:flex;gap:10px;align-items:center"><span class="eyebrow">Section Y</span><span data-c="cut"></span></div>`;
    new Combo(this.hud.bl.querySelector("[data-c=cut]"), { unit: "mm", width: "130px", value: Math.round((y0 + y1) / 2),
      presets: Array.from({ length: 9 }, (_, i) => Math.round(y0 + (y1 - y0) * (i + 0.5) / 9)), onChange: (v) => { plane.constant = v; } });
  }
  onProbe(e) {
    if (!this.m || !this.mesh || e.buttons) { this.probe.style.display = "none"; return; }
    const hit = this.stage.pick(e, [this.mesh]);
    const r = this.hostEl.getBoundingClientRect();
    if (!hit) { this.probe.style.display = "none"; this.highlight(-1); return; }
    let html;
    if (this.m.kind === "part") {
      const f = this.faceOf[hit.faceIndex], fi = this.m.faces[f];
      this.highlight(f);
      html = `face <b>#${f}</b> · ${esc(fi.type)} · ${fmt(fi.area_mm2, 4)} mm²`;
    } else {
      const fc = hit.face, ids = [fc.a, fc.b, fc.c];
      const p = hit.point;
      let best = ids[0], bd = Infinity;
      for (const i of ids) { const d = (this.pos[3 * i] - p.x) ** 2 + (this.pos[3 * i + 1] - p.y) ** 2 + (this.pos[3 * i + 2] - p.z) ** 2; if (d < bd) { bd = d; best = i; } }
      const f = this.field();
      html = `${esc(f.name)} <b>${fmt(this.value(this.k, best), 4)}</b> ${f.unit !== "-" ? esc(f.unit) : ""} <span class="muted">· (${fmt(this.base[3 * best], 3)}, ${fmt(this.base[3 * best + 1], 3)}) mm</span>`;
    }
    this.probe.innerHTML = html;
    this.probe.style.display = "block";
    this.probe.style.left = `${e.clientX - r.left}px`; this.probe.style.top = `${e.clientY - r.top}px`;
  }
}

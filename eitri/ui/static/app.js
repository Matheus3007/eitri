import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (id) => document.getElementById(id);
const api = async (url, opts) => {
  const r = await fetch(url, opts);
  const j = await r.json();
  if (j && j.error) throw new Error(j.error);
  return j;
};
const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "–" :
  typeof v === "number" ? (Math.abs(v) >= 1000 ? v.toFixed(0) : Number(v.toPrecision(d)).toString()) : String(v));
const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]);

const state = { project: null, run: null, jobTimer: null, jobSel: null };

// ---------- tabs ----------
for (const t of document.querySelectorAll(".tab")) {
  t.onclick = () => selectTab(t.dataset.pane);
}
function selectTab(name) {
  for (const t of document.querySelectorAll(".tab")) t.classList.toggle("sel", t.dataset.pane === name);
  for (const p of document.querySelectorAll(".pane")) p.classList.toggle("sel", p.id === name);
  if (name === "viewer") viewer.resize();
  if (name === "jobs") refreshJobs();
}

// ---------- projects & runs ----------
async function loadProjects() {
  const ps = await api("/api/projects");
  $("projects").innerHTML = ps.map((p) => `<div class="item" data-name="${p.name}">${esc(p.title)}
      <small>${p.type || "?"} · ${p.n_runs} runs${p.reference ? " · vs " + esc(p.reference.split("/").slice(-2, -1)[0]) : ""}</small></div>`).join("");
  for (const el of $("projects").children) el.onclick = async () => {
    await openProject(el.dataset.name);
    if (state.runs?.length) await openRun(state.runs[0].path);
  };
}

async function openProject(name) {
  state.project = name;
  state.run = null;
  for (const el of $("projects").children) el.classList.toggle("sel", el.dataset.name === name);
  const p = await api(`/api/project?name=${encodeURIComponent(name)}`);
  $("specText").value = p.spec;
  $("crumb").textContent = name;
  $("runs").innerHTML = p.runs.map((r) => {
    const pill = r.ok === undefined || r.ok === null ? "" : `<span class="pill ${r.ok ? "ok" : "bad"}">${r.ok ? "ok" : "fail"}</span>`;
    const nums = r.energy_j ? `${fmt(r.energy_j)} J · ${fmt(r.mass_g)} g` :
      r.peak_kn !== undefined ? `peak ${fmt(r.peak_kn)} kN · ${fmt(r.impulse_ns)} N·s` : "";
    const sub = [r.kind, nums].filter(Boolean).join(" · ");
    return `<div class="item ${r.depth ? "child" : ""}" data-path="${r.path}">${esc(r.depth ? r.name : r.name.replace(/-[a-z]+$/, ""))}${pill}${sub ? `<small>${esc(sub)}</small>` : ""}</div>`;
  }).join("") || `<div class="item muted">no runs yet</div>`;
  for (const el of $("runs").children) if (el.dataset.path) el.onclick = () => openRun(el.dataset.path);
  state.runs = p.runs;
}

async function openRun(path) {
  for (const el of $("runs").children) el.classList.toggle("sel", el.dataset.path === path);
  $("overview").innerHTML = `<p class="muted">loading…</p>`;
  const r = await api(`/api/run?path=${encodeURIComponent(path)}`);
  state.run = r;
  setHash();
  $("crumb").textContent = `${state.project} / ${r.name}`;
  renderOverview(r);
  renderCompare(r);
  $("animSel").innerHTML = `<option value="">animation… (${r.anims.length})</option>` + r.anims.map((a, i) =>
    `<option value="${i}">${esc(a.label || a.source)}</option>`).join("");
}

function renderOverview(r) {
  const h = r.headline || {};
  const cards = [["energy", h.energy_j, "J"], ["mass", h.mass_g, "g"], ["MOI", h.moi_gm2, "g·m²"],
    ["imbalance", h.imbalance_gmm, "g·mm"], ["spin SF", h.sf_spin, ""], ["impact SF", h.sf_impact, ""],
    ["first yield", h.yield_force_kn, "kN"]];
  let html = r.headline ? `<div class="cards">${cards.map(([k, v, u]) => `<div class="card"><div class="k">${k}</div><div class="v">${fmt(v)} <small class="muted">${u}</small></div></div>`).join("")}</div>` : "";
  if (r.topo) {
    html += `<h3>Topology designs</h3><table><tr><th>MOI target</th><th>energy</th><th>mass</th><th>imbalance</th><th>verdict</th><th>worse than the reference</th></tr>${r.topo.map((d) =>
      `<tr><td class="num">×${fmt(d.moi_frac)}</td><td class="num">${fmt(d.energy_j)} J</td><td class="num">${fmt((d.mass_kg || 0) * 1e3)} g</td><td class="num">${fmt(d.imbalance_gmm)} g·mm</td>
       <td class="${d.ok ? "good" : "bad"}">${d.ok ? "pass" : "fail"}</td><td class="muted">${esc(worseSummary(d))}</td></tr>`).join("")}</table>`;
  }
  if (h.error) html += `<p class="bad">${esc(h.error)}</p>`;
  if (r.checks?.length) {
    html += `<h3>Checks ${h.ok ? '<span class="pill ok">all pass</span>' : '<span class="pill bad">' + (h.failed || []).length + " fail</span>"}</h3>
      <table><tr><th>check</th><th>value</th><th>limit</th><th></th><th>detail</th></tr>${r.checks.map((c) =>
        `<tr><td>${c.name}</td><td>${fmt(c.value, 4)}</td><td>${fmt(c.limit, 4)}</td><td class="${c.ok ? "good" : "bad"}">${c.ok ? "pass" : "FAIL"}</td><td class="muted">${esc(c.detail || "")}</td></tr>`).join("")}</table>`;
  }
  const imgs = r.images.filter((i) => !i.includes("compare_"));
  if (imgs.length) html += `<div class="grid">${imgs.map((img) => `<figure><figcaption class="muted">${esc(img.split("/").slice(-3).join("/"))}</figcaption><img class="shot" src="/files/${img}"></figure>`).join("")}</div>`;
  if (r.files.some((f) => f.endsWith("/history.jsonl"))) html += `<h3>Search history <button id="histPlay">▶ replay</button></h3>
    <canvas id="hist" width="900" height="380" style="width:100%;max-width:900px;background:#11161c;border:1px solid var(--line);border-radius:8px"></canvas>
    <div class="muted">each dot is one evaluated design: x mass, y energy; filled = passes every check; color = phase (global search, Newton repair, CMA-ES refine)</div>`;
  if (r.summary) {
    const S = r.summary;
    html += `<h3>Optimization summary</h3><div class="cards">
      <div class="card"><div class="k">designs evaluated</div><div class="v">${S.n_trials}</div></div>
      <div class="card"><div class="k">pass every check</div><div class="v ${S.n_feasible ? "good" : "bad"}">${S.n_feasible}</div></div>
      <div class="card"><div class="k">${S.n_feasible ? "winner" : "closest (none passed)"}</div><div class="v">#${S.best_trial} <small class="muted">${esc(S.best_phase || "")}</small></div></div></div>
      ${S.fine_mesh_checks?.length ? `<table><tr><th>fine-mesh check</th><th>objective</th><th>verdict</th><th>failed</th></tr>${S.fine_mesh_checks.map((c) =>
        `<tr><td>trial ${c.trial}</td><td class="num">${fmt(c.objective, 4)}</td><td class="${c.ok ? "good" : "bad"}">${c.ok ? "pass" : "fail"}</td><td class="muted">${esc((c.failed || []).join(", "))}</td></tr>`).join("")}</table>` : ""}
      ${S.impact_gate && Object.keys(S.impact_gate).length ? `<h3>Explicit-hit gate</h3><table><tr><th>trial</th><th>hit</th><th>verdict</th><th>worse than the reference</th></tr>${Object.entries(S.impact_gate).flatMap(([tr, g]) =>
        Object.entries(g.cases || {}).map(([c, v]) => `<tr><td>${tr}</td><td>${c}</td><td class="${v.worse?.length ? "bad" : "good"}">${v.error ? "error" : v.worse?.length ? "worse" : "ok"}</td><td class="muted">${esc((v.worse || []).join("; "))}</td></tr>`)).join("")}</table>` : ""}`;
  }
  $("overview").innerHTML = html;
  const hf = r.files.find((f) => f.endsWith("/history.jsonl"));
  if (hf) drawHistory(hf);
}

async function drawHistory(file) {
  const txt = await (await fetch(`/files/${file}`)).text();
  const recs = txt.split("\n").filter(Boolean).map((l) => JSON.parse(l)).filter((h) => h.mass_kg && h.energy_j);
  const cv = $("hist"), g = cv.getContext("2d");
  if (!recs.length) return;
  const xs = recs.map((h) => h.mass_kg * 1e3), ys = recs.map((h) => h.energy_j);
  const pad = 44, W = cv.width, H = cv.height;
  const T = state.run?.targets || {};
  const mb = T.mass_max_g, re = T.ref_energy_j;
  const x0 = Math.min(...xs, mb || Infinity) - 1, x1 = Math.max(...xs, mb || -Infinity) + 1;
  const y0 = Math.min(...ys, re || Infinity) - 2, y1 = Math.max(...ys, re || -Infinity) + 2;
  const X = (v) => pad + (v - x0) / (x1 - x0 || 1) * (W - 2 * pad), Y = (v) => H - pad - (v - y0) / (y1 - y0 || 1) * (H - 2 * pad);
  const colors = { global: "#6c8ebf", "repair": "#e8763a", "repair-fd": "#8a5a3a", refine: "#3fb27f", start: "#d9a441" };
  const draw = (upto) => {
    g.clearRect(0, 0, W, H);
    g.strokeStyle = "#2a333e"; g.fillStyle = "#8b97a6"; g.font = "12px system-ui";
    g.strokeRect(pad, pad, W - 2 * pad, H - 2 * pad);
    g.fillText(`${x0.toFixed(0)} g`, pad, H - pad + 16); g.fillText(`${x1.toFixed(0)} g`, W - pad - 40, H - pad + 16);
    g.fillText(`${y0.toFixed(0)} J`, 4, H - pad); g.fillText(`${y1.toFixed(0)} J`, 4, pad + 10);
    g.setLineDash([5, 4]); g.strokeStyle = "#8b97a6";
    if (mb) { g.beginPath(); g.moveTo(X(mb), pad); g.lineTo(X(mb), H - pad); g.stroke(); g.fillText(`budget ${mb.toFixed(0)} g`, X(mb) + 4, pad + 12); }
    if (re) { g.beginPath(); g.moveTo(pad, Y(re)); g.lineTo(W - pad, Y(re)); g.stroke(); g.fillText(`reference ${re.toFixed(0)} J`, W - pad - 120, Y(re) - 4); }
    g.setLineDash([]);
    let lx = pad;
    for (const [name, c] of Object.entries({ global: colors.global, "Newton repair": colors.repair, "repair step (FD)": colors["repair-fd"], "CMA-ES": colors.refine })) {
      g.fillStyle = c; g.beginPath(); g.arc(lx + 5, H - 12, 4, 0, 2 * Math.PI); g.fill(); g.fillStyle = "#8b97a6"; g.fillText(name, lx + 13, H - 8); lx += g.measureText(name).width + 34;
    }
    let best = null;
    for (let i = 0; i < upto; i++) {
      const h = recs[i], c = colors[h.phase] || "#6c8ebf";
      g.beginPath(); g.arc(X(xs[i]), Y(ys[i]), h.ok ? 4 : 3, 0, 2 * Math.PI);
      if (h.ok) { g.fillStyle = c; g.fill(); if (!best || ys[i] > ys[best]) best = i; } else { g.strokeStyle = c; g.stroke(); }
    }
    if (best !== null) { g.strokeStyle = "#fff"; g.beginPath(); g.arc(X(xs[best]), Y(ys[best]), 8, 0, 2 * Math.PI); g.stroke(); }
    g.fillStyle = "#d8dee6"; g.fillText(`${upto} / ${recs.length} designs`, W - pad - 120, pad - 10);
  };
  draw(recs.length);
  $("histPlay").onclick = () => {
    let k = 0;
    const step = () => { k = Math.min(recs.length, k + Math.max(1, Math.round(recs.length / 150))); draw(k); if (k < recs.length) requestAnimationFrame(step); };
    step();
  };
}

function worseSummary(d) {
  // "case: item" entries -> "case (item, item)", one per load case
  const by = new Map();
  for (const w of d.worse_than_reference || d.failed || []) {
    const [g, item] = w.includes(": ") ? w.split(": ") : [w, ""];
    if (!by.has(g)) by.set(g, []);
    if (item) by.get(g).push(item.replace(/ \[kN\]$/, ""));
  }
  return [...by].map(([g, items]) => items.length ? `${g} (${items.join(", ")})` : g).join("; ");
}

function mdTables(md) {
  // markdown tables and headings -> HTML (enough for eitri's compare.md files)
  const out = [];
  let rows = [];
  const flush = () => {
    if (!rows.length) return;
    const [head, , ...body] = rows;
    const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
    out.push(`<table><tr>${cells(head).map((c) => `<th>${esc(c)}</th>`).join("")}</tr>${body.map((l) =>
      `<tr>${cells(l).map((c) => `<td class="${c.includes("WORSE") ? "worse" : ""}">${esc(c).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")}</td>`).join("")}</tr>`).join("")}</table>`);
    rows = [];
  };
  for (const l of md.split("\n")) {
    if (l.trim().startsWith("|")) { rows.push(l); continue; }
    flush();
    if (l.startsWith("#")) out.push(`<h3>${esc(l.replace(/^#+\s*/, ""))}</h3>`);
    else if (l.trim()) out.push(`<p>${esc(l)}</p>`);
  }
  flush();
  return out.join("");
}

function renderCompare(r) {
  let html = "";
  for (const img of r.images.filter((i) => i.includes("compare_"))) html += `<img class="shot" src="/files/${img}">`;
  for (const [path, md] of Object.entries(r.markdown)) html += `<div class="muted">${path.split("/").slice(-2).join("/")}</div>` + mdTables(md);
  $("compare").innerHTML = html || `<p class="muted">no comparison in this run (needs a [reference])</p>`;
}

// ---------- spec & jobs ----------
$("save").onclick = async () => {
  if (!state.project) return;
  const r = await fetch(`/api/spec?name=${encodeURIComponent(state.project)}`, { method: "PUT", body: $("specText").value }).then((x) => x.json());
  $("specMsg").innerHTML = r.ok ? `<span class="good">saved</span>` : `<span class="bad">${esc(r.error)}</span>`;
};
for (const b of document.querySelectorAll("button[data-cmd]")) {
  b.onclick = async () => {
    if (!state.project) return;
    const args = b.dataset.args ? b.dataset.args.split(" ") : [];
    const j = await api("/api/job", { method: "POST", body: JSON.stringify({ cmd: b.dataset.cmd, project: state.project, args }) });
    state.jobSel = j.id;
    selectTab("jobs");
  };
}
async function refreshJobs() {
  const js = await api("/api/jobs");
  $("jobList").innerHTML = `<table><tr><th>#</th><th>command</th><th>project</th><th>status</th></tr>${js.map((j) =>
    `<tr data-id="${j.id}" style="cursor:pointer"><td>${j.id}</td><td>${j.cmd} ${esc(j.args.join(" "))}</td><td>${j.project}</td>
     <td class="${j.running ? "" : j.returncode === 0 ? "good" : "bad"}">${j.running ? "running…" : "exit " + j.returncode}</td></tr>`).join("")}</table>`;
  for (const tr of $("jobList").querySelectorAll("tr[data-id]")) tr.onclick = () => { state.jobSel = +tr.dataset.id; refreshJobs(); };
  if (state.jobSel) {
    const l = await api(`/api/job?id=${state.jobSel}`);
    const el = $("jobLog");
    const atEnd = el.scrollTop + el.clientHeight >= el.scrollHeight - 20;
    el.textContent = l.log;
    if (atEnd) el.scrollTop = el.scrollHeight;
  }
  clearTimeout(state.jobTimer);
  if (document.querySelector('.tab.sel').dataset.pane === "jobs") state.jobTimer = setTimeout(refreshJobs, 1500);
}

// ---------- viewer ----------
const COLORMAP = [[0.0, [0.05, 0.03, 0.2]], [0.25, [0.35, 0.05, 0.5]], [0.5, [0.8, 0.2, 0.3]], [0.75, [0.98, 0.55, 0.1]], [1.0, [0.99, 0.95, 0.55]]];
const OVER = [0.0, 0.76, 1.0]; // above the max: cyan, like the static PNGs ("past yield")
function cmap(t, out, i) {
  if (t > 1.0001) { out[i] = OVER[0]; out[i + 1] = OVER[1]; out[i + 2] = OVER[2]; return; }
  t = Math.max(0, Math.min(1, t));
  for (let k = 1; k < COLORMAP.length; k++) {
    if (t <= COLORMAP[k][0]) {
      const [t0, c0] = COLORMAP[k - 1], [t1, c1] = COLORMAP[k];
      const f = (t - t0) / (t1 - t0);
      out[i] = c0[0] + f * (c1[0] - c0[0]); out[i + 1] = c0[1] + f * (c1[1] - c0[1]); out[i + 2] = c0[2] + f * (c1[2] - c0[2]);
      return;
    }
  }
}

const viewer = {
  init() {
    this.wrap = $("canvasWrap");
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setClearColor(0x0f1216);
    this.wrap.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(35, 1, 0.1, 10000);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const d = new THREE.DirectionalLight(0xffffff, 0.9); d.position.set(1, 2, 3); this.scene.add(d);
    const d2 = new THREE.DirectionalLight(0xffffff, 0.35); d2.position.set(-2, -1, -1); this.scene.add(d2);
    window.addEventListener("resize", () => this.resize());
    const loop = () => { requestAnimationFrame(loop); this.tick(); this.renderer.render(this.scene, this.camera); };
    loop();
  },
  resize() {
    const w = this.wrap.clientWidth, h = this.wrap.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  },
  async load(dir) {
    $("status").textContent = "loading…";
    const m = await api(`/files/${dir}/manifest.json`);
    // no ".bin" in URLs and a text content type: download managers must not grab these (see server.py)
    const data = async (part) => (await fetch(`/api/data?dir=${encodeURIComponent(dir)}&part=${part}`)).arrayBuffer();
    const mesh = await data("mesh");
    const frames = await data("frames");
    const vis = m.has_vis ? new Uint8Array(await data("vis")) : null;
    const n = m.n_nodes, nt = m.n_tris;
    this.m = m;
    this.base = new Float32Array(mesh, 0, n * 3);
    this.tris = new Uint32Array(mesh, n * 12, nt * 3);
    this.frames = new Float32Array(frames);
    this.vis = vis;
    this.stride = (m.has_disp ? n * 3 : 0) + m.fields.length * n;
    if (this.obj) { this.scene.remove(this.obj); this.obj.geometry.dispose(); }
    const g = new THREE.BufferGeometry();
    this.pos = new Float32Array(n * 3); this.pos.set(this.base);
    this.col = new Float32Array(n * 3);
    g.setAttribute("position", new THREE.BufferAttribute(this.pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(this.col, 3));
    g.setIndex(new THREE.BufferAttribute(new Uint32Array(this.tris), 1));
    const flat = m.kind !== "explicit";
    const mat = flat ? new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide })
      : new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide });
    this.obj = new THREE.Mesh(g, mat);
    this.scene.add(this.obj);
    g.computeBoundingSphere();
    const c = g.boundingSphere.center, r = g.boundingSphere.radius;
    this.controls.target.copy(c);
    const dist = r / Math.sin(THREE.MathUtils.degToRad(this.camera.fov / 2)) * 1.08;  // whole part in view
    this.camera.position.set(c.x, c.y - (flat ? 0 : dist * 0.55), c.z + dist * (flat ? 1 : 0.85));
    this.camera.up.set(0, flat ? 1 : 0, flat ? 0 : 1);
    this.camera.near = r / 100; this.camera.far = r * 100; this.camera.updateProjectionMatrix();
    this.controls.update();
    $("field").innerHTML = m.fields.map((f, i) => `<option value="${i}" ${f.name === m.default_field ? "selected" : ""}>${f.name}</option>`).join("");
    $("scale").value = m.disp_scale_default || 1;
    $("vmax").value = "";
    $("time").max = m.times.length - 1; $("time").value = 0;
    this.k = 0; this.playing = false; $("play").textContent = "▶";
    this.show(0);
    $("status").textContent = m.title;
  },
  show(k) {
    if (!this.m) return;
    const m = this.m, n = m.n_nodes;
    this.k = k;
    const off = k * this.stride;
    const sc = +$("scale").value || 0;
    if (m.has_disp) for (let i = 0; i < n * 3; i++) this.pos[i] = this.base[i] + sc * this.frames[off + i];
    const fi = +$("field").value || 0, f = m.fields[fi];
    const foff = off + (m.has_disp ? n * 3 : 0) + fi * n;
    const lo = f.name === "of yield" ? 0 : Math.max(0, f.min);
    let hi = parseFloat($("vmax").value);
    if (!(hi > lo)) hi = f.name === "of yield" ? 1.0 : (f.name === "plastic strain" ? Math.min(f.max, 0.15) : f.max);
    // plastic strain: material that hasn't yielded is plain steel grey, so only the yielded zones carry color
    const elastic = f.name === "plastic strain" ? 0.002 : -Infinity;
    for (let i = 0; i < n; i++) {
      const v = this.frames[foff + i];
      if (v < elastic) { this.col[i * 3] = 0.62; this.col[i * 3 + 1] = 0.65; this.col[i * 3 + 2] = 0.69; continue; }
      cmap((v - lo) / (hi - lo || 1), this.col, i * 3);
    }
    if (this.vis) {
      const v = this.vis.subarray(k * m.n_tris, (k + 1) * m.n_tris);
      const idx = [];
      for (let t = 0; t < m.n_tris; t++) if (v[t]) idx.push(this.tris[3 * t], this.tris[3 * t + 1], this.tris[3 * t + 2]);
      this.obj.geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(idx), 1));
    }
    this.obj.geometry.attributes.position.needsUpdate = true;
    this.obj.geometry.attributes.color.needsUpdate = true;
    if (m.kind === "explicit") this.obj.geometry.computeVertexNormals();
    $("tlabel").textContent = `${fmt(m.times[k], 4)} ${m.time_unit}`;
    const grad = COLORMAP.map(([t, c]) => `rgb(${c.map((x) => Math.round(x * 255)).join(",")}) ${t * 100}%`).join(",");
    $("legend").style.display = "block";
    $("legend").innerHTML = `<div>${f.name} ${f.unit !== "-" ? "[" + f.unit + "]" : ""}</div><div class="bar" style="background:linear-gradient(90deg,${grad})"></div>
      <div class="ends"><span>${fmt(lo)}</span><span>${fmt(hi)}</span></div><div class="muted">cyan: above max${f.name === "plastic strain" ? " · grey: not yielded" : ""}</div>`;
  },
  tick() {
    if (!this.playing || !this.m) return;
    const now = performance.now();
    if (now - (this.last || 0) < 1000 / (+$("fps").value || 12)) return;
    this.last = now;
    const k = (this.k + 1) % this.m.times.length;
    $("time").value = k;
    this.show(k);
  },
};
viewer.init();
$("time").oninput = () => viewer.show(+$("time").value);
$("field").onchange = $("scale").onchange = $("vmax").onchange = () => viewer.show(viewer.k);
$("play").onclick = () => { viewer.playing = !viewer.playing; $("play").textContent = viewer.playing ? "❚❚" : "▶"; };
$("animSel").onchange = async () => {
  const a = state.run?.anims[+$("animSel").value];
  if (!a) return;
  try {
    let dir = a.anim;
    if (a.source !== "ready") {
      $("status").textContent = "converting (first time only)…";
      dir = (await api(`/api/anim?path=${encodeURIComponent(state.run.path)}&source=${encodeURIComponent(a.source)}`, { method: "POST" })).anim;
    }
    await viewer.load(dir);
  } catch (e) { console.error("anim load failed", e.stack); $("status").textContent = "error: " + e.message; }
};

// ---------- deep links: #project=..&run=..&tab=..&anim=.. ----------
function setHash(extra = {}) {
  const h = new URLSearchParams(location.hash.slice(1));
  if (state.project) h.set("project", state.project);
  if (state.run) h.set("run", state.run.path);
  for (const [k, v] of Object.entries(extra)) h.set(k, v);
  history.replaceState(null, "", "#" + h.toString());
}
async function fromHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  if (h.get("project")) await openProject(h.get("project"));
  if (h.get("run")) await openRun(h.get("run"));
  else if (state.runs?.length) await openRun(state.runs[0].path);  // the latest run
  if (h.get("tab")) selectTab(h.get("tab"));
  if (h.get("anim") !== null && h.get("anim") !== "") {
    $("animSel").value = h.get("anim");
    await $("animSel").onchange();
    if (h.get("frame")) { $("time").value = h.get("frame"); viewer.show(+h.get("frame")); }
  }
}
for (const t of document.querySelectorAll(".tab")) t.addEventListener("click", () => setHash({ tab: t.dataset.pane }));

loadProjects().then(fromHash);

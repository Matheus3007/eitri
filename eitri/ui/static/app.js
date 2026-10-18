// Eitri UI shell: project/run navigation, tabs, live updates (hot reload + new runs), and the Simulation, Part,
// Compare, Spec and Jobs panes. Everything restorable lives in the URL hash.
import { $, h, esc, fmt, api, icon, Select, toast, splitLabel } from "./ui.js";
import { SimView } from "./stage.js";
import * as overview from "./overview.js";
import { METRICS, HIT_METRICS, kindLabel, when, mdTables } from "./overview.js";

const TABS = [["overview", "Overview", "overview"], ["compare", "Compare", "compare"], ["simulation", "Simulation", "sim"],
  ["part", "Part", "part"], ["spec", "Spec", "spec"], ["jobs", "Jobs", "jobs"]];
const state = { project: null, projects: [], runs: [], run: null, tab: "overview", sim: 0, step: 0, runCache: new Map() };

// ---------- hash ----------
function readHash() { return Object.fromEntries(new URLSearchParams(location.hash.slice(1))); }
function writeHash(extra = {}) {
  const p = new URLSearchParams();
  if (state.project) p.set("project", state.project);
  if (state.run) p.set("run", state.run.path);
  p.set("tab", state.tab);
  for (const [k, v] of Object.entries(extra)) if (v !== null && v !== undefined) p.set(k, v);
  history.replaceState(null, "", "#" + p.toString());
}

// ---------- tabs ----------
$("tabs").innerHTML = TABS.map(([id, label, ic]) => `<div class="tab" data-tab="${id}">${icon(ic)}<span>${label}</span></div>`).join("");
for (const t of $("tabs").children) t.onclick = () => selectTab(t.dataset.tab);
function moveInk() {
  const t = $("tabs").querySelector(".tab.sel"), ink = $("tabInk");
  if (!t) return;
  const r = t.getBoundingClientRect(), pr = $("tabInk").parentElement.getBoundingClientRect();
  ink.style.left = `${r.left - pr.left + 10}px`; ink.style.width = `${r.width - 20}px`;
}
window.addEventListener("resize", moveInk);
document.fonts?.ready.then(moveInk);
function selectTab(tab) {
  const order = TABS.map(([t]) => t), from = order.indexOf(state.tab), to = order.indexOf(tab);
  const changed = state.tab !== tab || !document.querySelector(".pane.sel");
  state.tab = tab;
  for (const t of $("tabs").children) t.classList.toggle("sel", t.dataset.tab === tab);
  for (const p of document.querySelectorAll(".pane")) p.classList.toggle("sel", p.id === tab);
  moveInk();
  if (changed && from >= 0) {  // slide in from the side the tab lies on
    const dir = to > from ? 1 : -1;
    $(tab).animate([{ opacity: 0, transform: `translateX(${dir * 36}px)` }, { opacity: 1, transform: "none" }],
      { duration: 380, easing: "cubic-bezier(.2,.8,.2,1)" });
  }
  writeHash();
  if (tab === "simulation") showSimulation();
  if (tab === "part") showPart();
  if (tab === "compare") showCompare();
  if (tab === "jobs") refreshJobs();
  if (tab === "spec") showSpec();
}

// ---------- projects & runs ----------
const projectSel = new Select($("projectDD"), { label: "Project", width: "250px", search: true, minMenu: 300,
  onChange: (v) => openProject(v, { openLatest: true }) });

async function loadProjects() {
  state.projects = await api("/api/projects");
  projectSel.setOptions(state.projects.map((p) => ({ value: p.name, label: p.title, sub: `${p.type || "?"} · ${p.n_runs}` })), state.project);
}

const RUN_ICON = { eval: "eval", optimize: "optimize", refine: "optimize", topo: "topo", impact: "impact", faces: "image", "explicit hit": "impact" };
function runLabel(r) {
  if (r.depth) return { name: r.name, icon: r.kind === "explicit hit" || /explicit|steel|tpu/.test(r.name) ? "impact" : "eval" };
  return { name: kindLabel(r.kind), icon: RUN_ICON[r.kind] || "folder" };
}
function runSub(r) {
  const bits = [];
  if (r.energy_j) bits.push(`${fmt(r.energy_j)} J`, `${fmt(r.mass_g)} g`);
  if (r.peak_kn !== undefined && r.peak_kn !== null) bits.push(`${fmt(r.peak_kn)} kN`, `${fmt(r.impulse_ns)} N·s`);
  return bits.filter(Boolean).join(" · ");
}

async function openProject(name, { openLatest = false, keepRun = false } = {}) {
  const changed = name !== state.project;
  state.project = name;
  projectSel.set(name, false);
  const p = await api(`/api/project?name=${encodeURIComponent(name)}`);
  state.runs = p.runs; state.spec = p.spec;
  const meta = state.projects.find((x) => x.name === name) || {};
  $("sideHead").innerHTML = `<div class="eyebrow">Project</div><div class="title">${esc(meta.title || name)}</div>
    <div class="meta"><span>type <b>${esc(meta.type || "?")}</b></span>${meta.reference ? `<span>ref <b>${esc(meta.reference.split("/").slice(-2, -1)[0])}</b></span>` : ""}
    <span><b>${p.runs.filter((r) => !r.depth).length}</b> runs</span></div>`;
  renderRuns();
  if (changed) { state.runCache.clear(); disposeCompare(); }
  if (keepRun && state.run) return;
  if (openLatest || changed) {
    if (p.runs.length) await openRun(p.runs[0].path);
    else { state.run = null; $("overview").innerHTML = `<div class="empty">${icon("folder")}<h3>No runs yet</h3>Run an evaluation from the Run menu.</div>`; refreshActive(); }
  }
}

function renderRuns() {
  // a run log: day headers, a spine with one node per run (filled = verdict), children branching off it
  let html = "", day = null;
  const rows = state.runs.map((r) => ({ r, day: r.depth ? null : when(r.name).split(" · ")[0] }));
  let curDay = null;
  for (const x of rows) { if (x.day !== null) curDay = x.day; x.d = curDay; }
  rows.forEach((x, i) => {
    const r = x.r;
    if (!r.depth && x.d && x.d !== day) { day = x.d; html += `<div class="log-day"><span class="eyebrow">${esc(day)}</span><span class="rule"></span></div>`; }
    const first = i === 0 || rows[i - 1].d !== x.d, last = i === rows.length - 1 || rows[i + 1].d !== x.d;
    const L = runLabel(r), v = r.ok === true ? "ok" : r.ok === false ? "bad" : "";
    html += `<div class="run ${r.depth ? "child" : ""} ${first ? "first" : ""} ${last ? "last" : ""} ${state.run?.path === r.path ? "sel" : ""}" data-path="${esc(r.path)}">
      <span class="time">${r.depth ? "" : esc(when(r.name).split(" · ")[1] || "")}</span><span class="node ${v} ${r.depth ? "" : "fill"}"></span>
      <span class="what"><div class="nm">${esc(L.name)}</div><div class="st">${esc(runSub(r))}</div></span>
      <span class="mark ${v}">${v === "ok" ? "PASS" : v === "bad" ? "FAIL" : ""}</span></div>`;
  });
  $("runs").innerHTML = html || `<div class="empty" style="padding:30px 10px">no runs yet</div>`;
  for (const el of $("runs").querySelectorAll(".run")) el.onclick = () => openRun(el.dataset.path);
}

async function getRun(path, fresh = false) {
  if (!fresh && state.runCache.has(path)) return state.runCache.get(path);
  const r = await api(`/api/run?path=${encodeURIComponent(path)}`);
  r._entry = state.runs.find((x) => x.path === path);
  const i = state.runs.findIndex((x) => x.path === path);
  r._children = [];
  if (i >= 0 && !state.runs[i].depth) for (const x of state.runs.slice(i + 1)) { if (!x.depth) break; r._children.push(x); }
  state.runCache.set(path, r);
  return r;
}

async function openRun(path, { fresh = false } = {}) {
  for (const el of $("runs").querySelectorAll(".run")) el.classList.toggle("sel", el.dataset.path === path);
  const r = await getRun(path, fresh);
  const runChanged = state.run?.path !== path;
  state.run = r;
  if (runChanged) { state.sim = 0; state.step = 0; simLoaded = null; partLoaded = null; }
  writeHash();
  renderOverview();
  refreshActive();
}

function refreshActive() {
  if (state.tab === "simulation") showSimulation();
  if (state.tab === "part") showPart();
  if (state.tab === "compare") showCompare();
  if (state.tab === "spec") showSpec();
}

// ---------- overview ----------
let thumb = null;
const ctx = { openSim: (i) => { state.sim = i; simLoaded = null; selectTab("simulation"); }, openRun: (p) => openRun(p), selectTab };
let shownRun = null;
function renderOverview() {
  const r = state.run;
  const el = $("overview");
  const fresh = shownRun !== r.path;
  shownRun = r.path;
  const top = fresh ? 0 : el.scrollTop;
  thumb?.dispose(); thumb = null;
  el.innerHTML = overview.render(r, ctx);
  el.classList.toggle("enter", fresh);
  overview.wire(el, r, ctx);
  el.scrollTop = top;
  const host = el.querySelector("#thumbHost");
  if (host && r.steps?.length) {
    const t = thumb = new SimView(host, { compact: true });
    t.hud.rc.style.display = "none";
    t.stage.autoRotate = true;
    t.stage.fitScale = 0.62;
    t.stage.controls.enableZoom = false;
    loadPartInto(t, r, 0).then(() => { t.hud.tl.innerHTML = ""; t.hud.tr.innerHTML = ""; t.stage.fit("iso"); }).catch((e) => console.error("part preview:", e.message));
  }
}

async function loadPartInto(view, r, i) {
  const s = r.steps[i];
  view.loading("Tessellating STEP…");
  const { part } = await api(`/api/part?step=${encodeURIComponent(s.path)}`, { method: "POST" });
  const own = s.path.split("/").slice(0, -1).join("/") === r.path;
  await view.loadPart(part, { title: s.label, mass_g: own && r.headline?.mass_g ? r.headline.mass_g : null });
}

async function loadSimInto(view, r, i, opts) {
  const a = r.anims[i];
  let dir = a.anim;
  if (a.source !== "ready") {
    view.loading("Converting results (first time only)…");
    dir = (await api(`/api/anim?path=${encodeURIComponent(r.path)}&source=${encodeURIComponent(a.source)}`, { method: "POST" })).anim;
    a.source = "ready"; a.anim = dir;
  }
  view.title = splitLabel(a.label)[0];
  await view.loadAnim(dir, opts);
}

// ---------- simulation ----------
let sim = null, simSel = null, simLoaded = null, pendingFrame = null;
function ensureSimPane() {
  if (sim) return;
  $("simulation").innerHTML = `<div class="cmp-bar"><span class="eyebrow">Simulation</span><span id="simPick"></span><div style="flex:1"></div>
    <span class="muted" style="font-size:11.5px">drag to orbit · scroll to zoom · hover to probe · space / ← → to step</span></div><div id="simHost" style="flex:1;display:flex;min-height:0"></div>`;
  simSel = new Select($("simPick"), { width: "460px", minMenu: 460, onChange: (v) => { state.sim = v; simLoaded = null; showSimulation(); } });
  sim = new SimView($("simHost"), { onFrame: (k) => writeHash({ sim: state.sim, frame: k }) });
}
async function showSimulation() {
  ensureSimPane();
  const r = state.run;
  if (!r) return sim.empty("No run selected");
  simSel.setOptions(r.anims.map((a, i) => ({ value: i, label: splitLabel(a.label || a.source)[0], sub: splitLabel(a.label)[1] })), state.sim);
  if (!r.anims.length) { simLoaded = null; return sim.empty("This run has no simulations to animate"); }
  const key = `${r.path}#${state.sim}`;
  if (simLoaded === key) return;
  simLoaded = key;
  try {
    await loadSimInto(sim, r, state.sim);
    if (pendingFrame !== null) { sim.seek(pendingFrame); pendingFrame = null; }
  } catch (e) { simLoaded = null; sim.empty(`Error: ${e.message}`); }
}

// ---------- part ----------
let partView = null, partSel = null, partLoaded = null;
function ensurePartPane() {
  if (partView) return;
  $("part").innerHTML = `<div class="cmp-bar"><span class="eyebrow">Part</span><span id="partPick"></span><div style="flex:1"></div>
    <span class="muted" style="font-size:11.5px">hover a face for its type and area</span><a class="btn" id="stepDl" download>${icon("download")} STEP</a></div><div id="partHost" style="flex:1;display:flex;min-height:0"></div>`;
  partSel = new Select($("partPick"), { width: "420px", onChange: (v) => { state.step = v; partLoaded = null; showPart(); } });
  partView = new SimView($("partHost"));
}
async function showPart() {
  ensurePartPane();
  const r = state.run;
  if (!r) return partView.empty("No run selected");
  partSel.setOptions((r.steps || []).map((s, i) => ({ value: i, label: s.label, sub: "STEP" })), state.step);
  if (!r.steps?.length) { $("stepDl").style.display = "none"; partLoaded = null; return partView.empty("This run has no STEP part"); }
  $("stepDl").style.display = ""; $("stepDl").href = `/files/${r.steps[state.step].path}`;
  const key = `${r.path}#${state.step}`;
  if (partLoaded === key) return;
  partLoaded = key;
  try { await loadPartInto(partView, r, state.step); } catch (e) { partLoaded = null; partView.empty(`Error: ${e.message}`); }
}

// ---------- compare ----------
let cmp = null;
function disposeCompare() { if (cmp) { cmp.A.view?.dispose(); cmp.B.view?.dispose(); $("compare").innerHTML = ""; cmp = null; } }
function viewsOf(r) {
  return [...(r.steps || []).map((s, i) => ({ value: `part:${i}`, label: `Part · ${s.label}`, group: "Part" })),
    ...(r.anims || []).map((a, i) => ({ value: `sim:${i}`, label: splitLabel(a.label || a.source)[0], sub: splitLabel(a.label)[1], group: "Simulation" })),
    ...(r.images || []).map((p) => ({ value: `img:${p}`, label: p.split("/").pop().replace(/\.png$/, "").replace(/_/g, " "), sub: p.split("/runs/").pop().split("/").slice(0, -1).join("/"), group: "Figure" }))];
}
async function showCompare() {
  const r = state.run;
  if (!r) { $("compare").innerHTML = `<div class="empty">No run selected</div>`; return; }
  if (cmp && cmp.for === r.path) return;
  disposeCompare();
  const pane = $("compare");
  pane.innerHTML = `<div class="cmp-bar">
      <div class="cmp-side a"><span class="letter">A</span><span data-c="runA"></span><span data-c="viewA"></span></div>
      <button class="btn icon" id="cmpSwap" title="Swap A and B">${icon("swap")}</button>
      <button class="btn on" id="cmpLink" title="Link camera and time">${icon("link")} Linked</button>
      <div class="cmp-side b" style="justify-content:flex-end"><span data-c="viewB"></span><span data-c="runB"></span><span class="letter">B</span></div></div>
    <div class="cmp-panes"><div class="cmp-pane" data-p="A"></div><div class="cmp-pane" data-p="B"></div></div>
    <div class="cmp-bottom"><div class="inner"><div class="card" id="cmpDiff"></div><div class="card" id="cmpRef"></div></div></div>`;
  const all = await api("/api/catalog");
  const runOpts = all.map((x) => ({ value: x.path, label: x.depth ? x.name : `${kindLabel(x.kind)} · ${when(x.path.split("/").pop()) || x.name}`, group: x.project, indent: !!x.depth,
    sub: x.energy_j ? `${fmt(x.energy_j)} J` : x.peak_kn ? `${fmt(x.peak_kn)} kN` : "" }));
  cmp = { for: r.path, linked: true, A: {}, B: {} };
  const bPath = r.path, aPath = r.reference?.path || (all.find((x) => x.path !== r.path && x.project === r.project)?.path ?? r.path);
  for (const side of ["A", "B"]) {
    const S = cmp[side];
    S.host = pane.querySelector(`[data-p=${side}]`);
    S.runSel = new Select(pane.querySelector(`[data-c=run${side}]`), { width: "300px", minMenu: 440, search: true, options: runOpts, value: side === "A" ? aPath : bPath,
      onChange: (v) => setCompareRun(side, v) });
    S.viewSel = new Select(pane.querySelector(`[data-c=view${side}]`), { width: "250px", minMenu: 360, onChange: (v) => setCompareView(side, v) });
  }
  $("cmpSwap").onclick = () => {
    const a = cmp.A.runSel.value, av = cmp.A.viewSel.value, b = cmp.B.runSel.value, bv = cmp.B.viewSel.value;
    cmp.A.runSel.set(b, false); cmp.B.runSel.set(a, false); setCompareRun("A", b, bv); setCompareRun("B", a, av);
  };
  $("cmpLink").onclick = (e) => { cmp.linked = !cmp.linked; e.currentTarget.classList.toggle("on", cmp.linked); e.currentTarget.innerHTML = `${icon("link")} ${cmp.linked ? "Linked" : "Unlinked"}`; };
  await setCompareRun("B", bPath);
  await setCompareRun("A", aPath);
}
function matchingView(v, r) {
  const opts = viewsOf(r);
  if (!v) return opts[0]?.value;
  const kind = v.split(":")[0];
  if (kind === "img") { const name = v.split("/").pop(); return opts.find((o) => o.value.endsWith("/" + name))?.value || opts[0]?.value; }
  if (kind === "sim") {
    const lab = splitLabel(cmp.B.run?.anims[+v.split(":")[1]]?.label)[0];
    return opts.find((o) => o.value.startsWith("sim:") && o.label === lab)?.value || opts.find((o) => o.value.startsWith("sim:"))?.value || opts[0]?.value;
  }
  return opts.find((o) => o.value.startsWith(kind + ":"))?.value || opts[0]?.value;
}
async function setCompareRun(side, path, view = null) {
  const S = cmp[side];
  S.run = await getRun(path);
  const opts = viewsOf(S.run);
  const pref = view || (side === "B" ? (opts.find((o) => o.value.startsWith("part:")) || opts[0])?.value : matchingView(cmp.B.viewSel.value, S.run));
  S.viewSel.setOptions(opts, pref);
  renderCompareTables();
  await setCompareView(side, S.viewSel.value);
}
async function setCompareView(side, v) {
  const S = cmp[side], other = cmp[side === "A" ? "B" : "A"];
  if (!v) { S.view?.dispose(); S.view = null; S.host.innerHTML = `<div class="empty">nothing to show for this run</div>`; return; }
  const kind = v.slice(0, v.indexOf(":")), arg = v.slice(v.indexOf(":") + 1);
  if (kind === "img") { S.view?.dispose(); S.view = null; S.host.innerHTML = `<div class="img-host"><img src="/files/${esc(arg)}"></div>`; return; }
  if (!S.view) {
    S.host.innerHTML = "";
    S.view = new SimView(S.host, { compact: true, onFrame: (k, view) => {
      if (cmp?.linked && other.view?.m && !S.syncing) { other.syncing = true; other.view.seekTime(view.m.times[k]); other.syncing = false; }
    } });
    S.view.stage.controls.addEventListener("change", () => {
      if (!cmp?.linked || S.syncing || !other.view) return;
      other.syncing = true;
      other.view.stage.camera.position.copy(S.view.stage.camera.position);
      other.view.stage.camera.up.copy(S.view.stage.camera.up);
      other.view.stage.controls.target.copy(S.view.stage.controls.target);
      other.view.stage.camera.quaternion.copy(S.view.stage.camera.quaternion);
      other.view.stage.camera.updateMatrixWorld();
      other.syncing = false;
    });
  }
  try {
    if (kind === "part") await loadPartInto(S.view, S.run, +arg);
    else await loadSimInto(S.view, S.run, +arg);
  } catch (e) { S.view.empty(`Error: ${e.message}`); }
}
function renderCompareTables() {
  if (!cmp?.A.run || !cmp?.B.run) return;
  const A = cmp.A.run, B = cmp.B.run;
  const vals = (r) => (r.headline && !r.headline.error ? r.headline : r.metrics || null);
  const a = vals(A), b = vals(B);
  const list = A.metrics || B.metrics ? HIT_METRICS : METRICS;
  const rows = a && b ? list.filter(([k]) => typeof a[k] === "number" || typeof b[k] === "number") : [];
  const nm = (r) => (r._entry?.depth ? r._entry.name : `${r.project} · ${kindLabel(r._entry?.kind || r.name.split("-").pop())}`);
  $("cmpDiff").innerHTML = `<div class="card-h"><h3>A vs B</h3><div class="right"><span class="muted" style="font-size:11.5px">Δ = B relative to A · green where B is better</span></div></div>
    <div class="card-b flush">${rows.length ? `<table class="data"><tr><th>Quantity</th><th class="num" style="color:var(--live)">A · ${esc(nm(A))}</th><th class="num" style="color:var(--accent-2)">B · ${esc(nm(B))}</th><th class="num">Δ</th></tr>
    ${rows.map(([k, label, unit, dir]) => {
      const x = a[k], y = b[k], d = typeof x === "number" && typeof y === "number" && x ? (y - x) / Math.abs(x) * 100 : null;
      const cls = d === null || Math.abs(d) < 0.05 || !dir ? "" : (d > 0) === (dir > 0) ? "better" : "worse";
      return `<tr><td>${label}</td><td class="num">${fmt(x, 4)} <span class="faint">${unit}</span></td><td class="num">${fmt(y, 4)} <span class="faint">${unit}</span></td><td class="num ${cls}">${d === null ? "–" : `${d >= 0 ? "+" : ""}${fmt(d, 3)} %`}</td></tr>`;
    }).join("")}</table>` : `<div class="empty" style="padding:24px">pick two runs with results to compare numbers</div>`}</div>`;
  const md = Object.entries(B.markdown || {}).find(([p]) => /compare/.test(p));
  $("cmpRef").innerHTML = `<div class="card-h"><h3>B against its reference</h3><div class="right"><span class="muted" style="font-size:11.5px">${md ? esc(md[0].split("/").pop()) : ""}</span></div></div>
    <div class="card-b flush">${md ? mdTables(md[1]) : `<div class="empty" style="padding:24px">no reference comparison in B</div>`}</div>`;
}

// ---------- spec ----------
function showSpec() {
  const el = $("spec");
  if (!el.dataset.ready) {
    el.innerHTML = `<div class="page"><div class="doc-h"><div><div class="eyebrow">Specification</div><h1 id="specTitle"></h1>
      <div class="when">units mm · deg · kg · rpm · m/s · ms — [design] values: a number (fixed) or [min, max] (free) · Ctrl+S saves</div></div>
      <div class="toolbar"><span class="msg" id="specMsg"></span><button class="btn" id="specRevert">Revert</button><button class="btn primary" id="specSave">Save</button></div></div>
      <div class="editor"><div class="gutter" id="specGutter"></div><div class="code"><pre id="specHl" aria-hidden="true"></pre><textarea id="specText" spellcheck="false"></textarea></div></div></div>`;
    el.dataset.ready = "1";
    const ta = $("specText"), gut = $("specGutter"), hl = $("specHl");
    const lines = () => {
      gut.textContent = Array.from({ length: ta.value.split("\n").length }, (_, i) => i + 1).join("\n");
      hl.innerHTML = tomlHighlight(ta.value) + "\n";
    };
    ta.oninput = () => { lines(); $("specMsg").innerHTML = `<span class="warn">unsaved changes</span>`; };
    ta.onscroll = () => { gut.scrollTop = ta.scrollTop; hl.scrollTop = ta.scrollTop; hl.scrollLeft = ta.scrollLeft; };
    ta.onkeydown = (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "s") { e.preventDefault(); $("specSave").click(); } };
    $("specSave").onclick = async () => {
      const r = await fetch(`/api/spec?name=${encodeURIComponent(state.project)}`, { method: "PUT", body: ta.value }).then((x) => x.json());
      $("specMsg").innerHTML = r.ok ? `<span class="ok">saved · validated</span>` : `<span class="bad">${esc(r.error)}</span>`;
      if (r.ok) state.spec = ta.value;
    };
    $("specRevert").onclick = () => { ta.value = state.spec; lines(); $("specMsg").textContent = ""; };
    el._lines = lines;
  }
  if ($("specText").dataset.project !== state.project) {
    $("specText").value = state.spec || ""; $("specText").dataset.project = state.project; el._lines(); $("specMsg").textContent = "";
  }
  $("specTitle").textContent = `${state.project} / spec.toml`;
}

function tomlHighlight(src) {
  const esc = (t) => String(t).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]);  // keep quotes for the string rule
  return src.split("\n").map((line) => {
    const m = /^(\s*)(\[[^\]]*\])(.*)$/.exec(line);
    let code = line, comment = "";
    const hash = (() => { let q = false; for (let i = 0; i < line.length; i++) { if (line[i] === '"') q = !q; if (line[i] === "#" && !q) return i; } return -1; })();
    if (hash >= 0) { code = line.slice(0, hash); comment = line.slice(hash); }
    if (m && hash !== 0) return `${esc(m[1])}<span class="t-sec">${esc(m[2])}</span>${comment ? `<span class="t-com">${esc(line.slice(m[1].length + m[2].length))}</span>` : esc(m[3])}`;
    const kv = /^(\s*)([A-Za-z0-9_.-]+)(\s*=\s*)(.*)$/.exec(code);
    let out = esc(code);
    if (kv) {
      const val = esc(kv[4]).replace(/("[^"]*")|(\b-?\d+(?:\.\d+)?(?:e-?\d+)?\b)|(\btrue\b|\bfalse\b)/g, (t, str, num, b) =>
        str ? `<span class="t-str">${str}</span>` : num ? `<span class="t-num">${num}</span>` : `<span class="t-bool">${b}</span>`);
      out = `${esc(kv[1])}<span class="t-key">${esc(kv[2])}</span>${esc(kv[3])}${val}`;
    }
    return out + (comment ? `<span class="t-com">${esc(comment)}</span>` : "");
  }).join("\n");
}

// ---------- jobs ----------
const COMMANDS = [["eval", [], "Evaluate", "eval"], ["eval", ["--fine"], "Evaluate · fine mesh", "eval"], ["optimize", [], "Optimize", "optimize"],
  ["topo", [], "Topology", "topo"], ["impact", ["--case", "all"], "Explicit hit · steel", "impact"],
  ["impact", ["--case", "bite", "--opponent", "tpu", "--no-ref"], "Explicit hit · TPU wall", "impact"]];
const runMenu = new Select($("runMenu"), { width: "96px", minMenu: 240, placeholder: "Run",
  options: COMMANDS.map(([, , label, ic], i) => ({ value: i, label, icon: ic })),
  onChange: async (i) => {
    runMenu.set(null, false);
    if (!state.project || i === null) return;
    const [cmd, args, label] = COMMANDS[i];
    const meta = state.projects.find((p) => p.name === state.project) || {};
    // a design run carries its own spec_fixed.toml: run on that design, else on the project spec
    const spec = cmd === "topo" ? null : jobTarget();
    if (!spec && meta.type === "topo" && cmd !== "topo") {
      toast(`A topology project has no single design: open one of its designs (an eval under a topology run), then run <b>${esc(label)}</b>`, 7000);
      return;
    }
    const j = await api("/api/job", { method: "POST", body: JSON.stringify({ cmd, project: state.project, args, spec }) });
    state.jobSel = j.id;
    toast(`Started <b>${esc(label)}</b> on ${esc(spec ? spec.split("/runs/").pop() : state.project + " / spec.toml")}`);
    selectTab("jobs");
  } });
runMenu.set(null, false);
runMenu.root.classList.add("dd-primary");

function jobTarget() {
  const r = state.run;
  if (!r) return null;
  return r.files.find((f) => f === `${r.path}/spec_fixed.toml`) || r.files.find((f) => f === `${r.path}/best/spec_fixed.toml`) || null;
}

let jobTimer = null;
async function refreshJobs() {
  clearTimeout(jobTimer);
  try {
    const js = await api("/api/jobs");
    const running = js.filter((j) => j.running).length;
    $("jobsBadge").classList.toggle("on", running > 0);
    $("jobsBadge").innerHTML = running ? `<span class="spinner"></span>${running} running` : "";
    $("jobsBadge").onclick = () => selectTab("jobs");
    if (state.tab === "jobs") await renderJobs(js);
  } finally { jobTimer = setTimeout(refreshJobs, state.tab === "jobs" ? 1500 : 6000); }
}
async function renderJobs(js) {
  const el = $("jobs");
  if (!el.dataset.ready) {
    el.innerHTML = `<div class="page"><div class="hero" style="border:0;padding-bottom:12px"><div><div class="eyebrow">Background</div><h1>Jobs</h1>
      <div class="when">the eitri CLI in detached processes — they keep running across server restarts</div></div></div>
      <div class="card"><div class="card-b flush" id="jobList"></div></div><div class="section-title"><h2>Log</h2><span class="muted" id="jobWhich"></span></div><pre class="console" id="jobLog">select a job</pre></div>`;
    el.dataset.ready = "1";
  }
  if (!state.jobSel && js.length) state.jobSel = js[0].id;
  $("jobList").innerHTML = js.length ? `<table class="data"><tr><th>#</th><th>Command</th><th>Project</th><th>Started</th><th>Status</th></tr>${js.map((j) =>
    `<tr class="click ${j.id === state.jobSel ? "sel" : ""}" data-id="${j.id}"><td class="mono">${j.id}</td><td class="mono">${esc(j.cmd)} ${esc(j.args.join(" "))}</td><td>${esc(j.project)}</td>
     <td class="mono muted">${new Date(j.started * 1000).toLocaleTimeString()}</td>
     <td>${j.running ? `<span class="tag live"><span class="spinner"></span> running</span>` : j.returncode === 0 ? `<span class="tag ok">done</span>` : `<span class="tag bad">exit ${j.returncode ?? "?"}</span>`}</td></tr>`).join("")}</table>`
    : `<div class="empty" style="padding:30px">no jobs yet — start one from the Run menu</div>`;
  for (const tr of $("jobList").querySelectorAll("tr[data-id]")) tr.onclick = () => { state.jobSel = +tr.dataset.id; refreshJobs(); };
  if (state.jobSel) {
    const l = await api(`/api/job?id=${state.jobSel}`);
    const el2 = $("jobLog"), atEnd = el2.scrollTop + el2.clientHeight >= el2.scrollHeight - 30;
    el2.textContent = l.log || "(no output yet)";
    const jj = js.find((x) => x.id === state.jobSel); $("jobWhich").textContent = jj ? `#${jj.id} · ${jj.cmd} ${jj.args.join(" ")}` : "–";
    if (atEnd) el2.scrollTop = el2.scrollHeight;
  }
}

// ---------- live: hot reload + new data ----------
let boot = null, dataTimer = null;
function connectEvents() {
  const es = new EventSource("/api/events");
  es.addEventListener("hello", (e) => {
    const b = JSON.parse(e.data).boot;
    $("live").classList.add("on");
    if (boot && b !== boot) { location.reload(); return; }  // the server restarted with new code
    boot = b;
  });
  es.addEventListener("static", (e) => {
    const files = JSON.parse(e.data).files;
    if (files.every((f) => f.endsWith(".css"))) $("css").href = `/static/styles.css?v=${Date.now()}`;
    else location.reload();
  });
  es.addEventListener("data", (e) => {
    const names = JSON.parse(e.data).projects;
    clearTimeout(dataTimer);
    dataTimer = setTimeout(() => onData(names), 600);
  });
  es.onerror = () => $("live").classList.remove("on");
}
async function onData(names) {
  const before = new Set(state.runs.map((r) => r.path));
  await loadProjects();
  if (!names.includes(state.project)) return;
  await openProject(state.project, { keepRun: true });
  const added = state.runs.filter((r) => !before.has(r.path));
  if (added.length) toast(`New in <b>${esc(state.project)}</b>: ${added.map((r) => esc(runLabel(r).name)).slice(0, 3).join(", ")}`);
  if (state.run) {
    const old = state.run;
    const r = await getRun(old.path, true);
    if (JSON.stringify([r.files, r.anims.length, r.headline]) !== JSON.stringify([old.files, old.anims.length, old.headline])) {
      state.run = r;
      renderOverview();
      if (state.tab === "simulation" && !sim?.m) showSimulation();
    }
  }
}

// ---------- boot ----------
(async () => {
  const H = readHash();
  await loadProjects();
  const name = H.project && state.projects.some((p) => p.name === H.project) ? H.project : state.projects[0]?.name;
  if (name) {
    await openProject(name);
    if (H.run && state.runs.some((r) => r.path === H.run) && H.run !== state.run?.path) await openRun(H.run);
    if (H.sim) state.sim = +H.sim;
    if (H.frame) pendingFrame = +H.frame;
  }
  selectTab(TABS.some(([t]) => t === H.tab) ? H.tab : "overview");
  connectEvents();
  refreshJobs();
})();

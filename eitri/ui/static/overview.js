// Overview: verdict and key numbers first (with the difference to the reference), then the evidence --
// grouped checks with their margins, the part and its simulations -- then what this kind of run adds
// (optimization search, topology designs, explicit hits), then reports and files.
import { esc, fmt, icon, h, lightbox, splitLabel } from "./ui.js";

// direction: +1 more is better, -1 less is better, 0 neutral
export const METRICS = [
  ["energy_j", "Stored energy", "J", 1, true], ["mass_g", "Mass", "g", -1, true], ["moi_gm2", "MOI", "g·m²", 1],
  ["imbalance_gmm", "Imbalance", "g·mm", -1], ["sf_spin", "Spin SF", "", 1], ["sf_impact", "Impact SF", "", 1],
  ["yield_force_kn", "First yield", "kN", 1], ["tip_speed_mps", "Tip speed", "m/s", 1], ["rpm", "Top speed", "rpm", 1],
  ["spinup_s", "Spin-up 95 %", "s", -1], ["bite_mm", "Bite", "mm", 1], ["effective_mass_kg", "Effective mass", "kg", 1],
  ["peak_force_kn", "Est. peak hit", "kN", 0], ["gyro_nm", "Gyro couple", "N·m", -1],
];
export const HIT_METRICS = [
  ["peak_force_kn", "Peak contact force", "kN", 0, true], ["impulse_ns", "Impulse", "N·s", 1, true],
  ["equivalent_half_sine_ms", "Equiv. half-sine", "ms", 1], ["max_plastic_strain_body", "Peak plastic strain (body)", "", -1],
  ["weapon_energy_lost_j", "Energy into the hit", "J", 1], ["plastic_work_j", "Plastic work", "J", -1],
];

const CHECK_NAMES = {
  strut_width: "Strut width", pocket_to_outline: "Pocket-to-outline wall", style_walls: "Constant walls", solid_valid: "Valid solid",
  sharp_corners: "Inside corners filleted", mass_max: "Mass budget", mass_min: "Minimum mass", imbalance: "Balance",
  safety_factor_spinning: "Spinning stress", impact_safety_factor: "Impact stress", ref_energy: "Energy ≥ reference",
  ref_mass: "Mass ≤ reference", ref_tooth_width: "Tooth width ≥ reference", ref_hit_bite: "Bite hit (in-plane)",
  ref_hit_tip3: "Tip hit, outer 3 mm", ref_hit_tip8: "Tip hit, outer 8 mm", ref_flat_hit: "Flat hit (out-of-plane)",
  bolt_shear: "Bolt shear", shaft_bending: "Shaft bending", style_arm_slot: "Arm slot", spinup: "Spin-up time",
};
const GROUPS = [
  ["Geometry & manufacturing", (n) => /^(strut|pocket|style|solid|sharp|step|min_|lock|tooth_)/.test(n)],
  ["Mass & balance", (n) => /^(mass|imbalance|ref_mass)/.test(n)],
  ["Strength", (n) => /(safety|bolt|shaft|stress)/.test(n)],
  ["Against the reference", (n) => n.startsWith("ref_")],
  ["Other", () => true],
];
const checkName = (n) => CHECK_NAMES[n] || n.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());

function delta(v, ref, dir) {
  if (typeof v !== "number" || typeof ref !== "number" || !ref) return `<div class="dbar"></div><div class="dtxt">no reference</div>`;
  const d = (v - ref) / Math.abs(ref) * 100;
  const cls = Math.abs(d) < 0.05 || dir === 0 ? "" : (d > 0) === (dir > 0) ? "better" : "worse";
  const w = Math.min(50, Math.abs(d) / 30 * 50);  // the half-bar spans +-30 %
  return `<div class="dbar"><span class="f ${cls}" style="left:${d >= 0 ? 50 : 50 - w}%;width:${Math.max(w, 0.6)}%"></span><span class="mid"></span></div>
    <div class="dtxt ${cls}">${Math.abs(d) < 0.05 ? "= reference" : `${d > 0 ? "+" : "−"}${fmt(Math.abs(d), 2)} % vs ref`}</div>`;
}

export function kpis(values, ref, list = METRICS, max = 8) {
  const rows = list.filter(([k]) => typeof values?.[k] === "number").slice(0, max);
  if (!rows.length) return "";
  return `<div class="kpis">${rows.map(([k, label, unit, dir]) => `<div class="kpi">
    <div class="k">${label}</div><div class="val"><span data-k="${k}" data-v="${values[k]}">${fmt(values[k], 4)}</span><small>${unit}</small></div>${delta(values[k], ref?.[k], dir)}</div>`).join("")}</div>`;
}

function marginBar(c) {
  if (typeof c.value !== "number" || typeof c.limit !== "number" || !c.limit) return `<div class="margin"><div class="nums"><span>${esc(fmt(c.value, 4))}</span></div></div>`;
  // "at most" checks by name; the rest pass at or above their limit
  const minType = !/(^mass_max|imbalance|^ref_mass|_max$|spinup|gyro|shaft)/.test(c.name);
  const r = Math.max(0, Math.min(2, c.value / c.limit));
  return `<div class="margin" title="${minType ? "at least" : "at most"} ${fmt(c.limit, 4)}">
    <div class="nums"><span>${fmt(c.value, 4)}</span><span class="lim">${minType ? "≥" : "≤"} ${fmt(c.limit, 4)}</span></div>
    <div class="bar"><i style="width:${r * 50}%"></i><b style="left:50%"></b></div></div>`;
}

export function checksCard(checks) {
  const groups = GROUPS.map(([g, f]) => [g, []]);
  for (const c of checks) groups[GROUPS.findIndex(([, f]) => f(c.name))][1].push(c);
  const nFail = checks.filter((c) => !c.ok).length;
  return `<div class="card"><div class="card-h"><h3>Checks</h3><div class="right">
      <span class="tag ${nFail ? "bad" : "ok"}">${nFail ? `${nFail} failing` : "all pass"}</span><span class="muted mono" style="font-size:11px">${checks.length}</span></div></div>
    <div class="checks">${groups.filter(([, cs]) => cs.length).map(([g, cs]) => {
      const f = cs.filter((c) => !c.ok).length;
      return `<div class="grp ${f ? "" : "closed"}">${icon("chev", "chev")}<span class="eyebrow">${g}</span>
        <span class="pips">${cs.map((c) => `<i class="${c.ok ? "" : "f"}"></i>`).join("")}</span>
        <span class="count">${f ? `<span class="bad">${f} fail</span> · ` : ""}${cs.length}</span></div>
        <div class="rows"><div>${cs.map((c) => `<div class="check ${c.ok ? "" : "fail"}"><span class="dot"></span><span class="name">${esc(checkName(c.name))}</span>
          ${marginBar(c)}<span class="detail" title="${esc(c.detail || "")}">${esc(c.detail || "")}</span></div>`).join("")}</div></div>`;
    }).join("")}</div></div>`;
}

const KIND = { eval: "Evaluation", optimize: "Optimization", refine: "Local refinement", topo: "Topology optimization",
  impact: "Explicit impact", faces: "Face map", compare: "Comparison", "explicit hit": "Explicit hit" };
export const kindLabel = (k) => KIND[k] || (k ? k[0].toUpperCase() + k.slice(1) : "Run");
export function when(name) {
  const m = /^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})/.exec(name);
  if (!m) return "";
  const d = new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]);
  return d.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" }) + " · " + d.toTimeString().slice(0, 5);
}

export function worseSummary(d) {
  const by = new Map();
  for (const w of d.worse_than_reference || d.failed || []) {
    const [g, item] = w.includes(": ") ? w.split(": ") : [w, ""];
    if (!by.has(g)) by.set(g, []);
    if (item) by.get(g).push(item.replace(/ \[kN\]$/, ""));
  }
  return [...by].map(([g, items]) => items.length ? `${g} (${items.join(", ")})` : g).join("; ");
}

export function mdTables(md) {
  const out = []; let rows = [];
  const flush = () => {
    if (!rows.length) return;
    const [head, , ...body] = rows;
    const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
    out.push(`<table class="data"><tr>${cells(head).map((c, i) => `<th class="${i ? "num" : ""}">${esc(c)}</th>`).join("")}</tr>${body.map((l) =>
      `<tr>${cells(l).map((c, i) => `<td class="${i ? "num" : ""} ${c.includes("WORSE") ? "worse" : ""}">${esc(c).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")}</td>`).join("")}</tr>`).join("")}</table>`);
    rows = [];
  };
  for (const l of md.split("\n")) {
    if (l.trim().startsWith("|")) { rows.push(l); continue; }
    flush();
    if (l.startsWith("#")) out.push(`<div class="eyebrow" style="padding:14px 14px 6px">${esc(l.replace(/^#+\s*/, ""))}</div>`);
    else if (l.trim()) out.push(`<p class="muted" style="padding:0 14px">${esc(l)}</p>`);
  }
  flush();
  return out.join("");
}

function simIcon(a) { const t = `${a.source} ${a.label}`.toLowerCase(); return /explicit|reference hit/.test(t) ? "impact" : /topolog/.test(t) ? "topo" : "sim"; }

export function simsStrip(r, one = false) {
  if (!r.anims?.length) return "";
  return `<div class="sims ${one ? "one" : ""}">${r.anims.map((a, i) => `<div class="sim" data-sim="${i}"><div class="k">${icon(simIcon(a))}</div>
    <div style="min-width:0"><div class="t">${esc(splitLabel(a.label || a.source)[0])}</div><div class="s">${esc(splitLabel(a.label)[1])}${a.source === "ready" ? "" : " · converts on first open"}</div></div>
    <span class="go">${icon("next")}</span></div>`).join("")}</div>`;
}

// figures grouped by what they show; line drawings are inked onto the dark sheet, color maps stay on paper
const FIG_CATS = [
  ["Comparison", /compare_/, "ink"], ["Geometry", /profile_|faces|outline/, "ink"], ["Stress", /stress_/, "paper"],
  ["Drive", /spinup_/, "ink"], ["Topology", /density|pocket|topo/, "paper"], ["Search", /pareto|history/, "ink"], ["Other", /./, "paper"],
];
function figures(r) {
  const groups = FIG_CATS.map(([name, , look]) => ({ name, look, items: [] }));
  for (const img of r.images) groups[FIG_CATS.findIndex(([, re]) => re.test(img.split("/").pop()))].items.push(img);
  return groups.filter((g) => g.items.length);
}
const figName = (p) => p.split("/").pop().replace(/\.png$/, "").replace(/_/g, " ");
const figWhere = (p) => p.split("/").slice(0, -1).join("/").split("/runs/").pop();
function plates(r) {
  const gs = figures(r);
  if (!gs.length) return "";
  return `<div class="plates" data-plates><div class="cats">${gs.map((g, i) => `<div class="cat ${i ? "" : "sel"}" data-cat="${i}"><span class="nm">${g.name}</span><span class="n">${g.items.length}</span></div>`).join("")}</div>
    <div class="view"><div class="sheet"><img alt=""></div><div class="caption"><span class="t"></span><span class="w"></span></div><div class="strip"></div></div></div>`;
}
function wirePlates(el, r) {
  const root = el.querySelector("[data-plates]");
  if (!root) return;
  const gs = figures(r), img = root.querySelector(".sheet img"), strip = root.querySelector(".strip");
  let cat = 0, fig = 0;
  const show = (c, f) => {
    cat = c; fig = f;
    const g = gs[c], src = g.items[f];
    for (const x of root.querySelectorAll(".cat")) x.classList.toggle("sel", +x.dataset.cat === c);
    img.classList.add("swap");
    setTimeout(() => {
      img.onload = () => img.classList.remove("swap");
      img.src = `/files/${src}`; img.className = `${g.look} swap`;
      if (img.complete) img.classList.remove("swap");
    }, 120);
    img.onclick = () => lightbox(`/files/${src}`);
    root.querySelector(".caption .t").textContent = figName(src);
    root.querySelector(".caption .w").textContent = figWhere(src);
    strip.style.display = g.items.length > 1 ? "" : "none";
    strip.innerHTML = g.items.map((p, i) => `<div class="th ${i === f ? "sel" : ""}" data-f="${i}" title="${esc(figName(p))}" style="background-image:url('/files/${esc(p)}')"></div>`).join("");
    for (const t of strip.querySelectorAll(".th")) t.onclick = () => show(cat, +t.dataset.f);
  };
  for (const c of root.querySelectorAll(".cat")) c.onclick = () => show(+c.dataset.cat, 0);
  show(0, 0);
}

function files(r) {
  // deliverables first (STEP, reports, specs); solver decks and logs behind a toggle
  const fs = r.files.filter((f) => !f.endsWith(".png"));
  if (!fs.length) return "";
  const key = (f) => /\.(step|md|toml)$|report\.json$|summary\.json$|metrics\.json$|impact\.json$|topo\.json$/.test(f);
  const main = fs.filter(key), rest = fs.filter((f) => !key(f));
  const row = (f) => `<a class="file" href="/files/${esc(f)}" target="_blank" title="${esc(f)}">${icon(f.endsWith(".step") ? "part" : "file")}<span>${esc(f.split("/runs/").pop())}</span></a>`;
  return `<div class="files">${main.map(row).join("")}</div>${rest.length ? `<details class="more"><summary>${rest.length} solver files (decks, logs, time histories)</summary>
    <div class="files">${rest.map(row).join("")}</div></details>` : ""}`;
}

function topoSection(r) {
  const designs = r.topo;
  const children = r._children || [];
  return `<div class="card"><div class="card-b flush"><table class="data"><tr><th>MOI target</th><th class="num">Energy</th><th class="num">Mass</th><th class="num">Imbalance</th><th>Verdict</th><th>Worse than the reference</th></tr>
    ${designs.map((d) => {
      const tag = `moi${String(d.moi_frac.toFixed(3)).replace(".", "p")}`;
      const ch = children.find((c) => c.path.includes(`/${tag}/`) && c.path.includes("/runs/")) || children.find((c) => c.path.includes(`/${tag}/`));
      return `<tr class="click" data-open="${esc(ch?.path || "")}"><td class="mono">× ${fmt(d.moi_frac, 4)}</td><td class="num">${fmt(d.energy_j, 4)} J</td>
        <td class="num">${fmt((d.mass_kg || 0) * 1e3, 4)} g</td><td class="num">${fmt(d.imbalance_gmm, 3)} g·mm</td>
        <td><span class="tag ${d.ok ? "ok" : "bad"}">${d.ok ? "pass" : "fail"}</span></td><td class="muted">${esc(worseSummary(d))}</td></tr>`;
    }).join("")}</table></div></div>`;
}

function optimizeSection(r) {
  const S = r.summary;
  let html = `<div class="kpis" style="margin-bottom:20px"><div class="kpi"><div class="k">Designs evaluated</div><div class="val">${fmt(S.n_trials)}</div></div>
      <div class="kpi"><div class="k">Pass every check</div><div class="val ${S.n_feasible ? "ok" : "bad"}">${fmt(S.n_feasible)}</div></div>
      <div class="kpi"><div class="k">${S.n_feasible ? "Winner" : "Closest (none passed)"}</div><div class="val">#${S.best_trial}<small>${esc(S.best_phase || "")}</small></div></div></div>`;
  if (r.files.some((f) => f.endsWith("/history.jsonl"))) html += `<div class="card"><div class="card-h"><h3>Design space</h3><div class="right">
      <span class="muted" style="font-size:11.5px">each dot one design · filled = passes · ring = winner</span><button class="btn" id="histPlay">${icon("play")} Replay</button></div></div>
      <div class="card-b"><canvas class="chart" id="hist" height="380"></canvas></div></div>`;
  if (S.fine_mesh_checks?.length) html += `<div class="card"><div class="card-h"><h3>Fine-mesh verification</h3></div><div class="card-b flush"><table class="data">
    <tr><th>Trial</th><th class="num">Objective</th><th>Verdict</th><th>Failed</th></tr>${S.fine_mesh_checks.map((c) => `<tr><td class="mono">#${c.trial}</td>
    <td class="num">${fmt(c.objective, 4)}</td><td><span class="tag ${c.ok ? "ok" : "bad"}">${c.ok ? "pass" : "fail"}</span></td><td class="muted">${esc((c.failed || []).join(", "))}</td></tr>`).join("")}</table></div></div>`;
  if (S.impact_gate && Object.keys(S.impact_gate).length) html += `<div class="card"><div class="card-h"><h3>Explicit-hit gate</h3></div><div class="card-b flush"><table class="data">
    <tr><th>Trial</th><th>Hit</th><th>Verdict</th><th>Worse than the reference</th></tr>${Object.entries(S.impact_gate).flatMap(([tr, g]) => Object.entries(g.cases || {}).map(([c, v]) =>
      `<tr><td class="mono">#${tr}</td><td>${esc(c)}</td><td><span class="tag ${v.worse?.length || v.error ? "bad" : "ok"}">${v.error ? "error" : v.worse?.length ? "worse" : "ok"}</span></td><td class="muted">${esc((v.worse || []).join("; "))}</td></tr>`)).join("")}</table></div></div>`;
  return html;
}

function impactSection(r) {
  const cases = r.impact?.cases || {};
  return Object.entries(cases).map(([c, by]) => {
    const d = by.design || {}, ref = by.reference;
    const rows = HIT_METRICS.filter(([k]) => typeof d[k] === "number");
    const regs = Object.keys(d.plastic_strain_by_region || {});
    return `<div class="eyebrow" style="margin:18px 0 8px">${esc(c)} hit · ${ref ? "design vs reference, same setup" : "design only"}</div>
      ${kpis(d, ref, HIT_METRICS, 6)}<div style="height:14px"></div>
      ${regs.length ? `<div class="card"><div class="card-h"><h3>Peak plastic strain by region</h3><div class="right"><span class="muted" style="font-size:11.5px">tip = contact denting, excluded from the gate</span></div></div>
        <div class="card-b flush"><table class="data"><tr><th>Region</th><th class="num">Design</th>${ref ? `<th class="num">Reference</th><th class="num">Δ</th>` : ""}</tr>
        ${regs.map((g) => { const a = d.plastic_strain_by_region[g], b = ref?.plastic_strain_by_region?.[g];
          const worse = ref && g !== "tip" && a > b * 1.02 + 1e-4;
          return `<tr><td>${esc(g)}</td><td class="num ${worse ? "worse" : ""}">${fmt(a, 4)}</td>${ref ? `<td class="num">${fmt(b, 4)}</td><td class="num ${worse ? "worse" : a < b ? "better" : ""}">${b ? (a >= b ? "+" : "") + fmt((a - b) / b * 100, 3) + " %" : "–"}</td>` : ""}</tr>`; }).join("")}
        </table></div></div>` : ""}`;
  }).join("");
}

function titleblock(r, verdict) {
  const kind = r._entry?.kind || r.name.split("-").pop();
  const name = r._entry?.depth ? r._entry.name : kindLabel(kind);
  const m = /(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})/.exec(r.path.split("/runs/").pop());
  const w = m ? `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}` : "";
  const mat = r.headline?.material || "–";
  return `<header class="titleblock"><div><div class="k">${esc(r.project)} · ${esc(kindLabel(kind))}</div><div class="name" title="${esc(name)}">${esc(name)}</div></div>
    <div><div class="k">Run</div><div class="v" title="${esc(r.path)}">${esc(r.path.split("/runs/").pop().split("/")[0].replace(/-[a-z]+$/, ""))}</div></div>
    <div><div class="k">Date</div><div class="v">${esc(w || "–")}</div></div>
    <div><div class="k">Material</div><div class="v">${esc(mat)}</div></div>
    <div><div class="k">Reference</div><div class="v">${esc(r.reference?.name || "–")}</div></div>
    ${verdict ? `<div class="verdict ${verdict.ok ? "ok" : "bad"}"><div><div class="k">Verdict</div><div class="big">${verdict.title}</div></div><div class="why">${verdict.detail}</div></div>` : `<div><div class="k">Verdict</div><div class="v">–</div></div>`}</header>`;
}

export function render(r, ctx) {
  const H = r.headline || {}, ref = r.reference?.headline;
  const secs = [];  // [title, note, html]
  let verdict = null;
  if (r.headline && !H.error) {
    const failed = H.failed || [];
    verdict = { ok: H.ok, title: H.ok ? "PASS" : "FAIL", detail: H.ok ? `every check${r.reference ? `<br>holds against ${esc(r.reference.name)}` : ""}` : esc(failed.map(checkName).join(", ")) };
    secs.push(["Performance", r.reference ? `bars: difference to ${esc(r.reference.name)}, ±30 % full scale` : "", kpis(H, ref)]);
    secs.push(["Integrity", "every check with its margin; the part as exported", `<div class="grid2"><div>${r.checks?.length ? checksCard(r.checks) : ""}</div><div>
      ${r.steps?.length ? `<div class="card"><div class="card-h"><h3>Part</h3><div class="right"><button class="btn ghost" data-go="part">${icon("part")} Inspect</button></div></div>
        <div class="thumb"><div class="stage-host" id="thumbHost"></div><div class="cap"><span class="eyebrow">${esc(r.steps[0].path.split("/").pop())}</span></div></div></div>` : ""}</div></div>`]);
  } else if (H.error) {
    verdict = { ok: false, title: "ERROR", detail: esc(H.error) };
  }
  if (r.topo) {
    const n = r.topo.filter((d) => d.ok).length;
    verdict = { ok: n > 0, title: `${n}/${r.topo.length}`, detail: n ? "designs hold everywhere the reference holds" : "no design holds everywhere the reference does" };
    secs.push(["Designs", "one per MOI target · open a row for its evaluation", topoSection(r)]);
  }
  if (r.summary) secs.push(["Search", "global TPE → Newton repair → CMA-ES refine", optimizeSection(r)]);
  if (r.impact) secs.push(["Explicit hits", "", impactSection(r)]);
  if (r.metrics) {
    const M = r.metrics;
    verdict = { ok: !(M.failed_elements > 0), title: M.failed_elements ? "TORN" : "INTACT", detail: `peak ${fmt(M.peak_force_kn, 3)} kN · ${fmt(M.impulse_ns, 3)} N·s · ${fmt(M.pulse_ms, 3)} ms pulse` };
    secs.push(["Hit", "", kpis(M, null, HIT_METRICS.concat([["pulse_ms", "Pulse", "ms", 0], ["added_mass_pct", "Added mass", "%", -1]]), 8)]);
    if (M.plastic_strain_by_region) secs.push(["Regions", "peak plastic strain and failed elements", `<div class="card"><div class="card-b flush"><table class="data">
      <tr><th>Region</th><th class="num">Plastic strain</th><th class="num">Failed elements</th></tr>${Object.entries(M.plastic_strain_by_region).map(([g, v]) =>
        `<tr><td>${esc(g)}</td><td class="num">${fmt(v, 4)}</td><td class="num">${fmt(M.failed_by_region?.[g] ?? 0)}</td></tr>`).join("")}</table></div></div>`]);
  }
  if (r.anims?.length) secs.push(["Simulations", `${r.anims.length} · open one to play it`, simsStrip(r)]);
  if (r.images?.length) secs.push(["Figures", `${r.images.length} · grouped by what they show`, plates(r)]);
  const fl = files(r);
  if (fl) secs.push(["Files", "", fl]);
  return `<div class="page">${titleblock(r, verdict)}${secs.map(([t, note, html], i) => `<section class="sec"><div class="sec-h"><span class="no">${String(i + 1).padStart(2, "0")}</span><h2>${t}</h2><span class="rule"></span>${note ? `<span class="note">${note}</span>` : ""}</div>${html}</section>`).join("")}</div>`;
}

const lastVals = new Map();
function countUp(el) {
  for (const sp of el.querySelectorAll("[data-v]")) {
    const k = sp.dataset.k, to = +sp.dataset.v, from = lastVals.get(k);
    lastVals.set(k, to);
    if (from === undefined || from === to || !Number.isFinite(from)) continue;
    const t0 = performance.now(), ms = 520;
    const step = (now) => {
      const u = Math.min(1, (now - t0) / ms), e = 1 - Math.pow(1 - u, 3);
      sp.textContent = fmt(from + (to - from) * e, 4);
      if (u < 1) requestAnimationFrame(step); else sp.textContent = fmt(to, 4);
    };
    requestAnimationFrame(step);
  }
}

export function wire(el, r, ctx) {
  countUp(el);
  wirePlates(el, r);
  for (const g of el.querySelectorAll(".checks .grp")) g.onclick = () => g.classList.toggle("closed");
  for (const s of el.querySelectorAll("[data-sim]")) s.onclick = () => ctx.openSim(+s.dataset.sim);
  for (const t of el.querySelectorAll("tr[data-open]")) if (t.dataset.open) t.onclick = () => ctx.openRun(t.dataset.open);
  for (const b of el.querySelectorAll("[data-go]")) b.onclick = () => ctx.selectTab(b.dataset.go);
  const hf = r.files.find((f) => f.endsWith("/history.jsonl"));
  if (hf && el.querySelector("#hist")) drawHistory(el.querySelector("#hist"), hf, r.targets || {}, el.querySelector("#histPlay"));
}

async function drawHistory(cv, file, T, playBtn) {
  const txt = await (await fetch(`/files/${file}`)).text();
  const recs = txt.split("\n").filter(Boolean).map((l) => JSON.parse(l)).filter((x) => x.mass_kg && x.energy_j);
  if (!recs.length) return;
  const dpr = Math.min(2, window.devicePixelRatio), W = cv.clientWidth, H = 380;
  cv.width = W * dpr; cv.height = H * dpr;
  const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
  const xs = recs.map((x) => x.mass_kg * 1e3), ys = recs.map((x) => x.energy_j);
  const mb = T.mass_max_g, re = T.ref_energy_j, P = { l: 56, r: 20, t: 18, b: 44 };
  const x0 = Math.min(...xs, mb || Infinity) - 1, x1 = Math.max(...xs, mb || -Infinity) + 1;
  const y0 = Math.min(...ys, re || Infinity) - 2, y1 = Math.max(...ys, re || -Infinity) + 2;
  const X = (v) => P.l + (v - x0) / (x1 - x0 || 1) * (W - P.l - P.r), Y = (v) => H - P.b - (v - y0) / (y1 - y0 || 1) * (H - P.t - P.b);
  const colors = { global: "#6f87a8", repair: "#ff5b1f", "repair-fd": "#8a4a2a", refine: "#35e0d0", start: "#ffb547" };
  const draw = (upto) => {
    g.clearRect(0, 0, W, H);
    g.font = "10.5px 'JetBrains Mono', monospace"; g.strokeStyle = "rgba(255,255,255,0.05)"; g.fillStyle = "#7a838f";
    for (let i = 0; i <= 5; i++) {
      const xv = x0 + (x1 - x0) * i / 5, yv = y0 + (y1 - y0) * i / 5;
      g.beginPath(); g.moveTo(X(xv), P.t); g.lineTo(X(xv), H - P.b); g.stroke();
      g.beginPath(); g.moveTo(P.l, Y(yv)); g.lineTo(W - P.r, Y(yv)); g.stroke();
      g.textAlign = "center"; g.fillText(`${xv.toFixed(0)}`, X(xv), H - P.b + 16);
      g.textAlign = "right"; g.fillText(`${yv.toFixed(0)}`, P.l - 8, Y(yv) + 3);
    }
    g.textAlign = "center"; g.fillText("mass [g]", (P.l + W - P.r) / 2, H - 8);
    g.save(); g.translate(14, (P.t + H - P.b) / 2); g.rotate(-Math.PI / 2); g.fillText("energy [J]", 0, 0); g.restore();
    if (mb && re) { g.fillStyle = "rgba(47,212,143,0.05)"; g.fillRect(P.l, P.t, X(mb) - P.l, Y(re) - P.t); }
    g.setLineDash([5, 4]); g.strokeStyle = "rgba(255,255,255,0.35)"; g.fillStyle = "#b6bdc7"; g.textAlign = "left";
    if (mb) { g.beginPath(); g.moveTo(X(mb), P.t); g.lineTo(X(mb), H - P.b); g.stroke(); g.fillText(`budget ${mb.toFixed(0)} g`, X(mb) + 6, P.t + 12); }
    if (re) { g.beginPath(); g.moveTo(P.l, Y(re)); g.lineTo(W - P.r, Y(re)); g.stroke(); g.fillText(`reference ${re.toFixed(0)} J`, P.l + 8, Y(re) - 6); }
    g.setLineDash([]);
    let best = null;
    for (let i = 0; i < upto; i++) {
      const x = recs[i], c = colors[x.phase] || colors.global;
      g.beginPath(); g.arc(X(xs[i]), Y(ys[i]), x.ok ? 3.6 : 2.6, 0, 2 * Math.PI);
      if (x.ok) { g.fillStyle = c; g.fill(); if (best === null || ys[i] > ys[best]) best = i; } else { g.strokeStyle = c; g.globalAlpha = 0.55; g.stroke(); g.globalAlpha = 1; }
    }
    if (best !== null) { g.strokeStyle = "#fff"; g.lineWidth = 1.5; g.beginPath(); g.arc(X(xs[best]), Y(ys[best]), 8, 0, 2 * Math.PI); g.stroke(); g.lineWidth = 1; }
    let lx = W - P.r;
    g.textAlign = "right";
    for (const [name, c] of Object.entries({ "CMA-ES": colors.refine, "repair step": colors["repair-fd"], "Newton repair": colors.repair, global: colors.global })) {
      g.fillStyle = "#b6bdc7"; g.fillText(name, lx, P.t + 4); const w = g.measureText(name).width;
      g.fillStyle = c; g.beginPath(); g.arc(lx - w - 9, P.t, 4, 0, 2 * Math.PI); g.fill(); lx -= w + 30;
    }
    g.fillStyle = "#eceff3"; g.textAlign = "left"; g.fillText(`${upto} / ${recs.length}`, P.l + 8, P.t + 4);
  };
  draw(recs.length);
  if (playBtn) playBtn.onclick = () => {
    let k = 0;
    const step = () => { k = Math.min(recs.length, k + Math.max(1, Math.round(recs.length / 180))); draw(k); if (k < recs.length) requestAnimationFrame(step); };
    step();
  };
}

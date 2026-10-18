// Shared helpers and the two dropdown components: Select (pick from a list, optional search, groups) and
// Combo (an editable value with presets: type any number or pick one).

export const $ = (id) => document.getElementById(id);
export const h = (html) => { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstElementChild; };
export const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

// animation labels are "title · where": split at the last separator
export function splitLabel(l) { const s = String(l || ""), i = s.lastIndexOf(" · "); return i < 0 ? [s, ""] : [s.slice(0, i), s.slice(i + 3)]; }

export async function api(url, opts) {
  const r = await fetch(url, opts);
  const j = await r.json();
  if (j && j.error) throw new Error(j.error);
  return j;
}

export function fmt(v, d = 3) {
  if (v === null || v === undefined || Number.isNaN(v) || v === "") return "–";
  if (typeof v !== "number") return String(v);
  if (!Number.isFinite(v)) return v > 0 ? "∞" : "–∞";
  const a = Math.abs(v);
  if (a >= 10000) return Math.round(v).toLocaleString("en-US").replace(/,/g, " ");
  if (a >= 1000) return v.toFixed(0);
  if (a === 0) return "0";
  if (a < 1e-3) return v.toExponential(1);
  return Number(v.toPrecision(d)).toString();
}

export const ICONS = {
  overview: '<rect x="3" y="3" width="7" height="9" rx="1"/><rect x="14" y="3" width="7" height="5" rx="1"/><rect x="14" y="12" width="7" height="9" rx="1"/><rect x="3" y="16" width="7" height="5" rx="1"/>',
  compare: '<rect x="3" y="4" width="8" height="16" rx="1"/><rect x="13" y="4" width="8" height="16" rx="1"/>',
  sim: '<path d="M3 12h3l3-8 4 16 3-8h5"/>',
  part: '<path d="M12 2 21 7v10l-9 5-9-5V7z"/><path d="M12 22V12M21 7l-9 5-9-5"/>',
  spec: '<path d="M14 3H6a1 1 0 0 0-1 1v16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V8z"/><path d="M14 3v5h5M9 13h6M9 17h6"/>',
  jobs: '<path d="m5 8 4 4-4 4M12 17h7"/>',
  play: '<path d="M7 4.5v15l12-7.5z" fill="currentColor" stroke="none"/>',
  pause: '<rect x="6" y="5" width="4" height="14" rx="1" fill="currentColor" stroke="none"/><rect x="14" y="5" width="4" height="14" rx="1" fill="currentColor" stroke="none"/>',
  prev: '<path d="M15 6 9 12l6 6"/>', next: '<path d="m9 6 6 6-6 6"/>',
  first: '<path d="M6 5v14M17 6l-6 6 6 6"/>', last: '<path d="M18 5v14M7 6l6 6-6 6"/>',
  loop: '<path d="M17 2l3 3-3 3"/><path d="M4 11V9a4 4 0 0 1 4-4h12M7 22l-3-3 3-3"/><path d="M20 13v2a4 4 0 0 1-4 4H4"/>',
  chev: '<path d="m6 9 6 6 6-6"/>', check: '<path d="M5 12.5 10 17 19 7"/>', x: '<path d="M6 6l12 12M18 6 6 18"/>',
  swap: '<path d="M7 4 3 8l4 4M3 8h14M17 20l4-4-4-4M21 16H7"/>', link: '<path d="M9 15 15 9M10 6l1-1a5 5 0 0 1 7 7l-1 1M14 18l-1 1a5 5 0 0 1-7-7l1-1"/>',
  download: '<path d="M12 4v11M7 10l5 5 5-5M5 20h14"/>', iso: '<path d="M12 2 21 7v10l-9 5-9-5V7z"/>',
  top: '<rect x="4" y="4" width="16" height="16" rx="2"/><circle cx="12" cy="12" r="2"/>', front: '<rect x="3" y="9" width="18" height="6" rx="1"/>',
  fit: '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>', section: '<path d="M3 12h18"/><path d="M6 6l12 12" opacity=".5"/>',
  edges: '<path d="M4 4h16v16H4z"/><path d="M4 12h16M12 4v16" opacity=".5"/>', run: '<path d="M7 4.5v15l12-7.5z"/>',
  bolt: '<path d="M13 2 4 14h7l-1 8 9-12h-7z"/>', folder: '<path d="M3 7a1 1 0 0 1 1-1h5l2 2h9a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1z"/>',
  eval: '<circle cx="12" cy="12" r="8"/><path d="M12 8v4l3 2"/>', optimize: '<path d="M3 17l5-5 4 4 8-9"/><path d="M15 7h5v5"/>',
  topo: '<path d="M4 20c4-1 5-6 8-8s7-2 8-8"/><circle cx="7" cy="8" r="2"/><circle cx="16" cy="16" r="2"/>',
  impact: '<path d="M3 21 12 12M14 4l6 6-8 3-1-1z"/>', file: '<path d="M14 3H6v18h12V7z"/><path d="M14 3v4h4"/>',
  image: '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="m21 17-5-5-9 8"/>',
  probe: '<circle cx="12" cy="12" r="3"/><path d="M12 2v4M12 18v4M2 12h4M18 12h4"/>', alert: '<path d="M12 3 2 20h20zM12 10v4M12 17v.5"/>',
  live: '<circle cx="12" cy="12" r="3"/><path d="M5.6 5.6a9 9 0 0 0 0 12.8M18.4 5.6a9 9 0 0 1 0 12.8"/>',
};
export const icon = (name, cls = "") => `<svg class="i ${cls}" viewBox="0 0 24 24">${ICONS[name] || ""}</svg>`;

let openMenu = null;
document.addEventListener("mousedown", (e) => {
  if (openMenu && !openMenu.root.contains(e.target) && !openMenu.menu.contains(e.target)) openMenu.close();
});
window.addEventListener("resize", () => openMenu?.close());
document.addEventListener("scroll", (e) => { if (openMenu && !openMenu.menu.contains(e.target)) openMenu.close(); }, true);

function placeMenu(menu, anchor, minWidth) {
  const r = anchor.getBoundingClientRect();
  menu.style.minWidth = `${Math.max(minWidth || 0, r.width)}px`;
  menu.classList.add("open");
  const mh = menu.offsetHeight, mw = menu.offsetWidth;
  const below = window.innerHeight - r.bottom;
  menu.style.top = `${below > mh + 8 || below > r.top ? r.bottom + 4 : r.top - mh - 4}px`;
  menu.style.left = `${Math.min(r.left, window.innerWidth - mw - 8)}px`;
}

// Select: options [{value, label, sub, group, indent}]; search for long lists
export class Select {
  constructor(host, { label = "", options = [], value = null, onChange = () => {}, search = false, width = null, placeholder = "–", minMenu = 0 } = {}) {
    this.onChange = onChange; this.search = search; this.placeholder = placeholder; this.minMenu = minMenu;
    this.root = h(`<div class="dd"><div class="dd-field">${label ? `<span class="dd-label">${esc(label)}</span>` : ""}
      <button class="dd-value" type="button"></button><button class="dd-caret" type="button" tabindex="-1">${icon("chev")}</button></div></div>`);
    if (width) this.root.style.width = width;
    this.menu = h(`<div class="dd-menu"></div>`);
    document.body.appendChild(this.menu);
    host.replaceChildren(this.root);
    this.root.querySelector(".dd-field").onclick = () => (this.menu.classList.contains("open") ? this.close() : this.open());
    this.setOptions(options, value);
  }
  setOptions(options, value = this.value) {
    this.options = options;
    this.set(options.some((o) => o.value === value) ? value : options[0]?.value ?? null, false);
  }
  set(value, fire = true) {
    this.value = value;
    const o = this.options.find((x) => x.value === value);
    this.root.querySelector(".dd-value").textContent = o ? o.label : this.placeholder;
    this.root.querySelector(".dd-value").title = o ? o.label : "";
    if (fire) this.onChange(value, o);
  }
  open() {
    openMenu?.close();
    const render = (q = "") => {
      const ql = q.toLowerCase();
      let html = "", group = null;
      for (const [i, o] of this.options.entries()) {
        if (ql && !`${o.label} ${o.sub || ""} ${o.group || ""}`.toLowerCase().includes(ql)) continue;
        if (o.group && o.group !== group) { group = o.group; html += `<div class="dd-group">${esc(group)}</div>`; }
        html += `<div class="dd-opt ${o.value === this.value ? "sel" : ""} ${o.indent ? "indent" : ""}" data-i="${i}">
          ${o.icon ? icon(o.icon) : ""}<span>${esc(o.label)}</span>${o.sub ? `<span class="sub">${esc(o.sub)}</span>` : ""}</div>`;
      }
      list.innerHTML = html || `<div class="dd-opt muted">nothing matches</div>`;
      for (const el of list.querySelectorAll(".dd-opt[data-i]")) el.onclick = () => { this.close(); this.set(this.options[+el.dataset.i].value); };
    };
    this.menu.innerHTML = this.search ? `<input class="dd-search" placeholder="Search…">` : "";
    const list = h(`<div></div>`);
    this.menu.appendChild(list);
    render();
    placeMenu(this.menu, this.root, this.minMenu);
    this.root.classList.add("open");
    if (this.search) { const s = this.menu.querySelector(".dd-search"); s.oninput = () => render(s.value); s.focus(); }
    this.menu.querySelector(".dd-opt.sel")?.scrollIntoView({ block: "nearest" });
    openMenu = this;
  }
  close() { this.menu.classList.remove("open"); this.root.classList.remove("open"); if (openMenu === this) openMenu = null; }
}

// Combo: an editable number with presets. value null = "auto" (when allowAuto).
export class Combo {
  constructor(host, { label = "", unit = "", presets = [], value = null, onChange = () => {}, allowAuto = false, width = "130px", digits = 4 } = {}) {
    Object.assign(this, { onChange, presets, allowAuto, digits, unit });
    this.root = h(`<div class="dd"><div class="dd-field">${label ? `<span class="dd-label">${esc(label)}</span>` : ""}
      <input class="dd-input" spellcheck="false">${unit ? `<span class="dd-unit">${esc(unit)}</span>` : ""}
      <button class="dd-caret" type="button" tabindex="-1">${icon("chev")}</button></div></div>`);
    this.root.style.width = width;
    this.input = this.root.querySelector("input");
    this.menu = h(`<div class="dd-menu"></div>`);
    document.body.appendChild(this.menu);
    host.replaceChildren(this.root);
    this.root.querySelector(".dd-caret").onclick = () => (this.menu.classList.contains("open") ? this.close() : this.open());
    this.input.onkeydown = (e) => {
      if (e.key === "Enter") { this.commit(); this.input.blur(); }
      if (e.key === "Escape") { this.show(); this.input.blur(); }
      if (e.key === "ArrowUp" || e.key === "ArrowDown") { e.preventDefault(); this.nudge(e.key === "ArrowUp" ? 1 : -1); }
    };
    this.input.onblur = () => this.commit();
    this.set(value, false);
  }
  show() { this.input.value = this.value === null ? "auto" : fmt(this.value, this.digits); }
  set(v, fire = true) { this.value = v; this.show(); if (fire) this.onChange(v); }
  setPresets(p) { this.presets = p; }
  commit() {
    const t = this.input.value.trim().toLowerCase();
    if (this.allowAuto && (t === "" || t === "auto")) return this.value === null ? this.show() : this.set(null);
    const v = parseFloat(t.replace(",", "."));
    if (Number.isFinite(v)) { if (v !== this.value) this.set(v); else this.show(); } else this.show();
  }
  nudge(dir) {
    const ps = this.presets.filter((p) => p !== null).sort((a, b) => a - b);
    const cur = this.value ?? ps[0] ?? 0;
    const nxt = dir > 0 ? ps.find((p) => p > cur + 1e-12) : [...ps].reverse().find((p) => p < cur - 1e-12);
    if (nxt !== undefined) this.set(nxt);
  }
  open() {
    openMenu?.close();
    const opts = [...(this.allowAuto ? [null] : []), ...this.presets];
    this.menu.innerHTML = opts.map((p, i) => `<div class="dd-opt ${p === this.value ? "sel" : ""}" data-i="${i}">
      <span class="mono">${p === null ? "auto" : fmt(p, this.digits)}</span>${p !== null && this.unit ? `<span class="sub">${esc(this.unit)}</span>` : ""}</div>`).join("");
    for (const el of this.menu.querySelectorAll(".dd-opt")) el.onclick = () => { this.close(); this.set(opts[+el.dataset.i]); };
    placeMenu(this.menu, this.root, 120);
    this.root.classList.add("open");
    openMenu = this;
  }
  close() { this.menu.classList.remove("open"); this.root.classList.remove("open"); if (openMenu === this) openMenu = null; }
}

export function toast(html, ms = 4200) {
  const t = h(`<div class="toast glass">${icon("live")}<div>${html}</div></div>`);
  $("toasts").appendChild(t);
  setTimeout(() => { t.classList.add("out"); setTimeout(() => t.remove(), 320); }, ms);
}

export function lightbox(src) {
  const lb = $("lightbox");
  lb.querySelector("img").src = src;
  lb.classList.add("open");
  lb.onclick = () => lb.classList.remove("open");
}

"""Design vs reference, the way the user reviews it: side-by-side silhouettes at the same scale, an outline
overlay, and a table of every compared quantity with the cases where the design is worse flagged.

Everything is drawn in the canonical frame (spin CCW seen from +Z, tooth tip at angle 0), so a CW-spinning
imported part and a generated hook line up.
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

from eitri.reference import CASES, REGION_CAP, REGIONS  # noqa: E402

REF_COLOR, DESIGN_COLOR = "#9aa0a6", "#c0392b"
CASE_LABELS = {
    "hit_bite": "in-plane hit on the bite (opponent at rest)",
    "hit_tip3": "in-plane hit on the outer 3 mm (tooth-to-tooth)",
    "hit_tip8": "in-plane hit on the outer 8 mm",
    "flat_hit": "3D hit on the flat of the tooth",
}


def canonical_wires(weapon, n: int = 900) -> list[np.ndarray]:
    """Outline + holes, sampled, mirrored to CCW spin and rotated so the tooth tip sits at angle 0."""
    face = weapon.profile()
    sy = -1.0 if getattr(weapon, "spin", "ccw") == "cw" else 1.0
    tx, ty = weapon.impact_face()[0]
    a = -math.atan2(sy * ty, tx)
    c, s = math.cos(a), math.sin(a)
    out = []
    for w in [face.outer_wire(), *face.inner_wires()]:
        xy = np.array([(p.X, sy * p.Y) for p in (w.position_at(t) for t in np.linspace(0, 1, n))])
        out.append(np.c_[c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]])
    return out


def _patch(wires, color, **kw) -> PathPatch:
    verts, codes = [], []
    for w in wires:
        verts += list(w) + [w[0]]
        codes += [MPath.MOVETO] + [MPath.LINETO] * (len(w) - 1) + [MPath.CLOSEPOLY]
    return PathPatch(MPath(verts, codes), facecolor=color, edgecolor="k", lw=0.8, **kw)


def _limits(all_wires):
    pts = np.concatenate([w for ws in all_wires for w in ws])
    lo, hi = pts.min(axis=0) - 5, pts.max(axis=0) + 5
    return (lo[0], hi[0]), (lo[1], hi[1])


def _subtitle(m: dict) -> str:
    return f"{m['energy_j']:.0f} J, {m['mass_kg'] * 1e3:.1f} g, imbalance {m['imbalance_gmm']:.2f} g*mm"


def sidebyside_png(items: list[tuple[str, object, dict]], path: Path) -> None:
    """items: (label, weapon, metrics) -- the reference first. Filled silhouettes, same scale."""
    wires = [canonical_wires(w) for _, w, _ in items]
    xl, yl = _limits(wires)
    fig, axes = plt.subplots(1, len(items), figsize=(6 * len(items), 6), squeeze=False)
    for i, (ax, (label, _, m), ws) in enumerate(zip(axes[0], items, wires)):
        ax.add_patch(_patch(ws, REF_COLOR if i == 0 else DESIGN_COLOR))
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{label}\n{_subtitle(m)}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def overlay_png(ref: tuple[str, object, dict], designs: list[tuple[str, object, dict]], path: Path) -> None:
    """Each design's outline (red) over the reference's silhouette (gray)."""
    ref_w = canonical_wires(ref[1])
    wires = [canonical_wires(w) for _, w, _ in designs]
    xl, yl = _limits([ref_w, *wires])
    fig, axes = plt.subplots(1, len(designs), figsize=(7 * len(designs), 7), squeeze=False)
    for ax, (label, _, m), ws in zip(axes[0], designs, wires):
        ax.add_patch(_patch(ref_w, REF_COLOR, alpha=0.55))
        for w in ws:
            ax.plot(w[:, 0], w[:, 1], color=DESIGN_COLOR, lw=1.4)
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        ax.set_title(f"{label} (red) over {ref[0]} (gray)\n{_subtitle(m)}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def rows(mine: dict, ref: dict, match_mass: bool = True) -> list[dict]:
    """Every compared quantity: {group, item, reference, design, ratio, worse (bool), note}."""
    out = []

    def add(group, item, r, d, higher_better=True, fmt="{:.1f}", need=None, note=""):
        ratio = d / r if r else None
        lim = need if need is not None else r
        worse = (d < lim * (1 - 1e-6)) if higher_better else (d > r * (1 + 1e-6))
        out.append({"group": group, "item": item, "reference": fmt.format(r), "design": fmt.format(d),
                    "ratio": ratio, "worse": bool(worse), "note": note})

    add("energy & mass", "stored energy [J]", ref["energy_j"], mine["energy_j"], fmt="{:.0f}")
    if "mass_kg" in ref:
        add("energy & mass", "mass [g]", ref["mass_kg"] * 1e3, mine["mass_kg"] * 1e3,
            higher_better=False, note="" if match_mass else "not a constraint (match_mass = false)")
        if not match_mass:
            out[-1]["worse"] = False
    for k, label, fmt in (("moi_kgm2", "MOI [g*m^2]", "{:.3f}"),):
        if k in ref and k in mine:
            add("energy & mass", label, ref[k] * 1e3, mine[k] * 1e3, fmt=fmt)
            out[-1]["worse"] = False  # informative: energy is what counts
    if "imbalance_gmm" in mine:
        out.append({"group": "energy & mass", "item": "static imbalance [g*mm]",
                    "reference": f"{ref.get('imbalance_gmm', float('nan')):.2f}",
                    "design": f"{mine['imbalance_gmm']:.3f}", "ratio": None, "worse": False, "note": ""})
    for k, label in (("hook_deg", "rake / hook [deg]"), ("face_depth_mm", "face depth [mm]")):
        if k in ref and k in mine:
            out.append({"group": "tooth", "item": label, "reference": f"{ref[k]:.1f}", "design": f"{mine[k]:.1f}",
                        "ratio": mine[k] / ref[k] if ref[k] else None, "worse": False, "note": "informative"})
    for d, rw in ref["tooth_width_mm"].items():
        if rw > 0:
            add("tooth", f"width {d} mm below the tip [mm]", rw, mine["tooth_width_mm"][d])
    for case in CASES:
        if case not in mine.get("cases", {}) or case not in ref.get("cases", {}):
            continue
        m, r = mine["cases"][case], ref["cases"][case]
        out.append({"group": CASE_LABELS[case], "item": "first yield",
                    "reference": f"{r['yield_force_kn']:.2f} kN, {r['region']}",
                    "design": f"{m['yield_force_kn']:.2f} kN, {m['region']}",
                    "ratio": m["yield_force_kn"] / r["yield_force_kn"] if r["yield_force_kn"] else None,
                    "worse": m["yield_force_kn"] < r["yield_force_kn"],
                    "note": "" if m["region"] == r["region"] else "first yields somewhere else"})
        for k in REGIONS:
            need = min(r["regions_kn"][k], REGION_CAP * r["yield_force_kn"])
            if not (math.isfinite(r["regions_kn"][k]) or math.isfinite(m["regions_kn"][k])):
                continue  # judged in neither (e.g. the tip under the contact patch)
            capped = r["regions_kn"][k] > need
            add(CASE_LABELS[case], f"{k} region holds [kN]", r["regions_kn"][k], m["regions_kn"][k], fmt="{:.2f}",
                need=need, note=f"needs {need:.2f} (capped at {REGION_CAP}x ref first yield)" if capped else "")
    return out


def table_md(rows_: list[dict], ref_name: str, design_name: str) -> str:
    lines = [f"| | {ref_name} | {design_name} | ratio | |", "|---|---|---|---|---|"]
    group = None
    for r in rows_:
        if r["group"] != group:
            group = r["group"]
            lines.append(f"| **{group}** | | | | |")
        ratio = "" if r["ratio"] is None else f"{r['ratio']:.3f}"
        flag = "**WORSE**" if r["worse"] else ""
        note = f" {r['note']}" if r["note"] else ""
        lines.append(f"| {r['item']} | {r['reference']} | {r['design']} | {ratio} | {flag}{note} |")
    n_worse = sum(r["worse"] for r in rows_)
    lines.append("")
    lines.append(f"{n_worse} item(s) worse than {ref_name}." if n_worse else f"Nothing worse than {ref_name}.")
    return "\n".join(lines) + "\n"


def write(run: Path, ref: tuple[str, object, dict], designs: list[tuple[str, object, dict]],
          match_mass: bool = True) -> dict:
    """compare_sidebyside.png, compare_overlay.png, compare.md (+ the rows, returned for report.json)."""
    sidebyside_png([ref, *designs], run / "compare_sidebyside.png")
    overlay_png(ref, designs, run / "compare_overlay.png")
    md, out = [], {}
    for name, _, m in designs:
        rs = rows(m, ref[2], match_mass)
        out[name] = rs
        md.append(f"## {name} vs {ref[0]}\n\n" + table_md(rs, ref[0], name))
    (run / "compare.md").write_text("\n".join(md))
    return out

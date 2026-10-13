"""PNG renders meant to be read by a human or by Claude (Read tool) to inspect a design."""

import math
from pathlib import Path

import build123d as bd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, FILL, ACCENT, BAD = "#1f2933", "#9aa5b1", "#d64545", "#d64545"


def _wire_xy(wire: bd.Wire, n: int = 600) -> np.ndarray:
    return np.array([(p.X, p.Y) for p in (wire.position_at(t) for t in np.linspace(0, 1, n))])


def profile_png(result: dict, path: Path, title: str = "") -> None:
    weapon, mp, hit = result["_weapon"], result["mass"], result["impact"]
    face = weapon.profile()
    fig, ax = plt.subplots(figsize=(8, 8))
    outer = _wire_xy(face.outer_wire())
    ax.fill(outer[:, 0], outer[:, 1], color=FILL, ec=INK, lw=1.2)
    for w in face.inner_wires():
        xy = _wire_xy(w, 200)
        ax.fill(xy[:, 0], xy[:, 1], color="white", ec=INK, lw=1)

    r_tip = weapon.tip_radius_mm()
    t = np.linspace(0, 2 * math.pi, 400)
    ax.plot(r_tip * np.cos(t), r_tip * np.sin(t), "--", color=INK, lw=0.6, alpha=0.5)
    cx, cy, _ = mp["cg_mm"]
    ax.plot([cx], [cy], "x", color=ACCENT, ms=10, mew=2, label=f"CG ({cx:.2f}, {cy:.2f}) mm")
    sign = 1 if getattr(weapon, "spin", "ccw") == "ccw" else -1
    ax.annotate("", xy=(0.25 * r_tip * math.cos(sign * 0.6), 0.25 * r_tip * math.sin(sign * 0.6)),
                xytext=(0.25 * r_tip, 0), arrowprops=dict(arrowstyle="->", color=ACCENT,
                                                          connectionstyle=f"arc3,rad={0.3 * sign}"))
    failed = result.get("failed", [])
    txt = "\n".join([
        f"material   {result['material']}",
        f"mass       {mp['mass_kg']:.3f} kg",
        f"MOI        {mp['moi_kgm2'] * 1e3:.2f} g*m^2",
        f"rpm        {result['drive']['top_rpm']:.0f}",
        f"energy     {hit['energy_j']:.0f} J",
        f"tip speed  {hit['tip_speed_mps']:.0f} m/s",
        f"bite       {hit['bite_mm']:.1f} mm",
        f"SF (yield) {result['stress']['safety_factor_yield']:.2f}",
        f"imbalance  {mp['static_imbalance_gmm']:.2f} g*mm",
        "OK" if result["ok"] else "FAILED: " + ", ".join(failed),
    ])
    ax.text(0.01, 0.99, txt, transform=ax.transAxes, va="top", family="monospace", fontsize=9,
            color=INK if result["ok"] else BAD,
            bbox=dict(boxstyle="round", fc="white", ec=INK, alpha=0.85))
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def spinup_png(result: dict, path: Path) -> None:
    curve = result.get("_spinup_curve")
    if not curve:
        return
    t, rpm = zip(*curve)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, rpm, color=INK)
    d = result["drive"]
    for frac, key in ((0.9, "spinup_s_90"), (0.95, "spinup_s_95")):
        if d[key] is not None:
            ax.axvline(d[key], ls=":", color=ACCENT)
            ax.text(d[key], frac * d["top_rpm"] * 0.5, f" {int(frac * 100)}% @ {d[key]:.2f}s",
                    color=ACCENT, fontsize=8)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("weapon rpm")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _tri_linear(elems: np.ndarray) -> np.ndarray:
    """Split each 6-node triangle into 4 linear ones for plotting (0-based)."""
    e = elems - 1
    return np.concatenate([e[:, [0, 3, 5]], e[:, [3, 1, 4]], e[:, [5, 4, 2]], e[:, [3, 4, 5]]])


def stress_png(result: dict, path: Path, yield_mpa: float) -> None:
    from matplotlib.tri import Triangulation

    from eitri.physics.fea.analysis import _vm, combined_stress

    run, s = result.get("_fea"), result["stress"]
    if run is None:
        return
    tri = Triangulation(run.mesh.nodes[:, 0], run.mesh.nodes[:, 1], _tri_linear(run.mesh.elems))
    f_y = s["yield_force_kn"]
    scale = f_y * 1e3 / run.applied_force_n if run.applied_force_n else 0.0
    panels = [
        ("spinning (no hit)", run.cases[0].result.von_mises, s["centrifugal"]["body_max_at_mm"],
         f"max body {s['centrifugal']['body_max_vm_mpa']:.0f} MPa, SF {s['centrifugal']['safety_factor_body']:.2f}"),
        (f"spinning + hit at yield force {f_y:.0f} kN", _vm(combined_stress(run, scale)), s["yield_location_mm"],
         f"est. peak hit {s['impact_at_estimated_peak']['force_kn']:.0f} kN -> impact SF "
         f"{s['impact_safety_factor']:.2f}; elastic if contact >= {s['min_elastic_contact_time_ms']:.2f} ms"),
    ]
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_over("#00c2ff")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    for i, (ax, (title, vm, at, sub)) in enumerate(zip(axes, panels)):
        is_hit = i == 1
        vmax = yield_mpa if is_hit else max(float(vm[run.cases[0].zones["body"]].max()) * 1.05, 1)
        cs = ax.tricontourf(tri, np.minimum(vm, vmax * 1.5), levels=np.linspace(0, vmax, 41), cmap=cmap,
                            extend="max")
        fig.colorbar(cs, ax=ax, shrink=0.75, label="von Mises [MPa]" + (" (cyan > yield)" if is_hit else ""))
        ax.plot(*at, "o", mfc="none", mec="#00ff88", ms=14, mew=2)
        (x0, y0), (x1, y1) = run.contact_segment
        ax.plot([x0, x1], [y0, y1], color="#00ff88", lw=4, solid_capstyle="butt")
        cz = run.cases[0].zones["contact"]
        ax.plot(run.mesh.nodes[cz, 0], run.mesh.nodes[cz, 1], ".", color="#00ff88", ms=1, alpha=0.4)
        for cx, cy, r in run.support_circles:
            ax.add_patch(plt.Circle((cx, cy), r, fill=False, ec="#00ff88", lw=1.2))
        ax.set_aspect("equal")
        ax.set_title(f"{title}\n{sub}", fontsize=10)
        ax.set_xlabel("x [mm]")
    fig.suptitle(f"{result['material']} — green: contact patch (+ excluded dent zone), supports, max body stress", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def pareto_png(history: list[dict], path: Path, objective: str, best: dict | None) -> None:
    pts = [h for h in history if "error" not in h and h.get("sf_impact") is not None]
    if not pts:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (key, label) in zip(axes, [("sf_impact", "impact safety factor"), ("mass_kg", "mass [kg]")]):
        for ok, color, name in ((False, "#c3c9d1", "infeasible"), (True, INK, "feasible")):
            sel = [h for h in pts if h["ok"] == ok]
            ax.scatter([h[key] for h in sel], [h["objective"] for h in sel], s=14, c=color, label=name)
        if best and "error" not in best:
            ax.scatter([best[key]], [best["objective"]], s=160, marker="*", c=ACCENT, label="best", zorder=5)
        ax.set_xlabel(label)
        ax.set_ylabel(objective)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)

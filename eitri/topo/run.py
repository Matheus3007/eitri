"""`eitri topo`: for each MOI target, optimize the density, extract pockets, build the balanced design and
judge it with the full pipeline (CalculiX, reference load cases, comparison report)."""

import json
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tomli_w  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402

from eitri import materials  # noqa: E402
from eitri.evaluate import build_weapon, single_tooth_bite_mm  # noqa: E402
from eitri.spec import Spec, load  # noqa: E402
from eitri.topo import extract, simp  # noqa: E402
from eitri.topo.domain import Domain  # noqa: E402


def targets(spec: Spec) -> tuple[float, float, str]:
    """(mass kg, reference MOI kg*m^2, where they come from)."""
    from eitri.reference import reference_metrics

    t = spec.topology
    ref = reference_metrics(spec.reference.spec, False, spec.analysis.hub_support) if spec.reference else None
    if t.mass_kg is not None:
        mass, src = t.mass_kg, "[topology] mass_kg"
    elif ref is not None and spec.reference.match_mass:
        mass, src = ref["mass_kg"], "the reference's mass"
        if spec.constraints.mass_kg.max is not None and spec.constraints.mass_kg.max < mass:
            mass, src = spec.constraints.mass_kg.max, "mass_kg.max (below the reference's)"
    elif spec.constraints.mass_kg.max is not None:
        mass, src = spec.constraints.mass_kg.max, "mass_kg.max"
    else:
        mass, src = None, "the base's mass"
    if ref is not None:
        moi = ref["moi_kgm2"]
    else:
        from eitri.evaluate import evaluate

        b = load(t.base)
        r = evaluate(b)
        moi = r["mass"]["moi_kgm2"]
        mass = mass if mass is not None else r["mass"]["mass_kg"]
    return mass, moi, src


def density_png(res: simp.TopoResult, path: Path, title: str) -> None:
    m = res.mesh
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.tripcolor(Triangulation(m.nodes[:, 0], m.nodes[:, 1], m.tris), facecolors=res.density, cmap="gray_r",
                 vmin=0, vmax=1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def run(spec: Spec, spec_path: Path, out: Path, log: Callable[[str], None] = print) -> list[dict]:
    from shapely.geometry import Polygon

    t = spec.topology
    mat = materials.get(spec.material.names()[0])
    bspec = load(t.base)
    base = build_weapon(bspec, bspec.fixed_design())
    bite = single_tooth_bite_mm(spec)
    env = bite if t.envelope else 0.0
    dom = Domain(base, t.keep, envelope_bite_mm=env)
    mass, moi_ref, src = targets(spec)
    contacts = t.contacts_mm or [bite, 3.0, 8.0]
    area = mass * (1 - 2e-5) / (mat.density * 1e-9 * base.params.thickness_mm)  # a hair under the budget
    omega, forces = 0.0, None
    if t.spinning and spec.reference is not None and t.contacts_mm is None:
        from eitri.physics import drive
        from eitri.reference import reference_metrics

        rc = reference_metrics(spec.reference.spec, False, spec.analysis.hub_support)["cases"]
        omega = drive.top_speed_rad_s(spec.drive)
        forces = [rc[k]["yield_force_kn"] * 1e3 for k in ("hit_bite", "hit_tip3", "hit_tip8", "flat_hit")]
        log(f"judged spinning at {omega * 30 / np.pi:.0f} rpm, hits at the reference's first-yield forces "
            f"{[round(f / 1e3, 2) for f in forces]} kN")
    log(f"domain {dom.profile.area:.0f} mm^2 ({'bite envelope' if env else 'base outline'}), keep "
        f"{sum(k.area for k in dom.keep):.0f} mm^2; mass {mass * 1e3:.1f} g ({src}); hits on {contacts} mm")
    outline = Polygon([(p.X, p.Y) for p in (dom.profile.outer_wire().position_at(u)
                                             for u in np.linspace(0, 1, 2000, endpoint=False))])
    rows = []
    for k, frac in enumerate(t.moi):
        tag = f"moi{frac:.3f}".replace(".", "p")
        d = out / tag
        d.mkdir(parents=True, exist_ok=True)
        log(f"  MOI {frac:.3f} x {moi_ref * 1e3:.3f} g*m^2:")
        res = simp.optimize(dom, mat.E, mat.nu, mat.density, mass, moi_ref * frac, contacts,
                            mesh_mm=t.mesh_mm, rmin_mm=t.rmin_mm, iters=t.iters, objective=t.objective,
                            flat_weight=t.flat_weight, omega_rad_s=omega, case_forces_n=forces, log=log)
        density_png(res, d / "density.png", f"{spec.name} {tag}: {t.objective}, mass {mass * 1e3:.1f} g")
        np.savez_compressed(d / "density.npz", nodes=res.mesh.nodes, tris=res.mesh.tris, density=res.density,
                            keep=res.keep)
        cs = extract.void_contours(res.mesh.nodes, res.mesh.tris, res.nodal_density(), outline)
        extract.save(d / "pockets.json", cs, {"moi_frac": frac, "objective": t.objective,
                                              "constraint_error": res.constraint_error})
        design = {"base": t.base, "pockets": str(d / "pockets.json"), "keep": ";".join(t.keep),
                  "fillet_mm": t.fillet_mm, "area_mm2": area, "envelope_bite_mm": env,
                  "min_web_mm": spec.constraints.min_ligament_mm}
        sd = spec.model_dump(mode="json", exclude_none=True)
        sd.pop("topology", None)
        sd["name"] = f"{spec.name}_{tag}"
        sd["weapon"]["type"] = "topo"
        sd["weapon"]["spin"] = base.spin
        sd["design"] = design
        (d / "spec.toml").write_text(tomli_w.dumps(sd))
        rows.append({"moi_frac": frac, "dir": str(d), "spec": str(d / "spec.toml"), "n_pockets": len(cs),
                     "topo_objective": res.history[-1]["objective"] if res.history else None})
    (out / "topo.json").write_text(json.dumps(rows, indent=2))
    return rows


def faces_map(spec: Spec, out: Path) -> dict:
    """Numbered boundary edges (side faces of the plate) and the keep/free regions, as PNG + JSON, so keep
    selectors (`edge:I:W`, `circle:...`) can be written by reading the picture."""
    t = spec.topology
    bspec = load(t.base if t else spec.design.get("base") or "")
    base = build_weapon(bspec, bspec.fixed_design())
    env = single_tooth_bite_mm(spec) if (t and t.envelope) else 0.0
    dom = Domain(base, (t.keep if t else []), envelope_bite_mm=env)
    edges = dom.edges()
    fig, ax = plt.subplots(figsize=(9, 9))
    from eitri.report.compare import _patch

    for f in dom.keep:
        rings = [np.array([(p.X, p.Y) for p in (w.position_at(u) for u in np.linspace(0, 1, 300))])
                 for w in [f.outer_wire(), *f.inner_wires()]]
        patch = _patch(rings, "#4c9be8", alpha=0.35)
        patch.set_edgecolor("none")
        ax.add_patch(patch)
    rows = []
    for i, e in enumerate(edges):
        xy = np.array([(p.X, p.Y) for p in (e.position_at(u) for u in np.linspace(0, 1, 60))])
        ax.plot(xy[:, 0], xy[:, 1], color="#1f2933", lw=1.2)
        m = e.position_at(0.5)
        ax.text(m.X, m.Y, str(i), fontsize=7, color="#c0392b", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
        rows.append({"id": i, "type": e.geom_type.name.lower(), "length_mm": round(e.length, 3),
                     "start": [round(e.start_point().X, 3), round(e.start_point().Y, 3)],
                     "end": [round(e.end_point().X, 3), round(e.end_point().Y, 3)],
                     "mid": [round(m.X, 3), round(m.Y, 3)]})
    tip, root = base.impact_face()
    ax.plot([tip[0], root[0]], [tip[1], root[1]], color="#00b36b", lw=3, label="impact face")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"{spec.name}: boundary edges (side faces) by id; blue = keep, rest free", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "faces.png", dpi=110)
    plt.close(fig)
    data = {"edges": rows, "keep": t.keep if t else [], "domain_area_mm2": dom.profile.area,
            "keep_area_mm2": sum(k.area for k in dom.keep)}
    (out / "faces.json").write_text(json.dumps(data, indent=1))
    return data

"""eitri command line. Every command is non-interactive and writes machine-readable output."""

import json
import math
import os
from datetime import datetime
from pathlib import Path

import tomli_w
import build123d as bd
import typer

from eitri import materials, spec as spec_mod
from eitri.evaluate import evaluate, public
from eitri.report import render

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)

TEMPLATE = """\
name = "{name}"

[weapon]
type = "{type}"         # hook (asymmetric single tooth, the default) | fang | disc | step (import a STEP)
spin = "ccw"            # seen from +Z; teeth face the spin direction

[material]
name = "AR500"          # or: candidates = ["AR500", "S7", "4340"]  (see `eitri materials`)

[drive]
rpm = 8000              # weapon rpm, or replace with a motor:
# motor = {{ kv = 190, voltage = 44.4, resistance_ohm = 0.04, current_limit_a = 150 }}
# ratio = 1.0           # motor turns per weapon turn

[context]               # the fight you design for
closing_speed_mps = 3.0
opponent_mass_kg = 13.6
restitution = 0.2
contact_time_ms = 0.71  # hit pulse; calibrated so a proven weapon (Rebot V3) sits at impact SF 1

[constraints]
mass_kg = {{ max = 4.5 }}
min_safety_factor = 2.0       # spinning, no hit
min_impact_safety_factor = 1.0  # tooth/body must not yield under the estimated peak hit
max_imbalance_gmm = 20
min_ligament_mm = 5
# max_spinup_s = 3.0

[objective]
maximize = "energy"     # energy | moi | effective_mass

[style]                 # looks are a hard constraint: aggressive and well made
min_hook_deg = 5
max_root_fillet_ratio = 0.4
min_pocket_sweep_deg = 15
min_wall_frac = 0.08    # pocket walls >= 8% of the tip radius: no thin-rim hoops

[design]                # number = fixed, [min, max] = free for the optimizer (eval uses the midpoint)
{design}"""

DESIGNS = {
    "hook": """\
od_mm = 230
thickness_mm = 6.5
face_depth_mm = 32      # radial depth of the impact face (>= the bite)
hook_deg = 12
root_fillet_mm = 3
back_deg = 24           # tooth back: from the tip down to back_r_frac * R, this far behind the tip
back_r_frac = 0.84
hub_d_mm = 52
spine_waist_deg = 102   # where the spine meets the hub, behind the tip
cw_point_deg = 143      # counterweight point, behind the tip (nominal: balance turns the lobe a little)
lobe_span_deg = 97
gullet_waist_deg = 39   # where the gullet meets the hub, ahead of the face
cw_size_frac = 0.8      # counterweight reach as a fraction of the bite envelope (balance comes from pockets)
wall_mm = 9.5           # thinnest counterweight-pocket wall; balance solves the actual (constant) wall
arm_wall_mm = 11        # arm slot wall
pocket_fillet_mm = 7    # every pocket corner is this radius
head_solid_mm = 30
arm_pocket = 1
bolt_count = 7
bolt_d_mm = 5.3
bolt_pcd_mm = 36.5
lock_d_mm = 29          # lobed pulley-lock cutout around the bolts (Rebot V3); 0 = plain bore_mm
lock_boss_d_mm = 10.5
lock_fillet_mm = 1
""",
    "fang": """\
od_mm = 300
thickness_mm = 12
hub_d_mm = 90
face_depth_mm = 30      # radial depth of the impact face
hook_deg = 10
root_fillet_mm = 5      # gullet fillet at the face root
land_deg = 8            # tooth head size behind the tip
arm_root_width_mm = 50
arm_sweep_deg = 45      # fang arm trails the tip by this much at the hub
arm_curve = 2.5         # >1: bend near the tip (forward-hooking fang)
blend_mm = 12
cw_radius_mm = 120      # counterweight reach (stays under the face root); its size/angle are solved
cw_sweep_deg = 30
cw_root_width_mm = 75
bore_mm = 25.4
bolt_count = 6
bolt_d_mm = 6.6
bolt_pcd_mm = 50
""",
    "disc": """\
od_mm = 300
thickness_mm = 12
n_teeth = 2
tooth_height_mm = 20
hook_deg = 10
root_fillet_mm = 4
land_deg = 8
ramp_deg = 90
bore_mm = 25.4
bolt_count = 6
bolt_d_mm = 6.6
bolt_pcd_mm = 50
pocket_style = "swept"
pockets_per_tooth = 2
pocket_phase_deg = -45
pocket_r_in_mm = 45
pocket_r_out_mm = 100
pocket_span_deg = 40
pocket_sweep_deg = 25
pocket_fillet_mm = 6
""",
}


def _run_dir(spec_path: Path, kind: str) -> Path:
    d = spec_path.parent / "runs" / f"{datetime.now():%Y%m%d-%H%M%S}-{kind}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _clean(o):
    """JSON-safe: inf/nan -> None."""
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def _summary(r: dict) -> str:
    if "error" in r:
        return f"[{r['material']}] INVALID DESIGN: {r['error']}"
    m, h, s = r["mass"], r["impact"], r["stress"]
    status = "OK" if r["ok"] else "FAIL " + ",".join(r["failed"])
    line = (f"[{r['material']}] {status} | mass {m['mass_kg']:.3f} kg | MOI {m['moi_kgm2'] * 1e3:.2f} g*m^2 | "
            f"{r['drive']['top_rpm']:.0f} rpm | E {h['energy_j']:.0f} J | tip {h['tip_speed_mps']:.0f} m/s | "
            f"bite {h['bite_mm']:.1f} mm | SF spin {s['safety_factor_yield']:.2f} | "
            f"imbalance {m['static_imbalance_gmm']:.2f} g*mm")
    if "yield_force_kn" in s:
        line += (f"\n    hit: impact SF {s['impact_safety_factor']:.2f} = yields at {s['yield_force_kn']:.0f} kN / "
                 f"est. peak {h['peak_force_kn']:.0f} kN, "
                 f"elastic if contact >= {s['min_elastic_contact_time_ms']:.2f} ms | "
                 f"impulse {h['impulse_ns']:.0f} N*s")
    return line


def export_step_verified(solid, path: Path, plate=None) -> dict:
    """Export, re-import and compare: a STEP that silently lost geometry must never ship.

    Plates compare the re-imported top face's area with the profile's (exact integration, 1e-6): a missing
    bolt hole is 0.3% of a beetleweight hook, below the ~0.1% noise of OCC's default volume integration on
    B-spline side faces, which therefore only serves as a gross check (0.5%)."""
    from eitri.geometry.common import face_props, to_nurbs

    def top_area(shape) -> float:
        tops = [f for f in shape.faces() if f.geom_type == bd.GeomType.PLANE and f.normal_at().Z > 0.999]
        return max(face_props(f).Mass() for f in tops) if tops else 0.0

    want = face_props(plate[0]).Mass() if plate is not None else None
    for attempt, shape in (("direct", solid), ("nurbs", None)):
        if shape is None:
            shape = to_nurbs(solid)
        bd.export_step(shape, str(path))
        back = bd.import_step(str(path))
        err = abs(back.volume - solid.volume) / solid.volume
        area_err = abs(top_area(back) - want) / want if want else 0.0
        if err < 5e-3 and area_err < 1e-6:
            return {"ok": True, "path": str(path), "volume_error": err, "profile_area_error": area_err,
                    "method": attempt}
    path.rename(path.with_suffix(".BROKEN.step"))
    return {"ok": False, "path": str(path.with_suffix(".BROKEN.step")), "volume_error": err,
            "profile_area_error": area_err}


def fixed_spec_toml(spec: spec_mod.Spec, design: dict, material: str) -> str:
    """The spec with every design value fixed and a single material: ready for `eval`."""
    d = spec.model_dump(mode="json", exclude_none=True)
    d["design"] = design
    d["material"] = {"name": material}
    return tomli_w.dumps(d)


def _write_outputs(run: Path | None, spec: spec_mod.Spec, spec_path: Path, results: list[dict]) -> dict:
    out = {"spec": str(spec_path), "free_params_at_midpoint": sorted(spec.free_params()),
           "results": [_clean(public(r)) for r in results]}
    if run is None:
        return out
    for r in results:
        if "error" in r:
            continue
        tag = r["material"]
        render.profile_png(r, run / f"profile_{tag}.png", f"{spec.name} - {tag}")
        render.spinup_png(r, run / f"spinup_{tag}.png")
        render.stress_png(r, run / f"stress_{tag}.png", materials.get(tag).yield_)
    good = next((r for r in results if "error" not in r), None)
    if good:
        out["step_export"] = export_step_verified(good["_solid"], run / "weapon.step", good["_weapon"].plate())
        (run / "spec_fixed.toml").write_text(fixed_spec_toml(spec, good["design"], good["material"]))
    (run / "report.json").write_text(json.dumps(out, indent=2))
    out["run_dir"] = str(run)
    return out


@app.command()
def new(project_dir: Path, name: str = typer.Option(None, help="defaults to the directory name"),
        weapon: str = typer.Option("hook", "--type", help="hook | fang | disc")):
    """Create a project directory with a template spec.toml."""
    if weapon not in DESIGNS:
        raise typer.BadParameter(f"--type must be one of {sorted(DESIGNS)}")
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / "spec.toml"
    if path.exists():
        raise typer.BadParameter(f"{path} already exists")
    path.write_text(TEMPLATE.format(name=name or project_dir.name, type=weapon, design=DESIGNS[weapon]))
    typer.echo(str(path))


@app.command("materials")
def list_materials():
    """List the material database."""
    for m in materials.all_materials().values():
        typer.echo(f"{m.name:12} yield {m.yield_:5.0f} MPa  uts {m.uts:5.0f} MPa  "
                   f"charpy {m.charpy_j:3.0f} J  {m.hardness:14}  {m.description}")


@app.command("eval")
def eval_cmd(
    spec_path: Path,
    json_out: bool = typer.Option(False, "--json", help="print the full report as JSON"),
    no_files: bool = typer.Option(False, "--no-files", help="don't write a run directory"),
    fine: bool = typer.Option(False, "--fine", help="verification mesh (od/75) instead of the fast one"),
    no_fea: bool = typer.Option(False, "--no-fea", help="skip FEA, analytic stress estimate only"),
):
    """Evaluate the design in a spec (ranges -> midpoint) for every material candidate."""
    spec = spec_mod.load(spec_path)
    if no_fea:
        spec.analysis.fea = False
    run = None if no_files else _run_dir(spec_path, "eval")
    results = [evaluate(spec, material=m, fine=fine, keep_fea_dir=run / "fea" / m if run else None)
               for m in spec.material.names()]
    out = _write_outputs(run, spec, spec_path, results)
    if json_out:
        typer.echo(json.dumps(out, indent=2))
    else:
        for r in results:
            typer.echo(_summary(r))
        if "run_dir" in out:
            typer.echo(f"run: {out['run_dir']}")
    raise typer.Exit(0 if all(r.get("ok") for r in results) else 1)


@app.command("optimize")
def optimize_cmd(
    spec_path: Path,
    trials: int = typer.Option(200, help="number of candidate designs to evaluate"),
    workers: int = typer.Option(max(1, min(8, (os.cpu_count() or 2) - 1)), help="parallel processes"),
    seed: int = typer.Option(0),
    verify_top: int = typer.Option(5, help="fine-mesh check up to this many top candidates"),
):
    """Optimize the [design] ranges for the [objective] under [constraints]; verify the best on a fine mesh."""
    from eitri.optimize import optimize

    spec = spec_mod.load(spec_path)
    run = _run_dir(spec_path, "optimize")
    typer.echo(f"optimizing {sorted(spec.free_params())} over {spec.material.names()} "
               f"({trials} trials, {workers} workers) -> {run}")
    res = optimize(spec, trials, workers, seed, run / "history.jsonl", log=typer.echo)
    render.pareto_png(res.history, run / "pareto.png", spec.objective.name, res.best)

    best = res.best
    if best is None or "error" in best:
        typer.echo("no valid design found; see history.jsonl")
        raise typer.Exit(1)
    # Verify the top candidates on the fine mesh; the winner is the best one that still passes there.
    ranked = sorted((h for h in res.history if h["ok"]), key=lambda h: -h["score"])[:verify_top] or [best]
    verified = []
    for cand in ranked:
        v = evaluate(spec, design=cand["design"], material=cand["material"], fine=True,
                     keep_fea_dir=run / "best" / "fea")
        verified.append((cand, v))
        typer.echo(f"  fine-mesh check trial {cand['trial']}: {spec.objective.name} "
                   f"{cand.get('objective', float('nan')):.4g} -> {'ok' if v.get('ok') else 'FAIL ' + ','.join(v.get('failed', [v.get('error', '')]))}")
        if v.get("ok"):
            break
    best, verify = next(((c, v) for c, v in verified if v.get("ok")), verified[0])
    (run / "best").mkdir(exist_ok=True)
    _write_outputs(run / "best", spec, spec_path, [verify])
    b0 = res.baseline
    summary = {
        "feasible_found": res.best_feasible,
        "n_trials": len(res.history),
        "n_feasible": sum(h["ok"] for h in res.history),
        "baseline": {k: b0.get(k) for k in ("ok", "objective", "mass_kg", "sf_spinning", "sf_impact", "failed")},
        "best_trial": best["trial"],
        "fine_mesh_checks": [{"trial": c["trial"], "objective": c.get("objective"), "ok": bool(v.get("ok")),
                              "failed": v.get("failed", [v.get("error")])} for c, v in verified],
        "best_verified_fine_mesh": _clean(public(verify)),
        "best_spec": str(run / "best" / "spec_fixed.toml"),
    }
    (run / "summary.json").write_text(json.dumps(_clean(summary), indent=2))
    typer.echo("" if res.best_feasible else "WARNING: no design met every constraint; best = least violating")
    if b0.get("objective") is not None and best.get("objective"):
        typer.echo(f"baseline {spec.objective.name} {b0['objective']:.4g} "
                   f"({'ok' if b0['ok'] else 'FAIL ' + ','.join(b0.get('failed', []))}) -> "
                   f"best {best['objective']:.4g} (trial {best['trial']})")
    typer.echo("verified on fine mesh:")
    typer.echo(_summary(verify))
    typer.echo(f"run: {run}\nbest spec: {summary['best_spec']}")
    raise typer.Exit(0 if res.best_feasible and verify.get("ok") else 1)


@app.command()
def check(spec_path: Path):
    """Fast geometry + constraint checks only (no files). Exit code 1 if anything fails."""
    spec = spec_mod.load(spec_path)
    ok = True
    for m in spec.material.names():
        r = evaluate(spec, material=m)
        if "error" in r:
            typer.echo(_summary(r))
            ok = False
            continue
        for c in r["checks"]:
            mark = "PASS" if c["ok"] else "FAIL"
            val = "" if c["value"] is None else f" {c['value']:.3f}"
            lim = "" if c["limit"] is None else f" (limit {c['limit']:.3f})"
            typer.echo(f"[{m}] {mark} {c['name']}{val}{lim}")
        ok &= r["ok"]
    raise typer.Exit(0 if ok else 1)


@app.command()
def show(spec_path: Path, port: int = 3939):
    """Send the design to the OCP CAD Viewer panel in VS Code (needs the `viewer` extra)."""
    try:
        from ocp_vscode import set_port, show as ocp_show
    except ImportError:
        raise typer.BadParameter("install the viewer extra: uv sync --extra viewer")
    from eitri.evaluate import build_weapon

    spec = spec_mod.load(spec_path)
    set_port(port)
    ocp_show(build_weapon(spec, spec.fixed_design()).solid(), names=[spec.name])


if __name__ == "__main__":
    app()

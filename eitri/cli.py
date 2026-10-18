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
contact_time_ms = 0.67  # hit pulse; calibrated so a proven weapon (Rebot V3) sits at impact SF 1

[constraints]
mass_kg = {{ max = 4.5 }}
min_safety_factor = 2.0       # spinning, no hit
min_impact_safety_factor = 1.0  # tooth/body must not yield under the estimated peak hit
max_imbalance_gmm = 0.1     # CG dead on the axis (every generator solves its balance by Newton)
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
    if good and good.get("reference_profile") and spec.reference is not None:
        out["compare"] = _write_compare(run, spec.reference, [(spec.name, good)])
    if good:
        out["step_export"] = export_step_verified(good["_solid"], run / "weapon.step", good["_weapon"].plate())
        (run / "spec_fixed.toml").write_text(fixed_spec_toml(spec, good["design"], good["material"]))
    (run / "report.json").write_text(json.dumps(out, indent=2))
    out["run_dir"] = str(run)
    return out


def _write_compare(run: Path, ref_spec, designs: list[tuple[str, dict]]) -> dict:
    """Default comparison against the [reference]: side-by-side, overlay and the flagged table."""
    from eitri import reference
    from eitri.report import compare

    ref_name, ref_weapon = reference.reference_weapon(ref_spec.spec)
    ref_m = next(r["reference_profile"]["reference"] for _, r in designs)
    items = [(name, r["_weapon"], r["reference_profile"]["design"]) for name, r in designs]
    return compare.write(run, (ref_name, ref_weapon, ref_m), items, ref_spec.match_mass)


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
    results = [evaluate(spec, material=m, fine=fine, keep_fea_dir=run / "fea" / m if run else None,
                        full_reference=run is not None) for m in spec.material.names()]
    out = _write_outputs(run, spec, spec_path, results)
    if json_out:
        typer.echo(json.dumps(out, indent=2))
    else:
        for r in results:
            typer.echo(_summary(r))
        if "run_dir" in out:
            typer.echo(f"run: {out['run_dir']}")
    raise typer.Exit(0 if all(r.get("ok") for r in results) else 1)


def _finish_optimize(spec, spec_path: Path, run: Path, res, verify_top: int, impact_gate: int = 1) -> None:
    """Fine-mesh verification of the top candidates, then (with a [reference] and OpenCourant installed) the
    explicit-hit gate on up to `impact_gate` of the fine-mesh survivors; outputs, summary; exits with the verdict."""
    render.pareto_png(res.history, run / "pareto.png", spec.objective.name, res.best)
    best = res.best
    if best is None or "error" in best:
        typer.echo("no valid design found; see history.jsonl")
        raise typer.Exit(1)
    from eitri.physics.explicit import impact3d

    use_gate = impact_gate > 0 and spec.reference is not None and impact3d.available()
    want = impact_gate if use_gate else 1
    # Verify the top candidates on the fine mesh; the winner is the best one that still passes there.
    ranked = sorted((h for h in res.history if h["ok"]), key=lambda h: -h["score"])[:verify_top] or [best]
    verified = []
    for cand in ranked:
        v = evaluate(spec, design=cand["design"], material=cand["material"], fine=True,
                     keep_fea_dir=run / "best" / "fea", full_reference=True)
        verified.append((cand, v))
        typer.echo(f"  fine-mesh check trial {cand['trial']}: {spec.objective.name} "
                   f"{cand.get('objective', float('nan')):.4g} -> {'ok' if v.get('ok') else 'FAIL ' + ','.join(v.get('failed', [v.get('error', '')]))}")
        if sum(bool(x.get("ok")) for _, x in verified) >= want:
            break
    gates = {}
    if use_gate:
        for cand, v in [(c, x) for c, x in verified if x.get("ok")][:impact_gate]:
            typer.echo(f"  explicit-hit gate, trial {cand['trial']}:")
            g = impact3d.gate(spec, cand["design"], cand["material"], run / "impact" / f"trial{cand['trial']}",
                              log=typer.echo)
            gates[cand["trial"]] = g
            v["impact_gate"] = g
            if not g["ok"]:
                v["ok"] = False
                v.setdefault("failed", []).append("impact_gate")
            else:
                break
    elif impact_gate > 0 and spec.reference is not None:
        typer.echo("  explicit-hit gate skipped: OpenCourant not installed (see tools/opencourant.sh)")
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
        "best_phase": best.get("phase"),
        "fine_mesh_checks": [{"trial": c["trial"], "objective": c.get("objective"), "ok": bool(v.get("ok")),
                              "failed": v.get("failed", [v.get("error")])} for c, v in verified],
        "best_verified_fine_mesh": _clean(public(verify)),
        "impact_gate": _clean({str(k): v for k, v in gates.items()}),
        "best_spec": str(run / "best" / "spec_fixed.toml"),
    }
    (run / "summary.json").write_text(json.dumps(_clean(summary), indent=2))
    typer.echo("" if res.best_feasible else "WARNING: no design met every constraint; best = least violating")
    if b0.get("objective") is not None and best.get("objective"):
        typer.echo(f"baseline {spec.objective.name} {b0['objective']:.4g} "
                   f"({'ok' if b0['ok'] else 'FAIL ' + ','.join(b0.get('failed', []))}) -> "
                   f"best {best['objective']:.4g} (trial {best['trial']}, {best.get('phase')})")
    typer.echo("verified on fine mesh:")
    typer.echo(_summary(verify))
    typer.echo(f"run: {run}\nbest spec: {summary['best_spec']}")
    raise typer.Exit(0 if res.best_feasible and verify.get("ok") else 1)


WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))


@app.command("optimize")
def optimize_cmd(
    spec_path: Path,
    trials: int = typer.Option(200, help="global candidates (TPE)"),
    workers: int = typer.Option(WORKERS, help="parallel processes"),
    seed: int = typer.Option(0),
    verify_top: int = typer.Option(5, help="fine-mesh check up to this many top candidates"),
    warm_start: Path | None = typer.Option(None, help="history.jsonl of an earlier run: seed with its best designs"),
    warm_k: int = typer.Option(12, help="how many earlier designs to seed"),
    polish: bool = typer.Option(True, help="Newton repair/climb of the best candidates, then CMA-ES around them"),
    local_trials: int | None = typer.Option(None, help="CMA-ES designs in the polish phase (default trials/4)"),
    radius: float = typer.Option(0.15, help="CMA-ES box: +- this fraction of each parameter's range"),
    impact_gate: int = typer.Option(1, help="explicit-hit gate on up to this many fine-mesh survivors (0 = off)"),
):
    """Optimize the [design] ranges for the [objective] under [constraints]; verify the best on a fine mesh."""
    from eitri.optimize import optimize, warm_designs

    spec = spec_mod.load(spec_path)
    run = _run_dir(spec_path, "optimize")
    typer.echo(f"optimizing {sorted(spec.free_params())} over {spec.material.names()} "
               f"({trials} trials, {workers} workers) -> {run}")
    warm = warm_designs(warm_start, warm_k) if warm_start else []
    res = optimize(spec, trials, workers, seed, run / "history.jsonl", log=typer.echo, warm=warm, polish=polish,
                   local_trials=local_trials, radius=radius)
    _finish_optimize(spec, spec_path, run, res, verify_top, impact_gate)


def _start_design(start: str) -> dict:
    """A design from a spec (`spec_fixed.toml`, ranges -> midpoint) or a history trial (`history.jsonl#123`)."""
    path, _, trial = start.partition("#")
    if trial:
        for line in Path(path).read_text().splitlines():
            rec = json.loads(line)
            if rec["trial"] == int(trial):
                return rec["design"]
        raise typer.BadParameter(f"no trial {trial} in {path}")
    return spec_mod.load(path).fixed_design()


@app.command("refine")
def refine_cmd(
    spec_path: Path,
    start: str = typer.Option(..., help="design to start from: spec_fixed.toml, or history.jsonl#<trial>"),
    workers: int = typer.Option(WORKERS, help="parallel processes"),
    local_trials: int = typer.Option(48, help="CMA-ES designs after the Newton repair/climb (0 = none)"),
    radius: float = typer.Option(0.15, help="CMA-ES box: +- this fraction of each parameter's range"),
    seed: int = typer.Option(0),
    verify_top: int = typer.Option(5),
    impact_gate: int = typer.Option(1, help="explicit-hit gate on up to this many fine-mesh survivors (0 = off)"),
):
    """Local search from one design: Newton repair (if it fails checks) or climb, then CMA-ES around it. The
    spec's [design] ranges bound the search; values outside them are clipped."""
    from eitri.optimize import OptimizeResult, _polish, _record, _worker, int_params, violations
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    spec = spec_mod.load(spec_path)
    spec.analysis.threads = 1
    design = spec.fixed_design() | _start_design(start)
    free = spec.free_params()
    design = {k: (min(max(v, free[k][0]), free[k][1]) if k in free else v) for k, v in design.items()}
    run = _run_dir(spec_path, "refine")
    typer.echo(f"refining from {start} -> {run}")
    history, full = [], {}
    mat = spec.material.names()[0]
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool, \
            (run / "history.jsonl").open("w") as hist:
        def record(result, phase, number=None):
            rec = _record(len(history), result, phase)
            rec["violation"] = sum(violations(result))
            history.append(rec)
            full[rec["trial"]] = result
            hist.write(json.dumps(rec) + "\n")
            hist.flush()
            return rec

        record(pool.submit(_worker, spec, design, mat).result(), "start")
        h0 = history[0]
        typer.echo(f"  start: {'OK' if h0['ok'] else 'fails ' + ','.join(h0.get('failed', [h0.get('error', '')]))}")
        _polish(spec, history, full, pool, record, int_params(spec.weapon.type), workers, seed, typer.echo,
                local_trials, radius)
    feasible = [h for h in history if h["ok"]]
    best = max(feasible, key=lambda h: h["score"]) if feasible else min(history, key=lambda h: h["violation"])
    _finish_optimize(spec, spec_path, run, OptimizeResult(history, best, bool(feasible), history[0]), verify_top,
                     impact_gate)


@app.command("compare")
def compare_cmd(
    specs: list[Path],
    ref: Path | None = typer.Option(None, help="reference spec (default: the first spec's [reference])"),
    fine: bool = typer.Option(False, "--fine", help="verification mesh"),
    out: Path | None = typer.Option(None, help="output directory (default: <first spec>/runs/<time>-compare)"),
):
    """Compare designs (their specs' fixed designs) with a reference: side-by-side silhouettes, overlay, and
    every load case / region / tooth width side by side, with whatever is worse than the reference flagged."""
    from eitri.spec import Reference

    loaded = [spec_mod.load(p) for p in specs]
    r = Reference(spec=str(ref)) if ref else next((s.reference for s in loaded if s.reference), None)
    if r is None:
        raise typer.BadParameter("no reference: pass --ref or give the specs a [reference] section")
    run = out or _run_dir(specs[0], "compare")
    run.mkdir(parents=True, exist_ok=True)
    designs = []
    for path, s in zip(specs, loaded):
        s.reference = r
        res = evaluate(s, fine=fine, full_reference=True)
        if "error" in res:
            raise typer.BadParameter(f"{path}: {res['error']}")
        typer.echo(_summary(res))
        names = [n for n, _ in designs]
        name = s.name if s.name not in names else f"{s.name} ({path.parent.name})"
        designs.append((name, res))
    rows = _write_compare(run, r, designs)
    (run / "compare.json").write_text(json.dumps(_clean(rows), indent=2))
    typer.echo((run / "compare.md").read_text())
    typer.echo(f"run: {run}")


@app.command("impact")
def impact_cmd(
    spec_path: Path,
    case: str = typer.Option("bite", help="bite | tip3 | all"),
    ref: bool = typer.Option(True, help="also run the [reference] weapon, same setup, and compare"),
    tstop: float = typer.Option(1.0, help="simulated time [ms]"),
    threads: int = typer.Option(8, help="solver threads per run"),
    opponent: str = typer.Option("steel", help="steel (failure test) | tpu (3D-printed TPU wall, see tpu.py)"),
):
    """Explicit 3D impact (OpenCourant): the spinning weapon hits the opponent; contact pulse, energy, plastic
    strain and failure per region, VTK frames for the visualizer. Needs tools/opencourant.sh (see it)."""
    from eitri.evaluate import single_tooth_bite_mm
    from eitri.physics import drive as drive_mod
    from eitri.physics.explicit import impact3d

    spec = spec_mod.load(spec_path)
    run = _run_dir(spec_path, "impact")
    cases = ["bite", "tip3"] if case == "all" else [case]
    targets = [("design", spec)]
    if ref and spec.reference is not None:
        targets.append(("reference", spec_mod.load(spec.reference.spec)))
    rows: dict = {c: {} for c in cases}
    for tag, sp in targets:
        from eitri.evaluate import build_weapon

        w = build_weapon(sp, sp.fixed_design())
        mat = materials.get(sp.material.names()[0])
        omega = drive_mod.top_speed_rad_s(sp.drive)
        J = (sp.drive.motor.rotor_inertia_kgm2 if sp.drive.motor else 0.0) * sp.drive.ratio ** 2
        for c in cases:
            folder = run / f"{tag}_{c}_{opponent}"
            setup = impact3d.ImpactSetup(case=c, tstop_ms=tstop, threads=threads, opponent=opponent)
            typer.echo(f"  {tag} {c}: running ...")
            r = impact3d.simulate(w, mat, omega, sp.context.opponent_mass_kg, sp.context.closing_speed_mps,
                                  single_tooth_bite_mm(sp), J, folder, setup)
            if not r.ok:
                typer.echo(f"  {tag} {c}: FAILED\n{r.log[-1500:]}")
                raise typer.Exit(1)
            (folder / "metrics.json").write_text(json.dumps(_clean(r.metrics), indent=1))
            rows[c][tag] = impact3d.summary(r.metrics)
            typer.echo(f"  {tag} {c}: " + ", ".join(f"{k} {v:.4g}" for k, v in rows[c][tag].items()
                                                  if isinstance(v, float)))
    out = {"cases": rows}
    if all("reference" in v for v in rows.values()):
        md = impact3d.compare_md(rows, spec_mod.load(spec.reference.spec).name, spec.name)
        (run / "impact_compare.md").write_text(md)
        typer.echo(md)
    (run / "impact.json").write_text(json.dumps(_clean(out), indent=1))
    typer.echo(f"run: {run}")


@app.command("ui")
def ui_cmd(port: int = typer.Option(8765), host: str = typer.Option("127.0.0.1")):
    """Local web app: browse projects and runs, compare with the reference, animate every simulation (static
    FEA load ramp, explicit hits, topology iterations), edit specs and launch eval/optimize/topo/impact."""
    from eitri.ui.server import serve

    serve(host, port)


@app.command("faces")
def faces_cmd(spec_path: Path):
    """Numbered map of the design domain's boundary edges (plate side faces) and keep regions: faces.png +
    faces.json, for writing [topology] keep selectors such as edge:12:6."""
    from eitri.topo.run import faces_map

    spec = spec_mod.load(spec_path)
    out = _run_dir(spec_path, "faces")
    d = faces_map(spec, out)
    typer.echo(f"{len(d['edges'])} edges, domain {d['domain_area_mm2']:.0f} mm^2, keep {d['keep_area_mm2']:.0f} mm^2")
    typer.echo(f"run: {out}")


@app.command("topo")
def topo_cmd(
    spec_path: Path,
    no_eval: bool = typer.Option(False, "--no-eval", help="only the density fields and pockets"),
    fine: bool = typer.Option(False, "--fine", help="evaluate the designs on the verification mesh"),
    impact_gate: int = typer.Option(1, help="explicit-hit gate on up to this many passing designs (0 = off)"),
):
    """Topology optimization of the [topology] base's free region, one design per MOI target; each design
    is balanced (Newton), evaluated and compared like any other (spec.toml per design, run it with eval)."""
    from eitri.topo.run import run

    spec = spec_mod.load(spec_path)
    if spec.topology is None:
        raise typer.BadParameter("the spec needs a [topology] section")
    out = _run_dir(spec_path, "topo")
    rows = run(spec, spec_path, out, log=typer.echo)
    if not no_eval:
        for r in rows:
            dspec = spec_mod.load(r["spec"])
            run_d = Path(r["dir"]) / "eval"
            run_d.mkdir(exist_ok=True)
            res = evaluate(dspec, fine=fine, keep_fea_dir=run_d / "fea", full_reference=True)
            o = _write_outputs(run_d, dspec, Path(r["spec"]), [res])
            r["ok"], r["failed"] = res.get("ok", False), res.get("failed", [res.get("error")])
            if "error" not in res:
                r |= {"energy_j": res["impact"]["energy_j"], "mass_kg": res["mass"]["mass_kg"],
                      "imbalance_gmm": res["mass"]["static_imbalance_gmm"]}
                if o.get("compare"):
                    worse = [x["group"] + ": " + x["item"] for x in next(iter(o["compare"].values())) if x["worse"]]
                    r["worse_than_reference"] = worse
            typer.echo(f"  MOI {r['moi_frac']:.3f}: " + _summary(res).replace("\n", "\n    "))
        from eitri.physics.explicit import impact3d

        if impact_gate and spec.reference is not None and impact3d.available():
            # explicit-hit gate on the passing designs, highest MOI first
            for r in sorted((r for r in rows if r.get("ok")), key=lambda r: -r["moi_frac"])[:impact_gate]:
                typer.echo(f"  explicit-hit gate, MOI {r['moi_frac']:.3f}:")
                g = impact3d.gate(spec_mod.load(r["spec"]), None, None, Path(r["dir"]) / "impact", log=typer.echo)
                r["impact_gate"] = g
                r["ok"] = r["ok"] and g["ok"]
        (out / "topo.json").write_text(json.dumps(_clean(rows), indent=2))
    typer.echo(f"run: {out}")


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

"""Spec + design -> every metric and constraint verdict. This is the function the CLI and optimizer share."""

import math
from dataclasses import asdict

from eitri import materials
from eitri.geometry.disc import Disc
from eitri.geometry.fang import Fang
from eitri.geometry.hook import Hook
from eitri.geometry.step_part import StepPart
from eitri.geometry.weapon import Check, Weapon
from eitri.physics import drive, impact
from eitri.physics.fea import analysis as fea
from eitri.physics.fea.ccx import CCX
from eitri.physics.mass import mass_props
from eitri.spec import Spec


WEAPONS = {"disc": Disc, "fang": Fang, "hook": Hook, "step": StepPart}


def single_tooth_bite_mm(spec: Spec) -> float:
    return spec.context.closing_speed_mps * 2 * math.pi / drive.top_speed_rad_s(spec.drive) * 1e3


def build_weapon(spec: Spec, design: dict[str, float]) -> Weapon:
    if spec.weapon.type == "hook" and not design.get("envelope_bite_mm"):
        design = {**design, "envelope_bite_mm": single_tooth_bite_mm(spec)}  # the outline follows the bite
    return WEAPONS[spec.weapon.type].from_dict(design, spec.weapon.spin)


def lame_bore_hoop_mpa(density: float, nu: float, omega: float, r_in_mm: float, r_out_mm: float) -> float:
    """Peak hoop stress (at the bore) of a spinning annular disc, plane stress (Timoshenko)."""
    a, b = r_in_mm * 1e-3, r_out_mm * 1e-3
    return (3 + nu) / 4 * density * omega**2 * (b**2 + (1 - nu) / (3 + nu) * a**2) * 1e-6


def evaluate(spec: Spec, design: dict[str, float] | None = None, material: str | None = None,
             weapon: Weapon | None = None, fine: bool = False, keep_fea_dir=None) -> dict:
    design = design if design is not None else spec.fixed_design()
    mat = materials.get(material or spec.material.names()[0])
    c = spec.constraints

    checks: list[Check] = []
    try:
        weapon = weapon or build_weapon(spec, design)
        checks += weapon.geometric_checks(c.min_ligament_mm)
        checks += weapon.style_checks(spec.style)
        solid = weapon.solid()
    except Exception as e:  # invalid parameter combination: report it, don't crash the loop
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "design": design, "material": mat.name}
    checks.append(Check("solid_valid", solid.is_valid))

    mp = mass_props(solid, mat.density)
    dr = drive.simulate(spec.drive, mp.moi_kgm2)
    p = weapon.params
    hit = impact.analyze(mp.moi_kgm2, dr.top_rad_s, weapon.tip_radius_mm(), weapon.n_teeth(),
                         weapon.tooth_depth_mm(), weapon.hook_deg(), spec.context)

    # Analytic cross-check: solid annulus out to the tip radius (ignores teeth/pockets).
    hoop = lame_bore_hoop_mpa(mat.density, mat.nu, dr.top_rad_s, getattr(p, "bore_mm", 0.0) / 2,
                              weapon.tip_radius_mm())
    lame = {"centrifugal_bore_hoop_mpa": hoop}
    fea_run = None
    if spec.analysis.fea and CCX:
        od = 2 * weapon.tip_radius_mm()
        size = od / 75 if fine else (spec.analysis.mesh_mm or od / 25)
        try:
            fea_run = fea.run(weapon, mat.E, mat.nu, mat.density, dr.top_rad_s, hit.peak_force_kn * 1e3,
                              hit.bite_mm, size, threads=spec.analysis.threads, keep_dir=keep_fea_dir)
        except Exception as e:
            return {"ok": False, "error": f"FEA failed: {type(e).__name__}: {e}", "design": design,
                    "material": mat.name}
        stress = fea.summarize(fea_run, mat.yield_, hit.impulse_ns)
        stress["safety_factor_yield"] = stress["centrifugal"]["safety_factor_body"]
    else:
        stress = {"method": "lame_annulus_estimate" + ("" if CCX else " (ccx not installed)"),
                  "safety_factor_yield": mat.yield_ / hoop if hoop > 0 else math.inf}
    stress["analytic"] = lame

    if c.mass_kg.max is not None:
        checks.append(Check("mass_max", mp.mass_kg <= c.mass_kg.max, mp.mass_kg, c.mass_kg.max))
    if c.mass_kg.min is not None:
        checks.append(Check("mass_min", mp.mass_kg >= c.mass_kg.min, mp.mass_kg, c.mass_kg.min))
    checks.append(Check("imbalance", mp.static_imbalance_gmm <= c.max_imbalance_gmm,
                        mp.static_imbalance_gmm, c.max_imbalance_gmm, "static imbalance g*mm"))
    if c.max_spinup_s is not None and dr.spinup_s_90 is not None:
        checks.append(Check("spinup_90", dr.spinup_s_90 <= c.max_spinup_s, dr.spinup_s_90, c.max_spinup_s))
    checks.append(Check("safety_factor_spinning", stress["safety_factor_yield"] >= c.min_safety_factor,
                        stress["safety_factor_yield"], c.min_safety_factor, stress["method"]))
    if c.min_impact_safety_factor is not None and stress.get("impact_safety_factor") is not None:
        checks.append(Check("impact_safety_factor", stress["impact_safety_factor"] >= c.min_impact_safety_factor,
                            stress["impact_safety_factor"], c.min_impact_safety_factor,
                            "hit force at first body yield (while spinning) / estimated peak hit force"))

    objective = {"energy": hit.energy_j, "moi": mp.moi_kgm2, "effective_mass": hit.effective_mass_kg}
    return {
        "ok": all(ch.ok for ch in checks),
        "failed": [ch.name for ch in checks if not ch.ok],
        "objective": {"name": spec.objective.maximize, "value": objective[spec.objective.maximize]},
        "material": mat.name,
        "design": weapon.param_dict(),
        "derived": weapon.derived(),
        "mass": mp.to_dict(),
        "drive": dr.to_dict(),
        "impact": hit.to_dict(),
        "stress": stress,
        "checks": [asdict(ch) for ch in checks],
        "_weapon": weapon,
        "_solid": solid,
        "_spinup_curve": dr.spinup_curve,
        "_fea": fea_run,
    }


def public(result: dict) -> dict:
    """Result without the in-memory objects (what goes into report.json)."""
    return {k: v for k, v in result.items() if not k.startswith("_")}

"""Spec + design -> every metric and constraint verdict. This is the function the CLI and optimizer share."""

import math
from dataclasses import asdict

from eitri import materials
from eitri.geometry.common import worst_concave_corner
from eitri.geometry.disc import Disc
from eitri.geometry.fang import Fang
from eitri.geometry.hook import Hook
from eitri.geometry.step_part import StepPart
from eitri.geometry.topo import Topo
from eitri.geometry.weapon import Check, Weapon
from eitri.physics import drive, impact, shaft
from eitri.physics.fea import analysis as fea
from eitri.physics.fea.ccx import CCX
from eitri.physics.mass import mass_props
from eitri.spec import Spec


WEAPONS = {"disc": Disc, "fang": Fang, "hook": Hook, "step": StepPart, "topo": Topo}


def single_tooth_bite_mm(spec: Spec) -> float:
    return spec.context.closing_speed_mps * 2 * math.pi / drive.top_speed_rad_s(spec.drive) * 1e3


def build_weapon(spec: Spec, design: dict[str, float]) -> Weapon:
    if spec.weapon.type == "hook" and not design.get("envelope_bite_mm"):
        design = {**design, "envelope_bite_mm": single_tooth_bite_mm(spec)}  # the outline follows the bite
    if spec.machining is not None and spec.machining.any():
        from eitri.geometry import machining

        return machining.build(spec, WEAPONS[spec.weapon.type], design)
    return WEAPONS[spec.weapon.type].from_dict(design, spec.weapon.spin)


def lame_bore_hoop_mpa(density: float, nu: float, omega: float, r_in_mm: float, r_out_mm: float) -> float:
    """Peak hoop stress (at the bore) of a spinning annular disc, plane stress (Timoshenko)."""
    a, b = r_in_mm * 1e-3, r_out_mm * 1e-3
    return (3 + nu) / 4 * density * omega**2 * (b**2 + (1 - nu) / (3 + nu) * a**2) * 1e-6


def evaluate(spec: Spec, design: dict[str, float] | None = None, material: str | None = None,
             weapon: Weapon | None = None, fine: bool = False, keep_fea_dir=None,
             full_reference: bool = False) -> dict:
    """`full_reference`: run the reference load cases even if other checks fail (for reports)."""
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
    sharp = worst_concave_corner(weapon.profile())
    checks.append(Check("sharp_corners", sharp is None or sharp.turn_deg <= c.max_sharp_corner_deg,
                        sharp.turn_deg if sharp else 0.0, c.max_sharp_corner_deg,
                        f"sharpest re-entrant corner (at {sharp.x:.1f}, {sharp.y:.1f} mm); stress raiser unless "
                        "filleted" if sharp else "no re-entrant corners"))

    mp = mass_props(solid, mat.density, weapon.plate(), getattr(weapon, "mass_parts", lambda: None)())
    dr = drive.simulate(spec.drive, mp.moi_kgm2)
    hit = impact.analyze(mp.moi_kgm2, dr.top_rad_s, weapon.tip_radius_mm(), weapon.n_teeth(),
                         weapon.tooth_depth_mm(), weapon.hook_deg(), spec.context)

    # Analytic cross-check: solid annulus out to the tip radius (ignores teeth/pockets).
    hoop = lame_bore_hoop_mpa(mat.density, mat.nu, dr.top_rad_s, weapon.bore_r_mm(), weapon.tip_radius_mm())
    lame = {"centrifugal_bore_hoop_mpa": hoop}
    fea_run = None
    if spec.analysis.fea and CCX:
        od = 2 * weapon.tip_radius_mm()
        size = od / 75 if fine else (spec.analysis.mesh_mm or od / 25)
        try:
            fea_run = fea.run(weapon, mat.E, mat.nu, mat.density, dr.top_rad_s, hit.peak_force_kn * 1e3,
                              hit.bite_mm, size, threads=spec.analysis.threads, keep_dir=keep_fea_dir,
                              hub_support=spec.analysis.hub_support)
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

    # loads on the rest of the robot, and damage tolerance at the hot spot
    gyro = shaft.gyro_moment_nm(mp.moi_kgm2, dr.top_rad_s, spec.context.tilt_rate_dps)
    robot = {"gyro_moment_nm": gyro, "tilt_rate_dps": spec.context.tilt_rate_dps,
             "shaft_peak_force_kn": hit.peak_force_kn}
    if spec.shaft is not None:
        sh = spec.shaft
        off = sh.weapon_offset_mm if sh.weapon_offset_mm is not None else sh.bearing_span_mm / 2
        robot["shaft"] = shaft.shaft_loads(hit.peak_force_kn * 1e3, gyro, sh.bearing_span_mm, off, sh.d_mm)
        sf = sh.yield_mpa / robot["shaft"]["bending_stress_mpa"]
        robot["shaft"]["safety_factor"] = sf
        checks.append(Check("shaft_bending", sf >= sh.min_safety_factor, sf, sh.min_safety_factor,
                            "shaft yield / bending stress at the estimated peak hit + gyro couple"))
    k_ic = shaft.k_ic_mpa_sqrt_m(mat.charpy_j, mat.yield_)
    peak = stress.get("impact_at_estimated_peak", {}).get("body_max_vm_mpa")
    robot["fracture"] = {"k_ic_mpa_sqrt_m": k_ic, "hot_spot_stress_mpa": peak,
                         "critical_crack_mm": shaft.critical_crack_mm(k_ic, peak) if peak else None}
    jt = stress.get("joint")
    if jt and jt["hub_support"] == "bolts":
        bolts = {k: v for k, v in jt["at_estimated_peak_kn"].items() if k.startswith("bolt")}
        if bolts:
            j = spec.joint
            holes = {f"bolt{i}": r for i, (_, _, r) in enumerate(weapon.support_circles())}
            worst = max(bolts, key=bolts.get)
            d_minor = j.bolt_minor_d_mm or 0.78 * 2 * holes[worst]
            cap_kn = 0.6 * j.bolt_proof_mpa * math.pi * d_minor**2 / 4 * 1e-3
            sf = cap_kn / bolts[worst] if bolts[worst] > 0 else math.inf
            jt["bolt_shear_capacity_kn"], jt["bolt_safety_factor"] = cap_kn, sf
            checks.append(Check("bolt_shear", sf >= j.min_bolt_safety_factor, sf, j.min_bolt_safety_factor,
                                f"{worst} carries {bolts[worst]:.1f} kN at the estimated peak hit; shear capacity "
                                f"{cap_kn:.1f} kN (minor d {d_minor:.2f} mm, proof {j.bolt_proof_mpa:.0f} MPa)"))
    if c.min_energy_j is not None:
        checks.append(Check("energy_min", hit.energy_j >= c.min_energy_j, hit.energy_j, c.min_energy_j))

    ref_profile = None
    if spec.reference is not None:
        from eitri import reference

        ref = reference.reference_metrics(spec.reference.spec, fine, spec.analysis.hub_support)
        mine = reference.cheap_metrics(weapon, hit.energy_j, mp.to_dict())
        ref_profile = {"design": mine, "reference": ref}
        checks += reference.compare_cheap(mine, ref, spec.reference.match_mass)
        # the expensive load cases only for designs that pass everything else (or when asked: reports)
        if fea_run is not None and (full_reference or all(ch.ok for ch in checks)):
            partial = {"_weapon": weapon, "_fea": fea_run, "drive": dr.to_dict(), "impact": hit.to_dict()}
            mine |= reference.case_metrics(partial, mat, fine)
            checks += reference.compare_cases(mine, ref)
        else:
            checks.append(Check("ref_profile", False, None, None,
                                "load cases not compared with the reference: " +
                                ("other checks fail first" if fea_run is not None else "needs the FEA")))
    objective = {"energy": hit.energy_j, "moi": mp.moi_kgm2, "effective_mass": hit.effective_mass_kg,
                 "mass": mp.mass_kg}
    o = spec.objective
    return {
        "ok": all(ch.ok for ch in checks),
        "failed": [ch.name for ch in checks if not ch.ok],
        "objective": {"name": o.name, "value": objective[o.name], "score": o.sign * objective[o.name]},
        "material": mat.name,
        "design": weapon.param_dict(),
        "derived": weapon.derived(),
        "mass": mp.to_dict(),
        "drive": dr.to_dict(),
        "impact": hit.to_dict(),
        "stress": stress,
        "robot": robot,
        "checks": [asdict(ch) for ch in checks],
        "reference_profile": ref_profile,
        "_weapon": weapon,
        "_solid": solid,
        "_spinup_curve": dr.spinup_curve,
        "_fea": fea_run,
    }


def public(result: dict) -> dict:
    """Result without the in-memory objects (what goes into report.json)."""
    return {k: v for k, v in result.items() if not k.startswith("_")}

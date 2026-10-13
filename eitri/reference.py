"""Compare a design's failure profile with a reference weapon (e.g. the user's proven Rebot V3).

The rule is relative: the new design must not lose anything against the reference. Every load case is run on
both with the same code and mesh settings, and in every case each region of the weapon (tip, tooth, arm, hub)
must take at least the force at which the same region of the reference first yields (capped at 1.25x the
reference's first-yield force: past that the reference itself is yielding). That covers "how strong"
and "where it gives first": no spot may become weaker than the same spot on the reference.

Load cases (all first-yield forces while spinning, linear elastic, so they rank designs rather than predict
fracture):
    hit_bite    in-plane hit on the outer `bite` of the face, opponent at rest (the standard impact case)
    hit_tip3    in-plane hit on the outer 3 mm of the face (tooth-to-tooth, tip on tip)
    hit_tip8    in-plane hit on the outer 8 mm of the face
    flat_hit    3D: force normal to the plate on the top face near the tip (vertical spinner, wedge, floor)
plus geometry: the tooth must be at least as wide as the reference's at every depth below the tip, and
stored energy at least the reference's.
"""

import hashlib
import json
import math
from pathlib import Path

import build123d as bd
import numpy as np

from eitri.geometry.weapon import Check
from eitri.physics.fea import analysis as fea
from eitri.physics.fea.solid3d import oop_run, yield_scale

METRICS_VERSION = 1
TOOTH_DEPTHS_MM = (3, 5, 10, 15, 20, 25, 30)
REGIONS = ("tip", "tooth", "arm", "hub")
REGION_CAP = 1.25  # regions are compared up to this multiple of the reference's first-yield force
CASES = ("hit_bite", "hit_tip3", "hit_tip8", "flat_hit")
FORCE_RANKED = tuple("ref_" + c for c in CASES)  # checks the coarse-mesh margin applies to


def regions(weapon, xy: np.ndarray) -> dict[str, np.ndarray]:
    """Node masks by radius: tip (outer 6 mm), tooth (down to 12 mm below the face root), hub (< 0.4 R)."""
    r = np.hypot(xy[:, 0], xy[:, 1])
    R, depth = weapon.tip_radius_mm(), weapon.tooth_depth_mm()
    tip = r > R - 6
    tooth = ~tip & (r > R - depth - 12)
    hub = r < 0.4 * R
    return {"tip": tip, "tooth": tooth, "arm": ~(tip | tooth | hub), "hub": hub}


def region_of(weapon, x: float, y: float) -> str:
    masks = regions(weapon, np.array([[x, y]]))
    return next(k for k in REGIONS if masks[k][0])


def _profile(weapon, xy, cf_stress, hit_stress, hit_force_n, body, yield_mpa) -> dict:
    masks = regions(weapon, xy)
    per = {k: yield_scale(cf_stress, hit_stress, body & m, yield_mpa) * hit_force_n * 1e-3 for k, m in masks.items()}
    weakest = min(per, key=per.get)
    vm = fea._vm(cf_stress + per[weakest] * 1e3 / hit_force_n * hit_stress)
    i = int(np.argmax(np.where(body, vm, -np.inf)))
    return {"yield_force_kn": per[weakest], "region": weakest,
            "at_mm": [round(float(v), 2) for v in xy[i]], "regions_kn": per}


def _plane_case(weapon, mat, omega, contact_mm, size, run=None) -> dict:
    if run is None:
        run = fea.run(weapon, mat.E, mat.nu, mat.density, omega, 1e4, contact_mm, size)
    cf, imp = run.cases
    return _profile(weapon, run.mesh.nodes, cf.result.stress, imp.result.stress, run.applied_force_n,
                    cf.zones["body"], mat.yield_)


def tooth_widths(weapon) -> dict[str, float]:
    """Material along an arc at each depth below the tip, from 15 deg ahead of the tip to 45 deg behind."""
    face = weapon.profile()
    tx, ty = weapon.impact_face()[0]
    a_tip = math.atan2(ty, tx)
    back = -1 if weapon.spin == "ccw" else 1  # direction of "behind the tip"
    out = {}
    for d in TOOTH_DEPTHS_MM:
        r = weapon.tip_radius_mm() - d
        a0, a1 = a_tip - back * math.radians(15), a_tip + back * math.radians(45)
        arc = bd.ThreePointArc(*[(r * math.cos(a), r * math.sin(a)) for a in (a0, (a0 + a1) / 2, a1)])
        cut = arc & face
        out[str(d)] = float(sum(e.length for e in cut.edges())) if cut is not None else 0.0
    return out


def metrics(result: dict, mat, fine: bool) -> dict:
    """Failure profile of an evaluated design (`evaluate(...)` result with its in-memory FEA run)."""
    w, run = result["_weapon"], result["_fea"]
    omega = result["drive"]["top_rad_s"]
    size = 2 * w.tip_radius_mm() / (75 if fine else 25)
    oop = oop_run(w, mat.E, mat.nu, mat.density, omega, size_mm=2.0 if fine else 3.0)
    return {
        "energy_j": result["impact"]["energy_j"],
        "tooth_width_mm": tooth_widths(w),
        "cases": {
            "hit_bite": _plane_case(w, mat, omega, None, None, run),
            "hit_tip3": _plane_case(w, mat, omega, 3.0, size),
            "hit_tip8": _plane_case(w, mat, omega, 8.0, size),
            "flat_hit": _profile(w, oop.nodes[:, :2], oop.centrifugal.stress, oop.unit_hit.stress, 1.0, oop.body,
                                 mat.yield_),
        },
    }


def reference_metrics(path: str, fine: bool) -> dict:
    """Metrics of the reference spec's design, cached next to it (keyed by its text and this code's version)."""
    from eitri import materials
    from eitri.evaluate import evaluate
    from eitri.spec import load

    src = Path(path)
    key = hashlib.sha1(f"{src.read_text()}|{METRICS_VERSION}|{fine}".encode()).hexdigest()
    cache = src.with_name(".eitri_reference.json")
    data = json.loads(cache.read_text()) if cache.exists() else {}
    if key not in data:
        spec = load(src)
        mat = materials.get(spec.material.names()[0])
        r = evaluate(spec, fine=fine)
        if "error" in r:
            raise RuntimeError(f"reference {path} doesn't evaluate: {r['error']}")
        data[key] = metrics(r, mat, fine)
        cache.write_text(json.dumps(data, indent=1))
    return data[key]


def compare(mine: dict, ref: dict) -> list[Check]:
    """One check per criterion; values are ratios to the reference (>= 1 passes)."""
    checks = [Check("ref_energy", mine["energy_j"] >= ref["energy_j"], mine["energy_j"], ref["energy_j"],
                    "stored energy vs the reference")]
    ratios = {d: mine["tooth_width_mm"][d] / ref["tooth_width_mm"][d] for d in ref["tooth_width_mm"]
              if ref["tooth_width_mm"][d] > 0}
    d = min(ratios, key=ratios.get)
    checks.append(Check("ref_tooth_width", ratios[d] >= 1.0, ratios[d], 1.0,
                        f"tooth width / reference at every depth; narrowest {d} mm below the tip: "
                        f"{mine['tooth_width_mm'][d]:.1f} vs {ref['tooth_width_mm'][d]:.1f} mm"))
    for case in CASES:
        m, r = mine["cases"][case], ref["cases"][case]
        # each region must hold what the same region of the reference holds, up to 1.25x the reference's first
        # yield: beyond that the reference is already yielding, and its linear-elastic numbers there say little
        need = {k: min(r["regions_kn"][k], REGION_CAP * r["yield_force_kn"]) for k in REGIONS}
        rr = {k: m["regions_kn"][k] / need[k] for k in REGIONS if math.isfinite(need[k]) and need[k] > 0}
        k = min(rr, key=rr.get)
        checks.append(Check(f"ref_{case}", rr[k] >= 1.0, rr[k], 1.0,
                            f"first yield {m['yield_force_kn']:.2f} kN in the {m['region']} (reference "
                            f"{r['yield_force_kn']:.2f} kN in the {r['region']}); weakest vs reference: {k} "
                            f"{m['regions_kn'][k]:.2f} vs {need[k]:.2f} kN needed"))
    return checks

"""3D out-of-plane hit: something strikes the flat of the tooth (a vertical spinner, a wedge, the floor).

The plane-stress model can't see this, so the solid is meshed with second-order tetrahedra (C3D10) and loaded
by a force normal to the plate, spread over the top face near the tip; the hub is held at the bolt holes
(the pulley clamps there). Spinning stress (CENTRIF) is superposed and the hit force at first yield in the
body is found by bisection, as in the 2D impact case. The material right under the patch (local denting)
and around the bolts (support artefacts) is excluded.

Linear elastic, quasi-static: like the 2D cases, it ranks designs (first-yield force and where it happens);
it does not predict fracture.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import build123d as bd
import gmsh
import numpy as np

from eitri.physics.fea.ccx import Result, run_job

# gmsh 10-node tet -> CalculiX C3D10: midsides (3,1) and (3,2) come in the other order
_GMSH_TO_CCX = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]
_FLIP = [0, 2, 1, 3, 6, 5, 4, 7, 9, 8]  # swap corners 1, 2 (and their midsides): fixes a negative volume


@dataclass
class Mesh3D:
    nodes: np.ndarray  # (n, 3)
    elems: np.ndarray  # (m, 10) 1-based, CalculiX order


def mesh_solid(solid: bd.Solid, size_mm: float, refine: tuple[float, float, float, float] | None = None
               ) -> Mesh3D:
    """`refine` = (x, y, size, radius): finer elements around a point (any z)."""
    with tempfile.TemporaryDirectory() as tmp:
        brep = Path(tmp) / "solid.brep"
        bd.export_brep(solid, str(brep))
        gmsh.initialize(interruptible=False)
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("solid")
            gmsh.model.occ.importShapes(str(brep))
            gmsh.model.occ.synchronize()
            gmsh.option.setNumber("Mesh.MeshSizeMax", size_mm)
            gmsh.option.setNumber("Mesh.MeshSizeMin", refine[2] if refine else size_mm / 4)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
            gmsh.option.setNumber("Mesh.ElementOrder", 2)
            # straight-sided midsides: curving them onto small fillets (the R1 hub lock) can invert tets
            # ("nonpositive jacobian"); at these element sizes the geometric error is negligible
            gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)
            if refine:
                x, y, s, r = refine
                f = gmsh.model.mesh.field.add("MathEval")
                d = f"Sqrt((x-({x}))^2+(y-({y}))^2)"
                gmsh.model.mesh.field.setString(f, "F", f"Min({size_mm},Max({s},{s}+({size_mm}-{s})*({d}-{r})/{3 * r}))")
                gmsh.model.mesh.field.setAsBackgroundMesh(f)
            gmsh.model.mesh.generate(3)
            tags, coords, _ = gmsh.model.mesh.getNodes()
            renum = np.zeros(int(tags.max()) + 1, dtype=np.int64)
            renum[tags.astype(np.int64)] = np.arange(1, len(tags) + 1)
            xyz = coords.reshape(-1, 3)[np.argsort(renum[tags.astype(np.int64)])]
            types, _, enodes = gmsh.model.mesh.getElements(3)
            tet10 = [n for t, n in zip(types, enodes) if t == 11]
            if not tet10:
                raise RuntimeError("gmsh produced no 10-node tetrahedra")
            elems = renum[tet10[0].astype(np.int64)].reshape(-1, 10)[:, _GMSH_TO_CCX]
        finally:
            gmsh.finalize()
    c = xyz[elems[:, :4] - 1]
    vol = np.einsum("ij,ij->i", np.cross(c[:, 1] - c[:, 0], c[:, 2] - c[:, 0]), c[:, 3] - c[:, 0])
    elems[vol < 0] = elems[vol < 0][:, _FLIP]
    # keep only nodes used by the tets (gmsh also returns vertex/edge nodes of lower dimensions: all used)
    return Mesh3D(nodes=xyz, elems=elems)


def _deck(mesh: Mesh3D, E: float, nu: float, density_kg_m3: float, fixed: np.ndarray, name: str,
          omega: float = 0.0, cloads: list[tuple[int, int, float]] = ()) -> str:
    L = ["*HEADING", f"eitri {name}", "*NODE, NSET=Nall"]
    L += [f"{i + 1},{x:.9g},{y:.9g},{z:.9g}" for i, (x, y, z) in enumerate(mesh.nodes)]
    L.append("*ELEMENT, TYPE=C3D10, ELSET=Eall")
    L += [f"{i + 1}," + ",".join(map(str, e)) for i, e in enumerate(mesh.elems)]
    L += ["*MATERIAL, NAME=STEEL", "*ELASTIC", f"{E},{nu}", "*DENSITY", f"{density_kg_m3 * 1e-12:.6e}",
          "*SOLID SECTION, ELSET=Eall, MATERIAL=STEEL", "*BOUNDARY"]
    L += [f"{n},1,3,0." for n in fixed]
    L += ["*STEP", "*STATIC"]
    if omega:
        L += ["*DLOAD", f"Eall,CENTRIF,{omega**2:.9g},0.,0.,0.,0.,0.,1."]
    if cloads:
        L.append("*CLOAD")
        L += [f"{n},{d},{v:.9g}" for n, d, v in cloads]
    L += ["*NODE FILE", "U", "*EL FILE", "S", "*END STEP"]
    return "\n".join(L) + "\n"


def _vm(stress: np.ndarray) -> np.ndarray:
    sx, sy, sz, txy, tyz, tzx = stress.T
    return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * (txy**2 + tyz**2 + tzx**2))


@dataclass
class OopRun:
    nodes: np.ndarray  # (n, 3)
    centrifugal: Result
    unit_hit: Result  # 1 N total, normal to the plate
    body: np.ndarray  # bool mask: nodes judged (not under the patch, not at the bolts)


def oop_run(weapon, E: float, nu: float, density_kg_m3: float, omega_rad_s: float, size_mm: float = 3.0,
            patch_mm: float = 8.0, keep_dir: Path | None = None) -> OopRun:
    """Spinning + a force normal to the plate on the top face within `patch_mm` of the tooth tip."""
    solid = weapon.solid()
    tip = np.array(weapon.impact_face()[0], float)
    mesh = mesh_solid(solid, size_mm, refine=(tip[0], tip[1], patch_mm / 5, patch_mm + 2))
    xy, z = mesh.nodes[:, :2], mesh.nodes[:, 2]
    d_tip = np.hypot(*(xy - tip).T)

    fixed = np.zeros(len(xy), bool)
    support = np.zeros(len(xy), bool)
    for cx, cy, r in weapon.support_circles():
        d = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
        fixed |= np.abs(d - r) < 1e-3
        support |= d < r + max(r, 3.0)
    fixed_ids = np.nonzero(fixed)[0] + 1
    if not len(fixed_ids):
        raise RuntimeError("3D mesh: no nodes on the bolt holes")
    top = np.abs(z - z.max()) < 1e-6
    patch = np.nonzero(top & (d_tip < patch_mm))[0]
    if len(patch) < 3:
        raise RuntimeError("3D mesh: no top-face nodes under the out-of-plane patch")
    unit = [(int(i) + 1, 3, -1.0 / len(patch)) for i in patch]  # 1 N total, into the plate
    cf = run_job(_deck(mesh, E, nu, density_kg_m3, fixed_ids, "oop_centrifugal", omega=omega_rad_s),
                 len(xy), "oop_centrifugal", keep_dir=keep_dir)
    hit = run_job(_deck(mesh, E, nu, density_kg_m3, fixed_ids, "oop_hit", cloads=unit),
                  len(xy), "oop_hit", keep_dir=keep_dir)
    return OopRun(mesh.nodes, cf, hit, ~(support | (d_tip < patch_mm + 4.0)))


def yield_scale(cf_stress: np.ndarray, hit_stress: np.ndarray, mask: np.ndarray, yield_mpa: float) -> float:
    """Largest multiple of the hit stress that keeps von Mises (with spinning) below yield on `mask`."""
    if not mask.any():
        return float("inf")

    def peak(scale: float) -> float:
        return float(_vm(cf_stress[mask] + scale * hit_stress[mask]).max())

    if peak(0.0) >= yield_mpa:
        return 0.0
    lo, hi = 0.0, 1.0
    while peak(hi) < yield_mpa:
        lo, hi = hi, hi * 2
        if hi > 1e12:
            return float("inf")
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if peak(mid) < yield_mpa else (lo, mid)
    return lo


def out_of_plane(weapon, E: float, nu: float, density_kg_m3: float, omega_rad_s: float, yield_mpa: float,
                 size_mm: float = 3.0, patch_mm: float = 8.0, keep_dir: Path | None = None) -> dict:
    """Out-of-plane hit force at first body yield, and where."""
    run = oop_run(weapon, E, nu, density_kg_m3, omega_rad_s, size_mm, patch_mm, keep_dir)
    s = yield_scale(run.centrifugal.stress, run.unit_hit.stress, run.body, yield_mpa)
    vm = _vm(run.centrifugal.stress + s * run.unit_hit.stress)
    i = int(np.argmax(np.where(run.body, vm, -np.inf)))
    return {"yield_force_kn": s * 1e-3, "yield_location_mm": [round(float(v), 2) for v in run.nodes[i]],
            "mesh_nodes": len(run.nodes)}

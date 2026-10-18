"""Explicit 3D impact: the spinning weapon hits an opponent, solved by OpenCourant (OpenRadioss fork).

What the static model has to assume, this one computes: the contact-force pulse (peak, duration, impulse),
the energy actually transferred, and where the weapon yields, how much it deforms plastically and whether
any of it fails, with the spinning stress and the deceleration of the whole rotor included.

Model (units mm, ms, kg -> kN, GPa, J; see radioss.py):
    weapon    10-node tets (gmsh), refined at the tooth; Johnson-Cook plasticity (materials.johnson_cook())
              with Johnson-Cook damage at a constant fracture strain (failed elements are deleted)
    hub       a rigid body on the hub hole walls (pulley lock + bolts), free to turn about Z only, carrying the
              drive's reflected inertia (motor rotor x ratio^2)
    spin      the weapon and hub start at top speed (no prestress: the centrifugal field settles in the first
              ~0.05 ms, before contact)
    opponent  a block of the opponent's mass, closing on the axis at the closing speed; its face stands just
              ahead of the tooth, engaging `contact_mm` of the face: the bite (`bite`) or the outer 3 mm (`tip3`,
              tooth-to-tooth). Its front `skin_mm` is deformable (Johnson-Cook, `opponent_material`, default the
              weapon's own steel: tooth-to-tooth is steel on steel) on a rigid backing that carries the mass.
              A fully rigid opponent puts all the deformation in the tooth (it eroded ~1500 elements of Rebot's
              tip in 0.03 ms): pessimistic in absolute terms, though still fair for comparisons.
    contact   node-to-surface penalty (TYPE7), weapon nodes near the tooth against the block, friction 0.2

Linear-elastic/static results rank designs by first-yield force; this one shows what happens past it. Both
judge the same physics, so a design that yields later statically should also strain less here.
"""

import csv
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import build123d as bd
import numpy as np

from eitri.physics.explicit import radioss
from eitri.physics.fea.solid3d import mesh_solid

CASES = {"bite": None, "tip3": 3.0}  # contact depth on the face (None = the bite)


@dataclass
class ImpactSetup:
    case: str = "bite"
    mesh_mm: float = 3.0
    tip_mesh_mm: float = 1.0
    gap_mm: float = 1.5  # arc between the tip and the opponent's face at t = 0
    tstop_ms: float = 1.0
    anim_dt_ms: float = 0.01
    friction: float = 0.2
    threads: int = 8
    # mass scaling floor for the time step: gmsh leaves a few tiny tets at the hub's R1 lock fillets (inside the
    # rigid hub) that would set a ~5 ns step; 30 ns adds well under 1 % of mass (reported as added_mass_pct)
    dt_min_ms: float = 3e-5
    skin_mm: float = 10.0
    opponent_material: str | None = None  # default: the weapon's material
    # "steel": the failure test (a hardened-steel face on a rigid backing); "tpu": a 3D-printed TPU wall
    # (eitri/physics/explicit/tpu.py), what most beetleweights actually wear
    opponent: str = "steel"
    tpu: object = None  # tpu.TPUWall; default TPUWall()


@dataclass
class ImpactResult:
    folder: Path
    ok: bool
    metrics: dict = field(default_factory=dict)
    log: str = ""


def _box_mesh(lo, hi, h: float):
    """Hex mesh of an axis-aligned box in a local frame: nodes (n, 3), bricks (m, 8) 0-based."""
    ns = [max(1, int(math.ceil((b - a) / h))) for a, b in zip(lo, hi)]
    return _grid_mesh(*[np.linspace(a, b, k + 1) for a, b, k in zip(lo, hi, ns)])


def _grid_mesh(gx, gy, gz):
    """Hex mesh on the tensor grid gx x gy x gz: nodes (n, 3), bricks (m, 8) 0-based (i, j, k loop order)."""
    g = [np.asarray(gx), np.asarray(gy), np.asarray(gz)]
    ns = [len(a) - 1 for a in g]
    X, Y, Z = np.meshgrid(*g, indexing="ij")
    nodes = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    idx = np.arange(len(nodes)).reshape(X.shape)
    el = []
    for i in range(ns[0]):
        for j in range(ns[1]):
            for k in range(ns[2]):
                el.append([idx[i, j, k], idx[i + 1, j, k], idx[i + 1, j + 1, k], idx[i, j + 1, k],
                           idx[i, j, k + 1], idx[i + 1, j, k + 1], idx[i + 1, j + 1, k + 1], idx[i, j + 1, k + 1]])
    return nodes, np.array(el)


def build(weapon, mat, omega_rad_s: float, opponent_mass_kg: float, closing_mps: float, bite_mm: float,
          drive_inertia_kgm2: float, setup: ImpactSetup, folder: Path, name: str = "hit") -> dict:
    """Write the deck; returns the geometry bookkeeping the post-processing needs."""
    t = weapon.params.thickness_mm
    tip = np.array(weapon.impact_face()[0], float)
    R = float(np.hypot(*tip))
    s = 1.0 if getattr(weapon, "spin", "ccw") == "ccw" else -1.0
    c_len = CASES[setup.case] or bite_mm

    # --- weapon mesh -------------------------------------------------------------------------------
    m = mesh_solid(weapon.solid(), setup.mesh_mm, refine=(tip[0], tip[1], setup.tip_mesh_mm, 12.0))
    n_w = len(m.nodes)
    xy, z = m.nodes[:, :2], m.nodes[:, 2]
    r = np.hypot(xy[:, 0], xy[:, 1])

    # --- opponent block, in the frame of the hit point ---------------------------------------------
    a_tip = math.atan2(tip[1], tip[0])
    a_hit = a_tip + s * setup.gap_mm / R
    e_r = np.array([math.cos(a_hit), math.sin(a_hit), 0.0])
    e_t = s * np.array([-math.sin(a_hit), math.cos(a_hit), 0.0])  # the way the tooth moves
    e_z = np.array([0.0, 0.0, 1.0])
    def lin(a, b, h):
        return np.linspace(a, b, max(1, int(math.ceil((b - a) / h))) + 1)

    from eitri import materials

    omat = materials.get(setup.opponent_material) if setup.opponent_material else mat
    if setup.opponent == "tpu":
        from eitri.physics.explicit.tpu import TPUWall

        wall = setup.tpu or TPUWall()
        depth_r, half_z = 40.0, max(20.0, t)
        sk, co = wall.skin_mm, wall.core_mm
        gy = np.unique(np.concatenate([lin(0, sk, 0.5), lin(sk, sk + co, 1.5), lin(sk + co, 2 * sk + co, 0.5)]))
        loc, br = _grid_mesh(lin(R - c_len, R - c_len + depth_r, 1.5), gy, lin(-half_z, half_z, 2.0))
        thick_t = wall.thickness_mm
        yc = loc[br][:, :, 1].mean(axis=1)
        epart = np.where((yc < sk) | (yc > sk + co), 2, 3)  # skins: part 2, core: part 3
        backing = np.abs(loc[:, 1] - thick_t) < 1e-9  # the robot's mass hangs on the back skin
        vol_skin = 2 * sk * depth_r * 2 * half_z
        box_mass = wall.rho_s * 1e-9 * (vol_skin + wall.infill * co * depth_r * 2 * half_z)
    else:
        depth_r, thick_t, half_z = 30.0, 20.0, max(15.0, t)
        loc, br = _box_mesh((R - c_len, 0.0, -half_z), (R + depth_r - c_len, thick_t, half_z), 2.0)
        epart = np.full(len(br), 2)
        backing = loc[:, 1] >= setup.skin_mm - 1e-9  # rigid part of the opponent (behind the skin)
        box_mass = omat.density * 1e-9 * depth_r * thick_t * 2 * half_z
    box = loc[:, :1] * e_r + loc[:, 1:2] * e_t + loc[:, 2:3] * e_z
    if s < 0:  # the local frame is left-handed for CW spin: keep the bricks right-handed
        br = br[:, [0, 3, 2, 1, 4, 7, 6, 5]]
    n_b = len(box)
    rb_mass = max(opponent_mass_kg - box_mass, 0.05 * opponent_mass_kg)
    a_, b_, c_ = depth_r, thick_t, 2 * half_z
    J_box = opponent_mass_kg / 12 * np.array([b_**2 + c_**2, a_**2 + c_**2, a_**2 + b_**2])  # kg mm^2
    box_c = ((R - c_len + depth_r / 2) * e_r + thick_t / 2 * e_t)

    hub_id, opp_id = n_w + n_b + 1, n_w + n_b + 2
    d = radioss.Deck(name)
    d.nodes(m.nodes)
    d.nodes(box, first_id=n_w + 1)
    d.nodes(np.array([[0.0, 0.0, 0.0], box_c]), first_id=hub_id)

    jc = mat.johnson_cook()
    d.mat_johnson_cook(1, mat.name, mat.density * 1e-9, mat.E * 1e-3, mat.nu, jc["A"] * 1e-3, jc["B"] * 1e-3,
                       jc["n"], jc["C"], eps_dot0=1e-3)
    d.fail_johnson(1, jc_damage(jc["eps_f"]), eps_dot0=1e-3)
    d.prop_solid(1, "weapon tetra10", isolid=0, itetra10=1000)
    if setup.opponent == "tpu":
        wall.write(d, 2, 3, 900)
        d.prop_solid(2, "TPU bricks", isolid=17)  # fully integrated: large strains, no hourglass modes
        d.part(2, "opponent TPU skin", 2, 2)
        d.part(3, "opponent TPU gyroid core", 2, 3)
    else:
        ojc = omat.johnson_cook()
        d.mat_johnson_cook(2, f"opponent {omat.name}", omat.density * 1e-9, omat.E * 1e-3, omat.nu,
                           ojc["A"] * 1e-3, ojc["B"] * 1e-3, ojc["n"], ojc["C"], eps_dot0=1e-3)
        d.fail_johnson(2, jc_damage(ojc["eps_f"]), eps_dot0=1e-3)
        d.prop_solid(2, "opponent bricks", isolid=24)
        d.part(2, "opponent", 2, 2)
    d.part(1, "weapon", 1, 1)
    d.tetra10(1, m.elems)
    first = len(m.elems) + 1
    for pid in sorted(set(epart.tolist())):
        sel = np.nonzero(epart == pid)[0]
        d.brick(pid, br[sel] + n_w + 1, first_id=first)
        first += len(sel)
    opp_parts = sorted(set(epart.tolist()))

    # hub: everything inside the center cutout radius (lock walls and bosses) + the bolt hole walls
    on_hub = r < weapon.center_cutout_r_mm() + 0.05
    for cx, cy, rr in weapon.support_circles():
        on_hub |= np.abs(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) - rr) < 1e-3
    d.grnod(1, "hub walls", np.nonzero(on_hub)[0] + 1)
    J_drive = max(drive_inertia_kgm2 * 1e6, 1.0)  # kg mm^2 (a small floor keeps the master node stable)
    d.rbody(1, "hub + drive", hub_id, 1, mass=0.05, J=(J_drive, J_drive, J_drive, 0, 0, 0))
    d.grnod(2, "hub master", [hub_id])
    d.bcs(1, "hub turns about Z only", "111", "110", 2)

    d.grnod(3, "opponent backing", np.nonzero(backing)[0] + n_w + 1)
    d.rbody(2, "opponent", opp_id, 3, mass=rb_mass, J=(*J_box, 0, 0, 0))
    d.grnod(4, "weapon + hub", list(range(1, n_w + 1)) + [hub_id])
    d.inivel_axis(1, "spin", 4, s * omega_rad_s * 1e-3)
    # the rigid hub moves with its master node: it needs the spin as a rotational velocity of its own, or the
    # rim spins against a standing hub (all the energy goes into the arms)
    d.inivel_rot(3, "hub spin", 2, (0.0, 0.0, s * omega_rad_s * 1e-3))
    d.grnod(5, "opponent + master", list(range(n_w + 1, n_w + n_b + 1)) + [opp_id])
    d.inivel_tra(2, "closing", 5, tuple(-closing_mps * e_r))

    near = (r > R - weapon.tooth_depth_mm() - 15) & (np.hypot(xy[:, 0] - tip[0], xy[:, 1] - tip[1]) < 60)
    d.grnod(6, "tooth nodes", np.nonzero(near)[0] + 1)
    d.surf_part(1, "opponent surface", opp_parts)
    d.inter7(1, "tooth vs opponent", 6, 1, gapmin=0.2, fric=setup.friction)

    d.th("INTER", 1, "contact", ["FNX", "FNY", "FNZ", "FTX", "FTY", "FTZ"], [1])
    d.th("RBODY", 2, "hub and opponent", ["DEF"], [1, 2])
    d.th("PART", 3, "parts", ["DEF"], [1, *opp_parts])
    d.run(setup.tstop_ms, setup.anim_dt_ms, setup.anim_dt_ms / 10, anim_elem=("VONM", "EPSP"),
          dt_min_ms=setup.dt_min_ms)
    d.write(folder)
    return {"n_weapon_nodes": n_w, "n_elems": len(m.elems), "R": R, "a_tip": a_tip, "contact_mm": c_len,
            "e_r": e_r.tolist(), "opponent_mass_kg": rb_mass + box_mass, "hub_nodes": int(on_hub.sum())}


def jc_damage(eps_f_tension: float) -> tuple:
    """Johnson-Cook damage constants: eps_f = D1 + D2 exp(D3 * triaxiality), shaped like Johnson & Cook's 4340
    (D1 = 0.05, D3 = -2.12) and scaled so uniaxial tension (triaxiality 1/3) fails at `eps_f_tension`. Crushed
    material (negative triaxiality) then tolerates several times more strain, as real steel does; a constant
    fracture strain erodes the tooth wherever it is merely compressed against the opponent."""
    D1, D3 = 0.05, -2.12
    D2 = max(eps_f_tension - D1, 0.01) / math.exp(D3 / 3)
    return (D1, D2, D3, 0.0, 0.0)


def read_th(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.reader(open(path)))
    head = [h.strip() for h in rows[0]]
    data = np.array([[float(x) for x in r if x.strip()] for r in rows[1:] if r])
    out = {}
    for i, h in enumerate(head[: data.shape[1]]):
        k = h if h not in out else f"{h}#{i}"
        out[k] = data[:, i]
    return out


def postprocess(folder: Path, name: str, info: dict, weapon, omega_rad_s: float, setup: ImpactSetup) -> dict:
    th_csv = folder / f"{name}T01.csv"
    subprocess.run([str(radioss.WRAPPER), "th_to_csv", f"{name}T01"], cwd=folder, capture_output=True)
    th = read_th(th_csv)
    time = th["time"]
    keys = list(th)
    # contact force magnitude: the interface's normal force components (name varies with the version)
    # the interface's variables in the order requested (FNX FNY FNZ FTX FTY FTZ), whatever their header says.
    # Radioss stores interface forces as *cumulative impulses* (kN ms = N s): the force is their derivative.
    icols = [k for k in keys if k.lower().startswith("contact") and " var " in k]  # not CONTACT ENERGY
    if len(icols) >= 6:
        In = np.c_[[th[k] for k in icols[:3]]].T
        It = np.c_[[th[k] for k in icols[3:6]]].T
        fn = np.linalg.norm(np.gradient(In, time, axis=0), axis=1)
        ft = np.linalg.norm(np.gradient(It, time, axis=0), axis=1)
        normal_impulse = float(np.linalg.norm(In[-1]))
    else:
        fn = ft = np.zeros_like(time)
        normal_impulse = 0.0
    out = {"time_ms": [float(time[0]), float(time[-1])], "th_columns": keys,
           "peak_friction_kn": float(ft.max()) if ft.any() else 0.0}
    if "PERCENTAGE ADDED MASS" in th:
        out["added_mass_pct"] = float(th["PERCENTAGE ADDED MASS"].max())
    if fn.any():
        i = int(np.argmax(fn))
        above = np.nonzero(fn > 0.1 * fn[i])[0]
        impulse = normal_impulse
        out |= {"peak_force_kn": float(fn[i]), "peak_at_ms": float(time[i]),
                "pulse_ms": float(time[above[-1]] - time[above[0]]), "impulse_ns": impulse,
                # the half-sine with this impulse and peak has this duration: compare with contact_time_ms
                "equivalent_half_sine_ms": float(math.pi / 2 * impulse / fn[i])}
    for k in ("PLASTIC WORK", "KINETIC ENERGY", "ROTATION ENERGY", "INTERNAL ENERGY", "CONTACT ENERGY"):
        if k in th:
            out[k.lower().replace(" ", "_") + "_j"] = [float(th[k][0]), float(th[k][-1])]
    # fields from the last animation frame: plastic strain, failure, per region
    frames = sorted(folder.glob(f"{name}A[0-9][0-9][0-9]"))
    if frames:  # every frame as VTK + a ParaView collection with the times (for the visualizer)
        vdir = folder / "vtk"
        vdir.mkdir(exist_ok=True)
        entries = []
        for k, fr in enumerate(frames):
            out_v = vdir / f"{fr.name}.vtk"
            with open(out_v, "w") as fh:
                subprocess.run([str(radioss.WRAPPER), "anim_to_vtk", fr.name], cwd=folder, stdout=fh)
            entries.append(f'  <DataSet timestep="{k * setup.anim_dt_ms:.6g}" file="{out_v.name}"/>')
        (vdir / f"{name}.pvd").write_text('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n'
                                          "<Collection>\n" + "\n".join(entries) + "\n</Collection>\n</VTKFile>\n")
        out |= _field_summary(vdir / f"{frames[-1].name}.vtk", weapon, vdir / f"{frames[0].name}.vtk")
        for fr in frames:  # the binary animation files are converted; keep the folder small
            fr.unlink()
    elif (folder / "vtk").exists():  # already converted (post-processing again)
        vt = sorted((folder / "vtk").glob(f"{name}A[0-9][0-9][0-9].vtk"))
        if vt:
            out |= _field_summary(vt[-1], weapon, vt[0])
    return out


def _field_summary(vtk: Path, weapon, first: Path | None = None) -> dict:
    from eitri.reference import REGIONS, regions

    pts, cells, data = _read_vtk(vtk)
    part = data.get("PART_ID")
    eps = data.get("3DELEM_Plastic_Strain")
    alive = data.get("EROSION_STATUS")
    if part is None or eps is None:
        return {}
    cen = np.array([pts[c].mean(axis=0) for c in cells])
    # the layer next to the rigid hub is a boundary artifact (like the static model's 1.5 mm bearing layer)
    rr = np.hypot(cen[:, 0], cen[:, 1])
    hub_layer = rr < weapon.center_cutout_r_mm() + 2.0
    for cx, cy, rad in weapon.support_circles():
        hub_layer |= np.hypot(cen[:, 0] - cx, cen[:, 1] - cy) < rad + 2.0
    w = (part == 1) & ~hub_layer
    masks = regions(weapon, cen[:, :2])
    dead0 = 0
    if first is not None and alive is not None:  # status 0 at t = 0 too (e.g. elements inside the rigid hub)
        d0 = _read_vtk(first)[2].get("EROSION_STATUS")
        dead0 = int(((d0 == 0) & w).sum()) if d0 is not None else 0
    res = {"max_plastic_strain": float(eps[w].max()),
           "failed_elements": int(((alive == 0) & w).sum()) - dead0 if alive is not None else None,
           "plastic_strain_by_region": {k: float(eps[w & masks[k]].max()) if (w & masks[k]).any() else 0.0
                                        for k in REGIONS}}
    i = int(np.argmax(np.where(w, eps, -1)))
    res["max_plastic_at_mm"] = [round(float(v), 2) for v in cen[i]]
    if alive is not None:
        dead = (alive == 0) & w
        if first is not None:
            d0 = _read_vtk(first)[2].get("EROSION_STATUS")
            dead &= d0 == 1
        res["failed_by_region"] = {k: int((dead & masks[k]).sum()) for k in REGIONS}
    # plastic strain away from the contact (the tip region is where the tooth dents against the opponent)
    body = w & ~masks["tip"]
    res["max_plastic_strain_body"] = float(eps[body].max()) if body.any() else 0.0
    j = int(np.argmax(np.where(body, eps, -1)))
    res["max_plastic_body_at_mm"] = [round(float(v), 2) for v in cen[j]]
    return res


def _read_vtk(path: Path):
    """Minimal ASCII legacy VTK reader (points, cells, cell scalars) for anim_to_vtk output."""
    lines = path.read_text(errors="ignore").split("\n")
    i, pts, cells, data = 0, None, [], {}
    while i < len(lines):
        tok = lines[i].split()
        if not tok:
            i += 1
            continue
        if tok[0] == "POINTS":
            n = int(tok[1])
            vals = []
            i += 1
            while len(vals) < 3 * n:
                vals += lines[i].split()
                i += 1
            pts = np.array(vals, float).reshape(n, 3)
            continue
        if tok[0] == "CELLS":
            n = int(tok[1])
            for k in range(n):
                v = lines[i + 1 + k].split()
                cells.append([int(x) for x in v[1:]])
            i += n + 1
            continue
        if tok[0] == "CELL_DATA":
            ncell = int(tok[1])
            i += 1
            while i < len(lines):
                tok = lines[i].split()
                if tok and tok[0] == "SCALARS":
                    nm = tok[1]
                    i += 2  # LOOKUP_TABLE
                    vals = []
                    while len(vals) < ncell:
                        vals += lines[i].split()
                        i += 1
                    data[nm] = np.array(vals[:ncell], float)
                    continue
                if tok and tok[0] in ("POINT_DATA",):
                    break
                i += 1
            continue
        i += 1
    return pts, cells, data


def simulate(weapon, mat, omega_rad_s: float, opponent_mass_kg: float, closing_mps: float, bite_mm: float,
             drive_inertia_kgm2: float, folder: Path, setup: ImpactSetup | None = None) -> ImpactResult:
    setup = setup or ImpactSetup()
    name = f"hit_{setup.case}"
    info = build(weapon, mat, omega_rad_s, opponent_mass_kg, closing_mps, bite_mm, drive_inertia_kgm2, setup,
                 folder, name)
    r = radioss.run(folder, name, threads=setup.threads)
    if not r.ok:
        return ImpactResult(folder, False, info, (r.starter_log[-2000:] + r.engine_log[-2000:]))
    return ImpactResult(folder, True, info | postprocess(folder, name, info, weapon, omega_rad_s, setup))


COMPARED = [("peak_force_kn", "peak contact force [kN]", None), ("impulse_ns", "impulse [N s]", None),
            ("equivalent_half_sine_ms", "equivalent half-sine pulse [ms]", None),
            ("max_plastic_strain_body", "max plastic strain outside the tip", "lower"),
            ("failed_body", "failed elements outside the tip", "lower")]


def summary(m: dict) -> dict:
    """The comparable numbers of one run (energies in J)."""
    out = {k: m.get(k) for k, _, _ in COMPARED}
    fb = m.get("failed_by_region") or {}
    out["failed_body"] = sum(v for k, v in fb.items() if k != "tip")
    out["failed_tip"] = fb.get("tip")
    ke = m.get("kinetic_energy_j")
    out["weapon_energy_lost_j"] = (ke[0] - ke[1]) if ke else None
    pw = m.get("plastic_work_j")
    out["plastic_work_j"] = pw[1] if pw else None
    out["plastic_strain_by_region"] = m.get("plastic_strain_by_region")
    return out


def compare_md(rows: dict[str, dict[str, dict]], ref_name: str, design_name: str) -> str:
    """rows[case] = {"design": summary, "reference": summary}: a table, worse-than-reference flagged."""
    lines = [f"| | {ref_name} | {design_name} | |", "|---|---|---|---|"]
    for case, pair in rows.items():
        lines.append(f"| **{case}** | | | |")
        r, d = pair["reference"], pair["design"]
        for key, label, better in COMPARED:
            rv, dv = r.get(key), d.get(key)
            flag = ""
            if better == "lower" and rv is not None and dv is not None and dv > rv * 1.02 + 1e-9:
                flag = "**WORSE**"
            lines.append(f"| {label} | {_fmt(rv)} | {_fmt(dv)} | {flag} |")
        for reg in ("tooth", "arm", "hub"):
            rv = (r.get("plastic_strain_by_region") or {}).get(reg)
            dv = (d.get("plastic_strain_by_region") or {}).get(reg)
            flag = "**WORSE**" if rv is not None and dv is not None and dv > rv * 1.02 + 1e-4 else ""
            lines.append(f"| plastic strain, {reg} | {_fmt(rv)} | {_fmt(dv)} | {flag} |")
    return "\n".join(lines) + "\n"


def _fmt(v) -> str:
    return "-" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))


# --- final gate: the explicit hits against the reference ------------------------------------------------
IMPACT_VERSION = 1  # bump when the explicit model changes (reference cache key)
GATE_CASES = ("bite", "tip3")


def available() -> bool:
    import os

    oc = Path(os.environ.get("OPENCOURANT_PATH", Path.home() / "opt" / "OpenCourant"))
    return (oc / "exec").exists() and radioss.WRAPPER.exists()


def run_spec(spec, design: dict | None, material: str | None, folder: Path, case: str,
             setup: ImpactSetup | None = None) -> ImpactResult:
    """One explicit hit of a spec's design (default: its fixed design) with the spec's drive and fight."""
    from eitri import materials
    from eitri.evaluate import build_weapon, single_tooth_bite_mm
    from eitri.physics import drive as drive_mod

    w = build_weapon(spec, design if design is not None else spec.fixed_design())
    mat = materials.get(material or spec.material.names()[0])
    J = (spec.drive.motor.rotor_inertia_kgm2 if spec.drive.motor else 0.0) * spec.drive.ratio ** 2
    setup = setup or ImpactSetup()
    setup.case = case
    return simulate(w, mat, drive_mod.top_speed_rad_s(spec.drive), spec.context.opponent_mass_kg,
                    spec.context.closing_speed_mps, single_tooth_bite_mm(spec), J, folder, setup)


def reference_impact(ref_spec_path: str, case: str, setup: ImpactSetup | None = None) -> dict:
    """The reference's explicit-hit summary, cached next to its spec (key: spec text, case, setup, version)."""
    import hashlib
    import json
    from dataclasses import asdict

    from eitri.spec import load

    setup = setup or ImpactSetup()
    src = Path(ref_spec_path)
    key = hashlib.sha1(f"{src.read_text()}|{case}|{asdict(setup) | {'case': case, 'threads': 0}}|"
                       f"{IMPACT_VERSION}".encode()).hexdigest()[:16]
    folder = src.parent / ".eitri_reference_impact" / f"{case}_{key}"
    cached = folder / "summary.json"
    if cached.exists():
        return json.loads(cached.read_text())
    r = run_spec(load(src), None, None, folder, case, setup)
    if not r.ok:
        raise RuntimeError(f"reference impact ({case}) failed:\n{r.log[-1500:]}")
    s = summary(r.metrics)
    cached.write_text(json.dumps(s, indent=1))
    return s


def gate_failures(mine: dict, ref: dict) -> list[str]:
    """What is worse than the reference (2 % tolerance on plastic strain; failed elements outside the tip)."""
    out = []
    for reg in ("tooth", "arm", "hub"):
        r = (ref.get("plastic_strain_by_region") or {}).get(reg)
        d = (mine.get("plastic_strain_by_region") or {}).get(reg)
        if r is not None and d is not None and d > r * 1.02 + 1e-4:
            out.append(f"plastic strain in the {reg}: {d:.4f} vs {r:.4f}")
    if (mine.get("failed_body") or 0) > (ref.get("failed_body") or 0):
        out.append(f"failed elements outside the tip: {mine.get('failed_body')} vs {ref.get('failed_body')}")
    return out


def gate(spec, design: dict | None, material: str | None, folder: Path, log=print,
         setup: ImpactSetup | None = None) -> dict:
    """Both gate hits on a design and the reference; {"ok", "cases": {case: {design, reference, worse}}}."""
    out = {"ok": True, "cases": {}}
    for case in GATE_CASES:
        log(f"    explicit {case}: reference ...")
        ref = reference_impact(spec.reference.spec, case, setup)
        log(f"    explicit {case}: design ...")
        r = run_spec(spec, design, material, folder / case, case, setup)
        if not r.ok:
            out["ok"] = False
            out["cases"][case] = {"error": r.log[-800:]}
            continue
        mine = summary(r.metrics)
        worse = gate_failures(mine, ref)
        out["cases"][case] = {"design": mine, "reference": ref, "worse": worse}
        out["ok"] &= not worse
        log(f"    explicit {case}: {'OK' if not worse else 'WORSE: ' + '; '.join(worse)}")
    rows = {c: {"design": v["design"], "reference": v["reference"]} for c, v in out["cases"].items() if "design" in v}
    if rows:
        (folder / "impact_compare.md").write_text(compare_md(rows, "reference", "design"))
    return out

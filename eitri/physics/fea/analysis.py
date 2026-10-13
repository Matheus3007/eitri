"""Plane-stress FEA load cases for a flat spinner (disc, and later bar/eggbeater profiles).

centrifugal
    Spinning at top speed, no hit. Supported only by 3 tangential constraints at the bore (no rigid-body
    motion, free radial growth), so no artificial support stress.

impact
    The hit alone (superposed on centrifugal in post-processing): force F on the outer `contact_len` of one face
    (contact_len = bite, the depth the opponent engages). Equivalent static, d'Alembert form:
      - the hit torque decelerates the rotor; that is applied as tangential inertial loads
        -m * alpha x r over the whole disc (alpha = T_hit / I), so torque is reacted by inertia, as in reality
      - the net hit force (a balanced rotor's CG can't move) is reacted by the hub supports
        (bolt holes, else bore), fully fixed.
    Contact stress under the load patch is a local crushing/denting problem the elastic model can't judge,
    so results are split into zones and the safety factor uses the `body` zone.

Because the model is linear, the hit stress scales with F. `summarize` uses that to find the hit force at
which the body first yields while spinning (`yield_force_kn`) — a number that does not depend on the
uncertain contact time — and the shortest pulse the tooth survives elastically for the estimated impulse.
"""

from dataclasses import dataclass

import numpy as np

from eitri.physics.fea.ccx import LoadCase, Result, solve
from eitri.physics.fea.mesh import Mesh2D, dist_to_segment, edges_on_segment, mesh_profile, nodes_on_circle


@dataclass
class FeaCase:
    name: str
    result: Result
    zones: dict[str, np.ndarray]  # zone name -> boolean mask over nodes


@dataclass
class FeaRun:
    mesh: Mesh2D
    cases: list[FeaCase]
    contact_segment: tuple[tuple[float, float], tuple[float, float]]
    support_circles: list[tuple[float, float, float]]
    applied_force_n: float


def _nearest(mesh: Mesh2D, x: float, y: float) -> int:
    return int(np.argmin(np.hypot(mesh.nodes[:, 0] - x, mesh.nodes[:, 1] - y))) + 1


def _element_mass_midside(mesh: Mesh2D, thickness: float, density_t_mm3: float) -> np.ndarray:
    """Mass lumped on midside nodes (consistent load vector of a T6 under uniform body force: corners 0)."""
    n, e = mesh.nodes, mesh.elems - 1
    area = 0.5 * np.abs((n[e[:, 1], 0] - n[e[:, 0], 0]) * (n[e[:, 2], 1] - n[e[:, 0], 1])
                        - (n[e[:, 2], 0] - n[e[:, 0], 0]) * (n[e[:, 1], 1] - n[e[:, 0], 1]))
    m_node = np.zeros(mesh.n_nodes)
    for k in (3, 4, 5):
        np.add.at(m_node, e[:, k], area * thickness * density_t_mm3 / 3)
    return m_node


def run(weapon, E_mpa: float, nu: float, density: float, omega_rad_s: float, peak_force_n: float,
        contact_len_mm: float, mesh_size_mm: float, threads: int = 1, keep_dir=None) -> FeaRun:
    t = weapon.params.thickness_mm
    tip0 = weapon.impact_face()[0]
    # a contact patch only a few elements long gets a locally refined mesh at the tip
    refine = ([(tip0[0], tip0[1], contact_len_mm / 4, contact_len_mm + 2)]
              if contact_len_mm < 2 * mesh_size_mm else [])
    mesh = mesh_profile(weapon.profile(), mesh_size_mm, refine=refine)
    xy = mesh.nodes
    supports = weapon.support_circles()

    # --- centrifugal: 3 tangential constraints on the bore (or near the center) --------------------
    r0 = weapon.bore_r_mm() or 10.0
    fix_cf = {_nearest(mesh, r0, 0): (2,), _nearest(mesh, 0, r0): (1,), _nearest(mesh, -r0, 0): (2,)}
    cases = [LoadCase("centrifugal", omega_rad_s=omega_rad_s, fixed=fix_cf)]

    # --- impact --------------------------------------------------------------------------------------
    tip, root = (np.array(p) for p in weapon.impact_face())
    face_len = float(np.linalg.norm(root - tip))
    c_len = float(np.clip(contact_len_mm, 2.0, face_len))
    c_end = tip + (root - tip) * c_len / face_len
    patch = edges_on_segment(mesh, tip, c_end)
    if not patch:
        raise RuntimeError("no mesh edges found on the impact face contact patch")

    # Edge geometry for the load resultant (corner-to-corner chord is exact: the face is straight).
    edge_nodes = mesh.edges()
    loaded_len, f_vec, torque = 0.0, np.zeros(2), 0.0
    f_contact = np.zeros((mesh.n_nodes, 2))
    for el, face in patch:
        a, b, m = edge_nodes[el - 1, face - 3]
        loaded_len += np.linalg.norm(xy[b - 1] - xy[a - 1])
    pressure = peak_force_n / (loaded_len * t)
    for el, face in patch:
        a, b, m = edge_nodes[el - 1, face - 3]
        pa, pb = xy[a - 1], xy[b - 1]
        d = pb - pa
        centroid = xy[mesh.elems[el - 1, :3] - 1].mean(axis=0)
        normal_in = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        if normal_in @ (centroid - (pa + pb) / 2) < 0:
            normal_in = -normal_in
        f = pressure * np.linalg.norm(d) * t * normal_in
        # Consistent nodal forces of a uniform pressure on a quadratic edge: 1/6, 1/6, 4/6.
        # (Applied as CLOADs: ccx 2.17 segfaults on *DLOAD pressure at some CPS6 edges touching sharp tips.)
        f_contact[a - 1] += f / 6
        f_contact[b - 1] += f / 6
        f_contact[m - 1] += 4 * f / 6
        mid = (pa + pb) / 2
        f_vec += f
        torque += mid[0] * f[1] - mid[1] * f[0]

    density_t_mm3 = density * 1e-12
    m_node = _element_mass_midside(mesh, t, density_t_mm3)
    i_mesh = float((m_node * (xy[:, 0] ** 2 + xy[:, 1] ** 2)).sum())  # t*mm^2
    alpha = torque / i_mesh  # rad/s^2
    # d'Alembert inertial load -m * (alpha z x r) = -m * alpha * (-y, x)
    fx = m_node * alpha * xy[:, 1] + f_contact[:, 0]
    fy = -m_node * alpha * xy[:, 0] + f_contact[:, 1]
    loaded = np.nonzero((fx != 0) | (fy != 0))[0]
    cloads = [(int(i) + 1, 1, float(fx[i])) for i in loaded] + [(int(i) + 1, 2, float(fy[i])) for i in loaded]

    fix_imp: dict[int, tuple[int, ...]] = {}
    for cx, cy, r in supports:
        for n in nodes_on_circle(mesh, cx, cy, r):
            fix_imp[int(n)] = (1, 2)
    cases.append(LoadCase("impact", cloads=cloads, fixed=fix_imp))

    # --- zones ---------------------------------------------------------------------------------------
    # Contact zone = tooth tip material right under the patch (local denting, acceptable damage). It never
    # reaches into the body radius + root fillet, so tooth-root cracking is always judged.
    r_keep = getattr(weapon, "r_body", 0.0) + getattr(weapon.params, "root_fillet_mm", 0.0) + 1.0
    contact_zone = ((dist_to_segment(xy, tip, c_end) < max(0.75 * c_len, 3.0))
                    & (np.hypot(xy[:, 0], xy[:, 1]) > r_keep))
    support_zone = np.zeros(mesh.n_nodes, bool)
    for cx, cy, r in supports:
        support_zone |= np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) < r + max(r, 3.0)
    body = ~(contact_zone | support_zone)

    results = []
    for case in cases:
        res = solve(mesh, t, E_mpa, nu, density, case, threads=threads, keep_dir=keep_dir)
        zones = {"body": body, "contact": contact_zone, "support": support_zone}
        results.append(FeaCase(case.name, res, zones))
    return FeaRun(mesh, results, (tuple(tip), tuple(c_end)), supports, float(np.linalg.norm(f_vec)))


def _zone_max(run: FeaRun, vm: np.ndarray, zones: dict[str, np.ndarray]) -> dict:
    d = {}
    for zone, mask in zones.items():
        if mask.any():
            i = int(np.argmax(np.where(mask, vm, -np.inf)))
            d[f"{zone}_max_vm_mpa"] = float(vm[i])
            d[f"{zone}_max_at_mm"] = [round(float(v), 2) for v in run.mesh.nodes[i]]
    return d


def _vm(stress: np.ndarray) -> np.ndarray:
    sx, sy, sz, txy, tyz, tzx = stress.T
    return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * (txy**2 + tyz**2 + tzx**2))


def combined_stress(run: FeaRun, scale: float) -> np.ndarray:
    """Spinning + `scale` x the applied hit, nodal stress tensors."""
    cf, imp = (c.result.stress for c in run.cases)
    return cf + scale * imp


def summarize(run: FeaRun, yield_mpa: float, impulse_ns: float) -> dict:
    cf, imp = run.cases
    zones = cf.zones
    body = zones["body"]
    F = run.applied_force_n
    out = {"method": "fea_plane_stress_ccx", "mesh_nodes": run.mesh.n_nodes}

    c = _zone_max(run, cf.result.von_mises, zones)
    c["safety_factor_body"] = yield_mpa / c["body_max_vm_mpa"]
    c["max_disp_mm"] = float(np.linalg.norm(cf.result.disp, axis=1).max())
    out["centrifugal"] = c

    h = _zone_max(run, _vm(combined_stress(run, 1.0)), zones)
    h["force_kn"] = F * 1e-3
    h["safety_factor_body"] = yield_mpa / h["body_max_vm_mpa"]
    out["impact_at_estimated_peak"] = h

    # Largest hit force with the body still elastic (while spinning): bisection on the load scale.
    if c["body_max_vm_mpa"] >= yield_mpa:
        s_yield = 0.0
    else:
        lo, hi = 0.0, 1.0
        while _vm(combined_stress(run, hi))[body].max() < yield_mpa and hi < 1e6:
            lo, hi = hi, hi * 2
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if _vm(combined_stress(run, mid))[body].max() < yield_mpa else (lo, mid)
        s_yield = lo
    f_yield = s_yield * F
    i_y = int(np.argmax(np.where(body, _vm(combined_stress(run, max(s_yield, 1e-9))), -np.inf)))
    out["yield_force_kn"] = f_yield * 1e-3
    out["yield_location_mm"] = [round(float(v), 2) for v in run.mesh.nodes[i_y]]
    # half-sine pulse: F_peak = pi/2 * J / dt  ->  dt_min = pi/2 * J / F_yield
    out["min_elastic_contact_time_ms"] = (np.pi / 2 * impulse_ns / f_yield * 1e3) if f_yield > 0 else None
    out["impact_safety_factor"] = f_yield / F if F > 0 else None  # yield force / estimated peak hit force
    return out

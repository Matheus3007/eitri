"""Density topology optimization (SIMP) of the free region, with mass, MOI and balance held exactly.

Problem, on the domain's linear-triangle mesh (element densities x in [0, 1], keep elements fixed at 1):
    minimize   sum_k w_k * c_k(x) / c_k(x0)        compliance of each in-plane hit case (bite, tip patches)
    subject to mass(x) = M                          the mass budget (e.g. the reference weapon's mass)
               MOI(x)  = I                          stored energy target (E ~ I at the drive's top speed)
               first moment_x(x) = first moment_y(x) = 0    CG on the axis: balance is a constraint, not a fix
All four constraints are linear in the physical densities. Each iteration takes a projected-gradient step
(move limit) and projects onto {A x = b, bounds} by solving for the 4 multipliers with a semismooth Newton
method -- the same "Newton on the balance" idea as the generators, applied inside the optimizer.

Hits are the eitri impact model: a force on the outer `contact` mm of the impact face, the torque reacted by
d'Alembert inertia of the current density field (alpha = T / I(x)), the net force by the rigid hub (all hub
hole edges fixed). Stiffness E(x) = Emin + x^p (penalty p continued 1 -> 3); a density filter of radius
`rmin` sets the smallest feature. Compliance is the stiffness proxy; the extracted design is then judged by the
full pipeline (CalculiX, reference load cases, 3D out-of-plane), not by this model.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from eitri.physics.fea.mesh import dist_to_segment
from eitri.topo.fem import Plane, TriMesh, mesh_linear

EMIN = 1e-6


@dataclass
class TopoResult:
    mesh: TriMesh
    density: np.ndarray  # physical (filtered) element densities
    keep: np.ndarray  # bool per element
    history: list[dict] = field(default_factory=list)
    constraint_error: dict = field(default_factory=dict)

    def nodal_density(self) -> np.ndarray:
        n = len(self.mesh.nodes)
        acc, wsum = np.zeros(n), np.zeros(n)
        a = np.abs(self.mesh.areas)
        for k in range(3):
            np.add.at(acc, self.mesh.tris[:, k], a * self.density)
            np.add.at(wsum, self.mesh.tris[:, k], a)
        return acc / np.maximum(wsum, 1e-12)


def _filter_matrix(c: np.ndarray, areas: np.ndarray, rmin: float) -> sp.csr_matrix:
    tree = cKDTree(c)
    pairs = tree.query_pairs(rmin, output_type="ndarray")
    i = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(len(c))])
    j = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(len(c))])
    w = (rmin - np.linalg.norm(c[i] - c[j], axis=1)) * areas[j]
    H = sp.csr_matrix((w, (i, j)), shape=(len(c), len(c)))
    return sp.diags(1 / np.asarray(H.sum(axis=1)).ravel()) @ H


def project(z: np.ndarray, lo: np.ndarray, hi: np.ndarray, A: np.ndarray, b: np.ndarray, iters: int = 50):
    """x = clip(z - A^T lam, lo, hi) with A x = b: semismooth Newton on the multipliers. Returns (x, residual)."""
    lam = np.zeros(len(b))
    best = None
    for _ in range(iters):
        x = np.clip(z - A.T @ lam, lo, hi)
        r = A @ x - b
        nr = np.linalg.norm(r)
        if best is None or nr < best[1]:
            best = (x, nr, r)
        if nr < 1e-10 * (1 + np.linalg.norm(b)):
            break
        act = (x > lo) & (x < hi)  # where x moves with lam
        J = -(A[:, act] @ A[:, act].T)
        try:
            d = np.linalg.solve(J + 1e-12 * np.eye(len(b)), -r)
        except np.linalg.LinAlgError:
            d = np.linalg.lstsq(J, -r, rcond=None)[0]
        t = 1.0
        for _ in range(20):  # damped: the merit is the residual norm
            x1 = np.clip(z - A.T @ (lam + t * d), lo, hi)
            if np.linalg.norm(A @ x1 - b) < nr:
                break
            t *= 0.5
        lam = lam + t * d
        if not np.all(np.isfinite(lam)):
            break  # targets out of reach: keep the closest point found
    return best[0], best[2]


def optimize(domain, E: float, nu: float, density: float, mass_kg: float, moi_kgm2: float | None,
             contacts_mm: list[float], weights: list[float] | None = None, mesh_mm: float = 1.5,
             rmin_mm: float = 5.0, iters: int = 160, move: float = 0.1, objective: str = "stress",
             pnorm: float = 8.0, flat_weight: float = 1.0, flat_patch_mm: float = 8.0, check_grad: int = 0,
             log: Callable[[str], None] = print) -> TopoResult:
    """objective "stress": minimize the p-norm of the (relaxed, sqrt(x)-scaled) von Mises stress of each hit
    case -- i.e. maximize the force at first yield, what the reference comparison judges; "compliance":
    the classic stiffness objective (cheaper, but blind to stress concentrations: it thins hub necks).
    `flat_weight` > 0 adds the out-of-plane hit (DKT plate bending: a force normal to the plate within
    `flat_patch_mm` of the tip, bolts clamped -- the reference's flat_hit case) to the stress objective."""
    base = domain.base
    t = base.params.thickness_mm
    mesh = mesh_linear(domain.profile, mesh_mm)
    c = mesh.centroids
    a = np.abs(mesh.areas)
    keep = domain.keep_mask(c)
    free = ~keep
    rho_t = density * 1e-9 * t  # kg per mm^2 of plate
    fe = Plane(mesh, E, nu, t)

    # --- loads (unit total force per case) and supports ------------------------------------------------
    tip, root = (np.array(p, float) for p in base.impact_face())
    face_len = np.linalg.norm(root - tip)
    bnd = mesh.boundary_edges()
    xy = mesh.nodes
    hit_loads, torques = [], []
    on_face = dist_to_segment(xy, tip, root) < 1e-3
    face_edges = bnd[on_face[bnd[:, 0]] & on_face[bnd[:, 1]]]
    s_of = (xy - tip) @ ((root - tip) / face_len)  # distance along the face from the tip
    for cl in contacts_mm:
        f = np.zeros(2 * len(xy))
        d = (root - tip) / face_len
        normal_in = np.array([-d[1], d[0]])
        # the force pushes into the material: probe which side of the face the body is on
        probe = (tip + root) / 2 + 0.5 * normal_in
        if not domain.profile.is_inside((*probe, 0)):
            normal_in = -normal_in
        # uniform pressure on the outer `cl` mm of the face, whatever the mesh: each face edge takes its overlap
        sa, sb = np.minimum(s_of[face_edges[:, 0]], s_of[face_edges[:, 1]]), np.maximum(s_of[face_edges[:, 0]],
                                                                                        s_of[face_edges[:, 1]])
        overlap = np.clip(np.minimum(sb, min(cl, face_len)) - np.maximum(sa, 0), 0, None)
        for (i, j), ov in zip(face_edges, overlap / overlap.sum()):
            f[2 * i: 2 * i + 2] += ov / 2 * normal_in
            f[2 * j: 2 * j + 2] += ov / 2 * normal_in
        fx, fy = f[0::2], f[1::2]
        torques.append(float((xy[:, 0] * fy - xy[:, 1] * fx).sum()))
        hit_loads.append(f)
    hub_nodes = np.unique(np.concatenate([
        bnd[np.hypot(*xy[bnd[:, 0]].T) <= domain.hub_hole_r][:, 0],
        bnd[np.hypot(*xy[bnd[:, 1]].T) <= domain.hub_hole_r][:, 1]]))
    fixed = np.concatenate([2 * hub_nodes, 2 * hub_nodes + 1])
    # stress judged everywhere except right under each contact patch (denting) and at the rigid hub contact
    hub_layer = cKDTree(xy[hub_nodes]).query(c, distance_upper_bound=1.5)[0] < 1.5
    judged = []
    for cl in contacts_mm:
        end = tip + (root - tip) * min(cl, face_len) / face_len
        judged.append(~hub_layer & (dist_to_segment(c, tip, end) > max(0.75 * cl, 3.0)))
    weights = np.asarray(weights or [1.0] * len(contacts_mm), float)
    plate = None
    if objective == "stress" and flat_weight > 0:
        from eitri.topo.plate import PlateBending

        plate = PlateBending(mesh, E, nu, t)
        bnodes = np.unique(bnd.ravel())
        clamp = np.concatenate([bnodes[np.abs(np.hypot(xy[bnodes, 0] - cx, xy[bnodes, 1] - cy) - r) < 1e-3]
                                for cx, cy, r in base.support_circles()])
        plate_fixed = np.concatenate([3 * clamp, 3 * clamp + 1, 3 * clamp + 2])
        under = np.nonzero(np.hypot(*(xy - tip).T) < flat_patch_mm)[0]
        plate_load = np.zeros(plate.n)
        plate_load[3 * under] = 1.0 / len(under)
        near_bolt = np.zeros(len(c), bool)
        for cx, cy, r in base.support_circles():
            near_bolt |= np.hypot(c[:, 0] - cx, c[:, 1] - cy) < r + max(r, 3.0)
        plate_judged = ~near_bolt & (np.hypot(*(c - tip).T) > flat_patch_mm + 4.0)
        plate_S = plate.stress_matrix()
        weights = np.append(weights, flat_weight)

    # --- filter and the linear constraints ------------------------------------------------------------
    H = _filter_matrix(c, a, rmin_mm)
    Hff, Hfk = H[free][:, free], H[free][:, keep]
    r2 = (c ** 2).sum(axis=1)
    rows = np.stack([a, a * r2, a * c[:, 0], a * c[:, 1]]) * rho_t  # kg, kg*mm^2, kg*mm
    target = np.array([mass_kg, (moi_kgm2 or 0.0) * 1e6, 0.0, 0.0])
    if moi_kgm2 is None:  # no MOI target: mass and balance only
        rows, target = rows[[0, 2, 3]], target[[0, 2, 3]]
    const = rows[:, keep].sum(axis=1) + rows[:, free] @ (Hfk @ np.ones(keep.sum()))
    A = np.asarray((Hff.T @ rows[:, free].T).T)
    b = target - const
    scale = np.linalg.norm(A, axis=1)
    A_n, b_n = A / scale[:, None], b / scale

    # start: uniform density that meets mass, then projected onto all constraints
    x = np.full(free.sum(), 0.5)
    x, _ = project(x, np.zeros_like(x), np.ones_like(x), A_n, b_n, iters=100)

    def physical(x, beta=1.0):
        """Filtered, then Heaviside-projected densities (keep elements 1) and d projected / d filtered."""
        xf = np.ones(len(c))
        xf[free] = Hff @ x + Hfk @ np.ones(keep.sum())
        xf = np.clip(xf, 0, 1)
        th = np.tanh(beta * 0.5)
        den = th + np.tanh(beta * 0.5)
        xp = (th + np.tanh(beta * (xf - 0.5))) / den
        dxp = beta * (1 - np.tanh(beta * (xf - 0.5)) ** 2) / den
        xp[keep], dxp[keep] = 1.0, 0.0
        return xp, dxp

    c0_box = [None]

    def objgrad(x, p, beta=1.0):
        """Objective and its gradient with respect to the free design variables."""
        xf, dxp = physical(x, beta)
        Ee = EMIN + xf ** p * (1 - EMIN)
        q = np.where(xf > 0.1, xf, xf ** 3 / 0.01)  # inertia-load mass (no parasitic loads in voids)
        dq = np.where(xf > 0.1, 1.0, 3 * xf ** 2 / 0.01)
        m_e = rho_t * a * q
        inertia = float((m_e * r2).sum())
        loads = np.zeros((fe.n, len(contacts_mm)))
        for k, (f, T) in enumerate(zip(hit_loads, torques)):
            alpha = T / inertia  # per newton of hit
            fin = np.zeros(fe.n)
            # d'Alembert: -m alpha (z x r) = m alpha (y, -x), lumped to the element's corners
            for kk in range(3):
                n_ = mesh.tris[:, kk]
                np.add.at(fin, 2 * n_, m_e / 3 * alpha * xy[n_, 1])
                np.add.at(fin, 2 * n_ + 1, -m_e / 3 * alpha * xy[n_, 0])
            loads[:, k] = f + fin
        u, again = fe.solve(Ee, loads, fixed, return_solver=True)
        dEe = p * xf ** (p - 1) * (1 - EMIN)
        tri = mesh.tris
        if objective == "compliance":
            meas = np.einsum("ik,ik->k", loads, u)
            ee = fe.element_energy(u)  # (m, k)
            dmeas = -dEe[:, None] * ee
            lam = u  # self-adjoint: d(f.u)/dx = 2 u.df/dx - u.dK/dx u
            lam_scale = 2.0
        else:
            eta = np.sqrt(np.maximum(xf, 1e-9))  # stress relaxation: voids carry no stress
            deta = 0.5 / eta
            V = np.array([[1, -0.5, 0], [-0.5, 1, 0], [0, 0, 3]])
            meas, dmeas = np.zeros(len(contacts_mm)), np.zeros((len(c), len(contacts_mm)))
            lam = np.zeros_like(u)
            for k in range(len(contacts_mm)):
                s0 = fe.solid_stress(u[:, k])  # (m, 3)
                vm0 = np.sqrt(np.maximum(np.einsum("ei,ij,ej->e", s0, V, s0), 1e-30))
                vm = eta * vm0
                J = judged[k]
                pn = float((vm[J] ** pnorm).sum() ** (1 / pnorm))
                meas[k] = pn
                dpn_dvm = np.where(J, (vm / pn) ** (pnorm - 1), 0.0)  # d pn / d vm_e
                dmeas[:, k] = dpn_dvm * deta * vm0  # explicit (stress interpolation)
                # d vm / d u = eta (D B)^T V s0 / vm0 -> adjoint load
                w = (dpn_dvm * eta / vm0)[:, None] * (s0 @ V)  # (m, 3)
                ge = np.einsum("ej,jk,eki->ei", w, fe.D, fe.B)  # (m, 6)
                rhs = np.zeros(fe.n)
                np.add.at(rhs, fe.dofs.ravel(), ge.ravel())
                lam[:, k] = again(rhs)
                # - lam^T dK/dx u
                ue, le = u[fe.dofs, k], lam[fe.dofs, k]
                dmeas[:, k] -= dEe * np.einsum("ei,eij,ej->e", le, fe.ke, ue)
            lam_scale = 1.0
        if plate is not None:  # out-of-plane hit: p-norm of the surface stress, adjoint as in-plane
            V = np.array([[1, -0.5, 0], [-0.5, 1, 0], [0, 0, 3]])
            eta = np.sqrt(np.maximum(xf, 1e-9))
            w_, again_p = plate.solve(Ee, plate_load, plate_fixed)
            s0 = np.einsum("eij,ej->ei", plate_S, w_[plate.dofs])
            vm0 = np.sqrt(np.maximum(np.einsum("ei,ij,ej->e", s0, V, s0), 1e-30))
            vm = eta * vm0
            pn = float((vm[plate_judged] ** pnorm).sum() ** (1 / pnorm))
            dpn_dvm = np.where(plate_judged, (vm / pn) ** (pnorm - 1), 0.0)
            d_pl = dpn_dvm * (0.5 / eta) * vm0
            wv = (dpn_dvm * eta / vm0)[:, None] * (s0 @ V)
            ge = np.einsum("ej,eji->ei", wv, plate_S)
            rhs = np.zeros(plate.n)
            np.add.at(rhs, plate.dofs.ravel(), ge.ravel())
            lam_p = again_p(rhs)
            d_pl -= dEe * np.einsum("ei,eij,ej->e", lam_p[plate.dofs], plate.ke, w_[plate.dofs])
            meas = np.append(meas, pn)
            dmeas = np.hstack([dmeas, d_pl[:, None]])
        if c0_box[0] is None:
            c0_box[0] = meas.copy()
        c0 = c0_box[0]
        obj = float((weights * meas / c0).sum())
        comp = meas
        dc = dmeas
        for k, T in enumerate(torques):  # design-dependent inertia load: + lam^T df/dx
            alpha = T / inertia
            lx, ly = lam[0::2, k], lam[1::2, k]
            work = sum(lx[tri[:, kk]] * xy[tri[:, kk], 1] - ly[tri[:, kk]] * xy[tri[:, kk], 0] for kk in range(3))
            dc[:, k] += lam_scale * rho_t * a * dq / 3 * alpha * work
            # ... and through alpha = T / I(x): every element's inertia load scales with 1 / I
            lam_fin1 = float((m_e / 3 * work).sum())  # lam . (inertia load at alpha = 1)
            dc[:, k] += lam_scale * lam_fin1 * (-T / inertia**2) * rho_t * a * dq * r2
        g_phys = (dc * (weights / c0)).sum(axis=1)
        g = Hff.T @ (g_phys * dxp)[free]
        return obj, g, comp, xf

    if check_grad:
        rng = np.random.default_rng(0)
        x0 = np.clip(x + rng.uniform(-0.2, 0.2, len(x)), 0.05, 0.95)
        f0, g0, _, _ = objgrad(x0, 3.0)
        for i in rng.choice(len(x0), check_grad, replace=False):
            h = 1e-5
            xp, xm = x0.copy(), x0.copy()
            xp[i] += h
            xm[i] -= h
            fd = (objgrad(xp, 3.0)[0] - objgrad(xm, 3.0)[0]) / (2 * h)
            log(f"    grad check var {i}: adjoint {g0[i]:.6e} finite difference {fd:.6e}")
        return None

    hist = []
    p, beta = 1.0, 1.0
    step = move
    for it in range(iters):
        if it >= 20 and it % 8 == 0:
            p = min(3.0, p + 0.25)
        if p >= 3.0 and it % 15 == 0:
            beta = min(32.0, 2 * beta)  # push to 0/1: grey density is cheap mass that doesn't carry load
        obj, g, comp, xf = objgrad(x, p, beta)
        # adaptive move limit: shrink when the objective went up at the same penalty (zig-zag), else grow
        if hist and hist[-1]["p"] == p and hist[-1]["beta"] == beta and obj > hist[-1]["objective"]:
            step = max(0.01, step * 0.6)
        else:
            step = min(move, step * 1.1)
        z = x - step * g / (np.abs(g).max() + 1e-30)
        lo, hi = np.maximum(0, x - step), np.minimum(1, x + step)
        # constraints on the projected densities, re-linearized here (Newton): A_l x = b_l
        xp_now, dxp_now = physical(x, beta)
        A_l = np.asarray((Hff.T @ (rows[:, free] * dxp_now[free]).T).T)
        b_l = target - rows @ xp_now + A_l @ x
        sc = np.linalg.norm(A_l, axis=1)
        x_new, res = project(z, lo, hi, A_l / sc[:, None], b_l / sc)
        res = res * sc / scale
        change = float(np.abs(x_new - x).max())
        x = x_new
        hist.append({"it": it, "p": p, "beta": beta, "objective": obj, "change": change,
                     objective: comp.tolist(), "constraint_residual": float(np.linalg.norm(res * scale))})
        if it % 10 == 0 or it == iters - 1:
            log(f"    topo it {it:3d}: p {p:.2f} objective {obj:.4f} change {change:.3f} "
                f"grey {float(((xf > 0.1) & (xf < 0.9))[free].mean()):.2f}")
        if p >= 3.0 and beta >= 32 and change < 0.005:
            break
    xf, _ = physical(x, beta)
    err = rows @ xf - target
    if moi_kgm2 is None:
        err = np.insert(err, 1, 0.0)
    return TopoResult(mesh, xf, keep, hist, {"mass_kg": float(err[0]), "moi_kgm2": float(err[1] * 1e-6),
                                              "first_moment_kgmm": [float(err[2]), float(err[3])]})

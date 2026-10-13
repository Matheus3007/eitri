"""Topology optimization: plate element, exact constraint projection, adjoint gradients; machined features."""

import build123d as bd
import numpy as np
import pytest
import scipy.sparse as sp

from eitri.topo.fem import mesh_linear
from eitri.topo.plate import PlateBending
from eitri.topo.simp import project


def test_dkt_plate_rigid_modes_and_cantilever():
    L, b, t, F, E, nu = 100.0, 20.0, 6.5, 100.0, 210000.0, 0.3
    m = mesh_linear((bd.Pos(L / 2, 0) * bd.Rectangle(L, b)).faces()[0], 2.5)
    pb = PlateBending(m, E, nu, t)
    x, y = m.nodes[:, 0], m.nodes[:, 1]
    K = sp.csc_matrix((pb.ke.ravel(), (pb.rows, pb.cols)), shape=(pb.n, pb.n))
    scale = abs(K).max()
    for u in (np.c_[np.ones_like(x), 0 * x, 0 * x], np.c_[y, np.ones_like(x), 0 * x], np.c_[x, 0 * x, -np.ones_like(x)]):
        u = u.ravel()
        assert abs(u @ (K @ u)) < 1e-9 * scale * len(x)  # translation and both rotations: no energy
    clamp, end = np.nonzero(x < 1e-6)[0], np.nonzero(x > L - 1e-6)[0]
    load = np.zeros(pb.n)
    load[3 * end] = F / len(end)
    u, _ = pb.solve(np.ones(len(m.tris)), load, np.concatenate([3 * clamp, 3 * clamp + 1, 3 * clamp + 2]))
    I = b * t**3 / 12
    assert u[3 * end].mean() == pytest.approx(F * L**3 / (3 * E * I), rel=0.04)
    s, c = pb.surface_stress(u), m.centroids
    mid = (c[:, 0] > 40) & (c[:, 0] < 60)
    assert np.median(np.abs(s[mid, 0]) / (F * (L - c[mid, 0]) * t / 2 / I)) == pytest.approx(1, rel=0.01)


def test_projection_meets_linear_constraints_exactly_within_bounds():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(4, 300))
    x_true = rng.uniform(0.2, 0.8, 300)
    b = A @ x_true
    x, r = project(rng.uniform(-0.5, 1.5, 300), np.zeros(300), np.ones(300), A, b)
    assert np.linalg.norm(r) < 1e-8 and x.min() >= 0 and x.max() <= 1


@pytest.mark.parametrize("objective", ["compliance", "stress"])
def test_topology_adjoint_gradient_matches_finite_differences(objective, capsys):
    from eitri.geometry.hook import Hook, HookParams
    from eitri.topo.domain import Domain
    from eitri.topo.simp import optimize

    h = Hook(HookParams(od_mm=230, thickness_mm=6.5, envelope_bite_mm=26.6))
    dom = Domain(h, ["hub:22", "rim:8", "tooth:18", "holes:3"])
    lines = []
    optimize(dom, 200000, 0.29, 7850, 0.45, None, [26.6, 3.0], mesh_mm=4.0, objective=objective, check_grad=4,
             log=lines.append)
    assert len(lines) == 4
    for line in lines:
        adj, fd = (float(v) for v in line.split("adjoint ")[1].split(" finite difference "))
        assert adj == pytest.approx(fd, rel=0.02, abs=1e-6)

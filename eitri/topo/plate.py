"""Kirchhoff plate bending on the topology mesh: DKT triangles (Batoz, Bathe & Ho 1980).

The out-of-plane hit (something strikes the flat of the tooth) is a plate-bending problem; the in-plane model
can't see it, and it is the case where a hub neck thinned for in-plane loads gives way first. Per node: w (mm),
theta_x = dw/dy, theta_y = -dw/dx. Bending stiffness E t^3 / 12(1 - nu^2); surface stress 6 M / t^2.
Validated against cantilever beam theory (tests/test_topo.py).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from eitri.topo.fem import TriMesh

_GAUSS = [(0.5, 0.0), (0.5, 0.5), (0.0, 0.5)]  # mid-side points, weight A/3 each


def _H(xi, eta, P, q, r, t):
    """d(beta_x)/d(xi, eta) and d(beta_y)/d(xi, eta) as 9-vectors (Batoz's Hx,xi Hx,eta Hy,xi Hy,eta)."""
    P4, P5, P6 = P
    q4, q5, q6 = q
    r4, r5, r6 = r
    t4, t5, t6 = t
    a, b = 1 - 2 * xi, 1 - 2 * eta
    Hx_xi = np.array([P6 * a + (P5 - P6) * eta, q6 * a - (q5 + q6) * eta, -4 + 6 * (xi + eta) + r6 * a - eta * (r5 + r6),
                      -P6 * a + eta * (P4 + P6), q6 * a - eta * (q6 - q4), -2 + 6 * xi + r6 * a + eta * (r4 - r6),
                      -eta * (P5 + P4), eta * (q4 - q5), -eta * (r5 - r4)])
    Hy_xi = np.array([t6 * a + eta * (t5 - t6), 1 + r6 * a - eta * (r5 + r6), -q6 * a + eta * (q5 + q6),
                      -t6 * a + eta * (t4 + t6), -1 + r6 * a + eta * (r4 - r6), -q6 * a - eta * (q4 - q6),
                      -eta * (t4 + t5), eta * (r4 - r5), -eta * (q4 - q5)])
    Hx_eta = np.array([-P5 * b - xi * (P6 - P5), q5 * b - xi * (q5 + q6), -4 + 6 * (xi + eta) + r5 * b - xi * (r5 + r6),
                       xi * (P4 + P6), xi * (q4 - q6), -xi * (r6 - r4),
                       P5 * b - xi * (P4 + P5), q5 * b + xi * (q4 - q5), -2 + 6 * eta + r5 * b + xi * (r4 - r5)])
    Hy_eta = np.array([-t5 * b - xi * (t6 - t5), 1 + r5 * b - xi * (r5 + r6), -q5 * b + xi * (q5 + q6),
                       xi * (t4 + t6), xi * (r4 - r6), -xi * (q4 - q6),
                       t5 * b - xi * (t4 + t5), -1 + r5 * b + xi * (r4 - r5), -q5 * b - xi * (q4 - q5)])
    return Hx_xi, Hx_eta, Hy_xi, Hy_eta


def element_B(xy: np.ndarray) -> tuple[list[np.ndarray], float]:
    """Curvature matrices (3 x 9) at the 3 integration points, and the area, for one triangle (3, 2)."""
    (x1, y1), (x2, y2), (x3, y3) = xy
    x12, x23, x31 = x1 - x2, x2 - x3, x3 - x1
    y12, y23, y31 = y1 - y2, y2 - y3, y3 - y1
    A2 = x31 * y12 - x12 * y31  # 2 A
    sides = [(x23, y23), (x31, y31), (x12, y12)]  # k = 4, 5, 6
    P, q, r, t = [], [], [], []
    for xx, yy in sides:
        l2 = xx * xx + yy * yy
        P.append(-6 * xx / l2)
        q.append(3 * xx * yy / l2)
        r.append(3 * yy * yy / l2)
        t.append(-6 * yy / l2)
    Bs = []
    for xi, eta in _GAUSS:
        Hx_xi, Hx_eta, Hy_xi, Hy_eta = _H(xi, eta, P, q, r, t)
        B = np.vstack([y31 * Hx_xi + y12 * Hx_eta,
                       -x31 * Hy_xi - x12 * Hy_eta,
                       -x31 * Hx_xi - x12 * Hx_eta + y31 * Hy_xi + y12 * Hy_eta]) / A2
        Bs.append(B)
    return Bs, A2 / 2


class PlateBending:
    """K(E) = sum_e E_e k_e on a TriMesh; 3 dofs per node (w, theta_x, theta_y)."""

    def __init__(self, mesh: TriMesh, E: float, nu: float, t: float):
        self.mesh, self.t = mesh, t
        self.Db = E * t**3 / (12 * (1 - nu**2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
        m = len(mesh.tris)
        self.ke = np.zeros((m, 9, 9))
        self.B = np.zeros((m, 3, 9))  # at the centroid-ish (mean of the 3 points), for stresses
        for e, tri in enumerate(mesh.tris):
            Bs, A = element_B(mesh.nodes[tri])
            self.ke[e] = sum(B.T @ self.Db @ B for B in Bs) * A / 3
            self.B[e] = sum(Bs) / 3
        dofs = (3 * mesh.tris[:, :, None] + np.arange(3)).reshape(-1, 9)
        self.dofs = dofs
        self.rows = np.repeat(dofs, 9, axis=1).ravel()
        self.cols = np.tile(dofs, (1, 9)).ravel()
        self.n = 3 * len(mesh.nodes)

    def solve(self, Ee: np.ndarray, load: np.ndarray, fixed: np.ndarray):
        K = sp.csc_matrix(((self.ke * Ee[:, None, None]).ravel(), (self.rows, self.cols)), shape=(self.n, self.n))
        free = np.setdiff1d(np.arange(self.n), fixed)
        lu = spla.splu(K[free][:, free].tocsc())
        u = np.zeros(self.n)
        u[free] = lu.solve(load[free])

        def again(rhs):
            v = np.zeros(self.n)
            v[free] = lu.solve(rhs[free])
            return v

        return u, again

    def surface_stress(self, u: np.ndarray) -> np.ndarray:
        """(m, 3) surface stress (sx, sy, txy) of fully dense material: 6 M / t^2, M = Db kappa."""
        kappa = np.einsum("eij,ej->ei", self.B, u[self.dofs])
        return kappa @ self.Db.T * 6 / self.t**2

    def stress_matrix(self) -> np.ndarray:
        """(m, 3, 9): surface stress per element = S_e u_e."""
        return np.einsum("ij,ejk->eik", self.Db * 6 / self.t**2, self.B)

"""Small, fast plane-stress FE for topology optimization: linear triangles (CST), numpy assembly, scipy solve.

CalculiX is the verification solver; this one only has to rank density fields quickly (a few hundred solves).
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import build123d as bd
import gmsh
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class TriMesh:
    nodes: np.ndarray  # (n, 2)
    tris: np.ndarray  # (m, 3) 0-based, CCW

    @property
    def centroids(self) -> np.ndarray:
        return self.nodes[self.tris].mean(axis=1)

    @property
    def areas(self) -> np.ndarray:
        p = self.nodes[self.tris]
        return 0.5 * ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                      - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))

    def boundary_edges(self) -> np.ndarray:
        """(k, 2) node pairs of edges used by one triangle only."""
        e = np.concatenate([self.tris[:, [0, 1]], self.tris[:, [1, 2]], self.tris[:, [2, 0]]])
        key = np.sort(e, axis=1)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        return e[cnt[inv.ravel()] == 1]


def mesh_linear(face: bd.Face, size_mm: float) -> TriMesh:
    with tempfile.TemporaryDirectory() as tmp:
        brep = Path(tmp) / "d.brep"
        bd.export_brep(face, str(brep))
        gmsh.initialize(interruptible=False)
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("d")
            gmsh.model.occ.importShapes(str(brep))
            gmsh.model.occ.synchronize()
            gmsh.option.setNumber("Mesh.MeshSizeMax", size_mm)
            gmsh.option.setNumber("Mesh.MeshSizeMin", size_mm / 3)
            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.model.mesh.generate(2)
            tags, coords, _ = gmsh.model.mesh.getNodes()
            renum = np.zeros(int(tags.max()) + 1, dtype=np.int64)
            renum[tags.astype(np.int64)] = np.arange(len(tags))
            xy = coords.reshape(-1, 3)[:, :2][np.argsort(renum[tags.astype(np.int64)])]
            types, _, enodes = gmsh.model.mesh.getElements(2)
            tri = renum[[n for t, n in zip(types, enodes) if t == 2][0].astype(np.int64)].reshape(-1, 3)
        finally:
            gmsh.finalize()
    m = TriMesh(xy, tri)
    flip = m.areas < 0
    m.tris[flip] = m.tris[flip][:, [0, 2, 1]]
    return m


class Plane:
    """Plane-stress stiffness with per-element modulus factors: K(E) = sum_e E_e * k_e."""

    def __init__(self, mesh: TriMesh, E: float, nu: float, t: float):
        self.mesh = mesh
        p = mesh.nodes[mesh.tris]  # (m, 3, 2)
        x, y = p[..., 0], p[..., 1]
        A = mesh.areas
        b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], axis=1)
        c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], axis=1)
        B = np.zeros((len(A), 3, 6))
        B[:, 0, 0::2], B[:, 1, 1::2] = b, c
        B[:, 2, 0::2], B[:, 2, 1::2] = c, b
        B /= (2 * A)[:, None, None]
        D = E / (1 - nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
        self.ke = t * A[:, None, None] * np.einsum("eki,kl,elj->eij", B, D, B)  # (m, 6, 6) at E
        self.B, self.D = B, D
        dofs = np.stack([2 * mesh.tris, 2 * mesh.tris + 1], axis=2).reshape(-1, 6)
        self.dofs = dofs
        self.rows = np.repeat(dofs, 6, axis=1).ravel()
        self.cols = np.tile(dofs, (1, 6)).ravel()
        self.n = 2 * len(mesh.nodes)

    def solve(self, Ee: np.ndarray, loads: np.ndarray, fixed: np.ndarray, return_solver: bool = False,
              springs: sp.spmatrix | None = None):
        """loads (n_dof, k) -> displacements (n_dof, k); `fixed`: constrained dof ids; `springs`: a constant
        support stiffness added to K (e.g. a rigid part bearing along surface normals). `return_solver`: also
        return a function solving K v = rhs with the same factorization (adjoint solves)."""
        K = sp.csc_matrix(((self.ke * Ee[:, None, None]).ravel(), (self.rows, self.cols)), shape=(self.n, self.n))
        if springs is not None:
            K = (K + springs).tocsc()
        free = np.setdiff1d(np.arange(self.n), fixed)
        u = np.zeros_like(loads)
        lu = spla.splu(K[free][:, free].tocsc())
        u[free] = lu.solve(loads[free])
        if not return_solver:
            return u

        def again(rhs: np.ndarray) -> np.ndarray:
            v = np.zeros_like(rhs)
            v[free] = lu.solve(rhs[free])
            return v

        return u, again

    def solid_stress(self, u: np.ndarray) -> np.ndarray:
        """Element stress (sx, sy, txy) of fully dense material for displacements u (n_dof,): (m, 3)."""
        return np.einsum("eij,ej->ei", self.B, u[self.dofs]) @ self.D.T

    def element_energy(self, u: np.ndarray) -> np.ndarray:
        """u_e^T k_e u_e per element (at unit modulus factor), for each load column: (m, k)."""
        ue = u[self.dofs]  # (m, 6, k)
        return np.einsum("eik,eij,ejk->ek", ue, self.ke, ue)

    def von_mises(self, u: np.ndarray, Ee: np.ndarray) -> np.ndarray:
        """Element von Mises (MPa if E in MPa, u in mm), stress scaled by the element modulus factor."""
        eps = np.einsum("eij,ej->ei", self.B, u[self.dofs])
        s = Ee[:, None] * eps @ self.D.T
        return np.sqrt(s[:, 0] ** 2 - s[:, 0] * s[:, 1] + s[:, 1] ** 2 + 3 * s[:, 2] ** 2)


def normal_springs(n_dof: int, nodes: np.ndarray, normals: np.ndarray, k: float) -> sp.csr_matrix:
    """k n n^T on each node's (x, y) dofs: the node is held along its normal only (frictionless bearing)."""
    r, c, v = [], [], []
    for i, (nx, ny) in zip(nodes, normals):
        d = [2 * i, 2 * i + 1]
        for a, na in zip(d, (nx, ny)):
            for b, nb in zip(d, (nx, ny)):
                r.append(a)
                c.append(b)
                v.append(k * na * nb)
    return sp.csr_matrix((v, (r, c)), shape=(n_dof, n_dof))


def boundary_normals(mesh: TriMesh) -> dict[int, np.ndarray]:
    """Outward unit normal at each boundary node (averaged over its boundary edges)."""
    xy, c = mesh.nodes, mesh.centroids
    e = np.concatenate([mesh.tris[:, [0, 1]], mesh.tris[:, [1, 2]], mesh.tris[:, [2, 0]]])
    owner = np.tile(np.arange(len(mesh.tris)), 3)
    key = np.sort(e, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    acc: dict[int, np.ndarray] = {}
    for idx in np.nonzero(cnt[inv.ravel()] == 1)[0]:
        a, b = e[idx]
        d = xy[b] - xy[a]
        nrm = np.array([d[1], -d[0]]) / np.linalg.norm(d)
        if nrm @ (c[owner[idx]] - (xy[a] + xy[b]) / 2) > 0:
            nrm = -nrm
        for n_ in (a, b):
            acc[int(n_)] = acc.get(int(n_), 0) + nrm
    return {k: v / np.linalg.norm(v) for k, v in acc.items() if np.linalg.norm(v) > 1e-9}

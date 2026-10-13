"""2D meshing of a weapon profile with gmsh: second-order triangles (CalculiX CPS6).

The profile face goes to gmsh as BREP, so gmsh meshes the exact OpenCascade geometry.
Node/edge selection for loads and supports is done geometrically on the result (see `select`), which keeps
it independent of gmsh's entity numbering.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import build123d as bd
import gmsh
import numpy as np


@dataclass
class Mesh2D:
    nodes: np.ndarray  # (n, 2) xy in mm; row i is node id i+1
    elems: np.ndarray  # (m, 6) node ids (1-based), gmsh/CalculiX order: 3 corners then 3 midsides

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def edges(self) -> np.ndarray:
        """(m, 3, 3) per element: for edges 1-2, 2-3, 3-1 -> (corner a, corner b, midside) node ids."""
        e = self.elems
        return np.stack([e[:, [0, 1, 3]], e[:, [1, 2, 4]], e[:, [2, 0, 5]]], axis=1)


def _refine(refine: list[tuple[float, float, float, float]], size_mm: float) -> None:
    """Size field: element size `s` within `r` of each point (x, y, s, r), growing linearly to the global
    size over another 3 r. A formula field, so no geometric points (an orphan node breaks the solve)."""
    fields = []
    for x, y, s, r in refine:
        f = gmsh.model.mesh.field.add("MathEval")
        d = f"Sqrt((x-({x}))^2+(y-({y}))^2)"
        gmsh.model.mesh.field.setString(f, "F", f"Min({size_mm},Max({s},{s}+({size_mm}-{s})*({d}-{r})/{3 * r}))")
        fields.append(f)
    mn = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(mn, "FieldsList", fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(mn)


def mesh_profile(face: bd.Face, size_mm: float, min_size_mm: float | None = None,
                 curvature_elems: int = 24, refine: list[tuple[float, float, float, float]] = ()) -> Mesh2D:
    """`refine`: (x, y, size, radius) local refinement, e.g. at a small contact patch."""
    min_size_mm = min([min_size_mm or size_mm / 6] + [s for _, _, s, _ in refine])
    with tempfile.TemporaryDirectory() as tmp:
        brep = Path(tmp) / "profile.brep"
        bd.export_brep(face, str(brep))
        gmsh.initialize(interruptible=False)
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("profile")
            gmsh.model.occ.importShapes(str(brep))
            gmsh.model.occ.synchronize()
            gmsh.option.setNumber("Mesh.MeshSizeMax", size_mm)
            gmsh.option.setNumber("Mesh.MeshSizeMin", min_size_mm)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature_elems)
            if refine:
                _refine(list(refine), size_mm)
            gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay: good quality triangles
            gmsh.option.setNumber("Mesh.ElementOrder", 2)
            gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)  # midside nodes on the true curves
            gmsh.model.mesh.generate(2)

            tags, coords, _ = gmsh.model.mesh.getNodes()
            xy = coords.reshape(-1, 3)[:, :2]
            renum = np.zeros(int(tags.max()) + 1, dtype=np.int64)
            renum[tags.astype(np.int64)] = np.arange(1, len(tags) + 1)
            order = np.argsort(renum[tags.astype(np.int64)])
            types, _, enodes = gmsh.model.mesh.getElements(2)
            tri6 = [n for t, n in zip(types, enodes) if t == 9]  # gmsh type 9 = 6-node triangle
            if not tri6:
                raise RuntimeError("gmsh produced no second-order triangles")
            elems = renum[tri6[0].astype(np.int64)].reshape(-1, 6)
        finally:
            gmsh.finalize()
    nodes = xy[order]
    # Mirrored (e.g. CW-spin) faces have a -Z normal and mesh clockwise; CalculiX needs CCW elements.
    n, e = nodes, elems - 1
    signed = ((n[e[:, 1], 0] - n[e[:, 0], 0]) * (n[e[:, 2], 1] - n[e[:, 0], 1])
              - (n[e[:, 2], 0] - n[e[:, 0], 0]) * (n[e[:, 1], 1] - n[e[:, 0], 1]))
    flip = signed < 0
    elems[flip] = elems[flip][:, [0, 2, 1, 5, 4, 3]]
    return Mesh2D(nodes=nodes, elems=elems)


# --- geometric selection ----------------------------------------------------------------------

def nodes_on_circle(mesh: Mesh2D, cx: float, cy: float, r: float, tol: float = 1e-3) -> np.ndarray:
    d = np.hypot(mesh.nodes[:, 0] - cx, mesh.nodes[:, 1] - cy)
    return np.nonzero(np.abs(d - r) < tol)[0] + 1


def dist_to_segment(pts: np.ndarray, a, b) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((pts - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)


def edges_on_segment(mesh: Mesh2D, a, b, tol: float = 1e-3) -> list[tuple[int, int]]:
    """(element id, CalculiX face label 3/4/5) of boundary edges lying on segment a-b."""
    on = dist_to_segment(mesh.nodes, a, b) < tol
    out = []
    for ei, el_edges in enumerate(mesh.edges()):
        for k, (n1, n2, nm) in enumerate(el_edges):
            if on[n1 - 1] and on[n2 - 1] and on[nm - 1]:
                out.append((ei + 1, 3 + k))
    return out


def boundary_normals(mesh: Mesh2D) -> dict[int, np.ndarray]:
    """Outward unit normal at every boundary node (node id -> (nx, ny)), averaged over its boundary edges."""
    e = mesh.edges().reshape(-1, 3)
    key = np.sort(e[:, :2], axis=1)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    owner = np.repeat(np.arange(len(mesh.elems)), 3)
    acc: dict[int, np.ndarray] = {}
    xy = mesh.nodes
    for i in np.nonzero(counts[inv.ravel()] == 1)[0]:
        a, b, m = e[i]
        d = xy[b - 1] - xy[a - 1]
        nrm = np.array([d[1], -d[0]]) / np.linalg.norm(d)
        centroid = xy[mesh.elems[owner[i], :3] - 1].mean(axis=0)
        if nrm @ (centroid - (xy[a - 1] + xy[b - 1]) / 2) > 0:
            nrm = -nrm  # point away from the material
        for n in (a, b, m):
            acc[int(n)] = acc.get(int(n), 0) + nrm
    return {n: v / np.linalg.norm(v) for n, v in acc.items() if np.linalg.norm(v) > 1e-9}

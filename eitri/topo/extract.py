"""Density field -> pocket outlines (closed point lists, saved as JSON for the `topo` weapon type).

The 0.5 iso-line of the nodal density bounds the voids; each void becomes a pocket (one reaching the outline
becomes a cut that runs past it): its contour is smoothed (periodic moving average, which rounds the mesh's
staircase) and resampled. Fillets,
clipping to the free region and the exact mass/balance are applied when the weapon is built
(eitri/geometry/topo.py), so the pockets here are only shapes.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402


def _area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _smooth(p: np.ndarray, spacing: float, passes: int = 3) -> np.ndarray:
    """Resample a closed polyline every `spacing` mm and smooth it with a periodic moving average."""
    q = np.vstack([p, p[:1]])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
    n = max(16, int(s[-1] / spacing))
    t = np.linspace(0, s[-1], n, endpoint=False)
    r = np.c_[np.interp(t, s, q[:, 0]), np.interp(t, s, q[:, 1])]
    for _ in range(passes):
        r = (np.roll(r, 1, axis=0) + 2 * r + np.roll(r, -1, axis=0)) / 4
    return r


def void_regions(nodes: np.ndarray, tris: np.ndarray, nodal_density: np.ndarray, outline_poly, level: float = 0.5,
                 min_area_mm2: float = 30.0, band_mm: float = 5.0):
    """Shapely polygons of the regions with density below `level`. Voids that reach the domain's outline
    (`outline_poly`: the outline without holes) are extended past it into a `band_mm` strip outside, so
    cutting them from the exact outline is a clean boolean (no slivers along the edge)."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    tri = Triangulation(nodes[:, 0], nodes[:, 1], tris)
    fig, ax = plt.subplots()
    cs = ax.tricontourf(tri, nodal_density, levels=[-1.0, level])
    plt.close(fig)
    region = Polygon()
    for path in cs.get_paths():
        for ring in path.to_polygons(closed_only=True):
            if len(ring) >= 4:
                region = region.symmetric_difference(Polygon(ring).buffer(0))  # even-odd: holes cancel
    band = outline_poly.buffer(band_mm) - outline_poly
    out = []
    for g in getattr(region, "geoms", [region]):
        if g.is_empty or g.area < min_area_mm2:
            continue
        if g.buffer(0.2).intersects(band):
            g = unary_union([g, band.intersection(g.buffer(3.0))])
        out.append(g)
    return out


def void_contours(nodes: np.ndarray, tris: np.ndarray, nodal_density: np.ndarray, outline_poly=None,
                  level: float = 0.5, min_area_mm2: float = 30.0, spacing_mm: float = 1.5) -> list[np.ndarray]:
    """Closed outlines (CCW point arrays) of the voids, smoothed and resampled."""
    if outline_poly is None:
        from shapely.geometry import MultiPoint
        outline_poly = MultiPoint(nodes).convex_hull
    out = []
    for g in void_regions(nodes, tris, nodal_density, outline_poly, level, min_area_mm2):
        for part in getattr(g, "geoms", [g]):
            p = _smooth(np.asarray(part.exterior.coords)[:-1], spacing_mm)
            out.append(p if _area(p) > 0 else p[::-1])
    return out


def save(path: Path, contours: list[np.ndarray], meta: dict) -> None:
    path.write_text(json.dumps({"meta": meta, "pockets": [np.round(c, 4).tolist() for c in contours]}))


def load(path: str | Path) -> list[np.ndarray]:
    return [np.asarray(c, float) for c in json.loads(Path(path).read_text())["pockets"]]

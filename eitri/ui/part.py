"""STEP part -> a viewer bundle: triangles per B-rep face (unshared, so creases stay sharp), the model's edges as
polylines and per-face descriptors (type, area) for picking. Cached next to the STEP.

    manifest.json   kind "part", counts, bbox, volume, faces [{type, area_mm2}]
    mesh.bin        float32 positions (n x 3), uint32 triangles (m x 3)
    edges.bin       float32 line segments (k x 2 x 3)
    faceid.bin      uint32 B-rep face index per triangle
"""

import json
from pathlib import Path

import numpy as np

_TYPES = {0: "plane", 1: "cylinder", 2: "cone", 3: "sphere", 4: "torus", 5: "bezier", 6: "bspline",
          7: "revolution", 8: "extrusion", 9: "offset", 10: "other"}


def build(step: Path, out: Path, deflection_mm: float = 0.03, angle_rad: float = 0.25) -> Path:
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
    from OCP.BRepGProp import BRepGProp
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.GCPnts import GCPnts_QuasiUniformDeflection
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_REVERSED
    from OCP.TopLoc import TopLoc_Location
    import build123d as bd

    part = bd.import_step(str(step))
    shape = part.wrapped
    BRepMesh_IncrementalMesh(shape, deflection_mm, False, angle_rad, True)

    pos, tris, fid, finfo = [], [], [], []
    base = 0
    for i, face in enumerate(part.faces(), start=1):
        f = face.wrapped
        loc = TopLoc_Location()
        t = BRep_Tool.Triangulation_s(f, loc)
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(f, props)
        finfo.append({"type": _TYPES.get(int(BRepAdaptor_Surface(f).GetType()), "other"),
                      "area_mm2": round(props.Mass(), 3)})
        if t is None:
            continue
        tr = loc.Transformation()
        p = np.array([[*(lambda q: (q.X(), q.Y(), q.Z()))(t.Node(k).Transformed(tr))]
                      for k in range(1, t.NbNodes() + 1)])
        tri = np.array([t.Triangle(k).Get() for k in range(1, t.NbTriangles() + 1)]) - 1
        if f.Orientation() == TopAbs_REVERSED:
            tri = tri[:, [0, 2, 1]]
        pos.append(p)
        tris.append(tri + base)
        fid.append(np.full(len(tri), i - 1))
        base += len(p)

    segs = []
    for edge in part.edges():
        e = edge.wrapped
        if BRep_Tool.Degenerated_s(e):
            continue
        c = BRepAdaptor_Curve(e)
        d = GCPnts_QuasiUniformDeflection(c, deflection_mm)
        if not d.IsDone() or d.NbPoints() < 2:
            continue
        q = np.array([[d.Value(k).X(), d.Value(k).Y(), d.Value(k).Z()] for k in range(1, d.NbPoints() + 1)])
        segs.append(np.stack([q[:-1], q[1:]], axis=1))

    P0 = np.concatenate(pos)
    (x0, y0, z0), (x1, y1, z1) = P0.min(axis=0), P0.max(axis=0)
    vol = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, vol)

    P = np.concatenate(pos).astype(np.float32)
    T = np.concatenate(tris).astype(np.uint32)
    out.mkdir(parents=True, exist_ok=True)
    (out / "mesh.bin").write_bytes(P.tobytes() + T.tobytes())
    (out / "edges.bin").write_bytes(np.concatenate(segs).astype(np.float32).tobytes() if segs else b"")
    (out / "faceid.bin").write_bytes(np.concatenate(fid).astype(np.uint32).tobytes())
    manifest = {"version": 1, "kind": "part", "title": step.name, "step": str(step.name),
                "n_nodes": int(len(P)), "n_tris": int(len(T)), "n_segments": int(sum(len(s) for s in segs)),
                "bbox_mm": [[float(x0), float(y0), float(z0)], [float(x1), float(y1), float(z1)]],
                # OCC's 3D volume of B-spline walled plates is ~60 mm^3 off (CLAUDE.md): display only
                "volume_mm3": vol.Mass(), "faces": finfo}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return out

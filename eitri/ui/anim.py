"""One animation format for every simulation, so a single viewer plays them all.

An animation is a folder:
    manifest.json   kind, title, times (+ unit), fields (name, unit, min, max), counts, flags
    mesh.bin        float32 positions (n x 3), then uint32 triangles (m x 3)   -- the surface, frame 0
    frames.bin      float32, per frame: displacement (n x 3, if has_disp), then each field (n)
    vis.bin         uint8, per frame: triangle visible (m)                       -- if has_vis (eroded cells)

Sources:
    explicit   OpenCourant VTK series (vtk/*.vtk + .pvd): the 3D surface, displacement, plastic strain, von
               Mises, speed; failed elements disappear (vis)
    fea        a static run's CalculiX results (fea/<mat>/centrifugal|impact .inp/.frd): the hit force ramps
               from 0 to the first-yield force on top of the spinning stress
    topo       a topology run's density history (density_history.npz): the density field, iteration by iteration
"""

import json
from pathlib import Path

import numpy as np

TET_FACES = [(0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)]
HEX_FACES = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]


def write(out: Path, kind: str, title: str, positions: np.ndarray, tris: np.ndarray, times, time_unit: str,
          fields: list[tuple[str, str]], frames: list[dict], disp: list[np.ndarray] | None = None,
          vis: list[np.ndarray] | None = None, extra: dict | None = None) -> Path:
    """frames[k][field name] = nodal values; disp[k] = (n, 3); vis[k] = (m,) bool."""
    out.mkdir(parents=True, exist_ok=True)
    pos = np.asarray(positions, np.float32)
    tri = np.asarray(tris, np.uint32)
    with open(out / "mesh.bin", "wb") as fh:
        fh.write(pos.tobytes())
        fh.write(tri.tobytes())
    with open(out / "frames.bin", "wb") as fh:
        for k, fr in enumerate(frames):
            if disp is not None:
                fh.write(np.asarray(disp[k], np.float32).tobytes())
            for name, _ in fields:
                fh.write(np.asarray(fr[name], np.float32).tobytes())
    if vis is not None:
        with open(out / "vis.bin", "wb") as fh:
            for v in vis:
                fh.write(np.asarray(v, np.uint8).tobytes())
    fmeta = []
    for name, unit in fields:
        allv = np.concatenate([np.asarray(fr[name], float) for fr in frames])
        allv = allv[np.isfinite(allv)]
        fmeta.append({"name": name, "unit": unit, "min": float(allv.min()) if allv.size else 0.0,
                      "max": float(allv.max()) if allv.size else 1.0})
    span = float(np.ptp(pos, axis=0).max()) if len(pos) else 1.0
    manifest = {"version": 1, "kind": kind, "title": title, "n_nodes": int(len(pos)), "n_tris": int(len(tri)),
                "times": [float(t) for t in times], "time_unit": time_unit, "fields": fmeta,
                "has_disp": disp is not None, "has_vis": vis is not None, "size_mm": span} | (extra or {})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return out


# --- legacy ASCII VTK (anim_to_vtk output), numpy-fast ------------------------------------------------
def read_vtk(path: Path) -> dict:
    """Points, cells, point and cell data of an ASCII legacy VTK file (numbers on fixed-width lines)."""
    lines = path.read_text(errors="ignore").split("\n")

    def numbers(i: int, count: int) -> np.ndarray:
        per = max(1, len(lines[i].split()))
        n = -(-count // per)
        return np.array(" ".join(lines[i:i + n]).split(), dtype=float)[:count]

    out: dict = {"point": {}, "cell": {}}
    sect, cnt = None, 0
    i = 0
    while i < len(lines):
        tok = lines[i].split()
        if not tok:
            i += 1
            continue
        key = tok[0]
        if key == "POINTS":
            n = int(tok[1])
            out["points"] = numbers(i + 1, 3 * n).reshape(n, 3)
        elif key == "CELLS":
            nc = int(tok[1])
            out["cells"] = [np.array(lines[i + 1 + k].split()[1:], dtype=np.int64) for k in range(nc)]
            i += nc
        elif key in ("POINT_DATA", "CELL_DATA"):
            sect, cnt = ("point" if key == "POINT_DATA" else "cell"), int(tok[1])
        elif key == "SCALARS" and sect:
            out[sect][tok[1]] = numbers(i + 2, cnt)  # line i + 1: LOOKUP_TABLE
        elif key == "VECTORS" and sect:
            out[sect][tok[1]] = numbers(i + 1, 3 * cnt).reshape(cnt, 3)
        i += 1
    return out


def _surface(cells: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Boundary triangles of tets (4 nodes) and hexes (8 nodes): (tris (m, 3), owner cell (m,))."""
    faces, owner = [], []
    for ci, c in enumerate(cells):
        if len(c) == 4:
            for f in TET_FACES:
                faces.append(tuple(c[list(f)]))
                owner.append(ci)
        elif len(c) == 8:
            for f in HEX_FACES:
                q = c[list(f)]
                faces.append((q[0], q[1], q[2], q[3]))
                owner.append(ci)
    key = {}
    for k, f in enumerate(faces):
        s = tuple(sorted(f))
        key.setdefault(s, []).append(k)
    tris, own = [], []
    for ks in key.values():
        if len(ks) != 1:
            continue
        f = faces[ks[0]]
        if len(f) == 3:
            tris.append(f)
            own.append(owner[ks[0]])
        else:
            tris += [(f[0], f[1], f[2]), (f[0], f[2], f[3])]
            own += [owner[ks[0]]] * 2
    return np.array(tris, np.int64), np.array(own, np.int64)


def from_explicit(vtk_dir: Path, out: Path, dt_ms: float, title: str = "explicit impact") -> Path:
    frames_vtk = sorted(vtk_dir.glob("*A[0-9][0-9][0-9].vtk"))
    if not frames_vtk:
        raise FileNotFoundError(f"no VTK frames in {vtk_dir}")
    f0 = read_vtk(frames_vtk[0])
    tris, owner = _surface(f0["cells"])
    used = np.unique(tris)
    remap = -np.ones(len(f0["points"]), np.int64)
    remap[used] = np.arange(len(used))
    tris_s = remap[tris]
    pos0 = f0["points"][used]
    part = f0["cell"].get("PART_ID")
    fields = [("plastic strain", "-"), ("von Mises", "MPa"), ("speed", "m/s"), ("part", "-")]
    frames, disp, vis, times = [], [], [], []
    # cell -> node averaging matrix over the surface nodes
    for k, fv in enumerate(frames_vtk):
        d = f0 if k == 0 else read_vtk(fv)
        p = d["points"][used]
        alive = d["cell"].get("EROSION_STATUS", np.ones(len(d["cells"])))
        eps = d["cell"].get("3DELEM_Plastic_Strain", np.zeros(len(d["cells"])))
        vm = d["cell"].get("3DELEM_Von_Mises", np.zeros(len(d["cells"]))) * 1e3
        nodal = {}
        for name, cv in (("plastic strain", eps), ("von Mises", vm), ("part", part if part is not None else eps * 0)):
            acc = np.zeros(len(used))
            cnt = np.zeros(len(used))
            tv = cv[owner]
            for j in range(3):
                np.add.at(acc, tris_s[:, j], tv)
                np.add.at(cnt, tris_s[:, j], 1)
            nodal[name] = acc / np.maximum(cnt, 1)
        vel = d["point"].get("Velocity")
        nodal["speed"] = np.linalg.norm(vel[used], axis=1) if vel is not None else np.zeros(len(used))
        frames.append(nodal)
        disp.append(p - pos0)
        vis.append(alive[owner] != 0)
        times.append(k * dt_ms)
    return write(out, "explicit", title, pos0, tris_s, times, "ms", fields, frames, disp, vis,
                 extra={"default_field": "plastic strain"})


# --- static FEA (2D CalculiX runs) ---------------------------------------------------------------------
def _read_inp(path: Path) -> tuple[np.ndarray, np.ndarray]:
    nodes, elems, mode = [], [], None
    for line in path.read_text().splitlines():
        if line.startswith("*"):
            u = line.upper()
            mode = "n" if u.startswith("*NODE") and "FILE" not in u and "PRINT" not in u else (
                "e" if u.startswith("*ELEMENT") else None)
            continue
        if mode == "n":
            v = line.split(",")
            nodes.append([float(v[1]), float(v[2])])
        elif mode == "e":
            elems.append([int(x) for x in line.split(",")[1:]])
    return np.array(nodes), np.array(elems) - 1


def from_fea(fea_dir: Path, out: Path, yield_force_kn: float, applied_force_kn: float, yield_mpa: float,
             n_frames: int = 41, title: str = "static FEA: spinning + hit up to first yield") -> Path:
    from eitri.physics.fea.ccx import parse_frd

    nodes, el = _read_inp(fea_dir / "impact.inp")
    cf = parse_frd(fea_dir / "centrifugal.frd", len(nodes))
    hit = parse_frd(fea_dir / "impact.frd", len(nodes))
    # 6-node triangles -> 4 linear ones for drawing
    tris = np.concatenate([el[:, [0, 3, 5]], el[:, [3, 1, 4]], el[:, [5, 4, 2]], el[:, [3, 4, 5]]])
    s_max = yield_force_kn / applied_force_kn if applied_force_kn else 1.0
    frames, disp, times = [], [], []
    for k in range(n_frames):
        sc = s_max * k / (n_frames - 1)
        sig = cf.stress + sc * hit.stress
        sx, sy, sz, txy, tyz, tzx = sig.T
        vm = np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * (txy**2 + tyz**2 + tzx**2))
        frames.append({"von Mises": vm, "of yield": vm / yield_mpa})
        u = cf.disp[:, :3] + sc * hit.disp[:, :3]
        disp.append(u)
        times.append(sc * applied_force_kn)
    pos = np.c_[nodes, np.zeros(len(nodes))]
    return write(out, "fea", title, pos, tris, times, "kN hit force", [("von Mises", "MPa"), ("of yield", "-")],
                 frames, disp, extra={"default_field": "of yield", "disp_scale_default": 50.0})


def from_topo(npz: Path, out: Path, title: str = "topology: density by iteration") -> Path:
    d = np.load(npz)
    nodes, tris, hist = d["nodes"], d["tris"], d["density_history"]
    its = d["iterations"] if "iterations" in d else np.arange(len(hist))
    # element densities -> nodes (area weighted would be nicer; plain average is enough to look at)
    frames = []
    for x in hist:
        acc, cnt = np.zeros(len(nodes)), np.zeros(len(nodes))
        for j in range(3):
            np.add.at(acc, tris[:, j], x)
            np.add.at(cnt, tris[:, j], 1)
        frames.append({"density": acc / np.maximum(cnt, 1)})
    pos = np.c_[nodes, np.zeros(len(nodes))]
    return write(out, "topo", title, pos, tris, its, "iteration", [("density", "-")], frames,
                 extra={"default_field": "density"})

"""The UI's animation format and server helpers (no browser)."""

import json

import numpy as np

from eitri.ui import anim


def test_animation_roundtrip(tmp_path):
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
    tris = np.array([[0, 1, 2], [1, 3, 2]])
    frames = [{"a": np.arange(4) * k} for k in range(3)]
    disp = [np.full((4, 3), 0.1 * k) for k in range(3)]
    vis = [np.array([1, k != 2]) for k in range(3)]
    out = anim.write(tmp_path / "a", "test", "t", pos, tris, [0, 1, 2], "ms", [("a", "-")], frames, disp, vis)
    m = json.loads((out / "manifest.json").read_text())
    assert m["n_nodes"] == 4 and m["n_tris"] == 2 and m["fields"][0]["max"] == 6
    mesh = (out / "mesh.bin").read_bytes()
    assert np.allclose(np.frombuffer(mesh[:48], np.float32).reshape(4, 3), pos)
    assert (np.frombuffer(mesh[48:], np.uint32).reshape(2, 3) == tris).all()
    fr = np.frombuffer((out / "frames.bin").read_bytes(), np.float32).reshape(3, 4 * 3 + 4)
    assert np.allclose(fr[2, :12], 0.2) and np.allclose(fr[2, 12:], [0, 2, 4, 6])
    assert list(np.frombuffer((out / "vis.bin").read_bytes(), np.uint8)) == [1, 1, 1, 1, 1, 0]


def test_tet_surface_has_four_faces():
    tris, owner = anim._surface([np.array([0, 1, 2, 3])])
    assert len(tris) == 4 and (owner == 0).all()
    two, _ = anim._surface([np.array([0, 1, 2, 3]), np.array([1, 2, 3, 4])])
    assert len(two) == 6  # the shared face is interior


def test_vtk_reader(tmp_path):
    p = tmp_path / "f.vtk"
    p.write_text("# vtk DataFile Version 3.0\nx\nASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS 4 float\n"
                 "0 0 0 1 0 0 0 1 0\n0 0 1\nCELLS 1 5\n4 0 1 2 3\nCELL_TYPES 1\n10\nPOINT_DATA 4\n"
                 "VECTORS Velocity float\n1 0 0 1 0 0\n1 0 0 1 0 0\nCELL_DATA 1\nSCALARS PART_ID int 1\n"
                 "LOOKUP_TABLE default\n7\n")
    d = anim.read_vtk(p)
    assert d["points"].shape == (4, 3) and list(d["cells"][0]) == [0, 1, 2, 3]
    assert d["point"]["Velocity"][3, 0] == 1 and d["cell"]["PART_ID"][0] == 7

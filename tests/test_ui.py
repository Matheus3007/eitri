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


def test_part_bundle_from_step(tmp_path):
    import build123d as bd

    from eitri.ui import part

    step = tmp_path / "box.step"
    bd.export_step(bd.Box(10, 20, 4), str(step))
    out = part.build(step, tmp_path / "part")
    m = json.loads((out / "manifest.json").read_text())
    assert len(m["faces"]) == 6 and sorted(f["type"] for f in m["faces"]) == ["plane"] * 6
    assert np.allclose(np.array(m["bbox_mm"][1]) - np.array(m["bbox_mm"][0]), [10, 20, 4], atol=1e-6)
    assert abs(m["volume_mm3"] - 800) < 1e-6
    mesh = (out / "mesh.bin").read_bytes()
    assert len(mesh) == m["n_nodes"] * 12 + m["n_tris"] * 12
    assert len((out / "faceid.bin").read_bytes()) == m["n_tris"] * 4
    assert m["n_segments"] >= 12  # every box edge drawn


def test_explicit_titles_and_steps(tmp_path):
    from eitri.ui import server

    assert server._explicit_title("design_bite_steel") == "Explicit hit · bite · steel"
    assert server._explicit_title("reference_tip3_tpu") == "Reference hit · tip3 · tpu"
    (tmp_path / "hit_0001.out").write_text(
        "   CYCLE    TIME      TIME-STEP  ELEMENT\n"
        "       0  0.0000      0.3000E-04 NODE  1  0.0%\n"
        "    1000  0.0300      0.3000E-04 NODE  1  0.0%\n"
        "    2000  0.0600      0.3000E-04 NODE  1  0.0%\n")
    t = server._explicit_timeline(tmp_path, [0.0, 0.03, 0.045])
    assert t["steps"] == [0, 1000, 1500] and t["step_unit"] == "solver cycle"

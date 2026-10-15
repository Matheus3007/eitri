"""Explicit dynamics (OpenCourant): deck format, units and conservation on a small impact."""

import os
from pathlib import Path

import build123d as bd
import numpy as np
import pytest

from eitri.physics.explicit.radioss import Deck, run

OC = Path(os.environ.get("OPENCOURANT_PATH", Path.home() / "opt" / "OpenCourant"))
pytestmark = pytest.mark.skipif(not (OC / "exec").exists(), reason="OpenCourant not installed")


def test_rigid_block_bounces_off_a_steel_cube_conserving_energy(tmp_path):
    from eitri.physics.explicit.impact3d import read_th
    from eitri.physics.fea.solid3d import mesh_solid
    import subprocess
    from eitri.physics.explicit.radioss import WRAPPER

    m = mesh_solid(bd.Box(20, 20, 20), 5.0)
    n = len(m.nodes)
    d = Deck("cube")
    d.nodes(m.nodes)
    bx = np.array([[x, y, z] for z in (11.0, 21.0) for y in (-15, 15) for x in (-15, 15)])[[0, 1, 3, 2, 4, 5, 7, 6]]
    d.nodes(np.vstack([bx, [[0.0, 0.0, 16.0]]]), first_id=n + 1)
    d.mat_johnson_cook(1, "steel", 7.85e-6, 210.0, 0.3, 1.4, 0.55, 0.05)
    d.mat_elastic(2, "block", 7.85e-6, 210.0, 0.3)
    d.prop_solid(1, "tet10", itetra10=1000)
    d.prop_solid(2, "brick", isolid=24)
    d.part(1, "cube", 1, 1)
    d.part(2, "block", 2, 2)
    d.tetra10(1, m.elems)
    d.brick(2, np.array([[n + 1 + i for i in range(8)]]), first_id=len(m.elems) + 1)
    d.grnod_part(1, "cube", [1])
    d.grnod(2, "block + master", range(n + 1, n + 10))
    d.grnod(4, "block nodes", range(n + 1, n + 9))
    d.rbody(1, "block", n + 9, 4, mass=1.0, J=(100, 100, 100, 0, 0, 0))
    d.surf_part(1, "block", [2])
    d.grnod(3, "bottom", np.nonzero(m.nodes[:, 2] < -10 + 1e-6)[0] + 1)
    d.bcs(1, "bottom", "111", "000", 3)
    d.inivel_tra(1, "down", 2, (0.0, 0.0, -3.0))
    d.inter7(1, "contact", 1, 1, gapmin=0.2)
    d.run(0.8, 0.4, 0.002)
    d.write(tmp_path)
    r = run(tmp_path, "cube", threads=2)
    assert r.ok, r.starter_log[-800:] + r.engine_log[-800:]
    subprocess.run([str(WRAPPER), "th_to_csv", "cubeT01"], cwd=tmp_path, capture_output=True)
    th = read_th(tmp_path / "cubeT01.csv")
    m_block = 1.0 + 30 * 30 * 10 * 7.85e-6
    assert th["KINETIC ENERGY"][0] == pytest.approx(0.5 * m_block * 9.0, rel=1e-4)  # kg mm ms: J
    total = th["KINETIC ENERGY"] + th["INTERNAL ENERGY"] + th["CONTACT ENERGY"]
    assert np.abs(total - total[0]).max() < 0.01 * total[0]  # conserved within 1 %
    assert th["Z-MOMENTUM"][-1] > 0.9 * m_block * 3.0  # bounced back (elastic)

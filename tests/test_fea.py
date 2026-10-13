"""FEA validation: CalculiX plane-stress results against closed form, and invariants of the impact model."""

import build123d as bd
import numpy as np
import pytest

from eitri.evaluate import evaluate
from eitri.physics.fea.ccx import CCX, LoadCase, solve
from eitri.physics.fea.mesh import mesh_profile, nodes_on_circle

pytestmark = pytest.mark.skipif(CCX is None, reason="CalculiX (ccx) not installed")


def lame_hoop(r, a, b, rho, nu, w):
    a, b, r = a * 1e-3, b * 1e-3, r * 1e-3
    return (3 + nu) / 8 * rho * w**2 * (a**2 + b**2 + a**2 * b**2 / r**2 - (1 + 3 * nu) / (3 + nu) * r**2) * 1e-6


def test_spinning_annulus_matches_lame():
    a, b, t, w, rho, nu = 12.7, 150.0, 10.0, 838.0, 7850.0, 0.29
    m = mesh_profile((bd.Circle(b) - bd.Circle(a)).faces()[0], 12.0)
    near = lambda x, y: int(np.argmin(np.hypot(m.nodes[:, 0] - x, m.nodes[:, 1] - y))) + 1  # noqa: E731
    res = solve(m, t, 200000, nu, rho, LoadCase("cf", omega_rad_s=w,
                                                 fixed={near(a, 0): (2,), near(0, a): (1,), near(-a, 0): (2,)}))
    vm = res.von_mises  # radial stress is 0 on free edges, so von Mises = hoop there
    assert vm[nodes_on_circle(m, 0, 0, a) - 1].max() == pytest.approx(lame_hoop(a, a, b, rho, nu, w), rel=0.02)
    assert vm[nodes_on_circle(m, 0, 0, b) - 1].mean() == pytest.approx(lame_hoop(b, a, b, rho, nu, w), rel=0.02)


def test_disc_fea_close_to_analytic_and_mesh_converged(spec):
    coarse, fine = evaluate(spec)["stress"], evaluate(spec, fine=True)["stress"]
    assert coarse["yield_force_kn"] == pytest.approx(fine["yield_force_kn"], rel=0.05)
    assert coarse["centrifugal"]["body_max_vm_mpa"] == pytest.approx(fine["centrifugal"]["body_max_vm_mpa"], rel=0.05)


def test_yield_force_does_not_depend_on_contact_time(spec):
    a = evaluate(spec)["stress"]
    spec.context.contact_time_ms *= 2
    b = evaluate(spec)["stress"]
    assert a["yield_force_kn"] == pytest.approx(b["yield_force_kn"], rel=1e-3)
    assert b["impact_safety_factor"] == pytest.approx(2 * a["impact_safety_factor"], rel=1e-3)


def test_bigger_root_fillet_is_stronger(spec):
    d = spec.fixed_design()
    weak = evaluate(spec, design={**d, "root_fillet_mm": 1.0})["stress"]["yield_force_kn"]
    strong = evaluate(spec, design={**d, "root_fillet_mm": 8.0})["stress"]["yield_force_kn"]
    assert strong > weak

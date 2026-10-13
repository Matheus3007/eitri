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


def test_mirrored_profile_meshes_counterclockwise():
    from eitri.geometry.disc import Disc, DiscParams

    face = Disc(DiscParams(od_mm=200, thickness_mm=8, pocket_count=0), spin="cw").profile()
    m = mesh_profile(face, 10.0)
    n, e = m.nodes, m.elems - 1
    signed = ((n[e[:, 1], 0] - n[e[:, 0], 0]) * (n[e[:, 2], 1] - n[e[:, 0], 1])
              - (n[e[:, 2], 0] - n[e[:, 0], 0]) * (n[e[:, 1], 1] - n[e[:, 0], 1]))
    assert (signed > 0).all()


def test_bearing_hub_reactions_balance_the_hit():
    """The pulley lock (or the bolts) reacts the whole net hit force; the inertia relief takes the torque."""
    from eitri.geometry.hook import Hook, HookParams
    from eitri.physics.fea import analysis as fea

    h = Hook(HookParams(od_mm=230, thickness_mm=6.5, envelope_bite_mm=26.6))
    for mode in ("lock", "bolts"):
        run = fea.run(h, 200000, 0.29, 7850, 700.0, 1e4, 20.0, 9.2, hub_support=mode)
        j = fea.joint_loads(run)
        total = sum(j.values())
        f_dir = -np.array(run.applied[run.applied[:, 0] ** 2 + run.applied[:, 1] ** 2 > 0].sum(axis=0))
        assert np.linalg.norm(total) == pytest.approx(1.0, abs=0.03)
        assert (total @ f_dir) / np.linalg.norm(f_dir) == pytest.approx(1.0, abs=0.03)  # opposes the hit
        assert ("center" in j) == (mode == "lock") or mode == "bolts"

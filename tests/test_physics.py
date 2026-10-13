"""Physics regression tests against closed-form solutions."""

import math

import build123d as bd
import pytest

from eitri.evaluate import evaluate, lame_bore_hoop_mpa
from eitri.physics import drive, impact
from eitri.physics.mass import mass_props
from eitri.spec import Context, Drive

RHO = 7850.0


def test_solid_disc_moi():
    r, t = 0.15, 0.012
    mp = mass_props(bd.Cylinder(r * 1e3, t * 1e3), RHO)
    m = RHO * math.pi * r**2 * t
    assert mp.mass_kg == pytest.approx(m, rel=1e-6)
    assert mp.moi_kgm2 == pytest.approx(0.5 * m * r**2, rel=1e-3)


def test_annulus_moi():
    r1, r2, t = 0.02, 0.1, 0.01
    ring = bd.Cylinder(r2 * 1e3, t * 1e3) - bd.Cylinder(r1 * 1e3, t * 1e3)
    mp = mass_props(ring, RHO)
    m = RHO * math.pi * (r2**2 - r1**2) * t
    assert mp.moi_kgm2 == pytest.approx(0.5 * m * (r1**2 + r2**2), rel=1e-3)


def test_offset_mass_gives_imbalance_and_parallel_axis():
    s = bd.Pos(50, 0, 0) * bd.Cylinder(10, 10)
    mp = mass_props(s, RHO)
    assert mp.cg_offset_mm == pytest.approx(50, rel=1e-6)
    assert mp.static_imbalance_gmm == pytest.approx(mp.mass_kg * 1e3 * 50, rel=1e-6)
    i_cg = 0.5 * mp.mass_kg * 0.01**2
    assert mp.moi_kgm2 == pytest.approx(i_cg + mp.mass_kg * 0.05**2, rel=1e-6)


def test_plate_mass_props_match_the_solid_and_closed_form():
    # (profile, thickness) route: exact 2D integration, extruded analytically
    r1, r2, t = 20.0, 100.0, 10.0
    prof = (bd.Circle(r2) - bd.Circle(r1)).faces()[0]
    solid = bd.extrude(prof, amount=t / 2, both=True).solids()[0]
    a, b = mass_props(solid, RHO), mass_props(solid, RHO, (prof, t))
    m = RHO * math.pi * (r2**2 - r1**2) * t * 1e-9
    assert b.mass_kg == pytest.approx(m, rel=1e-9)
    assert b.moi_kgm2 == pytest.approx(0.5 * m * ((r1 * 1e-3) ** 2 + (r2 * 1e-3) ** 2), rel=1e-9)
    for i in range(3):
        for j in range(3):
            assert b.moi_tensor_cg_kgm2[i][j] == pytest.approx(a.moi_tensor_cg_kgm2[i][j], rel=1e-6, abs=1e-12)


def test_plate_first_moment_is_exact_on_bspline_profiles():
    # OCC's default integration misses B-spline-bounded faces by ~1e-4; the plate route must not
    from eitri.geometry.common import first_moment

    pts = [(60, 0), (20, 45), (-40, 30), (-55, -20), (0, -50), (45, -35)]
    face = bd.Face(bd.Wire([bd.Spline(*pts, (60, 0))]))
    n, poly = 200000, []
    w = face.outer_wire()
    poly = [(q.X, q.Y) for q in (w.position_at(k / n) for k in range(n))]
    x = [q[0] for q in poly]
    y = [q[1] for q in poly]
    cr = [x[k] * y[(k + 1) % n] - x[(k + 1) % n] * y[k] for k in range(n)]
    area = sum(cr) / 2
    sx = sum((x[k] + x[(k + 1) % n]) * cr[k] for k in range(n)) / 6
    a, mx, _ = first_moment(face)
    assert abs(a) == pytest.approx(abs(area), rel=1e-6)
    assert mx * math.copysign(1, area) == pytest.approx(sx, rel=1e-5)  # default integration: ~1e-4


def test_symmetric_disc_is_balanced(spec):
    r = evaluate(spec)
    assert r["mass"]["static_imbalance_gmm"] < 0.1


def test_single_tooth_disc_is_flagged_unbalanced(spec):
    r = evaluate(spec, design={**spec.fixed_design(), "n_teeth": 1, "ramp_deg": 200})
    assert "imbalance" in r["failed"]


def test_spinup_matches_rc_exponential():
    # No current limit, efficiency 1 -> w(t) = w0 (1 - exp(-t/tau)), tau = J R / Kt^2 (ratio 1)
    kv, v, R, J = 200.0, 40.0, 0.05, 0.05
    d = Drive(motor=dict(kv=kv, voltage=v, resistance_ohm=R, speed_efficiency=1.0))
    res = drive.simulate(d, J)
    kt = 60 / (2 * math.pi * kv)
    tau = J * R / kt**2
    assert res.spinup_s_90 == pytest.approx(tau * math.log(10), rel=1e-3)
    assert res.top_rpm == pytest.approx(kv * v, rel=1e-9)


def test_spinup_current_limited_is_linear():
    # Pure current limit -> constant torque -> w(t) = (Kt I / J) t
    kv, v, I, J, ratio = 200.0, 40.0, 100.0, 0.05, 2.0
    d = Drive(motor=dict(kv=kv, voltage=v, current_limit_a=I, speed_efficiency=1.0), ratio=ratio)
    res = drive.simulate(d, J)
    kt = 60 / (2 * math.pi * kv)
    w_top = kv * v * 2 * math.pi / 60 / ratio
    assert res.spinup_s_90 == pytest.approx(0.9 * w_top * J / (ratio * kt * I), rel=1e-3)


def test_impact_limits():
    ctx = Context(opponent_mass_kg=1e9, restitution=0.0)
    hit = impact.analyze(0.05, 800.0, 150.0, 2, 20.0, 0.0, ctx)
    # infinitely heavy opponent, plastic: tooth stops, all weapon energy is lost
    assert hit.weapon_speed_drop_pct == pytest.approx(100, rel=1e-6)
    assert hit.weapon_energy_lost_j == pytest.approx(hit.energy_j, rel=1e-6)
    assert hit.effective_mass_kg == pytest.approx(0.05 / 0.15**2)


def test_bite_is_capped_by_tooth_height():
    ctx = Context(closing_speed_mps=50.0)
    hit = impact.analyze(0.05, 100.0, 150.0, 2, 20.0, 0.0, ctx)
    assert hit.bite_kinematic_mm > 20 and hit.bite_mm == 20


def test_lame_solid_disc_center_stress():
    # Solid disc (a -> 0): sigma_max = (3+nu)/8 rho w^2 b^2 ; with a tiny hole it doubles (stress concentration)
    nu, w, b = 0.29, 800.0, 150.0
    hole = lame_bore_hoop_mpa(RHO, nu, w, 1e-6, b)
    solid = (3 + nu) / 8 * RHO * w**2 * (b * 1e-3) ** 2 * 1e-6
    assert hole == pytest.approx(2 * solid, rel=1e-6)


def test_invalid_design_reports_error_instead_of_crashing(spec):
    r = evaluate(spec, design={**spec.fixed_design(), "n_teeth": 4, "ramp_deg": 85, "land_deg": 10})
    assert r["ok"] is False and "error" in r


def test_minimize_mass_objective_and_energy_floor(spec):
    spec.objective.minimize = "mass"
    spec.constraints.min_energy_j = 1e9
    r = evaluate(spec)
    assert r["objective"]["name"] == "mass"
    assert r["objective"]["score"] == pytest.approx(-r["mass"]["mass_kg"])
    assert "energy_min" in r["failed"]


def test_shaft_loads_closed_form():
    from eitri.physics.shaft import gyro_moment_nm, shaft_loads

    assert gyro_moment_nm(1e-3, 700.0, 180.0) == pytest.approx(1e-3 * 700 * math.pi)
    s = shaft_loads(10e3, 0.0, 40.0, 20.0, 10.0)  # centered: M = F L / 4, each bearing F / 2
    assert s["bending_moment_nm"] == pytest.approx(10e3 * 0.04 / 4)
    assert s["bearing_a_kn"] == pytest.approx(5.0) and s["bearing_b_kn"] == pytest.approx(5.0)
    assert s["bending_stress_mpa"] == pytest.approx(32 * 100 / (math.pi * 0.01**3) * 1e-6)
    o = shaft_loads(10e3, 0.0, 40.0, -10.0, 10.0)  # overhung 10 mm: M = F o, A takes F (L + o) / L
    assert o["bending_moment_nm"] == pytest.approx(100.0) and o["bearing_a_kn"] == pytest.approx(12.5)


def test_toughness_from_charpy_is_sane():
    from eitri.physics.shaft import critical_crack_mm, k_ic_mpa_sqrt_m

    k = k_ic_mpa_sqrt_m(50, 1170)  # 4340 at ~40 HRC: K_IC ~100 MPa*sqrt(m)
    assert 80 < k < 130
    assert k_ic_mpa_sqrt_m(27, 1250) < k  # tougher steel, higher K_IC
    assert critical_crack_mm(k, 1000) > critical_crack_mm(k, 2000)

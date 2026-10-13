"""Fang (asymmetric single-tooth bar) geometry tests: balance is solved, the tooth leads, the part is clean."""

import math

import pytest

from eitri.evaluate import evaluate
from eitri.geometry.fang import BalanceError, Fang, FangParams
from eitri.physics.mass import mass_props


def fang(spin="ccw", **kw) -> Fang:
    return Fang(FangParams(**{"od_mm": 300, "thickness_mm": 12, **kw}), spin)


@pytest.mark.parametrize("kw", [
    {},
    dict(arm_sweep_deg=65, arm_curve=3, hook_deg=15, cw_sweep_deg=30, arm_root_width_mm=38, cw_root_width_mm=82,
         land_deg=6),
    dict(hook_deg=0, arm_root_width_mm=45, thickness_mm=20),
])
def test_counterweight_balances_the_fang(kw):
    f = fang(**kw)
    mp = mass_props(f.solid(), 7850, f.plate())
    assert f.solid().is_valid
    assert mp.static_imbalance_gmm < 2.0  # g*mm, a few hundredths of a gram at the rim
    assert f.derived()["balance_residual_mm3"] < Fang.BALANCE_TOL_MM3


def test_tooth_tip_is_the_outermost_point():
    f = fang()
    bb = f.solid().bounding_box()
    assert bb.max.X == pytest.approx(150, abs=1e-3)
    far = max(math.hypot(v.X, v.Y) for v in f.profile().vertices())
    assert far == pytest.approx(150, abs=1e-3)


def test_counterweight_stays_under_the_face_root():
    f = fang()
    r_face_root = 150 - f.params.face_depth_mm
    for v in f.profile().vertices():
        if v.X < 0:  # counterweight side
            assert math.hypot(v.X, v.Y) <= r_face_root + 1e-6


def test_cw_spin_is_a_mirror():
    a, b = fang().solid(), fang(spin="cw").solid()
    assert a.volume == pytest.approx(b.volume, rel=1e-9)
    (tx, ty), _ = fang(spin="cw").impact_face()
    assert (tx, ty) == pytest.approx((150, 0))


def test_hook_moves_face_root_behind_tip():
    _, (x, y) = fang(hook_deg=15).impact_face()
    assert math.degrees(math.atan2(y, x)) < 0  # root trails the tip for CCW spin


def test_unbalanceable_fang_is_reported():
    with pytest.raises(BalanceError):
        fang(arm_root_width_mm=65, land_deg=20, cw_radius_mm=105).profile()


def test_single_tooth_gets_full_revolution_bite(fang_spec):
    r = evaluate(fang_spec, design={**fang_spec.fixed_design()})
    w = r["drive"]["top_rad_s"]
    assert r["impact"]["bite_kinematic_mm"] == pytest.approx(3.0 * 2 * math.pi / w * 1e3, rel=1e-9)
    assert "imbalance" not in r["failed"]


def test_style_rules_apply_to_fang(fang_spec):
    r = evaluate(fang_spec, design={**fang_spec.fixed_design(), "hook_deg": 0.0})
    assert "style_hook" in r["failed"]

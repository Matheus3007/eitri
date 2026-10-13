"""Hook (gancho) generator and STEP import tests."""

import math

import build123d as bd
import pytest

from eitri.geometry.common import corners, offset_pieces, pocket_wall_mm, worst_concave_corner
from eitri.geometry.hook import BalanceError, Hook, HookParams
from eitri.geometry.step_part import StepPart, StepParams
from eitri.physics.mass import mass_props

BITE = 26.6  # Rebot V3 at 3 m/s


def hook(spin="ccw", **kw) -> Hook:
    return Hook(HookParams(**{"od_mm": 230, "thickness_mm": 6.5, "envelope_bite_mm": BITE, **kw}), spin)


@pytest.mark.parametrize("kw", [{}, dict(cw_size_frac=0.86), dict(hook_deg=18, face_depth_mm=36, wall_mm=10)])
def test_pockets_balance_the_outline(kw):
    h = hook(**kw)
    mp = mass_props(h.solid(), 7850, h.plate())
    assert h.solid().is_valid
    assert mp.static_imbalance_gmm < 0.5  # 5 mm^3 tolerance x 6.5 mm x 7.85 g/cm^3 = 0.26 g*mm
    assert h.derived()["balance_residual_mm3"] < Hook.BALANCE_TOL_MM3


def test_counterweight_stays_inside_bite_envelope():
    h = hook(cw_size_frac=0.86)
    h.profile()  # balance may turn the lobe: it must stay inside the envelope
    assert h.envelope_violation() <= 0.0


def test_bigger_counterweight_gets_a_bigger_thinner_walled_pocket():
    small, big = hook(cw_size_frac=0.76).derived(), hook(cw_size_frac=0.86).derived()
    assert big["cw_pocket_area_mm2"] > small["cw_pocket_area_mm2"]
    assert big["cw_wall_mm"] < small["cw_wall_mm"]


def test_pocket_walls_are_constant_and_respect_the_minimum():
    h = hook(cw_size_frac=0.86, wall_mm=10, arm_wall_mm=12)
    d = h.derived()
    assert d["cw_wall_mm"] >= 10
    assert pocket_wall_mm(h.profile(), h.r_hub) == pytest.approx(min(d["cw_wall_mm"], 12), abs=0.05)


def test_unbalanceable_outlines_say_why():
    with pytest.raises(BalanceError, match="tooth-heavy"):
        hook(cw_size_frac=0.6).profile()
    with pytest.raises(BalanceError, match="heavier than"):
        hook(cw_size_frac=0.95, wall_mm=14).profile()


@pytest.mark.parametrize("spin", ["ccw", "cw"])
def test_no_unfilleted_reentrant_corners(spin):
    # only the tip and the counterweight point may be sharp, and those are convex (no stress raiser)
    f = hook(spin=spin).profile()
    assert worst_concave_corner(f) is None or worst_concave_corner(f).turn_deg < 2.0
    sharp = [c for c in corners(f, 2.0)]
    assert all(not c.concave for c in sharp)
    assert len(sharp) <= 3  # tip, counterweight point, tooth-back blend at most


def test_hub_has_rebot_pulley_lock():
    h = hook()
    f = h.profile()
    center = [w for w in f.inner_wires() if bd.Face(w).center().length < 1e-3 and bd.Face(w).area > 100]
    assert len(center) == 1
    assert bd.Face(center[0]).area == pytest.approx(613.7, abs=1.0)  # Rebot V3's cutout: 614 mm^2
    assert h.bore_r_mm() == pytest.approx(13.0)  # bosses stand 1.5 mm into the 29 mm cutout
    assert len(h.support_circles()) == 7
    plain = hook(lock_d_mm=0, bore_mm=12)
    assert plain.bore_r_mm() == pytest.approx(6.0)


def test_tip_leads_and_is_outermost():
    h = hook()
    far = max(math.hypot(v.X, v.Y) for v in h.profile().vertices())
    assert far == pytest.approx(115, abs=1e-3)
    (tx, ty), (rx, ry) = h.impact_face()
    assert math.atan2(ry, rx) < math.atan2(ty, tx)  # positive hook: root trails the tip (CCW)


def test_step_import_recovers_generated_hook(tmp_path):
    h = hook(spin="cw", hook_deg=14)
    path = tmp_path / "hook.step"
    bd.export_step(bd.Pos(10, -5, 3) * h.solid(), str(path))  # off-axis in the file, like an assembly
    from eitri.cli import export_step_verified

    assert export_step_verified(h.solid(), tmp_path / "check.step", h.plate())["ok"]  # STEP keeps it all
    s = StepPart(StepParams(source=str(path), axis_x_mm=10, axis_y_mm=-5), spin="cw")
    assert s.tip_radius_mm() == pytest.approx(115, abs=1e-3)
    assert s.hook_deg() == pytest.approx(14, abs=0.1)
    assert s.tooth_depth_mm() == pytest.approx(h.tooth_depth_mm(), abs=0.5)
    assert len(s.support_circles()) == 7
    assert s.params.thickness_mm == pytest.approx(6.5, abs=1e-6)
    assert s.geometric_checks(5)[0].ok  # flat plate
    assert s.plate() is not None
    assert mass_props(s.solid(), 7850, s.plate()).mass_kg == pytest.approx(
        mass_props(h.solid(), 7850, h.plate()).mass_kg, rel=1e-4)


def test_windows_paths_map_to_wsl():
    from eitri.geometry.step_part import resolve_path

    assert str(resolve_path("C:\\Users\\me\\w.STEP")) == "/mnt/c/Users/me/w.STEP"


# --- shared 2D helpers -----------------------------------------------------------------------------

def test_corners_classify_convex_and_reentrant():
    ell = bd.Face(bd.Wire.make_polygon([(0, 0), (20, 0), (20, 10), (10, 10), (10, 20), (0, 20)], close=True))
    cs = corners(ell)
    assert len(cs) == 6
    assert sum(c.concave for c in cs) == 1  # the inside corner of the L
    assert all(c.turn_deg == pytest.approx(90) for c in cs)


def test_fillet_corners_rounds_only_what_is_asked():
    from eitri.geometry.common import fillet_corners

    ell = bd.Face(bd.Wire.make_polygon([(0, 0), (20, 0), (20, 10), (10, 10), (10, 20), (0, 20)], close=True))
    f = fillet_corners(ell, 3.0, only=[(10, 10)])
    assert worst_concave_corner(f) is None
    assert len([c for c in corners(f, 2.0) if not c.concave]) == 5
    assert f.area - ell.area == pytest.approx((1 - math.pi / 4) * 9, rel=1e-6)  # the fillet adds material


def test_offset_pieces_survives_splits_and_vanishing():
    dumbbell = (bd.Pos(-20, 0) * bd.Circle(10) + bd.Pos(20, 0) * bd.Circle(10)
                + bd.Rectangle(40, 4)).faces()[0]
    assert len(offset_pieces(dumbbell, -3)) == 2  # the 4 mm neck vanishes, two lobes remain
    assert offset_pieces(dumbbell, -12) == []
    assert offset_pieces(dumbbell, 2)[0].area > dumbbell.area


def test_arm_slot_that_cannot_fit_is_flagged_not_dropped():
    from eitri.spec import Style

    ok = {c.name: c.ok for c in hook().style_checks(Style())}
    assert ok["style_arm_slot"]
    crowded = {c.name: c.ok for c in hook(arm_wall_mm=16, arm_fillet_mm=12).style_checks(Style())}
    assert not crowded["style_arm_slot"]


def test_spine_shoulder_widens_the_tooth_below_its_back():
    from eitri.reference import tooth_widths

    plain, full = tooth_widths(hook()), tooth_widths(hook(spine_shoulder=0.1))
    assert full["3"] == pytest.approx(plain["3"], abs=0.05)  # the tip itself is untouched
    assert full["25"] > plain["25"] + 1.0

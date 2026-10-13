"""Hook (gancho) generator and STEP import tests."""

import math

import build123d as bd
import pytest

from eitri.geometry.hook import Hook, HookParams
from eitri.geometry.step_part import StepPart, StepParams
from eitri.physics.mass import mass_props

BITE = 26.6  # Rebot V3 at 3 m/s


def hook(spin="ccw", **kw) -> Hook:
    return Hook(HookParams(**{"od_mm": 230, "thickness_mm": 6.5, "envelope_bite_mm": BITE, **kw}), spin)


@pytest.mark.parametrize("kw", [{}, dict(cw_size_frac=0.8), dict(hook_deg=18, face_depth_mm=36, wall_mm=10)])
def test_pockets_balance_the_outline(kw):
    h = hook(**kw)
    mp = mass_props(h.solid(), 7850)
    assert h.solid().is_valid
    assert mp.static_imbalance_gmm < 2.0
    assert h.derived()["balance_residual_mm3"] < Hook.BALANCE_TOL_MM3


def test_counterweight_stays_inside_bite_envelope():
    h = hook(cw_size_frac=0.9)
    assert h.envelope_violation() <= 0.0


def test_bigger_counterweight_needs_bigger_pocket():
    small, big = hook(cw_size_frac=0.72).derived(), hook(cw_size_frac=0.85).derived()
    assert big["cw_pocket_span_deg"] > small["cw_pocket_span_deg"]


def test_pocket_is_a_local_cut_not_the_whole_inset():
    # regression: filleting a boolean result used to fillet (and return) the whole inset operand
    h = hook()
    h._cache["outline"] = h._outline()
    pocket = h._cw_pocket(10.0, -205.0)
    assert pocket.area < 0.25 * h._inset().area


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
    assert bd.import_step(str(path)).volume == pytest.approx(h.solid().volume, rel=1e-4)  # STEP keeps it all
    s = StepPart(StepParams(source=str(path), axis_x_mm=10, axis_y_mm=-5), spin="cw")
    assert s.tip_radius_mm() == pytest.approx(115, abs=1e-3)
    assert s.hook_deg() == pytest.approx(14, abs=0.1)
    assert s.tooth_depth_mm() == pytest.approx(h.tooth_depth_mm(), abs=0.5)
    assert len(s.support_circles()) == 7
    assert s.params.thickness_mm == pytest.approx(6.5, abs=1e-6)
    assert s.geometric_checks(5)[0].ok  # flat plate
    assert mass_props(s.solid(), 7850).mass_kg == pytest.approx(mass_props(h.solid(), 7850).mass_kg, rel=1e-4)


def test_windows_paths_map_to_wsl():
    from eitri.geometry.step_part import resolve_path

    assert str(resolve_path("C:\\Users\\me\\w.STEP")) == "/mnt/c/Users/me/w.STEP"

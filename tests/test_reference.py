"""3D out-of-plane FEA validation and the relative 'must not lose against the reference' rule."""

import math

import build123d as bd
import numpy as np
import pytest

from eitri.physics.fea.ccx import CCX, run_job
from eitri.physics.fea.solid3d import _deck, mesh_solid, yield_scale
from eitri.reference import compare


@pytest.mark.skipif(CCX is None, reason="CalculiX (ccx) not installed")
def test_c3d10_cantilever_matches_beam_theory():
    L, b, t, F, E = 100.0, 20.0, 6.5, 100.0, 210000.0
    m = mesh_solid(bd.Pos(L / 2, 0, 0) * bd.Box(L, b, t), 3.0)
    x = m.nodes[:, 0]
    end = np.nonzero(x > L - 1e-6)[0]
    r = run_job(_deck(m, E, 0.3, 7850, np.nonzero(x < 1e-6)[0] + 1, "cant",
                      cloads=[(int(i) + 1, 3, -F / len(end)) for i in end]), len(x), "cant")
    I = b * t**3 / 12
    assert -r.disp[end, 2].mean() == pytest.approx(F * L**3 / (3 * E * I), rel=0.03)
    surf = (x > 40) & (x < 60) & (np.abs(m.nodes[:, 2]) > t / 2 - 1e-6)
    assert np.median(np.abs(r.stress[surf, 0]) / (F * (L - x[surf]) * t / 2 / I)) == pytest.approx(1, rel=0.02)


def test_yield_scale_finds_the_superposed_limit():
    cf = np.zeros((3, 6)); cf[:, 0] = [100, 0, 0]
    hit = np.zeros((3, 6)); hit[:, 0] = [10, 20, 0]
    s = yield_scale(cf, hit, np.array([True, True, True]), 500)
    assert s == pytest.approx(25, rel=1e-9)  # node 2 reaches 500 first (20 * 25); node 1 would need 40
    assert yield_scale(cf, hit, np.array([True, False, False]), 500) == pytest.approx(40, rel=1e-9)


def _profile(overall, regions):
    return {"yield_force_kn": overall, "region": min(regions, key=regions.get), "at_mm": [0, 0],
            "regions_kn": regions}


def _metrics(energy, width, cases):
    return {"energy_j": energy, "tooth_width_mm": {"3": width}, "cases": cases}


def test_compare_is_per_region_and_ignores_regions_far_from_failure():
    ref_case = _profile(10, {"tip": 50, "tooth": 12, "arm": 15, "hub": 10})
    ref = _metrics(350, 13, {c: ref_case for c in ("hit_bite", "hit_tip3", "hit_tip8", "flat_hit")})
    # stronger overall, but its tooth region (12 -> 11) got weaker: must fail even though first yield is higher
    worse_tooth = _profile(11, {"tip": 60, "tooth": 11, "arm": 16, "hub": 12})
    # the tip region dropped 50 -> 13, but 13 > 1.25 x reference first yield (12.5): still compliant
    fine = _profile(11, {"tip": 13, "tooth": 13, "arm": 16, "hub": 11})
    mine = _metrics(360, 13, {"hit_bite": worse_tooth, "hit_tip3": fine, "hit_tip8": fine, "flat_hit": fine})
    res = {c.name: c for c in compare(mine, ref)}
    assert not res["ref_hit_bite"].ok and res["ref_hit_bite"].value == pytest.approx(11 / 12)
    assert res["ref_hit_tip3"].ok and res["ref_energy"].ok and res["ref_tooth_width"].ok
    narrow = compare(_metrics(360, 12, mine["cases"]), ref)
    assert not {c.name: c for c in narrow}["ref_tooth_width"].ok


def test_tooth_widths_of_a_disc_sector_are_its_arcs():
    from eitri.geometry.hook import Hook, HookParams
    from eitri.reference import tooth_widths

    h = Hook(HookParams(od_mm=230, thickness_mm=6.5, envelope_bite_mm=26.6))
    w = tooth_widths(h)
    assert all(w[str(a)] <= w[str(b)] + 1e-6 for a, b in ((3, 5), (5, 10), (10, 15)))  # tooth widens downward
    assert 0 < w["3"] < math.radians(60) * (115 - 3)

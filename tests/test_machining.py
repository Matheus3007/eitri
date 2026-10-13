"""Machined features: exact mass, balance with them, and a solid that matches the mass model."""

import pytest

from eitri.evaluate import build_weapon
from eitri.physics.mass import mass_props
from eitri.spec import Spec


def _spec(**mach) -> Spec:
    # a light counterweight: blind pockets remove only part of the plate, so the balance needs bigger ones
    return Spec.model_validate({"weapon": {"type": "hook"}, "material": {"name": "AR500"}, "drive": {"rpm": 6800},
                                "context": {"opponent_mass_kg": 1.36}, "analysis": {"fea": False},
                                "design": {"od_mm": 230, "thickness_mm": 6.5, "envelope_bite_mm": 26.6,
                                           "cw_size_frac": 0.76, "wall_mm": 6.0},
                                "machining": mach})


@pytest.mark.parametrize("mach", [
    {"pocket_depth_mm": 3.0},
    {"pocket_depth_mm": 4.0, "pocket_sides": 2},
    {"face_bevel_mm": 1.5, "face_bevel_sides": 2},
    {"counterbore_d_mm": 9.0, "counterbore_depth_mm": 3.0},
])
def test_machined_hook_is_balanced_and_its_solid_matches_the_mass_model(mach):
    spec = _spec(**mach)
    w = build_weapon(spec, spec.fixed_design())
    mp = mass_props(w.solid(), 7850, w.plate(), w.mass_parts())
    assert mp.static_imbalance_gmm < 0.1  # the Newton balance saw the features
    assert w.solid().is_valid
    # what the features remove: exact parts vs OCC on the real solid (compared as a difference: OCC's 3D volume
    # of plates with B-spline pocket walls is itself ~60 mm^3 off, the reason mass comes from 2D profiles)
    import build123d as bd

    plain = bd.extrude(w.profile(), amount=3.25, both=True).solids()[0]
    assert plain.volume - w.solid().volume == pytest.approx(sum(v for _, v, _, _ in w.mass_parts()[1:]), abs=6.0)
    if mach.get("pocket_sides") == 2:
        assert abs(mp.cg_mm[2]) < 1e-6  # symmetric pockets keep the CG mid-plane


def test_blind_pockets_are_heavier_than_through_ones_but_stay_balanced():
    plain, blind = _spec(), _spec(pocket_depth_mm=3.0)
    a = build_weapon(plain, plain.fixed_design())
    b = build_weapon(blind, blind.fixed_design())
    ma = mass_props(a.solid(), 7850, a.plate())
    mb = mass_props(b.solid(), 7850, b.plate(), b.mass_parts())
    assert mb.mass_kg > ma.mass_kg and mb.static_imbalance_gmm < 0.1

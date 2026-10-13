"""Geometry unit tests: the disc is what the parameters say it is."""

import math

import build123d as bd
import pytest

from eitri.geometry.disc import Disc, DiscParams
from tests.conftest import BASE_DESIGN


def disc(**kw) -> Disc:
    return Disc(DiscParams(**{**BASE_DESIGN, **kw}))


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_solid_valid_for_tooth_counts(n):
    s = disc(n_teeth=n, ramp_deg=min(90, 300 / n - 15)).solid()
    assert s.is_valid
    assert len(s.solids()) == 1


def test_extents_match_parameters():
    s = disc().solid()
    bb = s.bounding_box()
    assert bb.max.X == pytest.approx(150, abs=1e-3)  # tooth tip at angle 0
    assert bb.size.Z == pytest.approx(12, abs=1e-6)
    assert bb.min.Z == pytest.approx(-6, abs=1e-6)  # centered on z=0


def test_holes_cut_through():
    d = disc()
    assert len(d.profile().inner_wires()) == 1 + 6 + 5


def test_positive_hook_puts_root_behind_tip():
    assert disc(hook_deg=15).face_root_offset_deg() > 0
    assert disc(hook_deg=-15).face_root_offset_deg() < 0
    assert disc(hook_deg=0).face_root_offset_deg() == pytest.approx(0, abs=1e-9)


def test_face_root_lies_on_body_radius():
    d = disc(hook_deg=12)
    x, y = d.face_root(37.0)
    assert math.hypot(x, y) == pytest.approx(d.r_body)


def test_cw_is_mirror_of_ccw():
    a, b = disc().solid(), Disc(DiscParams(**BASE_DESIGN), spin="cw").solid()
    assert a.volume == pytest.approx(b.volume, rel=1e-9)
    assert b.bounding_box().max.X == pytest.approx(150, abs=1e-3)


def test_overfull_sector_is_rejected():
    d = disc(n_teeth=4, ramp_deg=85, land_deg=10)
    assert not next(c for c in d.geometric_checks(5) if c.name == "tooth_sector_fits").ok
    with pytest.raises(ValueError):
        d.solid()


def test_ligament_checks_flag_crowded_pockets():
    failed = {c.name for c in disc(pocket_count=8, pocket_d_mm=45).geometric_checks(5) if not c.ok}
    assert "lig_pocket_to_pocket" in failed


def test_step_round_trip_preserves_volume(tmp_path):
    s = disc().solid()
    p = tmp_path / "w.step"
    bd.export_step(s, str(p))
    back = bd.import_step(str(p))
    assert back.volume == pytest.approx(s.volume, rel=1e-6)

import pytest

from eitri.spec import Spec

BASE_DESIGN = dict(od_mm=300, thickness_mm=12, n_teeth=2, tooth_height_mm=20, hook_deg=0, land_deg=8,
                   ramp_deg=90, bore_mm=25.4, bolt_count=6, bolt_d_mm=6.6, bolt_pcd_mm=50, pocket_count=5,
                   pocket_d_mm=30, pocket_pcd_mm=120)


@pytest.fixture
def spec() -> Spec:
    return Spec.model_validate({"weapon": {"type": "disc"}, "material": {"name": "AR500"},
                                "drive": {"rpm": 8000}, "design": dict(BASE_DESIGN)})


FANG_DESIGN = dict(od_mm=300, thickness_mm=12)


@pytest.fixture
def fang_spec() -> Spec:
    return Spec.model_validate({"weapon": {"type": "fang"}, "material": {"name": "AR500"},
                                "drive": {"rpm": 8000}, "design": dict(FANG_DESIGN)})

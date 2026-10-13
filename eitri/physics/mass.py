"""Exact mass properties from the B-rep solid (OpenCascade integrates the true geometry, no meshing)."""

import math
from dataclasses import asdict, dataclass

import build123d as bd


@dataclass
class MassProps:
    volume_cm3: float
    mass_kg: float
    cg_mm: tuple[float, float, float]
    cg_offset_mm: float  # radial distance of the CG from the spin axis
    static_imbalance_gmm: float  # mass * CG offset
    moi_kgm2: float  # about the spin axis (Z through the origin)
    moi_tensor_cg_kgm2: list[list[float]]  # about the CG, for dynamic (couple) imbalance
    radius_of_gyration_mm: float

    def to_dict(self) -> dict:
        return asdict(self)


def mass_props(solid: bd.Solid, density_kg_m3: float) -> MassProps:
    vol_mm3 = solid.volume
    mass = density_kg_m3 * vol_mm3 * 1e-9
    c = solid.center(bd.CenterOf.MASS)
    tensor_unit = solid.matrix_of_inertia  # unit density, mm^5, about the CG
    to_si = density_kg_m3 * 1e-15
    tensor = [[v * to_si for v in row] for row in tensor_unit]
    e_mm = math.hypot(c.X, c.Y)
    izz = tensor[2][2] + mass * (e_mm * 1e-3) ** 2  # parallel axis to the spin axis
    return MassProps(
        volume_cm3=vol_mm3 * 1e-3,
        mass_kg=mass,
        cg_mm=(c.X, c.Y, c.Z),
        cg_offset_mm=e_mm,
        static_imbalance_gmm=mass * 1e3 * e_mm,
        moi_kgm2=izz,
        moi_tensor_cg_kgm2=tensor,
        radius_of_gyration_mm=math.sqrt(izz / mass) * 1e3,
    )

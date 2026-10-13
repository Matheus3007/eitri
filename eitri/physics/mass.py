"""Exact mass properties from the B-rep (OpenCascade integrates the true geometry, no meshing).

Plates (every generated weapon, and imported ones that pass the flat-plate test) are integrated from their 2D
profile with adaptive, error-controlled integration and extruded analytically. OCC's 3D volume integration
has no working error control (with an epsilon it returns wrong volumes) and its fixed-order default misses
the first moment of B-spline side faces by ~1000 mm^4: ~9 g*mm of fake imbalance on a beetleweight hook.
"""

import math
from dataclasses import asdict, dataclass

import build123d as bd
import numpy as np


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


def _plate(profile: bd.Face, t: float):
    """(volume, CG, unit-density inertia tensor about the CG) of `profile` extruded t, centered on z=0."""
    from eitri.geometry.common import face_props

    g = face_props(profile)
    a, c, m = g.Mass(), g.CentreOfMass(), g.MatrixOfInertia()  # m: area moments about the CG, z = 0
    lam = a * t**3 / 12  # through-thickness term
    tensor = [[t * m.Value(1, 1) + lam, t * m.Value(1, 2), 0.0],
              [t * m.Value(2, 1), t * m.Value(2, 2) + lam, 0.0],
              [0.0, 0.0, t * m.Value(3, 3)]]
    return a * t, bd.Vector(c.X(), c.Y(), 0.0), tensor


def plate_part(face: bd.Face, t: float, z: float) -> tuple[float, np.ndarray, np.ndarray]:
    """(volume, centroid, unit-density tensor about the centroid) of `face` extruded t, centered at height z."""
    v, c, T = _plate(face, t)
    return v, np.array([c.X, c.Y, z]), np.array(T)


def combine(parts) -> tuple[float, "bd.Vector", list[list[float]]]:
    """Sum of signed parts (sign, volume, centroid, tensor about centroid): volume, CG, tensor about the CG."""
    vol = sum(s * v for s, v, _, _ in parts)
    cg = sum(s * v * np.asarray(c) for s, v, c, _ in parts) / vol
    T = np.zeros((3, 3))
    for s, v, c, Ti in parts:
        d = np.asarray(c) - cg
        T += s * (np.asarray(Ti) + v * ((d @ d) * np.eye(3) - np.outer(d, d)))
    return vol, bd.Vector(*cg), T.tolist()


def mass_props(solid: bd.Solid, density_kg_m3: float, plate: tuple[bd.Face, float] | None = None,
               parts=None) -> MassProps:
    """`plate` = (profile, thickness) when the solid is that profile extruded symmetrically about z=0;
    `parts` = exact signed parts (machined plates, see geometry/machining.py)."""
    if parts:
        vol_mm3, c, tensor_unit = combine(parts)
    elif plate is not None:
        vol_mm3, c, tensor_unit = _plate(*plate)
    else:
        vol_mm3, c, tensor_unit = solid.volume, solid.center(bd.CenterOf.MASS), solid.matrix_of_inertia
    mass = density_kg_m3 * vol_mm3 * 1e-9
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

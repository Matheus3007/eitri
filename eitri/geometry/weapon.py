"""Common interface every weapon type (disc, drum, eggbeater) implements.

The physics, FEA, optimizer and report layers only talk to this interface, so adding a weapon type
means adding a geometry module, not touching the rest.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields

import build123d as bd


@dataclass
class Check:
    name: str
    ok: bool
    value: float | None = None
    limit: float | None = None
    detail: str = ""


class Weapon(ABC):
    params: object  # a dataclass of geometric parameters (mm, deg)

    @abstractmethod
    def solid(self) -> bd.Solid:
        """3D solid in mm, spin axis = Z, centered on the origin."""

    @abstractmethod
    def tip_radius_mm(self) -> float:
        """Radius of the striking surface."""

    @abstractmethod
    def n_teeth(self) -> int: ...

    @abstractmethod
    def tooth_depth_mm(self) -> float:
        """Radial depth of the impact face: the most the opponent can be engaged (caps the bite)."""

    @abstractmethod
    def hook_deg(self) -> float: ...

    @abstractmethod
    def profile(self) -> bd.Face:
        """2D outline in XY (with holes), for FEA and rendering."""

    @abstractmethod
    def impact_face(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """(tip, root) of the impact face the FEA loads."""

    @abstractmethod
    def support_circles(self) -> list[tuple[float, float, float]]:
        """(x, y, r) holes through which the hub drives the weapon (FEA supports)."""

    @property
    def r_body(self) -> float:
        """Radius below which material is 'body' (never excluded as tooth-tip denting in the FEA)."""
        return self.tip_radius_mm() - self.tooth_depth_mm()

    def plate(self) -> tuple[bd.Face, float] | None:
        """(profile, thickness) if the solid is the profile extruded symmetrically about z=0 (exact mass
        properties come from the 2D profile then); None for anything else."""
        return self.profile(), self.params.thickness_mm

    def bore_r_mm(self) -> float:
        """Smallest radius of the center cutout (where the centrifugal FEA case pins the part)."""
        return getattr(self.params, "bore_mm", 0.0) / 2

    def has_pulley_lock(self) -> bool:
        """A non-round center cutout that keys the weapon to its pulley (which then carries the hit)."""
        return False

    def center_cutout_r_mm(self) -> float:
        """Largest radius of the center cutout (bore, or the pulley lock): where the pulley/shaft bears."""
        return self.bore_r_mm()

    @abstractmethod
    def geometric_checks(self, min_ligament_mm: float) -> list[Check]:
        """Cheap analytic sanity checks on the parameters (ligaments, fits, angles)."""

    def derived(self) -> dict:
        """Solved/derived geometry worth reporting (e.g. a fang's balancing counterweight)."""
        return {}

    def style_checks(self, style) -> list[Check]:
        """Aesthetic rules from the spec's [style] section (see eitri.spec.Style)."""
        return []

    def param_dict(self) -> dict[str, float]:
        return {f.name: getattr(self.params, f.name) for f in fields(self.params)}

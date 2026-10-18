"""3D-printed TPU armour as an explicit-dynamics opponent: a solid TPU skin (the printed perimeters) over a
homogenized gyroid core (the infill), on both faces of a wall.

    skin   perimeters x line width of solid TPU 95A: Ogden hyperelastic (LAW42) + Prony viscous series
    core   gyroid at `infill` relative density, homogenized as a rate-dependent crushable foam (LAW70)

Numbers and where they come from (units here MPa, mm, ms; converted to GPa on writing):
    solid TPU 95A    E ~ 26 MPa quasi-static (Ultimaker TPU 95A datasheet tensile modulus), density 1210 kg/m^3,
                     elongation at break ~580 %. Ogden: mu1 = 8 MPa, alpha1 = 2 (neo-Hookean term, G0 ~ 8 MPa,
                     E ~ 24 MPa) plus a small high-alpha term for the stiffening at large stretch; nu = 0.48.
    rate dependence  Prony series g1 = 2 G0 (tau 1 us), g2 = 1 G0 (tau 0.1 ms): at impact rates (1e3-1e4 /s)
                     the instantaneous modulus is ~4x the quasi-static one. Split-Hopkinson tests on TPUs of
                     this hardness report 2-5x: the right order, not a measurement of this filament.
    gyroid core      Gibson-Ashby scaling for bending-dominated elastomer lattices: E* ~ C E_s rho^2 with
                     C = 0.5 -> ~0.3 MPa at 15 %; elastic-buckling plateau ~0.06 MPa; densification near
                     1 - 1.4 rho (0.79 at 15 %). Plateau x3 at 1e4 /s (same viscous origin as the solid).

How far to trust it: the order of magnitude of the contact pulse (duration, peak force) and how it compares
between two weapons -- not absolute plastic strains. Unknowns that matter: the real rate dependence of this
filament (+-2x on stiffness), the core's plateau and densification (+-2x: print settings, layer adhesion,
gyroid orientation), skin tearing (not modelled: TPU stretches ~6x before breaking, the tooth can still cut
it), and the wall's own bending between its mounts (the back skin is tied to the robot's mass here: local
crushing only, which shortens the pulse vs a wall that also flexes).

What would validate it, cheapest first:
  1. quasi-static compression of printed coupons (skin-only and wall-sandwich, 30x30 mm): fixes the
     core curve and the skin modulus;
  2. drop-weight test on a wall coupon with a steel striker (accelerometer or high-speed video):
     gives the pulse at ~5 m/s and calibrates the rate factor;
  3. a real hit: high-speed video (>= 5000 fps) of the weapon hitting a TPU wall, or an accelerometer on
     the weapon pod: weapon speed drop and contact duration to compare with the simulation.
"""

from dataclasses import dataclass


@dataclass
class TPUWall:
    skin_mm: float = 2.0  # 5 perimeters x 0.4 mm
    core_mm: float = 11.0  # wall thickness = 2 x skin + core
    infill: float = 0.15  # gyroid relative density
    E_s_mpa: float = 26.0
    rho_s: float = 1210.0  # kg/m^3
    rate_factor: float = 4.0  # instantaneous / quasi-static modulus (Prony)
    core_plateau_mpa: float = 0.06
    core_C: float = 0.5  # E* = C E_s rho^2

    @property
    def thickness_mm(self) -> float:
        return 2 * self.skin_mm + self.core_mm

    def write(self, deck, mid_skin: int, mid_core: int, fid0: int) -> None:
        G0 = self.E_s_mpa / 3.0  # MPa
        mu1, a1, mu2, a2 = 0.99 * G0, 2.0, 0.01 * G0 * 2 / 8, 8.0  # sum mu a / 2 = G0
        g_extra = (self.rate_factor - 1.0) * G0
        # tearing: TPU 95A breaks near 580 % elongation (the tooth cuts it; torn elements are deleted -- and
        # without that, crushed elements distort until the time step collapses)
        deck.fail_tensstrain(mid_skin, 4.5, 5.5)
        deck.fail_tensstrain(mid_core, 4.5, 5.5)
        deck.mat_ogden(mid_skin, "TPU 95A solid (printed perimeters)", self.rho_s * 1e-12 * 1e3, 0.48,
                       [mu1 * 1e-3, mu2 * 1e-3], [a1, a2],
                       prony_g=[2 / 3 * g_extra * 1e-3, 1 / 3 * g_extra * 1e-3], prony_tau=[1e-3, 0.1])
        rho = self.infill
        E_core = self.core_C * self.E_s_mpa * rho**2  # MPa
        eD = 1 - 1.4 * rho
        p = self.core_plateau_mpa
        e_y = p / E_core
        # nominal strain -> nominal stress (MPa): linear, plateau, densification toward the solid
        curve = [(0.0, 0.0), (e_y, p), (0.55 * eD / 0.79, 1.2 * p), (0.9 * eD, 4 * p), (eD, 20 * p),
                 (min(eD + 0.06, 0.97), 0.25 * self.E_s_mpa), (0.98, 2.0 * self.E_s_mpa)]
        deck.funct(fid0, "gyroid core loading (MPa->GPa)", [(e, s * 1e-3) for e, s in curve])
        deck.mat_foam_tab(mid_core, f"TPU gyroid {rho:.0%} (homogenized)", self.rho_s * rho * 1e-12 * 1e3,
                          E_core * 1e-3, 0.05, self.E_s_mpa * 1e-3, 0.98,
                          load=[(fid0, 1e-3, 1.0), (fid0, 10.0, self.rate_factor * 0.75)])

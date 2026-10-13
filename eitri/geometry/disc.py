"""Parametric disc spinner.

Outline, traversed CCW (the default spin direction), repeated for each of `n_teeth` sectors:

    dwell  : arc at r_body          (the gap in front of a tooth, where the opponent enters = bite)
    ramp   : spiral r_body -> r_tip over `ramp_deg`
    land   : arc at r_tip over `land_deg`
    face   : straight impact face from the tip back down to r_body, leaning by `hook_deg`
             (positive hook = tip leads the root, grabs; negative = deflects)

The concave corner at each face root gets a `root_fillet_mm` fillet.
Through features: center bore, bolt circle, and lightening pockets: either a ring of round holes or
"swept" blade-shaped pockets (one set per tooth, outer end trailing the spin, filleted corners), which is
the default look for an aggressive, intentional-looking weapon.
The disc is a constant-thickness extrusion centered on z = 0, spin axis = Z.
All lengths in mm, angles in degrees.
"""

import math
from dataclasses import dataclass

import build123d as bd

from eitri.geometry.weapon import Check, Weapon


@dataclass
class DiscParams:
    od_mm: float
    thickness_mm: float
    n_teeth: int = 2
    tooth_height_mm: float = 20.0
    hook_deg: float = 0.0
    root_fillet_mm: float = 3.0  # fillet where the impact face meets the body (kills the sharp-corner crack starter)
    land_deg: float = 8.0
    ramp_deg: float = 90.0
    bore_mm: float = 25.4
    bolt_count: int = 6
    bolt_d_mm: float = 6.6
    bolt_pcd_mm: float = 50.0
    pocket_style: str = "round"  # "none" | "round" | "swept"
    pocket_phase_deg: float = 0.0  # round: ring rotation; swept: pocket center relative to each tooth face
    # round pockets: a ring of circular holes
    pocket_count: int = 0
    pocket_d_mm: float = 30.0
    pocket_pcd_mm: float = 120.0
    # swept pockets: blade-shaped cutouts, `pockets_per_tooth` per tooth so the pattern follows the teeth
    pockets_per_tooth: int = 1
    pocket_r_in_mm: float = 45.0
    pocket_r_out_mm: float = 105.0
    pocket_span_deg: float = 50.0  # angular width of each pocket
    pocket_sweep_deg: float = 25.0  # how far the outer end trails the inner end, against the spin
    pocket_fillet_mm: float = 6.0

    def __post_init__(self):
        # The optimizer works in floats; counts are rounded here so the rest of the code can trust them.
        self.n_teeth = round(self.n_teeth)
        self.bolt_count = round(self.bolt_count)
        self.pocket_count = round(self.pocket_count)
        self.pockets_per_tooth = round(self.pockets_per_tooth)
        if self.pocket_style not in ("none", "round", "swept"):
            raise ValueError(f"pocket_style must be none|round|swept, got {self.pocket_style!r}")


def _polar(r: float, deg: float) -> tuple[float, float]:
    a = math.radians(deg)
    return (r * math.cos(a), r * math.sin(a))


class Disc(Weapon):
    def __init__(self, params: DiscParams, spin: str = "ccw"):
        self.params = params
        self.spin = spin

    @classmethod
    def from_dict(cls, d: dict[str, float], spin: str = "ccw") -> "Disc":
        return cls(DiscParams(**d), spin)

    # --- derived dimensions -------------------------------------------------------------
    @property
    def r_tip(self) -> float:
        return self.params.od_mm / 2

    @property
    def sector_deg(self) -> float:
        return 360.0 / self.params.n_teeth

    def face_root(self, tip_deg: float) -> tuple[float, float]:
        """Root point of the impact face whose tip is at angle `tip_deg` on r_tip."""
        R, rb, h = self.r_tip, self.r_body, math.radians(self.params.hook_deg)
        tx, ty = _polar(R, tip_deg)
        a = math.radians(tip_deg) + math.pi + h  # inward radial rotated by the hook angle
        dx, dy = math.cos(a), math.sin(a)
        disc = R**2 * math.cos(h) ** 2 - R**2 + rb**2
        if disc < 0:
            raise ValueError(f"hook {self.params.hook_deg} deg too steep for tooth height")
        s = R * math.cos(h) - math.sqrt(disc)
        return (tx + s * dx, ty + s * dy)

    def face_root_offset_deg(self) -> float:
        """How far (deg) the face root sits behind the tip; >0 for positive hook (undercut)."""
        x, y = self.face_root(0.0)
        return -math.degrees(math.atan2(y, x))

    def dwell_deg(self) -> float:
        p = self.params
        return self.sector_deg - p.land_deg - p.ramp_deg - self.face_root_offset_deg()

    # --- geometry -----------------------------------------------------------------------
    def outline(self) -> bd.Wire:
        p, R, rb, S = self.params, self.r_tip, self.r_body, self.sector_deg
        if self.dwell_deg() < 0:
            raise ValueError(
                f"land {p.land_deg} + ramp {p.ramp_deg} + face {self.face_root_offset_deg():.1f} deg "
                f"exceeds the {S:.1f} deg tooth sector"
            )
        edges: list[bd.Edge] = []
        for k in range(p.n_teeth):
            tip = (k + 1) * S  # tip of the tooth that closes this sector
            root_prev = self.face_root(k * S)
            root_prev_deg = k * S - self.face_root_offset_deg()
            ramp_start = tip - p.land_deg - p.ramp_deg
            land_start = tip - p.land_deg
            # dwell: from the previous face root forward to the ramp start
            if ramp_start - root_prev_deg > 1e-6:
                mid = (ramp_start + root_prev_deg) / 2
                edges.append(bd.ThreePointArc(root_prev, _polar(rb, mid), _polar(rb, ramp_start)))
            # ramp: Archimedean spiral, linear in angle
            if p.ramp_deg > 1e-6:
                n = 9
                pts = [
                    _polar(rb + (R - rb) * i / (n - 1), ramp_start + p.ramp_deg * i / (n - 1))
                    for i in range(n)
                ]
                edges.append(bd.Spline(*pts))
            # land: arc at the tip radius
            if p.land_deg > 1e-6:
                edges.append(
                    bd.ThreePointArc(_polar(R, land_start), _polar(R, tip - p.land_deg / 2), _polar(R, tip))
                )
            # impact face
            edges.append(bd.Line(_polar(R, tip), self.face_root(tip)))
        wire = bd.Wire(edges)
        if self.spin == "cw":
            wire = wire.mirror(bd.Plane.XZ)
        return wire

    def holes(self) -> list[tuple[float, float, float]]:
        """(x, y, diameter) of every circular through hole: bore, bolts, round pockets."""
        p = self.params
        out = [(0.0, 0.0, p.bore_mm)] if p.bore_mm > 0 else []
        for i in range(p.bolt_count):
            out.append((*_polar(p.bolt_pcd_mm / 2, 360 * i / p.bolt_count), p.bolt_d_mm))
        if p.pocket_style == "round":
            for i in range(p.pocket_count):
                ang = p.pocket_phase_deg + 360 * i / p.pocket_count
                out.append((*self._mirror(_polar(p.pocket_pcd_mm / 2, ang)), p.pocket_d_mm))
        return out

    # swept pockets ------------------------------------------------------------------------------
    @property
    def n_swept(self) -> int:
        p = self.params
        return p.n_teeth * p.pockets_per_tooth if p.pocket_style == "swept" else 0

    def _swept_theta(self, u: float) -> float:
        """Angular offset (deg) of the pocket walls at normalized radius u in [0, 1]: trailing curve."""
        return -self.params.pocket_sweep_deg * u**1.5

    def swept_pocket(self, center_deg: float) -> bd.Face:
        p = self.params
        r0, r1, half = p.pocket_r_in_mm, p.pocket_r_out_mm, p.pocket_span_deg / 2
        n = 7
        us = [i / (n - 1) for i in range(n)]
        lead = [_polar(r0 + (r1 - r0) * u, center_deg + half + self._swept_theta(u)) for u in us]
        trail = [_polar(r0 + (r1 - r0) * u, center_deg - half + self._swept_theta(u)) for u in us]
        th_out = center_deg + self._swept_theta(1.0)
        edges = [
            bd.Spline(*lead),
            bd.ThreePointArc(lead[-1], _polar(r1, th_out), trail[-1]),
            bd.Spline(*reversed(trail)),
            bd.ThreePointArc(trail[0], _polar(r0, center_deg), lead[0]),
        ]
        face = bd.Face(bd.Wire(edges))
        if p.pocket_fillet_mm > 0:
            face = bd.fillet(face.vertices(), p.pocket_fillet_mm).faces()[0]
        return face.mirror(bd.Plane.XZ) if self.spin == "cw" else face

    def swept_pocket_centers(self) -> list[float]:
        p = self.params
        step = self.sector_deg / p.pockets_per_tooth
        return [k * self.sector_deg + p.pocket_phase_deg + j * step
                for k in range(p.n_teeth) for j in range(p.pockets_per_tooth)]

    def swept_min_spoke_mm(self) -> float:
        """Narrowest spoke between adjacent swept pockets, measured perpendicular to the spoke."""
        p = self.params
        gap_deg = 360 / self.n_swept - p.pocket_span_deg
        r0, r1 = p.pocket_r_in_mm, p.pocket_r_out_mm
        widths = []
        for i in range(11):
            u = i / 10
            r = r0 + (r1 - r0) * u
            dtheta_dr = math.radians(p.pocket_sweep_deg) * 1.5 * math.sqrt(u) / (r1 - r0)
            widths.append(math.radians(gap_deg) * r * math.cos(math.atan(r * dtheta_dr)))
        return min(widths)

    def _mirror(self, pt: tuple[float, float]) -> tuple[float, float]:
        return (pt[0], -pt[1]) if self.spin == "cw" else pt

    def impact_face(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """(tip, root) of the impact face of the tooth at angle 0, in final (spin-mirrored) coordinates."""
        return self._mirror(_polar(self.r_tip, 0.0)), self._mirror(self.face_root(0.0))

    def support_circles(self) -> list[tuple[float, float, float]]:
        """(x, y, radius) of the holes through which the hub drives the disc: bolts, else the bore."""
        p = self.params
        if p.bolt_count:
            return [(x, y, d / 2) for x, y, d in self.holes()[1 if p.bore_mm > 0 else 0:][: p.bolt_count]]
        return [(0.0, 0.0, p.bore_mm / 2)]

    def profile(self) -> bd.Face:
        """2D cross-section in the XY plane (used for FEA and rendering)."""
        face = bd.Face(self.outline())
        r = self.params.root_fillet_mm
        if r > 0:
            roots = [self._mirror(self.face_root(k * self.sector_deg)) for k in range(self.params.n_teeth)]
            verts = [min(face.vertices(), key=lambda v: math.hypot(v.X - x, v.Y - y)) for x, y in roots]
            face = bd.fillet(verts, r).faces()[0]
        cutters = [bd.Pos(x, y) * bd.Circle(d / 2) for x, y, d in self.holes()]
        cutters += [self.swept_pocket(c) for c in self.swept_pocket_centers()] if self.n_swept else []
        if cutters:
            face = face - cutters
            face = face.faces()[0] if hasattr(face, "faces") else face
        return face

    def solid(self) -> bd.Solid:
        t = self.params.thickness_mm
        s = bd.extrude(self.profile(), amount=t / 2, both=True)
        return s.solids()[0]

    # --- weapon interface ---------------------------------------------------------------
    def tip_radius_mm(self) -> float:
        return self.r_tip

    def n_teeth(self) -> int:
        return self.params.n_teeth

    def tooth_depth_mm(self) -> float:
        return self.params.tooth_height_mm

    def hook_deg(self) -> float:
        return self.params.hook_deg

    def style_checks(self, style) -> list[Check]:
        p, out = self.params, []
        if style.min_hook_deg is not None:
            out.append(Check("style_hook", p.hook_deg >= style.min_hook_deg, p.hook_deg, style.min_hook_deg,
                             "teeth must lean into the spin"))
        if style.max_root_fillet_ratio is not None and p.tooth_height_mm > 0:
            ratio = p.root_fillet_mm / p.tooth_height_mm
            out.append(Check("style_gullet", ratio <= style.max_root_fillet_ratio, ratio,
                             style.max_root_fillet_ratio, "root fillet / tooth height"))
        if style.min_pocket_sweep_deg is not None and self.n_swept:
            out.append(Check("style_pocket_sweep", p.pocket_sweep_deg >= style.min_pocket_sweep_deg,
                             p.pocket_sweep_deg, style.min_pocket_sweep_deg))
        return out

    def geometric_checks(self, min_ligament_mm: float) -> list[Check]:
        p, lig = self.params, min_ligament_mm
        checks = [
            Check("tooth_sector_fits", self.dwell_deg() >= 0, self.dwell_deg(), 0.0,
                  "dwell (bite gap) angle left after land + ramp + face, deg"),
            Check("tooth_height_positive", self.r_body > 0 and p.tooth_height_mm > 0, p.tooth_height_mm),
        ]
        hub_outer = p.bore_mm / 2
        if p.bolt_count:
            r_bolt = p.bolt_pcd_mm / 2
            checks.append(Check("lig_bore_to_bolts", r_bolt - p.bolt_d_mm / 2 - p.bore_mm / 2 >= lig,
                                r_bolt - p.bolt_d_mm / 2 - p.bore_mm / 2, lig))
            if p.bolt_count > 1:
                gap = 2 * r_bolt * math.sin(math.pi / p.bolt_count) - p.bolt_d_mm
                checks.append(Check("lig_bolt_to_bolt", gap >= lig, gap, lig))
            hub_outer = r_bolt + p.bolt_d_mm / 2
        if self.n_swept:
            fits = p.pocket_r_out_mm > p.pocket_r_in_mm and p.pocket_span_deg > 0
            checks.append(Check("swept_pocket_shape", fits, p.pocket_r_out_mm - p.pocket_r_in_mm, 0.0))
            inner = p.pocket_r_in_mm - hub_outer
            outer = self.r_body - p.pocket_r_out_mm
            spoke = self.swept_min_spoke_mm() if fits else 0.0
            max_fillet = min(p.pocket_r_out_mm - p.pocket_r_in_mm,
                             math.radians(p.pocket_span_deg) * p.pocket_r_in_mm) / 2
            checks.append(Check("lig_hub_to_pockets", inner >= lig, inner, lig))
            checks.append(Check("lig_pockets_to_rim", outer >= lig, outer, lig, "measured to the body (dwell) radius"))
            checks.append(Check("lig_spoke", spoke >= lig, spoke, lig, "narrowest spoke between swept pockets"))
            checks.append(Check("pocket_fillet_fits", p.pocket_fillet_mm < max_fillet * 0.95,
                                p.pocket_fillet_mm, max_fillet * 0.95))
        elif p.pocket_style == "round" and p.pocket_count:
            r_p = p.pocket_pcd_mm / 2
            inner = r_p - p.pocket_d_mm / 2 - hub_outer
            outer = self.r_body - (r_p + p.pocket_d_mm / 2)
            checks.append(Check("lig_hub_to_pockets", inner >= lig, inner, lig))
            checks.append(Check("lig_pockets_to_rim", outer >= lig, outer, lig,
                                "measured to the body (dwell) radius"))
            if p.pocket_count > 1:
                gap = 2 * r_p * math.sin(math.pi / p.pocket_count) - p.pocket_d_mm
                checks.append(Check("lig_pocket_to_pocket", gap >= lig, gap, lig))
        else:
            checks.append(Check("lig_hub_to_rim", self.r_body - hub_outer >= lig, self.r_body - hub_outer, lig))
        return checks

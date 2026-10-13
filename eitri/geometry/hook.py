"""Asymmetric single-tooth "hook" (gancho): the user's preferred layout, modelled on Rebot V3's weapon.

Canonical frame: spin CCW seen from +Z, tooth tip at angle 0. Angles below are "degrees behind the tip"
(the outline is traced from the tip backwards, against the spin, all the way round to the face root).

Outline (one continuous contour, sharp only at the tip, the face root and the counterweight point):
    face     : straight impact face, tip -> root, `face_depth_mm` deep, leaning `hook_deg`
    back     : tooth back from the tip down to (`back_r_frac` * R, `back_deg` behind)
    spine    : dives from the tooth back to a waist at the hub (`spine_waist_deg` behind), then out to the
               counterweight point P (`cw_point_deg` behind)
    lobe     : counterweight lobe from P over `lobe_span_deg`, bulging out
    gullet   : dives from the lobe to a waist at the hub (`gullet_waist_deg` ahead of the face) and up to
               the face root: the open bite zone in front of the tooth

Bite envelope: with one tooth the opponent advances `bite` per revolution, so a feature phi deg behind the
tooth meets it bite*phi/360 closer in. The counterweight (P and the lobe) is drawn out to that envelope minus
`cw_clearance_mm` -- as big as it can be without becoming a second tooth -- which leaves the outline
counterweight-heavy on purpose (`cw_size_frac` sets how much of the envelope it uses).

Balance: pockets are the outline inset by `wall_mm`, masked to sectors. The arm pocket is a fixed style
feature; the counterweight pocket's angular span and center are solved by Newton iteration so the center of
mass lands on the spin axis (the pocket grows until it balances).
"""

import math
from dataclasses import dataclass

import build123d as bd
import numpy as np

from eitri.geometry.common import (facing_up, first_moment, hooked_face_root, nearest_vertices, polar,
                                   single_face, to_nurbs)
from eitri.geometry.weapon import Check, Weapon


@dataclass
class HookParams:
    od_mm: float
    thickness_mm: float
    face_depth_mm: float = 32.0
    hook_deg: float = 12.0
    root_fillet_mm: float = 3.0
    back_deg: float = 24.0
    back_r_frac: float = 0.84
    hub_d_mm: float = 52.0
    spine_waist_deg: float = 102.0
    cw_point_deg: float = 143.0
    lobe_span_deg: float = 97.0
    gullet_waist_deg: float = 39.0  # ahead of the face
    cw_size_frac: float = 0.72  # counterweight reach as a fraction of the bite envelope (1 = touching it)
    cw_clearance_mm: float = 4.0  # hard minimum gap to the bite envelope
    wall_mm: float = 8.0  # counterweight pocket wall thickness, following the outline
    arm_wall_mm: float = 11.0  # arm slot wall thickness
    web_mm: float = 6.0  # material kept around the hub
    pocket_fillet_mm: float = 7.0
    arm_pocket: float = 1.0  # 1 = slot in the fang arm (style), 0 = solid arm
    head_solid_mm: float = 30.0  # solid tooth head before the arm pocket starts
    bore_mm: float = 12.0
    bolt_count: int = 7
    bolt_d_mm: float = 5.3
    bolt_pcd_mm: float = 36.5
    envelope_bite_mm: float = 0.0  # 0 = use the bite computed from the spec's rpm and closing speed

    def __post_init__(self):
        self.bolt_count = round(self.bolt_count)
        self.arm_pocket = round(self.arm_pocket)


class BalanceError(ValueError):
    pass


def _sector(r0: float, r1: float, a0: float, a1: float) -> bd.Face:
    """Annular sector between radii r0..r1 and CCW angles a0..a1 (deg, a1 > a0, span < 360)."""
    mid = (a0 + a1) / 2
    edges = [
        bd.ThreePointArc(polar(r1, a0), polar(r1, mid), polar(r1, a1)),
        bd.Line(polar(r1, a1), polar(r0, a1)),
        bd.ThreePointArc(polar(r0, a1), polar(r0, mid), polar(r0, a0)),
        bd.Line(polar(r0, a0), polar(r1, a0)),
    ]
    return facing_up(bd.Face(bd.Wire(edges)))


def _fillet_all(face: bd.Face, r: float) -> bd.Face:
    # Rebuild first: build123d's fillet() finds its target through the vertices' topo_parent, which for a
    # boolean result is the *operand* (here the whole inset), not this face.
    face = facing_up(bd.Face(face.outer_wire(), face.inner_wires()))
    for rr in (r, 0.6 * r, 0.3 * r):
        try:
            return single_face(bd.fillet(face.vertices(), rr))
        except Exception:
            continue
    return face


class Hook(Weapon):
    BALANCE_TOL_MM3 = 5.0

    def __init__(self, params: HookParams, spin: str = "ccw"):
        self.params = params
        self.spin = spin
        self._cache: dict = {}

    @classmethod
    def from_dict(cls, d: dict, spin: str = "ccw") -> "Hook":
        return cls(HookParams(**d), spin)

    # --- envelope & key points (canonical frame; `behind` angles are positive degrees) ------------
    @property
    def R(self) -> float:
        return self.params.od_mm / 2

    @property
    def r_hub(self) -> float:
        return self.params.hub_d_mm / 2

    @property
    def bite(self) -> float:
        if self.params.envelope_bite_mm <= 0:
            raise ValueError("hook needs envelope_bite_mm (evaluate fills it from the spec)")
        return self.params.envelope_bite_mm

    def envelope(self, behind_deg: float) -> float:
        return self.R - self.bite * behind_deg / 360

    def _pt(self, r: float, behind_deg: float) -> np.ndarray:
        return np.array(polar(r, -behind_deg))

    def _face_root(self):
        return hooked_face_root(self.R, self.R - self.params.face_depth_mm, 0.0, self.params.hook_deg)

    def _outline(self) -> bd.Face:
        p, R = self.params, self.R
        T = np.array(polar(R, 0.0))
        F = np.array(self._face_root())
        rw = self.r_hub  # the spine waist sits on the hub radius, the gullet waist a little outside
        rB = p.back_r_frac * R
        a_B, a_Ws = p.back_deg, p.spine_waist_deg
        a_P = p.cw_point_deg
        a_L = a_P + p.lobe_span_deg
        a_G = 360 - p.gullet_waist_deg

        def cw_r(a):
            return min(p.cw_size_frac * self.envelope(a), self.envelope(a) - p.cw_clearance_mm)

        r_P, r_L = cw_r(a_P), cw_r(a_L)
        a_M = (a_P + a_L) / 2
        P, L = self._pt(r_P, a_P), self._pt(r_L, a_L)
        M = self._pt(cw_r(a_M) * 1.04, a_M)  # slight bulge so the lobe reads as a lobe
        rg = rw + 5.0
        r_F = float(np.hypot(*F))
        a_F = -math.degrees(math.atan2(F[1], F[0])) % 360  # face root, degrees behind the tip

        def lerp(a, b, f):
            return a + (b - a) * f

        # Control points as fractions of each span, measured off Rebot V3's GanchoUnidente.
        back_mid = self._pt(R - (R - rB) * 0.3, a_B * 0.45)
        spine = [B := self._pt(rB, a_B),
                 self._pt(lerp(rw, rB, 0.56), lerp(a_B, a_Ws, 0.21)),
                 self._pt(lerp(rw, rB, 0.33), lerp(a_B, a_Ws, 0.33)),
                 self._pt(lerp(rw, rB, 0.10), lerp(a_B, a_Ws, 0.55)),
                 self._pt(rw, a_Ws),
                 self._pt(lerp(rw, r_P, 0.18), a_P - 7),
                 self._pt(lerp(rw, r_P, 0.58), a_P + 2.6),
                 P]
        gullet = [L,
                  self._pt(lerp(rg, r_L, 0.54), lerp(a_L, a_G, 0.225)),
                  self._pt(lerp(rg, r_L, 0.26), lerp(a_L, a_G, 0.384)),
                  self._pt(lerp(rg, r_L, 0.05), lerp(a_L, a_G, 0.64)),
                  self._pt(rg, a_G),
                  self._pt(lerp(rg, r_F, 0.13), 360 - 0.345 * p.gullet_waist_deg),
                  self._pt(lerp(rg, r_F, 0.38), 360.0),
                  self._pt(lerp(rg, r_F, 0.68), a_F + 360 - 0.3),
                  F]
        edges = [
            bd.ThreePointArc(tuple(T), tuple(back_mid), tuple(B)),
            bd.Spline(*[tuple(x) for x in spine]),
            bd.ThreePointArc(tuple(P), tuple(M), tuple(L)),
            bd.Spline(*[tuple(x) for x in gullet]),
            bd.Line(tuple(F), tuple(T)),
        ]
        face = facing_up(bd.Face(bd.Wire(edges)))
        if not face.is_valid:
            raise ValueError("outline self-intersects")
        corners = [tuple(F)] if p.root_fillet_mm > 0 else []
        if corners:
            face = single_face(bd.fillet(nearest_vertices(face, corners), p.root_fillet_mm))
        # soften the lobe/gullet junction (the only non-feature corner), like the R15 on Rebot
        try:
            face = single_face(bd.fillet(nearest_vertices(face, [tuple(L)]), 0.15 * p.lobe_span_deg))
        except Exception:
            pass
        return face

    # --- pockets ---------------------------------------------------------------------------------
    def _inset(self, wall: float | None = None) -> bd.Face:
        wall = self.params.wall_mm if wall is None else wall
        key = f"inset{wall}"
        if key not in self._cache:
            inset = bd.offset(self._cache["outline"], amount=-wall, kind=bd.Kind.ARC)
            self._cache[key] = to_nurbs(single_face(inset))  # offset curves don't survive STEP export
        return self._cache[key]

    def _arm_pocket(self) -> bd.Face | None:
        p = self.params
        if not p.arm_pocket:
            return None
        r0, r1 = self.r_hub + p.web_mm, self.R - p.head_solid_mm
        # CCW angles: the arm lies between the gullet waist (ahead) and the spine waist (behind)
        a0, a1 = -(p.spine_waist_deg - 12), p.gullet_waist_deg - 12
        if r1 <= r0 + 5 or a1 <= a0:
            return None
        shape = self._inset(p.arm_wall_mm) & _sector(r0, r1, a0, a1)
        faces = shape.faces() if shape is not None else []
        if not faces:
            return None
        return _fillet_all(max(faces, key=lambda f: f.area), p.pocket_fillet_mm)

    def _cw_window(self) -> tuple[float, float]:
        """CCW angle range the counterweight pocket may occupy (between the two waists, minus webs)."""
        p = self.params
        return -(360 - p.gullet_waist_deg - 10), -(p.spine_waist_deg + 10)

    def _cw_pocket(self, half: float, center: float) -> bd.Face | None:
        p = self.params
        lo, hi = self._cw_window()
        a0, a1 = max(center - half, lo), min(center + half, hi)
        if a1 - a0 < 2:
            return None
        shape = self._inset() & _sector(self.r_hub + p.web_mm, 2 * self.R, a0, a1)
        faces = shape.faces() if shape is not None else []
        if not faces:
            return None
        return _fillet_all(max(faces, key=lambda f: f.area), p.pocket_fillet_mm)

    # --- assembly + balance ------------------------------------------------------------------------
    def _holes(self) -> list[tuple[float, float, float]]:
        p = self.params
        out = [(0.0, 0.0, p.bore_mm)] if p.bore_mm > 0 else []
        for i in range(p.bolt_count):
            out.append((*polar(p.bolt_pcd_mm / 2, 360 * i / p.bolt_count), p.bolt_d_mm))
        return out

    def _moment(self, base_m: np.ndarray, half: float, center: float) -> np.ndarray:
        pk = self._cw_pocket(half, center)
        if pk is None:
            return base_m
        a, mx, my = first_moment(pk)
        return base_m - np.array([mx, my])

    def _build(self):
        if "face" in self._cache:
            return self._cache
        self._cache["outline"] = outline = self._outline()
        base = outline
        arm = self._arm_pocket()
        if arm is not None:
            base = single_face(base - arm)
        _, mx, my = first_moment(base)
        base_m = np.array([mx, my])

        lo, hi = self._cw_window()
        # Start: pocket centered on the excess moment's direction, modest span.
        center = math.degrees(math.atan2(base_m[1], base_m[0]))
        while center > hi:  # unwrap into the (negative, behind-the-tooth) window
            center -= 360
        center = float(np.clip(center, lo + 5, hi - 5))
        half = 10.0  # grow from small: a pocket bigger than the lobe stops changing anything (flat Jacobian)
        x = np.array([half, center])
        m = self._moment(base_m, *x)
        for it in range(1, 25):
            if np.linalg.norm(m) < self.BALANCE_TOL_MM3:
                break
            J = np.zeros((2, 2))
            for k in range(2):
                dx = np.zeros(2)
                dx[k] = 0.5
                J[:, k] = (self._moment(base_m, *(x + dx)) - m) / 0.5
            if abs(np.linalg.det(J)) < 1e-6 * max(np.abs(J).max(), 1e-9) ** 2:
                x[0] *= 0.7  # saturated (pocket fills the lobe): back off and retry
                m = self._moment(base_m, *x)
                continue
            step = -np.linalg.solve(J, m)
            step *= min(1.0, 15.0 / max(abs(step).max(), 1e-9))  # damp: at most 15 deg per step
            x = x + step
            x[0] = float(np.clip(x[0], 1.0, (hi - lo) / 2))
            x[1] = float(np.clip(x[1], lo + x[0] * 0.2, hi - x[0] * 0.2))
            m = self._moment(base_m, *x)
        residual = float(np.linalg.norm(m))
        if residual > 20 * self.BALANCE_TOL_MM3:
            need = math.degrees(math.atan2(base_m[1], base_m[0]))
            raise BalanceError(f"pockets can't balance this outline (residual {residual:.0f} mm^3, excess moment "
                               f"points at {need:.0f} deg): the outline is tooth-heavy -> raise cw_size_frac, "
                               "widen lobe_span_deg or enable arm_pocket")
        face = base
        pk = self._cw_pocket(*x)
        if pk is not None:
            face = single_face(face - pk)
        cutters = [bd.Pos(cx, cy) * bd.Circle(d / 2) for cx, cy, d in self._holes()]
        if cutters:
            face = single_face(face - cutters)
        if self.spin == "cw":
            face = face.mirror(bd.Plane.XZ)
        self._cache.update(face=face, cw_pocket_span_deg=2 * x[0], cw_pocket_center_deg=x[1],
                           balance_residual_mm3=residual, balance_iterations=it)
        return self._cache

    # --- weapon interface --------------------------------------------------------------------------
    def profile(self) -> bd.Face:
        return self._build()["face"]

    def solid(self) -> bd.Solid:
        t = self.params.thickness_mm
        return bd.extrude(self.profile(), amount=t / 2, both=True).solids()[0]

    def tip_radius_mm(self) -> float:
        return self.R

    def n_teeth(self) -> int:
        return 1

    def tooth_depth_mm(self) -> float:
        return self.params.face_depth_mm

    def hook_deg(self) -> float:
        return self.params.hook_deg

    def _mirror(self, pt):
        return (pt[0], -pt[1]) if self.spin == "cw" else pt

    def impact_face(self):
        return self._mirror(polar(self.R, 0.0)), self._mirror(self._face_root())

    def support_circles(self):
        p = self.params
        if p.bolt_count:
            return [(x, y, d / 2) for x, y, d in self._holes()[1 if p.bore_mm > 0 else 0:]]
        return [(0.0, 0.0, p.bore_mm / 2)]

    def derived(self) -> dict:
        b = self._build()
        return {k: b[k] for k in ("cw_pocket_span_deg", "cw_pocket_center_deg", "balance_residual_mm3",
                                  "balance_iterations")} | {"envelope_bite_mm": self.bite,
                                                            "envelope_violation_mm": self.envelope_violation()}

    def envelope_violation(self) -> float:
        """How far (mm) the outline pokes outside the bite envelope, ignoring the tooth itself."""
        outline = self._build()["outline"].outer_wire()
        worst = 0.0
        for t in np.linspace(0, 1, 720):
            q = outline.position_at(t)
            behind = (-math.degrees(math.atan2(q.Y, q.X))) % 360
            if behind < self.params.back_deg + 5 or behind > 355:  # the tooth head may exceed: it's the tooth
                continue
            worst = max(worst, math.hypot(q.X, q.Y) - self.envelope(behind))
        return worst

    def geometric_checks(self, min_ligament_mm: float) -> list[Check]:
        p, lig = self.params, min_ligament_mm
        checks = [
            Check("bite_envelope", self.envelope_violation() <= 0.0, self.envelope_violation(), 0.0,
                  "counterweight must stay inside the bite spiral (never a second tooth)"),
            Check("face_depth_vs_bite", p.face_depth_mm >= self.bite, p.face_depth_mm, self.bite,
                  "the face should be at least as deep as the bite it takes"),
            Check("wall_thickness", p.wall_mm >= lig, p.wall_mm, lig),
        ]
        if p.bolt_count:
            r_bolt = p.bolt_pcd_mm / 2
            checks.append(Check("lig_bore_to_bolts", r_bolt - p.bolt_d_mm / 2 - p.bore_mm / 2 >= lig * 0.5,
                                r_bolt - p.bolt_d_mm / 2 - p.bore_mm / 2, lig * 0.5))
            checks.append(Check("lig_bolts_to_hub_edge", self.r_hub - r_bolt - p.bolt_d_mm / 2 >= lig * 0.5,
                                self.r_hub - r_bolt - p.bolt_d_mm / 2, lig * 0.5))
        return checks

    def style_checks(self, style) -> list[Check]:
        p, out = self.params, []
        if style.min_hook_deg is not None:
            out.append(Check("style_hook", p.hook_deg >= style.min_hook_deg, p.hook_deg, style.min_hook_deg))
        if style.max_root_fillet_ratio is not None:
            ratio = p.root_fillet_mm / p.face_depth_mm
            out.append(Check("style_gullet", ratio <= style.max_root_fillet_ratio, ratio,
                             style.max_root_fillet_ratio))
        return out

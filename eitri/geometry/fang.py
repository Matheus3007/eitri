"""Asymmetric single-tooth "fang" bar: the modern spinner layout (one impactor + a counterweight arm).

Spin axis Z, default spin CCW seen from +Z. The tooth tip sits at angle 0 on the tip radius.

    hub          : boss of `hub_d_mm` carrying the bore and bolt circle
    fang arm     : leaves the hub trailing the tip by `arm_sweep_deg`, then hooks forward into the tooth
                   (curvature concentrated toward the tip, set by `arm_curve`); width tapers from
                   `arm_root_width_mm` at the hub to the tooth head
    tooth head   : straight impact face of radial depth `face_depth_mm` leaning by `hook_deg`, a land arc at
                   the tip radius (`land_deg`, sets how chunky the head is), and a `root_fillet_mm` gullet
    counterweight: a shorter, swept "club" arm reaching `cw_radius_mm`, sweeping back by `cw_sweep_deg`,
                   `cw_root_width_mm` wide at the hub and flaring into a rounded club end. The club's end
                   width and angle are SOLVED (not free) so the part's center of mass sits on the spin axis
    blends       : `blend_mm` fillets wherever the arms meet the hub

Everything ahead of the impact face is open, so the single tooth gets the full bite of a revolution.
"""

import math
from dataclasses import dataclass

import build123d as bd
import numpy as np

from eitri.geometry.common import (corner_turn_deg, facing_up, first_moment, hooked_face_root, nearest_vertices,
                                   polar, single_face)
from eitri.geometry.weapon import Check, Weapon


@dataclass
class FangParams:
    od_mm: float
    thickness_mm: float
    hub_d_mm: float = 90.0
    face_depth_mm: float = 30.0
    hook_deg: float = 10.0
    root_fillet_mm: float = 5.0
    land_deg: float = 8.0
    arm_root_width_mm: float = 50.0
    arm_sweep_deg: float = 45.0
    arm_curve: float = 2.5  # >1 puts the bend toward the tip: trailing arm, forward-hooking tooth
    blend_mm: float = 12.0
    cw_radius_mm: float = 120.0
    cw_sweep_deg: float = 30.0
    cw_root_width_mm: float = 75.0
    bore_mm: float = 25.4
    bolt_count: int = 6
    bolt_d_mm: float = 6.6
    bolt_pcd_mm: float = 50.0

    def __post_init__(self):
        self.bolt_count = round(self.bolt_count)


class BalanceError(ValueError):
    pass


@dataclass
class _Built:
    face: bd.Face
    blend_used_mm: float
    cw_end_width_mm: float
    cw_end_deg: float
    residual_mm3: float
    iterations: int


def _swept_arm(r0: float, r1: float, end_deg: float, sweep_deg: float, curve: float,
               w0: float, w1: float, n: int = 13) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    """Leading/trailing edge points of an arm whose centerline runs from (r0, end-sweep) to (r1, end).

    Returns (lead, trail, end_center, end_tangent). Lead is the +90deg (forward for CCW) offset.
    """
    def center(u):
        return np.array(polar(r0 + (r1 - r0) * u, end_deg - sweep_deg * (1 - u**curve)))

    lead, trail = [], []
    for u in np.linspace(0, 1, n):
        d = center(min(u + 1e-4, 1)) - center(max(u - 1e-4, 0))
        nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        w = w0 + (w1 - w0) * u
        lead.append(center(u) + w / 2 * nrm)
        trail.append(center(u) - w / 2 * nrm)
    d = center(1.0) - center(1 - 1e-4)
    return lead, trail, center(1.0), d / np.linalg.norm(d)


class Fang(Weapon):
    BALANCE_TOL_MM3 = 0.5  # first-moment residual: 0.5 mm^3 x 12 mm plate ~ 0.05 g*mm (CG on the axis)
    MIN_CW_WIDTH_MM = 12.0

    def __init__(self, params: FangParams, spin: str = "ccw"):
        self.params = params
        self.spin = spin
        self._built: _Built | None = None

    @classmethod
    def from_dict(cls, d: dict, spin: str = "ccw") -> "Fang":
        return cls(FangParams(**d), spin)

    # --- key points ----------------------------------------------------------------------------
    @property
    def R(self) -> float:
        return self.params.od_mm / 2

    @property
    def r_hub(self) -> float:
        return self.params.hub_d_mm / 2

    @property
    def r_arm0(self) -> float:
        return 0.3 * self.r_hub  # arms start inside the hub so the union is clean

    def _tip(self):
        return polar(self.R, 0.0)

    def _face_root(self):
        return hooked_face_root(self.R, self.R - self.params.face_depth_mm, 0.0, self.params.hook_deg)

    def _land_end(self):
        return polar(self.R, -self.params.land_deg)

    def _mirror(self, pt):
        return (pt[0], -pt[1]) if self.spin == "cw" else pt

    # --- fang arm ------------------------------------------------------------------------------------
    def _fang_arm(self) -> bd.Face:
        p = self.params
        F, S, T = np.array(self._face_root()), np.array(self._land_end()), self._tip()
        H = (F + S) / 2
        lead, trail, _, _ = _swept_arm(self.r_arm0, float(np.hypot(*H)), math.degrees(math.atan2(H[1], H[0])),
                                       p.arm_sweep_deg, p.arm_curve, p.arm_root_width_mm,
                                       float(np.linalg.norm(F - S)))
        # pull the head ends exactly onto the face root / land end with a smooth u^3 correction
        us = np.linspace(0, 1, len(lead))
        dl, dt = F - lead[-1], S - trail[-1]
        lead = [tuple(pt + dl * u**3) for pt, u in zip(lead, us)]
        trail = [tuple(pt + dt * u**3) for pt, u in zip(trail, us)]
        edges = [
            bd.Spline(*lead),
            bd.Line(lead[-1], T),
            bd.ThreePointArc(T, polar(self.R, -p.land_deg / 2), tuple(S)),
            bd.Spline(*reversed(trail)),
            bd.Line(trail[0], lead[0]),
        ]
        return facing_up(bd.Face(bd.Wire(edges)))

    # --- counterweight club arm -------------------------------------------------------------------------
    def _cw_arm(self, w_end: float, end_deg: float) -> bd.Face:
        p = self.params
        r_end = p.cw_radius_mm - w_end / 2
        lead, trail, c, t = _swept_arm(self.r_arm0, r_end, end_deg, p.cw_sweep_deg, 1.5, p.cw_root_width_mm, w_end)
        cap_mid = c + w_end / 2 * t
        edges = [
            bd.Spline(*[tuple(x) for x in lead]),
            bd.ThreePointArc(tuple(lead[-1]), tuple(cap_mid), tuple(trail[-1])),
            bd.Spline(*[tuple(x) for x in reversed(trail)]),
            bd.Line(tuple(trail[0]), tuple(lead[0])),
        ]
        return facing_up(bd.Face(bd.Wire(edges)))

    def _hub(self) -> bd.Face:
        return bd.Face(bd.Wire([bd.Edge.make_circle(self.r_hub)]))

    def _max_cw_end_width(self) -> float:
        # Wider than this and the club's end cap wraps back around toward the hub: a ball, not a club.
        return 1.3 * (self.params.cw_radius_mm - self.r_hub)

    def _solve_cw(self, q: np.ndarray, w_end: float, end_deg: float) -> tuple[float, float]:
        """Club end width and end angle whose first moment (outside the hub) equals q (secant on width)."""
        hub, w_max = self._hub(), self._max_cw_end_width()
        prev = None
        for _ in range(12):
            try:
                cw = self._cw_arm(w_end, end_deg)
            except Exception:
                raise BalanceError(f"counterweight club ({w_end:.0f} mm wide, {self.params.cw_sweep_deg:.0f} deg "
                                   "sweep) self-intersects: reduce cw_sweep_deg or widen cw_root_width_mm") from None
            _, mx, my = first_moment(single_face(hub + cw))
            m = np.array([mx, my])
            if np.linalg.norm(m - q) < self.BALANCE_TOL_MM3 / 2:
                break
            turn = math.degrees(math.atan2(q[1], q[0]) - math.atan2(m[1], m[0]))
            end_deg += (turn + 180) % 360 - 180
            mag, target = float(np.linalg.norm(m)), float(np.linalg.norm(q))
            if prev and abs(mag - prev[1]) > 1e-9:
                w_new = w_end + (target - mag) * (w_end - prev[0]) / (mag - prev[1])
            else:
                w_new = w_end * target / mag
            if w_end >= w_max - 1e-9 and target > mag:
                raise BalanceError(f"counterweight too light even at the widest club ({w_max:.0f} mm): "
                                   "raise cw_radius_mm / cw_root_width_mm or lighten the fang")
            prev = (w_end, mag)
            w_end = float(np.clip(w_new, 0.5 * w_end, min(2 * w_end, w_max)))
            if w_end < self.MIN_CW_WIDTH_MM:
                raise BalanceError(f"counterweight club thinner than {self.MIN_CW_WIDTH_MM} mm: "
                                   "lower cw_radius_mm or cw_root_width_mm")
        return w_end, end_deg

    # --- full profile ------------------------------------------------------------------------------
    def _assemble(self, width: float, end_deg: float) -> bd.Face:
        p = self.params
        face = single_face(self._hub() + self._fang_arm() + self._cw_arm(width, end_deg))
        self._blend_used = p.blend_mm
        if p.blend_mm > 0:
            junctions = [(v.X, v.Y) for v in face.vertices() if abs(math.hypot(v.X, v.Y) - self.r_hub) < 0.05]
            # One junction at a time; a junction next to a short edge backs off instead of failing.
            for pt in junctions:
                if corner_turn_deg(face, pt) < 15:  # already smooth; OCC fillets of near-tangent corners fail slowly
                    continue
                for r in (p.blend_mm, 0.6 * p.blend_mm, 0.35 * p.blend_mm, 0.0):
                    if r == 0.0:
                        self._blend_used = 0.0  # a real corner left sharp
                        break
                    try:
                        face = single_face(bd.fillet(nearest_vertices(face, [pt]), r))
                        self._blend_used = min(self._blend_used, r)
                        break
                    except Exception:
                        continue
        if p.root_fillet_mm > 0:
            face = single_face(bd.fillet(nearest_vertices(face, [self._face_root()]), p.root_fillet_mm))
        face = single_face(bd.fillet(nearest_vertices(face, [self._land_end()]), 2.0))
        cutters = [bd.Pos(x, y) * bd.Circle(d / 2) for x, y, d in self._holes()]
        return single_face(face - cutters) if cutters else face

    def _build(self) -> _Built:
        if self._built:
            return self._built
        _, mx, my = first_moment(single_face(self._hub() + self._fang_arm()))
        q = -np.array([mx, my])
        width = self.params.cw_root_width_mm * 1.5
        end_deg = math.degrees(math.atan2(q[1], q[0])) + self.params.cw_sweep_deg / 2
        for it in range(1, 6):
            width, end_deg = self._solve_cw(q, width, end_deg)
            face = self._assemble(width, end_deg)
            _, mx, my = first_moment(face)
            residual = math.hypot(mx, my)
            if residual < self.BALANCE_TOL_MM3:
                break
            q = q - np.array([mx, my])
        if self.spin == "cw":
            face = face.mirror(bd.Plane.XZ)
        self._built = _Built(face, self._blend_used, width, end_deg, residual, it)
        return self._built

    def _holes(self) -> list[tuple[float, float, float]]:
        p = self.params
        out = [(0.0, 0.0, p.bore_mm)] if p.bore_mm > 0 else []
        for i in range(p.bolt_count):
            out.append((*polar(p.bolt_pcd_mm / 2, 360 * i / p.bolt_count), p.bolt_d_mm))
        return out

    # --- weapon interface --------------------------------------------------------------------------
    def profile(self) -> bd.Face:
        return self._build().face

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

    def impact_face(self):
        return self._mirror(self._tip()), self._mirror(self._face_root())

    def support_circles(self):
        p = self.params
        if p.bolt_count:
            return [(x, y, d / 2) for x, y, d in self._holes()[1 if p.bore_mm > 0 else 0:]]
        return [(0.0, 0.0, p.bore_mm / 2)]

    def derived(self) -> dict:
        b = self._build()
        return {"blend_used_mm": b.blend_used_mm, "cw_end_width_mm": b.cw_end_width_mm, "cw_end_deg": b.cw_end_deg,
                "balance_residual_mm3": b.residual_mm3, "balance_iterations": b.iterations}

    def geometric_checks(self, min_ligament_mm: float) -> list[Check]:
        p, lig = self.params, min_ligament_mm
        root_reach = math.hypot(self.r_arm0, p.arm_root_width_mm / 2)
        cw_reach = math.hypot(self.r_arm0, p.cw_root_width_mm / 2)
        checks = [
            Check("arm_root_inside_hub", root_reach < 0.97 * self.r_hub, root_reach, 0.97 * self.r_hub,
                  "fang arm root must start inside the hub boss"),
            Check("cw_root_inside_hub", cw_reach < 0.97 * self.r_hub, cw_reach, 0.97 * self.r_hub),
            Check("cw_below_tooth", p.cw_radius_mm <= self.R - p.face_depth_mm, p.cw_radius_mm,
                  self.R - p.face_depth_mm, "counterweight stays under the face root so only the tooth hits"),
            Check("cw_outside_hub", p.cw_radius_mm >= self.r_hub + 15, p.cw_radius_mm, self.r_hub + 15),
            Check("face_depth_fits", 0 < p.face_depth_mm < self.R - self.r_hub, p.face_depth_mm,
                  self.R - self.r_hub),
        ]
        if p.bolt_count:
            r_bolt = p.bolt_pcd_mm / 2
            checks.append(Check("lig_bore_to_bolts", r_bolt - p.bolt_d_mm / 2 - p.bore_mm / 2 >= lig,
                                r_bolt - p.bolt_d_mm / 2 - p.bore_mm / 2, lig))
            checks.append(Check("lig_bolts_to_hub_edge", self.r_hub - r_bolt - p.bolt_d_mm / 2 >= lig,
                                self.r_hub - r_bolt - p.bolt_d_mm / 2, lig))
            if p.bolt_count > 1:
                gap = 2 * r_bolt * math.sin(math.pi / p.bolt_count) - p.bolt_d_mm
                checks.append(Check("lig_bolt_to_bolt", gap >= lig, gap, lig))
        return checks

    def style_checks(self, style) -> list[Check]:
        p, out = self.params, []
        if style.min_hook_deg is not None:
            out.append(Check("style_hook", p.hook_deg >= style.min_hook_deg, p.hook_deg, style.min_hook_deg,
                             "tooth leans into the spin"))
        if style.max_root_fillet_ratio is not None:
            ratio = p.root_fillet_mm / p.face_depth_mm
            out.append(Check("style_gullet", ratio <= style.max_root_fillet_ratio, ratio,
                             style.max_root_fillet_ratio, "root fillet / face depth"))
        if p.blend_mm > 0:
            used = self._build().blend_used_mm
            out.append(Check("style_blends", used >= 0.5 * p.blend_mm, used, 0.5 * p.blend_mm,
                             "hub blends must be real fillets, not sharp junctions"))
        return out

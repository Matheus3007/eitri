"""Asymmetric single-tooth "hook" (gancho): the user's preferred layout, modelled on Rebot V3's weapon.

Canonical frame: spin CCW seen from +Z, tooth tip at angle 0. Angles below are "degrees behind the tip"
(the outline is traced from the tip backwards, against the spin, all the way round to the face root).

Outline (one continuous contour; sharp only at the two convex points, the tip and the counterweight point,
which carry no stress concentration -- every re-entrant corner is filleted):
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

Balance (the outline is drawn counterweight-heavy, then pockets take the excess out):
    arm pocket : fixed style slot in the tooth arm, `arm_wall_mm` from the outline
    cw pocket  : the counterweight lobe hollowed into a shell with a constant wall that follows the outline
                 (Rebot's comma pocket). Its wall thickness is solved so the pocket removes exactly the excess
                 moment (never thinner than `wall_mm`), and the lobe's angular position is nudged within the
                 bite envelope (`lobe_turn_deg`) to cancel the sideways remainder -- what a designer does by hand.
Every pocket corner is a tangent arc of `pocket_fillet_mm` (morphological opening, see `_pocket`).

Hub: bolt circle plus a lobed center cutout that keys the pulley (`lock_d_mm`; see `common.hub_lock`), or a
plain `bore_mm` when `lock_d_mm` = 0.
"""

import functools
import math
from dataclasses import dataclass

import build123d as bd
import numpy as np

from eitri.geometry.common import (facing_up, fillet_corners, first_moment, hooked_face_root, hub_lock,
                                   offset_pieces, pocket_wall_mm, polar, single_face, to_nurbs)
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
    spine_shoulder: float = 0.0  # >0 fills out the spine just below the tooth back (wider tooth); 0 = Rebot
    hub_d_mm: float = 52.0
    spine_waist_deg: float = 102.0
    cw_point_deg: float = 143.0  # nominal: balance turns the lobe by up to +-max_lobe_turn_deg
    lobe_span_deg: float = 97.0
    gullet_waist_deg: float = 39.0  # ahead of the face
    max_lobe_turn_deg: float = 30.0
    cw_size_frac: float = 0.8  # counterweight reach as a fraction of the bite envelope (1 = touching it)
    cw_clearance_mm: float = 4.0  # hard minimum gap to the bite envelope
    wall_mm: float = 9.5  # thinnest counterweight-pocket wall; balance thickens it as needed (Rebot: 11)
    arm_wall_mm: float = 11.0  # arm slot wall thickness
    web_mm: float = 0.0  # pockets start at hub_d/2 + web (Rebot: pockets reach the hub radius)
    pocket_fillet_mm: float = 7.0  # counterweight pocket corners
    arm_fillet_mm: float = 7.0  # arm slot corners (Rebot: R7.2); separate, the slot is narrow
    arm_pocket: float = 1.0  # 1 = slot in the fang arm (style), 0 = solid arm
    head_solid_mm: float = 30.0  # solid tooth head before the arm pocket starts
    bore_mm: float = 12.0  # plain round bore, only used when lock_d_mm = 0
    bolt_count: int = 7
    bolt_d_mm: float = 5.3
    bolt_pcd_mm: float = 36.5
    lock_d_mm: float = 29.0  # lobed pulley-lock cutout (0 = plain bore)
    lock_boss_d_mm: float = 10.5  # material boss left around each bolt inside the lock cutout
    lock_fillet_mm: float = 1.0
    envelope_bite_mm: float = 0.0  # 0 = use the bite computed from the spec's rpm and closing speed
    # set from [machining] (eitri/geometry/machining.py): pockets milled to this fraction of the plate (1 =
    # through), and the first moment (plate-equivalent mm^3, canonical frame) of material other features remove
    pocket_depth_frac: float = 1.0
    extra_mx_mm3: float = 0.0
    extra_my_mm3: float = 0.0

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


def _stadium(p0: tuple[float, float], p1: tuple[float, float], r: float) -> bd.Face:
    """Everything within r of the segment p0-p1."""
    (x0, y0), (x1, y1) = p0, p1
    length = math.hypot(x1 - x0, y1 - y0)
    return (bd.Pos((x0 + x1) / 2, (y0 + y1) / 2) * bd.Rot(0, 0, math.degrees(math.atan2(y1 - y0, x1 - x0)))
            * bd.SlotCenterToCenter(length, 2 * r)).faces()[0]


_hub_lock = functools.lru_cache(maxsize=8)(hub_lock)  # same hub on every candidate of an optimization


class Hook(Weapon):
    BALANCE_TOL_MM3 = 0.5  # first moment left after balancing: 0.5 mm^3 x 6.5 mm plate ~ 0.03 g*mm

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

    def _outline(self, turn: float = 0.0) -> bd.Face:
        p, R = self.params, self.R
        T = np.array(polar(R, 0.0))
        F = np.array(self._face_root())
        rw = self.r_hub  # the spine waist sits on the hub radius, the gullet waist a little outside
        rB = p.back_r_frac * R
        a_B, a_Ws = p.back_deg, p.spine_waist_deg
        a_P = p.cw_point_deg + turn
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
                 self._pt(lerp(rw, rB, min(0.56 + p.spine_shoulder, 0.95)), lerp(a_B, a_Ws, 0.21)),
                 self._pt(lerp(rw, rB, min(0.33 + p.spine_shoulder, 0.9)), lerp(a_B, a_Ws, 0.33)),
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
        face = fillet_corners(face, p.root_fillet_mm, only=[tuple(F)])
        # blend the tooth back into the spine (can kink inward for some proportions)
        face = fillet_corners(face, 0.1 * R, only=[tuple(B)])
        # soften the lobe/gullet junction (the only non-feature corner), like the R15 on Rebot
        return fillet_corners(face, 0.15 * p.lobe_span_deg, only=[tuple(L)])

    # --- pockets ---------------------------------------------------------------------------------
    def _state(self, turn: float) -> dict:
        """Outline, arm pocket and pre-balance first moment for one lobe position (cached). The arm pocket is
        cut once, from the un-turned outline: turning the lobe doesn't move the tooth arm."""
        key = round(turn, 6)
        if key not in self._states:
            p = self.params
            st = {"turn": turn, "outline": self._outline(turn), "insets": {}}
            if "arm" not in self._cache:
                r0, r1 = self.r_hub + p.web_mm, self.R - p.head_solid_mm
                # CCW angles: the arm lies between the gullet waist (ahead) and the spine waist (behind)
                a0, a1 = -(p.spine_waist_deg - 12), p.gullet_waist_deg - 12
                arm_st = st if turn == 0 else {"turn": 0.0, "outline": self._outline(0.0), "insets": {}}
                self._cache["arm"] = (self._pocket(arm_st, p.arm_wall_mm, r0, r1, a0, a1, final=True,
                                                   rho=p.arm_fillet_mm)
                                      if p.arm_pocket and r1 > r0 + 5 and a1 > a0 else None)
            arm = self._cache["arm"]
            st["base"] = single_face(st["outline"] - arm) if arm is not None else st["outline"]
            # first moment of what's left: a blind pocket only removes its depth fraction of the plate
            m = np.array(first_moment(st["outline"])[1:])
            if arm is not None:
                m -= p.pocket_depth_frac * np.array(first_moment(arm)[1:])
            st["base_m"] = m - np.array([p.extra_mx_mm3, p.extra_my_mm3])
            self._states[key] = st
        return self._states[key]

    def _inset(self, st: dict, wall: float) -> list[bd.Face]:
        key = round(wall, 6)
        if key not in st["insets"]:
            # offset curves don't survive STEP export (and offsets of offsets fail): NURBS right away
            st["insets"][key] = [to_nurbs(f) for f in offset_pieces(st["outline"], -wall)]
        return st["insets"][key]

    def _pocket(self, st: dict, wall: float, r0: float, r1: float, a0: float, a1: float,
                final: bool = False, rho: float | None = None) -> bd.Face | None:
        """The outline inset by `wall`, masked to the annular sector (r0..r1, a0..a1), with every corner rounded
        to R = rho (default pocket_fillet_mm) by a morphological opening (shrink by rho, grow back by rho).

        The opening fillets every re-entrant corner the pocket leaves in the material, turns cuts across the
        inset into rounded slot ends, drops slivers narrower than 2 rho, and changes continuously with the
        parameters (the balance solve relies on that; corner-by-corner fillets flip on and off as a cut sweeps
        past a corner of the inset). Shrinking distributes over intersections, so the shrunk shape is built
        directly (outline inset by wall + rho, sector shrunk by rho); only the grow-back offsets an offset.
        `final` converts the result to NURBS for STEP export (not needed while solving the balance)."""
        rho = self.params.pocket_fillet_mm if rho is None else rho
        if r1 - r0 <= 2 * rho + 1e-3:
            return None
        key = ("mask", r0, r1, a0, a1, rho)
        if key not in self._cache:
            mask = _sector(r0 + rho, r1 - rho, a0, a1)
            if rho > 0:
                mask = mask - [_stadium(polar(r0, a), polar(r1, a), rho) for a in (a0, a1)]
            self._cache[key] = mask if mask.faces() else None
        mask = self._cache[key]
        if mask is None:
            return None
        pieces = []
        for piece in self._inset(st, wall + rho):
            cut = piece & mask
            pieces += cut.faces() if cut is not None else []
        if not pieces:
            return None
        core = max(pieces, key=lambda f: f.area)
        if rho <= 0:
            return core
        grown = offset_pieces(core, rho)  # the inset is NURBS already: no offset-of-offset curves here
        if not grown:
            return None
        pocket = max(grown, key=lambda f: f.area)
        return to_nurbs(pocket) if final else pocket

    def _cw_window(self) -> tuple[float, float]:
        """CCW angle range the counterweight pocket may occupy (between the two waists, minus webs)."""
        p = self.params
        return -(360 - p.gullet_waist_deg - 10), -(p.spine_waist_deg + 10)

    def _cw_pocket(self, st: dict, wall: float, final: bool = False) -> bd.Face | None:
        p = self.params
        lo, hi = self._cw_window()
        return self._pocket(st, wall, self.r_hub + p.web_mm, 2 * self.R, lo, hi, final)

    # --- assembly + balance ------------------------------------------------------------------------
    def _bolts(self) -> list[tuple[float, float, float]]:
        p = self.params
        return [(*polar(p.bolt_pcd_mm / 2, 360 * i / p.bolt_count), p.bolt_d_mm) for i in range(p.bolt_count)]

    def _hub_cutters(self) -> list:
        p = self.params
        out = [bd.Pos(cx, cy) * bd.Circle(d / 2) for cx, cy, d in self._bolts()]
        if p.lock_d_mm > 0 and p.bolt_count:
            out.append(_hub_lock(p.lock_d_mm, p.lock_boss_d_mm, p.bolt_pcd_mm, p.bolt_count, p.lock_fillet_mm))
        elif p.bore_mm > 0:
            out.append(bd.Circle(p.bore_mm / 2))
        return out

    def bore_r_mm(self) -> float:
        p = self.params
        if p.lock_d_mm > 0 and p.bolt_count:
            return min(p.lock_d_mm / 2, p.bolt_pcd_mm / 2 - p.lock_boss_d_mm / 2)
        return p.bore_mm / 2

    def has_pulley_lock(self) -> bool:
        return self.params.lock_d_mm > 0 and self.params.bolt_count > 0

    def center_cutout_r_mm(self) -> float:
        p = self.params
        return p.lock_d_mm / 2 if p.lock_d_mm > 0 and p.bolt_count else p.bore_mm / 2

    def _residual(self, st: dict, wall: float) -> np.ndarray:
        """First moment (mm^3) left after hollowing the lobe with this wall (hub cutouts are symmetric)."""
        self._n_evals += 1
        pk = self._cw_pocket(st, wall)
        f = self.params.pocket_depth_frac
        return st["base_m"] - (f * np.array(first_moment(pk)[1:]) if pk is not None else 0.0)

    def _solve_wall(self, st: dict, guess: float | None = None) -> tuple[float, np.ndarray]:
        """Wall thickness whose pocket removes the excess moment's component along itself. Thicker wall =
        smaller pocket, monotonically, so a bracketed regula falsi (Illinois) is safe."""
        p = self.params
        base = st["base_m"]
        need = math.degrees(math.atan2(base[1], base[0]))
        if abs(need) < 90:
            raise BalanceError(f"the outline is tooth-heavy (excess moment points at {need:.0f} deg, toward the "
                               "tooth): raise cw_size_frac, widen lobe_span_deg or enable arm_pocket")
        u = base / np.linalg.norm(base)
        def g(w):
            r = self._residual(st, w)
            return r @ u, r

        lo = p.wall_mm
        g_lo, _ = g(lo)
        if g_lo > 0:
            raise BalanceError(f"the counterweight is heavier than its thinnest-walled pocket can fix "
                               f"({g_lo:.0f} mm^3 left): lower cw_size_frac or wall_mm")
        if guess is not None and guess > lo + 0.5:  # warm start: bracket tightly around the last answer
            gl, _ = g(guess - 0.5)
            if gl < 0:
                lo, g_lo = guess - 0.5, gl
        hi = max(lo + 1.0, guess + 0.5) if guess is not None else lo + 8.0
        g_hi, r_hi = g(hi)
        while g_hi < 0:  # still over-removing: thicken until the pocket is too small (it vanishes eventually)
            lo, g_lo = hi, g_hi
            hi += 8.0
            if hi > self.R:
                raise BalanceError("couldn't bracket the pocket wall")
            g_hi, r_hi = g(hi)
        side = 0
        w, r = hi, r_hi
        for _ in range(40):
            w = hi - g_hi * (hi - lo) / (g_hi - g_lo)
            gw, r = g(w)
            if abs(gw) < self.BALANCE_TOL_MM3 / 2 or hi - lo < 1e-4:
                break
            if gw < 0:
                lo, g_lo = w, gw
                if side == 1:
                    g_hi /= 2
                side = 1
            else:
                hi, g_hi = w, gw
                if side == -1:
                    g_lo /= 2
                side = -1
        return w, r

    def _solve_balance(self):
        """Lobe turn (outer, secant on the sideways remainder) x pocket wall (inner, along the excess)."""
        p = self.params
        t_min = (p.spine_waist_deg + 15) - p.cw_point_deg  # keep the lobe clear of both waists
        t_max = (360 - p.gullet_waist_deg - 15 - p.lobe_span_deg) - p.cw_point_deg
        t_min, t_max = max(t_min, -p.max_lobe_turn_deg), min(t_max, p.max_lobe_turn_deg)
        if t_min > 0 or t_max < 0:
            raise BalanceError("the lobe overlaps a waist: shrink lobe_span_deg or move cw_point_deg")

        def side(turn, guess=None):
            st = self._state(turn)
            w, r = self._solve_wall(st, guess)
            u = st["base_m"] / np.linalg.norm(st["base_m"])
            return u[0] * r[1] - u[1] * r[0], w, r, st  # sideways remainder (signed), CCW positive

        t0, (s0, w0, r0, st0) = 0.0, side(0.0)
        best = (np.linalg.norm(r0), t0, w0, r0, st0)
        if best[0] < self.BALANCE_TOL_MM3:
            return best[1:]
        # Turning the lobe further behind (+turn) swings its moment clockwise: sideways ~ -|m| * turn.
        t1 = float(np.clip(t0 + math.degrees(s0 / np.linalg.norm(st0["base_m"])), t_min, t_max))
        if abs(t1 - t0) < 1e-3:
            t1 = t0 + (1.0 if t0 < t_max else -1.0)
        s1, w1, r1, st1 = side(t1, w0)
        for _ in range(12):
            if np.linalg.norm(r1) < best[0]:
                best = (np.linalg.norm(r1), t1, w1, r1, st1)
            if best[0] < self.BALANCE_TOL_MM3 or s1 == s0:
                break
            t2 = float(np.clip(t1 - s1 * (t1 - t0) / (s1 - s0), t_min, t_max))
            if abs(t2 - t1) < 1e-5:
                break
            t0, s0, w0 = t1, s1, w1
            t1 = t2
            s1, w1, r1, st1 = side(t1, w0)
        if np.linalg.norm(r1) < best[0]:
            best = (np.linalg.norm(r1), t1, w1, r1, st1)
        residual, turn, wall, r, st = best
        if residual > 20 * self.BALANCE_TOL_MM3:
            raise BalanceError(f"can't balance within +-{p.max_lobe_turn_deg:.0f} deg of lobe turn (residual "
                               f"{residual:.0f} mm^3): change cw_point_deg, lobe_span_deg or cw_size_frac")
        return turn, wall, r, st

    def _build(self):
        if "face" in self._cache:
            return self._cache
        self._states, self._n_evals = {}, 0
        turn, wall, r, st = self._solve_balance()
        face = st["base"]
        pk = self._cw_pocket(st, wall, final=True)
        if pk is not None:
            face = single_face(face - pk)
        face = single_face(face - self._hub_cutters())
        pockets = [f for f in (self._cache.get("arm"), pk) if f is not None]
        filled = single_face(st["outline"] - self._hub_cutters())
        if self.spin == "cw":
            face = face.mirror(bd.Plane.XZ)
            filled = filled.mirror(bd.Plane.XZ)
            pockets = [f.mirror(bd.Plane.XZ) for f in pockets]
        self._cache.update(face=face, filled=facing_up(filled), pockets=[facing_up(f) for f in pockets], outline=st["outline"], lobe_turn_deg=turn, cw_wall_mm=wall,
                           cw_pocket_area_mm2=pk.area if pk is not None else 0.0,
                           balance_residual_mm3=float(np.linalg.norm(r)), balance_evals=self._n_evals)
        self._states = {}  # free the per-turn geometry
        return self._cache

    # --- weapon interface --------------------------------------------------------------------------
    def profile(self) -> bd.Face:
        return self._build()["face"]

    def pocket_faces(self) -> list[bd.Face]:
        return self._build()["pockets"]

    def filled_profile(self) -> bd.Face:
        """The profile with the lightening pockets left in (hub cutouts still cut)."""
        return self._build()["filled"]

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
            return [(*self._mirror((x, y)), d / 2) for x, y, d in self._bolts()]
        return [(0.0, 0.0, p.bore_mm / 2)]

    def derived(self) -> dict:
        b = self._build()
        return {k: float(b[k]) for k in ("lobe_turn_deg", "cw_wall_mm", "cw_pocket_area_mm2",
                                         "balance_residual_mm3", "balance_evals")} | {"envelope_bite_mm": self.bite,
                                                            "envelope_violation_mm": self.envelope_violation()}

    def envelope_violation(self) -> float:
        """How far (mm) the outline pokes outside the bite envelope, ignoring the tooth itself."""
        outline = self._cache.get("outline")
        outline = (outline if outline is not None else self._outline()).outer_wire()
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
            Check("wall_thickness", p.wall_mm >= lig, p.wall_mm, lig, "thinnest pocket wall allowed"),
        ]
        if p.bolt_count:
            r_bolt = p.bolt_pcd_mm / 2
            if p.lock_d_mm > 0:
                # the ring of boss around each bolt (clamped under the pulley, hence half the ligament)
                lig_lock = (p.lock_boss_d_mm - p.bolt_d_mm) / 2
                checks.append(Check("lig_bolt_to_lock", lig_lock >= lig * 0.5, lig_lock, lig * 0.5))
                engage = p.lock_d_mm / 2 - (r_bolt - p.lock_boss_d_mm / 2)
                checks.append(Check("lock_engagement", engage >= 0.5, engage, 0.5,
                                    "how far the bolt bosses stand into the lock cutout (Rebot: 1.5 mm)"))
            else:
                checks.append(Check("lig_bore_to_bolts", r_bolt - p.bolt_d_mm / 2 - p.bore_mm / 2 >= lig * 0.5,
                                    r_bolt - p.bolt_d_mm / 2 - p.bore_mm / 2, lig * 0.5))
            hub_lig = self.r_hub - r_bolt - p.bolt_d_mm / 2  # to the outline waists and the pockets
            checks.append(Check("lig_bolts_to_hub_edge", hub_lig >= lig, hub_lig, lig))
        return checks

    def style_checks(self, style) -> list[Check]:
        p, out = self.params, []
        if style.min_hook_deg is not None:
            out.append(Check("style_hook", p.hook_deg >= style.min_hook_deg, p.hook_deg, style.min_hook_deg))
        if style.max_root_fillet_ratio is not None:
            ratio = p.root_fillet_mm / p.face_depth_mm
            out.append(Check("style_gullet", ratio <= style.max_root_fillet_ratio, ratio,
                             style.max_root_fillet_ratio))
        if p.arm_pocket:  # the slot is part of the look: if it can't fit, say so instead of dropping it
            self._build()
            out.append(Check("style_arm_slot", self._cache.get("arm") is not None, None, None,
                             "arm slot requested (arm_pocket = 1) but the arm is too narrow for its walls "
                             "and fillets: lower arm_wall_mm / arm_fillet_mm or head_solid_mm"))
        if style.min_wall_frac is not None:
            wall = pocket_wall_mm(self.profile(), self.r_hub)
            need = style.min_wall_frac * self.R
            out.append(Check("style_walls", wall >= need, wall, need,
                             "thinnest wall between a pocket and the outline (no thin-rim 'hoops')"))
        return out

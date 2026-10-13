"""A weapon whose pockets come from topology optimization (`eitri topo`) inside a base weapon's domain.

The base (a generated design or an imported STEP, given by its spec) supplies the outline, the hub holes, the
impact face and the supports; its own pockets are filled. `keep` regions (see eitri/topo/domain.py) are never
cut. Pockets are smooth closed curves (from the density field) that get, at every build:
    offset    by delta_i (the balance knobs below)
    clipped   to the free region (domain minus keep)
    opened    shrink/grow by `fillet_mm`: every corner rounded, slivers dropped (no sharp re-entrant corners)
Balance (always solved, never tuned): Newton on the pocket offsets -- one common offset for the mass
(`area_mm2`, if set: the plate area that gives the mass budget) and two pockets' own offsets for the first
moment (the two big pockets whose directions from the axis are the most different) -- CG on the axis.
"""

import math
from dataclasses import dataclass
from functools import cached_property

import build123d as bd
import numpy as np

from eitri.geometry.balance import BalanceError, newton
from eitri.geometry.common import facing_up, first_moment, offset_pieces, pocket_wall_mm, single_face, to_nurbs
from eitri.geometry.weapon import Check, Weapon


@dataclass
class TopoParams:
    base: str  # spec of the base weapon (outline, hub, tooth)
    pockets: str  # JSON from `eitri topo`
    keep: str = "hub:22;rim:7;tooth:18;holes:3"  # ';'-separated keep selectors
    fillet_mm: float = 4.0
    area_mm2: float = 0.0  # plate area to hold (mass budget / (density * thickness)); 0 = whatever it comes to
    min_pocket_area_mm2: float = 25.0
    min_web_mm: float = 5.0  # thinnest strut left between two pockets
    envelope_bite_mm: float = 0.0  # domain grown to the bite envelope (as in the topology run); 0 = base outline
    # from [machining]: pocket depth fraction (1 = through), and area / first moment other features remove
    pocket_depth_frac: float = 1.0
    extra_area_mm2: float = 0.0
    extra_mx_mm3: float = 0.0
    extra_my_mm3: float = 0.0
    thickness_mm: float = 0.0  # from the base


class Topo(Weapon):
    BALANCE_TOL_MM3 = 0.5

    def __init__(self, params: TopoParams, spin: str = "ccw"):
        from eitri.evaluate import build_weapon
        from eitri.spec import load
        from eitri.topo import extract
        from eitri.topo.domain import Domain

        self.params = params
        bspec = load(params.base)
        self.base = build_weapon(bspec, bspec.fixed_design())
        self.spin = self.base.spin
        params.thickness_mm = self.base.params.thickness_mm
        self.domain = Domain(self.base, [k for k in params.keep.split(";") if k.strip()],
                             envelope_bite_mm=params.envelope_bite_mm)
        self._raw = extract.load(params.pockets)
        self._cache: dict = {}

    @classmethod
    def from_dict(cls, d: dict, spin: str = "ccw") -> "Topo":
        return cls(TopoParams(**{k: v for k, v in d.items() if k in TopoParams.__dataclass_fields__}), spin)

    # --- pockets -----------------------------------------------------------------------------------
    def _shapes(self) -> list[bd.Face]:
        if "shapes" not in self._cache:
            faces = []
            for pts in self._raw:
                # ~40 control points: smooth enough, and booleans/offsets on long B-splines are slow
                k = max(1, len(pts) // 40)
                w = bd.Spline(*[tuple(p) for p in pts[::k]], periodic=True)
                faces.append(facing_up(bd.Face(bd.Wire([w]))))
            self._cache["shapes"] = faces
        return self._cache["shapes"]

    @cached_property
    def _free_poly(self):
        """The free region as a shapely polygon (dense sampling of the exact faces: 0.3 mm spacing)."""
        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        def ring(w):
            n = max(64, int(w.length / 0.3))
            return [(p.X, p.Y) for p in (w.position_at(t) for t in np.linspace(0, 1, n, endpoint=False))]

        free = unary_union([Polygon(ring(f.outer_wire()), [ring(h) for h in f.inner_wires()])
                            for f in self.domain.free.faces()])
        # plus a strip outside the outline: cuts that reach the outline run past it (clean booleans)
        outline = Polygon(ring(self.domain.profile.outer_wire()))
        return free.union(outline.buffer(5.0) - outline)

    @staticmethod
    def _spline_face(ring: np.ndarray, spacing: float = 1.5) -> bd.Face:
        """One smooth periodic spline through a closed polygon ring (resampled every `spacing` mm)."""
        q = np.vstack([ring, ring[:1]])
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
        n = int(np.clip(s[-1] / spacing, 24, 200))
        t = np.linspace(0, s[-1], n, endpoint=False)
        pts = np.c_[np.interp(t, s, q[:, 0]), np.interp(t, s, q[:, 1])]
        return facing_up(bd.Face(bd.Wire([bd.Spline(*[tuple(p) for p in pts], periodic=True)])))

    def _pockets(self, deltas: np.ndarray, final: bool = False, turn_deg: float = 0.0) -> list[bd.Face]:
        """Offset, clip to the free region and open (shrink/grow by fillet_mm) on polygons -- shapely's buffers
        never silently drop a piece the way OCC's 2D offsets do -- then one smooth spline per pocket."""
        from shapely import affinity
        from shapely.geometry import Polygon

        from shapely.ops import unary_union

        r, web = self.params.fillet_mm, self.params.min_web_mm * 1.03  # margin for the spline fit
        polys = []
        for raw, d in zip(self._raw, deltas):
            poly = Polygon(raw)
            if abs(turn_deg) > 1e-9:
                poly = affinity.rotate(poly, turn_deg, origin=(0, 0))
            if abs(d) > 1e-9:
                poly = poly.buffer(float(d), quad_segs=16)
            polys.append(poly.intersection(self._free_poly))
        # struts between pockets at least `web` wide: each pocket gives way to its neighbours (by area order,
        # the big pockets keep their shape), then every pocket is opened (shrink/grow by r: rounded corners)
        pieces = []
        for g in polys:
            g = g.buffer(-r, quad_segs=16).buffer(r, quad_segs=16)
            pieces += [q for q in getattr(g, "geoms", [g]) if not q.is_empty]
        placed, out = [], []
        for g in sorted(pieces, key=lambda q: -q.area):  # big pockets keep their shape, small ones give way
            if placed:
                g = g.difference(unary_union(placed).buffer(web, quad_segs=16))
                g = g.buffer(-r, quad_segs=16).buffer(r, quad_segs=16)
            for part in sorted(getattr(g, "geoms", [g]), key=lambda q: -q.area):
                if part.is_empty or part.area < self.params.min_pocket_area_mm2:
                    continue
                if placed and part.distance(unary_union(placed)) < 0.9 * web:
                    continue  # a sliver left between two pockets
                placed.append(part)
                f = self._spline_face(np.asarray(part.exterior.coords)[:-1])
                out.append(to_nurbs(f) if final else f)
        return out

    def _profile_for(self, deltas, final=False, turn_deg: float = 0.0) -> bd.Face:
        face = self.domain.profile
        pk = self._pockets(deltas, final, turn_deg)
        return self._main_body(face - pk) if pk else face

    @staticmethod
    def _main_body(shape) -> bd.Face:
        """The piece that carries the hub (the largest): a cut can leave islands, which would fly off."""
        return max(shape.faces(), key=lambda f: f.area)

    def _knob_pairs(self) -> list[tuple[int, int]]:
        """Pocket pairs ranked by how well they steer the CG: big, and in directions from the axis that differ
        the most."""
        cs = []
        for f in self._shapes():
            a, mx, my = first_moment(f)
            cs.append((a, np.array([mx, my]) / a))
        pairs = []
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                ai, ci = cs[i]
                aj, cj = cs[j]
                s = ai * aj * abs(ci[0] * cj[1] - ci[1] * cj[0]) / (np.linalg.norm(ci) * np.linalg.norm(cj) + 1e-9)
                pairs.append((s, (i, j)))
        return [p for _, p in sorted(pairs, reverse=True)]

    def _build(self):
        if "face" in self._cache:
            return self._cache
        n = len(self._shapes())
        if n < 2:
            raise BalanceError("topology design needs at least two pockets to balance")
        pr = self.params
        dom = np.array(first_moment(self.domain.profile))
        hold_mass = pr.area_mm2 > 0
        sol = err = None
        for a, b in self._knob_pairs()[:4]:  # the best-conditioned pairs first; another pair if one can't reach
            for turn_knob in ([n < 3] if n < 3 else [False, True]) if hold_mass else [False]:
                def split(z, a=a, b=b, turn_knob=turn_knob):
                    d = np.zeros(n)
                    if hold_mass and not turn_knob:
                        d += z[0]
                    d[a] += z[-2]
                    d[b] += z[-1]
                    return d, (z[0] if turn_knob else 0.0)

                def resid(z, split=split):
                    d, turn = split(z)
                    # a blind pocket removes only its depth fraction of the plate
                    thru = np.array(first_moment(self._profile_for(d, turn_deg=turn)))
                    area, mx, my = dom + pr.pocket_depth_frac * (thru - dom) - [pr.extra_area_mm2, pr.extra_mx_mm3,
                                                                                 pr.extra_my_mm3]
                    return np.array(([10 * (area - pr.area_mm2)] if hold_mass else []) + [mx, my])

                k = 3 if hold_mass else 2
                lim = np.full(k, 6.0)
                if turn_knob:
                    lim[0] = 10.0  # degrees
                try:
                    sol = newton(resid, np.zeros(k), -lim, lim, tol=self.BALANCE_TOL_MM3, h=0.3)
                    break
                except BalanceError as e:
                    err = e
            if sol is not None:
                break
        if sol is None:
            raise err
        d, turn = split(sol.x)
        pockets = self._pockets(d, final=True, turn_deg=turn)
        face = self._main_body(self.domain.profile - pockets) if pockets else self.domain.profile
        # no mirroring: the domain is in the base weapon's own frame, which already carries its spin
        self._cache.update(face=face, pockets=[facing_up(f) for f in pockets], offsets=d, turn_deg=turn,
                           balance_residual_mm3=sol.norm, balance_evals=sol.evals, knob_pockets=[a, b])
        return self._cache

    # --- weapon interface (geometry from the build, the rest from the base) ------------------------
    def profile(self) -> bd.Face:
        return self._build()["face"]

    def filled_profile(self) -> bd.Face:
        return self.domain.profile

    def pocket_faces(self) -> list[bd.Face]:
        """Pocket faces clipped to the domain (cuts reaching the outline run past it)."""
        return [f2 for f in self._build()["pockets"] for f2 in ((f & self.domain.profile).faces() or [])]

    def solid(self) -> bd.Solid:
        t = self.params.thickness_mm
        return bd.extrude(self.profile(), amount=t / 2, both=True).solids()[0]

    def tip_radius_mm(self) -> float:
        return self.base.tip_radius_mm()

    def n_teeth(self) -> int:
        return self.base.n_teeth()

    def tooth_depth_mm(self) -> float:
        return self.base.tooth_depth_mm()

    def hook_deg(self) -> float:
        return self.base.hook_deg()

    def impact_face(self):
        return self.base.impact_face()

    def support_circles(self):
        return self.base.support_circles()

    def bore_r_mm(self) -> float:
        return self.base.bore_r_mm()

    def center_cutout_r_mm(self) -> float:
        return self.base.center_cutout_r_mm()

    def has_pulley_lock(self) -> bool:
        return self.base.has_pulley_lock()

    def derived(self) -> dict:
        b = self._build()
        return {"pocket_offsets_mm": [round(float(v), 4) for v in b["offsets"]], "pocket_turn_deg": b["turn_deg"],
                "balance_residual_mm3": float(b["balance_residual_mm3"]), "balance_evals": int(b["balance_evals"]),
                "balance_pockets": b["knob_pockets"], "n_pockets": len(self._shapes())}

    def geometric_checks(self, min_ligament_mm: float) -> list[Check]:
        face = self.profile()
        hub_r = self.domain.hub_hole_r
        pockets = [w for w in face.inner_wires()
                   if max(math.hypot(v.X, v.Y) for v in w.vertices()) > hub_r]
        gap = min((pockets[i].distance_to(pockets[j]) for i in range(len(pockets))
                   for j in range(i + 1, len(pockets))), default=math.inf)
        return [Check("strut_width", gap >= min_ligament_mm, gap, min_ligament_mm,
                      "thinnest web between two pockets"),
                Check("pocket_to_outline", pocket_wall_mm(face, hub_r) >= min_ligament_mm,
                      pocket_wall_mm(face, hub_r), min_ligament_mm)]

    def style_checks(self, style) -> list[Check]:
        if style.min_wall_frac is None:
            return []
        wall = pocket_wall_mm(self.profile(), self.domain.hub_hole_r)
        need = style.min_wall_frac * self.tip_radius_mm()
        return [Check("style_walls", wall >= need, wall, need,
                      "thinnest wall between a pocket and the outline (no thin-rim 'hoops')")]

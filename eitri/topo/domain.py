"""Design domain for keep/free optimization of a flat weapon.

The domain is a base weapon's outline (a generated design or an imported STEP) with its hub holes (bolts,
center cutout / pulley lock) but with every other pocket filled: the optimizer decides where material goes.
`keep` regions are never touched; the rest is free.

Keep selectors (a list of strings; coordinates in the base weapon's own frame, mm):
    hub:R              everything within R of the spin axis
    rim:W              a band W wide inside the outline (keeps the silhouette, the user's look)
    tooth:D            everything within D of the impact face (tip -> face root)
    holes:W            a ring W wide around every hub hole (bolts, center cutout)
    edge:I:W           a band W wide along boundary edge I of `eitri faces` (side face I of the plate)
    circle:X,Y,R       a disc
    box:X0,Y0,X1,Y1    a rectangle
"""

import math
from dataclasses import dataclass, field
from functools import cached_property

import build123d as bd
import numpy as np

from eitri.geometry.common import facing_up, offset_pieces, single_face


def _disc(x, y, r) -> bd.Face:
    return facing_up((bd.Pos(x, y) * bd.Circle(r)).faces()[0])


def _stadium(p0, p1, r) -> bd.Face:
    (x0, y0), (x1, y1) = p0, p1
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 1e-6:
        return _disc(x0, y0, r)
    return facing_up((bd.Pos((x0 + x1) / 2, (y0 + y1) / 2) * bd.Rot(0, 0, math.degrees(math.atan2(y1 - y0, x1 - x0)))
                      * bd.SlotCenterToCenter(length, 2 * r)).faces()[0])


def _band_along(edge: bd.Edge, w: float, n: int = 24) -> bd.Face:
    """Everything within w of an edge (union of stadiums along it)."""
    pts = [(p.X, p.Y) for p in (edge.position_at(t) for t in np.linspace(0, 1, n + 1))]
    out = _stadium(pts[0], pts[1], w)
    for a, b in zip(pts[1:], pts[2:]):
        out = out + _stadium(a, b, w)
    return single_face(out) if len(out.faces()) == 1 else max(out.faces(), key=lambda f: f.area)


def envelope_face(base, bite_mm: float, clearance_mm: float, n: int = 180) -> bd.Face:
    """The single-tooth bite envelope r(phi) = R - bite * phi / 360 - clearance (phi = degrees behind the tip,
    against the spin): material inside it can't become a second tooth. Closed by the straight line from the
    envelope's end (just ahead of the face) back to the tip."""
    R = base.tip_radius_mm()
    tx, ty = base.impact_face()[0]
    a_tip = math.atan2(ty, tx)
    back = -1.0 if base.spin == "ccw" else 1.0  # direction of "behind the tip"
    pts = []
    for phi in np.linspace(0, 359, n):
        r = R - bite_mm * phi / 360 - (clearance_mm if phi > 20 else clearance_mm * phi / 20)
        a = a_tip + back * math.radians(phi)
        pts.append((r * math.cos(a), r * math.sin(a)))
    return facing_up(bd.Face(bd.Wire([bd.Spline(*pts), bd.Line(pts[-1], pts[0])])))


@dataclass
class Domain:
    base: object  # the base Weapon (impact face, supports, spin, thickness come from it)
    keep_spec: list[str] = field(default_factory=list)
    # grow the domain past the base outline to the whole bite envelope (0 = the base outline only)
    envelope_bite_mm: float = 0.0
    envelope_clearance_mm: float = 4.0

    @cached_property
    def hub_hole_r(self) -> float:
        """Inner wires entirely inside this radius are hub holes (kept); the rest are pockets (filled)."""
        sup = self.base.support_circles()
        r = max([math.hypot(x, y) + rr for x, y, rr in sup] + [self.base.center_cutout_r_mm()])
        return r + 1.0

    def _is_hub_hole(self, wire: bd.Wire) -> bool:
        return max(math.hypot(p.X, p.Y) for p in (wire.position_at(t) for t in np.linspace(0, 1, 60))) \
            <= self.hub_hole_r

    @cached_property
    def profile(self) -> bd.Face:
        face = self.base.profile()
        outline = facing_up(bd.Face(face.outer_wire()))
        if self.envelope_bite_mm > 0:
            # keep the face and gullet clear: nothing may grow in front of the impact face
            env = envelope_face(self.base, self.envelope_bite_mm, self.envelope_clearance_mm)
            tip, root = (np.array(p, float) for p in self.base.impact_face())
            ahead = np.array([-tip[1], tip[0]]) / np.linalg.norm(tip) * (1 if self.base.spin == "ccw" else -1)
            strip = facing_up(bd.Face(bd.Wire.make_polygon(
                [tuple(root), tuple(tip), tuple(tip + 60 * ahead), tuple(root + 60 * ahead)], close=True)))
            grown = (outline + env) - strip
            outline = max(grown.faces(), key=lambda f: f.area)
        holes = [h for h in face.inner_wires() if self._is_hub_hole(h)]
        return facing_up(single_face(outline - [facing_up(bd.Face(h)) for h in holes])) if holes else outline

    def edges(self) -> list[bd.Edge]:
        """Boundary edges of the domain, numbered as `eitri faces` shows them (outline first, then holes)."""
        face = self.profile
        return [e for w in [face.outer_wire(), *face.inner_wires()] for e in w.edges()]

    def _selector(self, sel: str) -> bd.Face:
        kind, _, arg = sel.partition(":")
        vals = [float(v) for v in arg.replace(":", ",").split(",") if v.strip()] if arg else []
        dom = self.profile
        if kind == "hub":
            return _disc(0, 0, vals[0])
        if kind == "rim":
            inner = offset_pieces(bd.Face(dom.outer_wire()), -vals[0])
            band = bd.Face(dom.outer_wire())
            for f in inner:
                band = band - f
            return max(band.faces(), key=lambda f: f.area)
        if kind == "tooth":
            tip, root = self.base.impact_face()
            return _stadium(tip, root, vals[0])
        if kind == "holes":
            out = None
            for w in dom.inner_wires():
                ring = bd.Face(w)
                grown = offset_pieces(ring, vals[0])
                g = max(grown, key=lambda f: f.area) if grown else ring
                out = g if out is None else out + g
            return out
        if kind == "edge":
            return _band_along(self.edges()[int(vals[0])], vals[1])
        if kind == "circle":
            return _disc(*vals)
        if kind == "box":
            x0, y0, x1, y1 = vals
            return facing_up((bd.Pos((x0 + x1) / 2, (y0 + y1) / 2) * bd.Rectangle(abs(x1 - x0), abs(y1 - y0))).faces()[0])
        raise ValueError(f"unknown keep selector {sel!r} (hub, rim, tooth, holes, edge, circle, box)")

    @cached_property
    def keep(self) -> list[bd.Face]:
        out = []
        for sel in self.keep_spec:
            cut = self._selector(sel.strip()) & self.profile
            out += [facing_up(f) for f in cut.faces()] if cut is not None else []
        return out

    @cached_property
    def free(self) -> bd.Face | None:
        """Domain minus the keep regions (may be several faces: returned as a compound's faces list)."""
        f = self.profile
        for k in self.keep:
            f = f - k
        return f

    def polygons(self, face_or_shape, n: int = 400) -> list[np.ndarray]:
        """Sampled outline + holes of each face (for fast point-in-region tests)."""
        out = []
        for f in face_or_shape.faces():
            for w in [f.outer_wire(), *f.inner_wires()]:
                out.append(np.array([(p.X, p.Y) for p in (w.position_at(t) for t in np.linspace(0, 1, n))]))
        return out

    def keep_mask(self, pts: np.ndarray) -> np.ndarray:
        """Points (n, 2) inside any keep region."""
        from matplotlib.path import Path as MPath

        mask = np.zeros(len(pts), bool)
        for k in self.keep:
            inside = MPath(self.polygons(k)[0]).contains_points(pts)
            for hole in self.polygons(k)[1:]:
                inside &= ~MPath(hole).contains_points(pts)
            mask |= inside
        return mask

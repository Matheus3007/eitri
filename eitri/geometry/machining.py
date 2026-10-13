"""Machined (non-through) features on a plate weapon, from the spec's [machining] section.

    pocket_depth_mm      lightening pockets milled to this depth instead of cut through (0 = through);
                         `pocket_sides` = 1 (from the top) or 2 (half from each face: CG stays mid-plane)
    face_bevel_mm        45 deg chamfer of the impact face's top edge (`face_bevel_sides` = 2: both edges):
                         a sharper leading edge that bites
    counterbore_d_mm     counterbore around each bolt hole, `counterbore_depth_mm` deep, from the top (bolt heads
                         sit flush)

Everything stays exact and balanced:
    mass    the plate (2D profile with the blind pockets filled back) extruded analytically, minus each feature
            as an exact part (pocket prisms from their 2D faces, bevel = triangular prism, counterbore = annulus)
    balance the features' first moment is handed to the generator's Newton solver before it builds
            (hook/topo `pocket_depth_frac`, `extra_*`), so the CG lands on the axis with them
    FEA     2D: elements under a blind pocket / counterbore get the reduced thickness; 3D (out-of-plane case)
            and STEP use the real solid
"""

import math

import build123d as bd
import numpy as np

from eitri.geometry.common import facing_up, first_moment, single_face
from eitri.geometry.weapon import Check, Weapon


def _bevel_part(weapon, b: float, t: float, top: bool) -> tuple[float, np.ndarray, np.ndarray]:
    """(volume, centroid, tensor about centroid) of a 45 deg x b chamfer along the impact face's edge."""
    tip, root = (np.array(p, float) for p in weapon.impact_face())
    L = float(np.linalg.norm(root - tip))
    u = (root - tip) / L
    n = np.array([-u[1], u[0]])
    ahead = np.array([-tip[1], tip[0]]) * (1 if getattr(weapon, "spin", "ccw") == "ccw" else -1)
    if n @ ahead > 0:
        n = -n  # into the material: it is behind the face, against the spin
    vol = 0.5 * b * b * L
    c2 = (tip + root) / 2 + n * b / 3
    z = (t / 2 - b / 3) * (1 if top else -1)
    m = vol  # unit density
    T = np.zeros((3, 3))
    d3 = np.array([u[0], u[1], 0.0])
    T += m * L**2 / 12 * (np.eye(3) - np.outer(d3, d3))  # slender prism along the face
    return vol, np.array([c2[0], c2[1], z]), T


def bevel_wedges(weapon, mach, t: float) -> list[bd.Solid]:
    """45 deg x face_bevel_mm wedges along the impact face (top, and bottom if 2 sides), overshooting the plate."""
    tip, root = (np.array(p, float) for p in weapon.impact_face())
    L = float(np.linalg.norm(root - tip))
    u = (root - tip) / L
    n = np.array([-u[1], u[0]])
    ahead = np.array([-tip[1], tip[0]]) * (1 if getattr(weapon, "spin", "ccw") == "ccw" else -1)
    if n @ ahead > 0:
        n = -n  # into the material: it is behind the face, against the spin
    b, h = mach.face_bevel_mm, t / 2
    out = []
    for zs in ([1] if mach.face_bevel_sides == 1 else [1, -1]):
        tri = [(-1.0, zs * (h - b - 1)), (b + 1, zs * (h + 1)), (-1.0, zs * (h + 1))]
        pts = [(*(tip + nn * n), z) for nn, z in tri]
        out.append(bd.extrude(bd.Face(bd.Wire.make_polygon(pts, close=True)), amount=L, dir=(u[0], u[1], 0)))
    return out


def feature_parts(weapon, mach, t: float, exact: bool = True) -> list[tuple[float, float, np.ndarray, np.ndarray]]:
    """(sign, volume, centroid, tensor about centroid) of material the bevel and counterbores remove. `exact`:
    from the built profile (wedge and plate intersected, counterbores clipped to the material); else the
    analytic estimate the balance starts from (before the weapon exists)."""
    from eitri.physics.mass import plate_part

    out = []
    if mach.face_bevel_mm > 0:
        if exact:
            plate = bd.extrude(weapon.profile(), amount=t / 2, both=True).solids()[0]
            for w in bevel_wedges(weapon, mach, t):
                cut = plate & w
                v = cut.volume
                c = cut.center(bd.CenterOf.MASS)
                L = float(np.linalg.norm(np.subtract(*weapon.impact_face())))
                out.append((-1.0, v, np.array([c.X, c.Y, c.Z]), np.eye(3) * v * L**2 / 12))
        else:
            for top in ([True, False] if mach.face_bevel_sides == 2 else [True]):
                out.append((-1.0, *_bevel_part(weapon, mach.face_bevel_mm, t, top)))
    if mach.counterbore_d_mm > 0 and mach.counterbore_depth_mm > 0:
        h = mach.counterbore_depth_mm
        for cx, cy, r in weapon.support_circles():
            R = mach.counterbore_d_mm / 2
            if R <= r:
                continue
            ring = facing_up((bd.Pos(cx, cy) * bd.Circle(R)).faces()[0]) - facing_up(
                (bd.Pos(cx, cy) * bd.Circle(r)).faces()[0])
            if exact:
                ring = ring & weapon.profile()  # the lock cutout may take part of it
            for f in ring.faces():
                out.append((-1.0, *plate_part(f, h, t / 2 - h / 2)))
    return out


def plate_equivalent(parts, t: float) -> tuple[float, float, float]:
    """(area, first moment x, y) a set of removed parts amounts to, as a through-cut of the plate."""
    a = sum(-s * v for s, v, _, _ in parts) / t
    mx = sum(-s * v * c[0] for s, v, c, _ in parts) / t
    my = sum(-s * v * c[1] for s, v, c, _ in parts) / t
    return a, mx, my


class Machined(Weapon):
    """A plate weapon plus [machining] features. Everything not about the material is the inner weapon's."""

    def __init__(self, inner: Weapon, mach):
        self.inner = inner
        self.mach = mach
        self.params = inner.params
        self.spin = getattr(inner, "spin", "ccw")
        self._cache: dict = {}

    def __getattr__(self, name):  # anything else (derived geometry, solver internals): the inner weapon's
        return getattr(self.inner, name)

    @property
    def t(self) -> float:
        return self.params.thickness_mm

    def blind(self) -> list[bd.Face]:
        return self.inner.pocket_faces() if self.mach.pocket_depth_mm > 0 else []

    def profile(self) -> bd.Face:
        """Through-profile: blind pockets are material (with reduced thickness, see thickness_regions)."""
        if "profile" not in self._cache:
            face = self.inner.filled_profile() if self.blind() else self.inner.profile()
            self._cache["profile"] = face
        return self._cache["profile"]

    def thickness_regions(self) -> list[tuple[bd.Face, float]]:
        out = [(f, self.t - self.mach.pocket_depth_mm) for f in self.blind()]
        if self.mach.counterbore_d_mm > 0 and self.mach.counterbore_depth_mm > 0:
            for cx, cy, r in self.support_circles():
                out.append((facing_up((bd.Pos(cx, cy) * bd.Circle(self.mach.counterbore_d_mm / 2)).faces()[0]),
                            self.t - self.mach.counterbore_depth_mm))
        return out

    def mass_parts(self) -> list[tuple[float, float, np.ndarray, np.ndarray]]:
        from eitri.physics.mass import plate_part

        t, m = self.t, self.mach
        parts = [(1.0, *plate_part(self.profile(), t, 0.0))]
        for f in self.blind():
            if m.pocket_sides == 2:
                parts += [(-1.0, *plate_part(f, m.pocket_depth_mm / 2, z * (t / 2 - m.pocket_depth_mm / 4)))
                          for z in (1, -1)]
            else:
                parts.append((-1.0, *plate_part(f, m.pocket_depth_mm, t / 2 - m.pocket_depth_mm / 2)))
        if "features" not in self._cache:
            self._cache["features"] = feature_parts(self, m, t)
        return parts + self._cache["features"]

    def plate(self):
        return None  # not a prism: mass comes from mass_parts

    def solid(self) -> bd.Solid:
        if "solid" in self._cache:
            return self._cache["solid"]
        t, m = self.t, self.mach
        s = bd.extrude(self.profile(), amount=t / 2, both=True).solids()[0]
        cutters = []
        for f in self.blind():
            if m.pocket_sides == 2:
                h = m.pocket_depth_mm / 2
                cutters += [bd.Pos(0, 0, t / 2 - h) * bd.extrude(f, amount=h + 1),
                            bd.Pos(0, 0, -t / 2 - 1) * bd.extrude(f, amount=h + 1)]
            else:
                cutters.append(bd.Pos(0, 0, t / 2 - m.pocket_depth_mm) * bd.extrude(f, amount=m.pocket_depth_mm + 1))
        if m.counterbore_d_mm > 0 and m.counterbore_depth_mm > 0:
            for cx, cy, r in self.support_circles():
                cutters.append(bd.Pos(cx, cy, t / 2 - m.counterbore_depth_mm)
                               * bd.extrude(bd.Circle(m.counterbore_d_mm / 2), amount=m.counterbore_depth_mm + 1))
        for c in cutters:
            s = (s - c).solids()[0]
        for w in (bevel_wedges(self, m, t) if m.face_bevel_mm > 0 else []):
            # an explicit wedge: OCC's chamfer runs on along every tangent-continuous edge (root fillet, gullet)
            s = (s - w).solids()[0]
        self._cache["solid"] = s
        return s

    # --- the abstract interface: the inner weapon's ------------------------------------------------
    def tip_radius_mm(self):
        return self.inner.tip_radius_mm()

    def n_teeth(self):
        return self.inner.n_teeth()

    def tooth_depth_mm(self):
        return self.inner.tooth_depth_mm()

    def hook_deg(self):
        return self.inner.hook_deg()

    def impact_face(self):
        return self.inner.impact_face()

    def support_circles(self):
        return self.inner.support_circles()

    def bore_r_mm(self):
        return self.inner.bore_r_mm()

    def center_cutout_r_mm(self):
        return self.inner.center_cutout_r_mm()

    def has_pulley_lock(self):
        return self.inner.has_pulley_lock()

    def pocket_faces(self):
        return self.inner.pocket_faces()

    def derived(self) -> dict:
        return self.inner.derived() | {"machining": self.mach.model_dump()}

    def param_dict(self) -> dict:
        return self.inner.param_dict()

    def geometric_checks(self, min_ligament_mm: float) -> list[Check]:
        m, out = self.mach, self.inner.geometric_checks(min_ligament_mm)
        if m.pocket_depth_mm > 0:
            floor = self.t - m.pocket_depth_mm
            out.append(Check("pocket_floor", floor >= 1.5, floor, 1.5, "material left under a blind pocket"))
        return out

    def style_checks(self, style) -> list[Check]:
        return self.inner.style_checks(style)


def build(spec, weapon_cls, design: dict) -> Weapon:
    """Build a weapon with [machining]: feature moments go into the generator's balance first (an analytic
    estimate), then once more with the exact parts measured on the built weapon."""
    m = spec.machining
    probe = weapon_cls.from_dict(design, spec.weapon.spin)  # impact face and bolts don't need the balance
    t = probe.params.thickness_mm
    fields = getattr(type(probe.params), "__dataclass_fields__", {})
    base = dict(design)
    if "pocket_depth_frac" in fields:
        base["pocket_depth_frac"] = m.pocket_depth_mm / t if m.pocket_depth_mm > 0 else 1.0  # removed fraction

    def with_parts(parts) -> dict:
        if "extra_mx_mm3" not in fields:
            return base
        a, mx, my = plate_equivalent(parts, t)
        if getattr(probe, "spin", "ccw") == "cw" and weapon_cls.__name__ == "Hook":
            my = -my  # the hook balances in its canonical CCW frame and mirrors afterwards
        d = base | {"extra_mx_mm3": mx, "extra_my_mm3": my}
        if "extra_area_mm2" in fields:
            d["extra_area_mm2"] = a
        return d

    w = Machined(weapon_cls.from_dict(with_parts(feature_parts(probe, m, t, exact=False)), spec.weapon.spin), m)
    if "extra_mx_mm3" in fields and (m.face_bevel_mm > 0 or m.counterbore_d_mm > 0):
        exact = feature_parts(w, m, t)
        w = Machined(weapon_cls.from_dict(with_parts(exact), spec.weapon.spin), m)
    return w

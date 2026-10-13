"""A weapon imported from STEP (a single part, or one labelled part of an assembly).

Used to evaluate existing designs (and calibrate the models against weapons known to survive) with the same
pipeline as generated ones. Assumes a flat plate spinning about an axis parallel to the STEP's Z axis
through (axis_x_mm, axis_y_mm); a `flat_plate` check verifies the plate assumption the 2D FEA relies on.

Detected from the geometry:
    tip          : outline vertex at the largest radius that starts a straight face leading the spin
    impact face  : that straight edge (tip -> root), hook angle measured like the generated weapons
    supports     : circular holes off-center (bolts); FEA reacts the hit there
"""

import math
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import build123d as bd
import numpy as np

from eitri.geometry.weapon import Check, Weapon


@dataclass
class StepParams:
    source: str  # STEP path (Windows "C:\\..." paths are mapped to /mnt/c/...)
    part: str = ""  # assembly label; "" = the largest solid in the file
    axis_x_mm: float = 0.0  # spin axis location in the STEP's frame
    axis_y_mm: float = 0.0
    n_teeth: int = 1
    # filled in from the geometry:
    thickness_mm: float = 0.0
    bore_mm: float = 0.0

    def __post_init__(self):
        self.n_teeth = round(self.n_teeth)


def resolve_path(src: str) -> Path:
    if len(src) > 2 and src[1] == ":" and src[2] in "\\/":
        src = f"/mnt/{src[0].lower()}/" + src[3:].replace("\\", "/")
    return Path(src).expanduser()


@cache
def _load(path: str, part: str) -> bd.Solid:
    shape = bd.import_step(str(resolve_path(path)))
    if part:
        def find(s):
            for c in getattr(s, "children", None) or []:
                if c.label == part:
                    return c
                hit = find(c)
                if hit is not None:
                    return hit
            return None
        shape = find(shape)
        if shape is None:
            raise KeyError(f"no part labelled {part!r} in {path}")
    solids = shape.solids()
    if not solids:
        raise ValueError(f"{path}:{part} has no solids")
    return max(solids, key=lambda s: s.volume)


class StepPart(Weapon):
    def __init__(self, params: StepParams, spin: str = "ccw"):
        self.params = params
        self.spin = spin
        raw = _load(params.source, params.part)
        bb = raw.bounding_box()
        self._solid = bd.Pos(-params.axis_x_mm, -params.axis_y_mm, -(bb.min.Z + bb.max.Z) / 2) * raw
        params.thickness_mm = bb.size.Z
        tops = [f for f in self._solid.faces()
                if f.geom_type == bd.GeomType.PLANE and f.normal_at().Z > 0.999]
        if not tops:
            raise ValueError("no planar face normal to Z: is the spin axis along the STEP's Z?")
        top = max(tops, key=lambda f: f.area)
        self._profile = bd.Pos(0, 0, -top.center().Z) * top
        self._find_tooth()
        center_hole = [w for w in self._profile.inner_wires() if self._wire_contains_origin(w)]
        if center_hole:
            pts = self._sample(center_hole[0], 200)
            params.bore_mm = 2 * float(np.hypot(pts[:, 0], pts[:, 1]).min())

    @classmethod
    def from_dict(cls, d: dict, spin: str = "ccw") -> "StepPart":
        return cls(StepParams(**d), spin)

    # --- geometry helpers --------------------------------------------------------------------------
    @staticmethod
    def _sample(wire: bd.Wire, n: int) -> np.ndarray:
        return np.array([(p.X, p.Y) for p in (wire.position_at(t) for t in np.linspace(0, 1, n))])

    @staticmethod
    def _wire_contains_origin(wire: bd.Wire) -> bool:
        xy = StepPart._sample(wire, 200)
        ang = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
        return abs(ang[-1] - ang[0]) > np.pi  # winds around the axis

    def _spin_dir(self, x: float, y: float) -> np.ndarray:
        t = np.array([-y, x]) if self.spin == "ccw" else np.array([y, -x])
        return t / np.linalg.norm(t)

    def _find_tooth(self) -> None:
        outer = self._profile.outer_wire()
        verts = [(v.X, v.Y) for v in outer.vertices()]
        r_max = max(math.hypot(*v) for v in verts)
        best = None
        for e in outer.edges():
            if e.geom_type != bd.GeomType.LINE:
                continue
            a, b = (e.start_point().X, e.start_point().Y), (e.end_point().X, e.end_point().Y)
            tip, root = (a, b) if math.hypot(*a) >= math.hypot(*b) else (b, a)
            if math.hypot(*tip) < r_max - 1.0 or math.hypot(*root) > math.hypot(*tip) - 2.0:
                continue
            mid = (np.array(tip) + np.array(root)) / 2
            probe = mid + 1.0 * self._spin_dir(*mid)  # just ahead of the face, in the spin direction
            if self._profile.is_inside(bd.Vector(probe[0], probe[1], 0)):
                continue  # material ahead: this edge isn't a leading face
            if best is None or math.hypot(*tip) > math.hypot(*best[0]):
                best = (tip, root)
        if best is None:
            raise ValueError("couldn't find a straight impact face leading the spin at the tip radius "
                             "(check `spin` and the axis location)")
        self._tip, self._root = best
        self._r_max = r_max

    # --- weapon interface --------------------------------------------------------------------------
    def solid(self) -> bd.Solid:
        return self._solid

    def profile(self) -> bd.Face:
        return self._profile

    def tip_radius_mm(self) -> float:
        return math.hypot(*self._tip)

    def n_teeth(self) -> int:
        return self.params.n_teeth

    def tooth_depth_mm(self) -> float:
        return math.hypot(*self._tip) - math.hypot(*self._root)

    def hook_deg(self) -> float:
        tx, ty = self._tip
        rx, ry = self._root
        if self.spin == "cw":  # measure in the CCW-normalized frame used by the generators
            ty, ry = -ty, -ry
        inward = math.atan2(-ty, -tx)
        face = math.atan2(ry - ty, rx - tx)
        return math.degrees((face - inward + math.pi) % (2 * math.pi) - math.pi)

    def impact_face(self):
        return self._tip, self._root

    def support_circles(self):
        out = []
        for w in self._profile.inner_wires():
            edges = w.edges()
            if not all(e.geom_type == bd.GeomType.CIRCLE for e in edges):
                continue
            c, r = edges[0].arc_center, edges[0].radius
            same = all((e.arc_center - c).length < 1e-3 and abs(e.radius - r) < 1e-3 for e in edges)
            if same and math.hypot(c.X, c.Y) > 1.0:  # CAD often splits a hole into two half arcs
                out.append((c.X, c.Y, r))
        if not out:
            raise ValueError("no off-center circular holes (bolts) to support the FEA")
        return out

    def derived(self) -> dict:
        return {"thickness_mm": self.params.thickness_mm, "tip_xy_mm": list(self._tip),
                "face_root_xy_mm": list(self._root), "tooth_depth_mm": self.tooth_depth_mm(),
                "hook_deg": self.hook_deg(), "bore_min_d_mm": self.params.bore_mm,
                "n_bolts": len(self.support_circles())}

    def _plate_error(self) -> float:
        extruded = self._profile.area * self.params.thickness_mm
        return abs(extruded - self._solid.volume) / self._solid.volume

    def _is_prism(self) -> bool:
        """Exactly the profile extruded along Z: every face is a cap at z = +-t/2 or a wall with horizontal
        normals, and the two caps are the same area (no chamfers, counterbores, steps)."""
        from eitri.geometry.common import face_props

        t, caps = self.params.thickness_mm, {1: 0.0, -1: 0.0}
        for f in self._solid.faces():
            if f.geom_type == bd.GeomType.PLANE and abs(abs(f.normal_at().Z) - 1) < 1e-9:
                if abs(abs(f.center().Z) - t / 2) > 1e-6:
                    return False
                caps[round(f.normal_at().Z)] += face_props(f).Mass()
                continue
            for u in (0.1, 0.5, 0.9):
                for v in (0.1, 0.5, 0.9):
                    if abs(f.normal_at(u, v).Z) > 1e-6:
                        return False
        return abs(caps[1] - caps[-1]) <= 1e-6 * caps[1]

    def plate(self):
        # Exact-profile mass properties only if it really is a plate.
        return (self._profile, self.params.thickness_mm) if self._is_prism() else None

    def geometric_checks(self, min_ligament_mm: float) -> list[Check]:
        err = self._plate_error()
        return [Check("flat_plate", err < 0.03, err, 0.03,
                      "volume vs top-face area x thickness; 2D FEA assumes a flat plate")]

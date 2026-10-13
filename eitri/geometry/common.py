"""2D helpers shared by weapon geometries. Angles in degrees, lengths in mm, CCW positive."""

import math
from dataclasses import dataclass

import build123d as bd


def polar(r: float, deg: float) -> tuple[float, float]:
    a = math.radians(deg)
    return (r * math.cos(a), r * math.sin(a))


def hooked_face_root(r_tip: float, r_root: float, tip_deg: float, hook_deg: float) -> tuple[float, float]:
    """Root point (on r_root) of a straight impact face whose tip is at (r_tip, tip_deg).

    The face is the inward radial line rotated by `hook_deg`: positive hook puts the root behind the tip
    (the tip leads and grabs), negative leans the face back.
    """
    h = math.radians(hook_deg)
    tx, ty = polar(r_tip, tip_deg)
    a = math.radians(tip_deg) + math.pi + h
    dx, dy = math.cos(a), math.sin(a)
    disc = r_tip**2 * math.cos(h) ** 2 - r_tip**2 + r_root**2
    if disc < 0:
        raise ValueError(f"hook {hook_deg} deg too steep for this tooth depth")
    s = r_tip * math.cos(h) - math.sqrt(disc)
    return (tx + s * dx, ty + s * dy)


def nearest_vertices(face: bd.Face, points: list[tuple[float, float]]) -> list[bd.Vertex]:
    return [min(face.vertices(), key=lambda v: math.hypot(v.X - x, v.Y - y)) for x, y in points]


def single_face(shape) -> bd.Face:
    faces = shape.faces()
    if len(faces) != 1:
        raise ValueError(f"profile split into {len(faces)} faces (features don't connect)")
    return faces[0]


def face_props(face: bd.Face):
    """OCC surface properties with adaptive, error-controlled integration. The default (fixed Gauss order)
    misses faces bounded by B-splines/offset curves by ~1e-4 relative: tens of mm^3 of first moment on a
    hook, more than the whole balance tolerance, and booleans stop adding up. Checked against a 320k-point
    polygon to 1e-3 mm^3."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    g = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face.wrapped, g, 1e-9)
    return g


def first_moment(face: bd.Face) -> tuple[float, float, float]:
    """(area, area*cx, area*cy) of a planar face, mm^2 / mm^3."""
    g = face_props(face)
    c = g.CentreOfMass()
    return g.Mass(), g.Mass() * c.X(), g.Mass() * c.Y()


def facing_up(face: bd.Face) -> bd.Face:
    """Orient a planar face's normal to +Z (booleans won't merge faces wound in opposite directions)."""
    return -face if face.normal_at().Z < 0 else face


def corner_turn_deg(face: bd.Face, pt: tuple[float, float]) -> float:
    """How sharply the outline turns at the vertex nearest `pt`: 0 = smooth (tangent), 90 = square corner."""
    v = nearest_vertices(face, [pt])[0]
    p = bd.Vector(v.X, v.Y, 0)
    dirs = []
    for e in face.edges():
        if (e.start_point() - p).length < 1e-3:
            dirs.append(e.tangent_at(0))
        elif (e.end_point() - p).length < 1e-3:
            dirs.append(-e.tangent_at(1))
    if len(dirs) != 2:
        return 0.0
    cos = max(-1.0, min(1.0, dirs[0].dot(dirs[1])))
    return 180.0 - math.degrees(math.acos(cos))


def to_nurbs(shape):
    """Convert all curves/surfaces to NURBS. Offset curves (from 2D offsets) are silently dropped by the STEP
    writer, so anything built with `offset` must go through this before it reaches a solid."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert

    return type(shape)(BRepBuilderAPI_NurbsConvert(shape.wrapped, True).Shape())


@dataclass
class Corner:
    x: float
    y: float
    turn_deg: float  # 0 = tangent (smooth), 90 = square corner
    concave: bool  # re-entrant into the material: a stress raiser unless filleted


def corners(face: bd.Face, min_turn_deg: float = 0.5) -> list[Corner]:
    """Every vertex of a face's wires where the boundary kinks by more than `min_turn_deg`."""
    out = []
    for wire in [face.outer_wire(), *face.inner_wires()]:
        edges = wire.edges()
        for v in wire.vertices():
            p = bd.Vector(v.X, v.Y, 0)
            dirs = []  # unit tangents leaving the vertex along each adjacent edge
            for e in edges:
                if (e.start_point() - p).length < 1e-4:
                    dirs.append(e.tangent_at(0))
                elif (e.end_point() - p).length < 1e-4:
                    dirs.append(-e.tangent_at(1))
            if len(dirs) != 2:
                continue
            turn = 180.0 - math.degrees(math.acos(max(-1.0, min(1.0, dirs[0].dot(dirs[1])))))
            if turn < min_turn_deg:
                continue
            # The bisector points into the smaller wedge between the edges: material there = pointy (convex)
            # corner, void there = the material wraps around it (re-entrant).
            bis = (dirs[0] + dirs[1]).normalized()
            concave = not face.is_inside(p + bis * 0.02)
            out.append(Corner(v.X, v.Y, turn, concave))
    return out


def worst_concave_corner(face: bd.Face) -> Corner | None:
    cc = [c for c in corners(face) if c.concave]
    return max(cc, key=lambda c: c.turn_deg) if cc else None


def fillet_corners(face: bd.Face, r: float, min_turn_deg: float = 2.0,
                   only: list[tuple[float, float]] | None = None) -> bd.Face:
    """Fillet every corner sharper than `min_turn_deg` (or only those nearest the `only` points), one at a
    time, sharpest first, backing the radius off where the adjacent edges are too short. Corners that can't
    be filleted even at 0.3 r stay sharp, and the `sharp_corners` check reports them.

    Corner-by-corner on purpose: filleting all vertices at once fails as a whole if any one (e.g. a tangent
    junction between offset curve pieces) can't be filleted.
    """
    # Rebuild first: build123d's fillet() finds its target through the vertices' topo_parent, which for a
    # boolean result is the *operand*, not this face.
    face = facing_up(bd.Face(face.outer_wire(), face.inner_wires()))
    if r <= 0:
        return face
    targets = None if only is None else [bd.Vector(x, y, 0) for x, y in only]
    failed: list[bd.Vector] = []
    for _ in range(200):
        todo = [c for c in corners(face, min_turn_deg)
                if all((bd.Vector(c.x, c.y, 0) - q).length > 1e-3 for q in failed)
                and (targets is None or any((bd.Vector(c.x, c.y, 0) - q).length < 1e-3 for q in targets))]
        if not todo:
            return face
        c = max(todo, key=lambda c: c.turn_deg)
        v = nearest_vertices(face, [(c.x, c.y)])
        for k in (1.0, 0.7, 0.5, 0.3):
            try:
                face = facing_up(single_face(bd.fillet(v, r * k)))
                face = bd.Face(face.outer_wire(), face.inner_wires())
                break
            except Exception:
                continue
        else:
            failed.append(bd.Vector(c.x, c.y, 0))
    return face


def hub_lock(lock_d: float, boss_d: float, bolt_pcd: float, bolt_count: int, fillet: float,
             start_deg: float = 0.0) -> bd.Face:
    """Center cutout that keys the weapon to its pulley: a round hole of `lock_d` with a boss of material
    (`boss_d`) left around each bolt, so torque goes through the boss flanks and not only bolt friction
    (Rebot V3's GanchoUnidente pattern: 29 mm hole, 10.5 mm bosses on the 7 x M5 bolt circle, R1 fillets)."""
    hole = bd.Circle(lock_d / 2).faces()[0]
    bosses = [bd.Pos(*polar(bolt_pcd / 2, start_deg + 360 * i / bolt_count)) * bd.Circle(boss_d / 2)
              for i in range(bolt_count)]
    return fillet_corners(single_face(hole - bosses), fillet)


def pocket_wall_mm(face: bd.Face, hub_r: float) -> float:
    """Thinnest wall between a pocket (an inner wire reaching past `hub_r`; bolt holes and the center cutout
    don't count) and the outline. Exact distance (BRepExtrema), so it works on any profile."""
    outer = face.outer_wire()
    walls = [w.distance_to(outer) for w in face.inner_wires()
             if max(math.hypot(v.X, v.Y) for v in w.vertices()) > hub_r]
    return min(walls) if walls else math.inf


def offset_pieces(face: bd.Face, amount: float) -> list[bd.Face]:
    """2D offset of a face's outer wire (negative = shrink, round joins). Unlike build123d's `offset`, a shrink
    that splits the face into several pieces, or makes it vanish, returns those pieces (or []) instead of
    raising.

    Edges are first split wherever they are only C0 inside: an earlier offset can leave a B-spline with a
    kink in its middle, and OCC's 2D offset fails outright on those."""
    from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeOffset
    from OCP.GeomAbs import GeomAbs_JoinType, GeomAbs_Shape
    from OCP.ShapeUpgrade import ShapeUpgrade_ShapeDivideContinuity

    split = ShapeUpgrade_ShapeDivideContinuity(face.wrapped)
    split.SetBoundaryCriterion(GeomAbs_Shape.GeomAbs_C1)
    split.SetTolerance(1e-4)
    split.Perform()
    mk = BRepOffsetAPI_MakeOffset()
    mk.Init(GeomAbs_JoinType.GeomAbs_Arc)
    mk.AddWire(bd.Face(split.Result()).outer_wire().wrapped)
    try:
        mk.Perform(amount)
    except Exception:
        return []
    if not mk.IsDone():
        return []
    out = []
    for w in bd.Compound(mk.Shape()).wires() if not mk.Shape().IsNull() else []:
        try:
            f = bd.Face(w)
        except Exception:
            continue
        if w.is_closed and f.area > 1e-6:
            out.append(facing_up(f))
    return out

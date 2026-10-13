"""2D helpers shared by weapon geometries. Angles in degrees, lengths in mm, CCW positive."""

import math

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


def first_moment(face: bd.Face) -> tuple[float, float, float]:
    """(area, area*cx, area*cy) of a planar face, mm^2 / mm^3."""
    c = face.center(bd.CenterOf.MASS)
    a = face.area
    return a, a * c.X, a * c.Y


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

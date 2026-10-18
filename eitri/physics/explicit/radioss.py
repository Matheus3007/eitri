"""Write OpenCourant/OpenRadioss block-format decks and run them.

Units (consistent): mm, ms, kg  ->  force kN, stress GPa, energy J, velocity mm/ms = m/s, density kg/mm^3.
Cards are fixed-width (10/20 columns) as described in OpenCourant's hm_cfg_files (CFG/radiossYYYY/...).
The solver runs through tools/opencourant.sh (sets its runtime environment).
"""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

WRAPPER = Path(__file__).resolve().parents[3] / "tools" / "opencourant.sh"


def _i(*v) -> str:
    return "".join(f"{int(x):10d}" for x in v)


def _f(*v) -> str:
    return "".join(f"{float(x):20.10g}" for x in v)


def _ids(ids, per=10) -> list[str]:
    ids = [int(i) for i in ids]
    return [_i(*ids[k:k + per]) for k in range(0, len(ids), per)]


@dataclass
class Deck:
    """Accumulates starter cards; `write` emits <name>_0000.rad (starter) and <name>_0001.rad (engine)."""
    name: str
    cards: list[str] = field(default_factory=list)
    engine: list[str] = field(default_factory=list)

    def add(self, *lines: str) -> None:
        self.cards += list(lines)

    # --- model --------------------------------------------------------------------------------------
    def nodes(self, xyz: np.ndarray, first_id: int = 1) -> None:
        self.add("/NODE")
        self.add(*[f"{first_id + i:10d}{x:20.10g}{y:20.10g}{z:20.10g}" for i, (x, y, z) in enumerate(xyz)])

    def tetra10(self, part: int, elems: np.ndarray, first_id: int = 1) -> None:
        """elems: (m, 10) node ids, corners then midsides on edges 12, 23, 31, 14, 24, 34."""
        self.add(f"/TETRA10/{part}")
        for k, e in enumerate(elems):
            self.add(_i(first_id + k), _i(*e))

    def brick(self, part: int, elems: np.ndarray, first_id: int = 1) -> None:
        self.add(f"/BRICK/{part}")
        self.add(*[_i(first_id + k, *e) for k, e in enumerate(elems)])

    def part(self, pid: int, title: str, prop: int, mat: int) -> None:
        self.add(f"/PART/{pid}", title, _i(prop, mat, 0))

    def prop_solid(self, pid: int, title: str, isolid: int = 0, itetra10: int = 0, icontrol: int = 0) -> None:
        """`icontrol` = 1: distortion control (soft materials: foams, rubbers) -- keeps crushed elements from
        inverting."""
        self.add(f"/PROP/TYPE14/{pid}", title, _i(isolid, 0, 0, 0, itetra10, 0, 0, 0) + _f(0.0),
                 _f(0, 0, 0, 0, 0), _f(0, 0, 0, 0, 0), _i(0, 0, icontrol))

    def mat_elastic(self, mid: int, title: str, rho: float, E: float, nu: float) -> None:
        self.add(f"/MAT/LAW1/{mid}", title, _f(rho), _f(E, nu))

    def mat_johnson_cook(self, mid: int, title: str, rho: float, E: float, nu: float, A: float, B: float,
                         n: float, c: float = 0.0, eps_dot0: float = 1.0, eps_max: float = 0.0) -> None:
        """LAW2 (Johnson-Cook) without thermal softening. Stresses GPa, strain rate reference 1/ms."""
        self.add(f"/MAT/PLAS_JOHNS/{mid}", title, _f(rho), _f(E, nu) + _i(0, 0) + _f(0.0),
                 _f(A, B, n, eps_max, 0.0), _f(c, eps_dot0) + _i(0, 0) + _f(0.0, 0.0), _f(0, 0, 0, 0, 0))

    def funct(self, fid: int, title: str, xy) -> None:
        self.add(f"/FUNCT/{fid}", title, *[_f(x, y) for x, y in xy])

    def mat_ogden(self, mid: int, title: str, rho: float, nu: float, mu, alpha, prony_g=(), prony_tau=()) -> None:
        """LAW42 Ogden hyperelastic (up to 10 terms; initial shear modulus = sum mu_i alpha_i / 2) with a Prony
        viscous series: shear moduli `prony_g` (GPa) relaxing with times `prony_tau` (ms)."""
        mu, alpha = list(mu) + [0.0] * (10 - len(mu)), list(alpha) + [0.0] * (10 - len(alpha))
        m = len(prony_g)
        lines = [f"/MAT/LAW42/{mid}", title, _f(rho), _f(nu, 0.0) + _i(0, 0) + _f(0.0) + _i(m, 0),
                 _f(*mu[:5]), _f(*mu[5:]), _f(*alpha[:5]), _f(*alpha[5:])]
        for vals in (prony_g, prony_tau):
            vals = list(vals)
            lines += [_f(*vals[k:k + 5]) for k in range(0, len(vals), 5)]
        self.add(*lines)

    def mat_foam_tab(self, mid: int, title: str, rho: float, E0: float, nu: float, Emax: float, eps_max: float,
                     load: list[tuple[int, float, float]], unload: list[tuple[int, float, float]] = ()) -> None:
        """LAW70 tabulated foam: loading curves (function id, strain rate 1/ms, scale) stress(GPa) vs strain."""
        self.add(f"/MAT/LAW70/{mid}", title, _f(rho), _f(E0, nu, Emax, eps_max) + _i(0),
                 _f(0.0) + _i(0, len(load), len(unload), 0) + _f(0.0, 0.0),
                 *[_i(f) + _f(r, sc) for f, r, sc in load], *[_i(f) + _f(r, sc) for f, r, sc in unload])

    def impvel(self, iid: int, title: str, fid: int, direction: str, grnod: int, scale: float) -> None:
        """Imposed velocity: function `fid` (of time) x `scale` along X/Y/Z on the nodes of `grnod`."""
        self.add(f"/IMPVEL/{iid}", title, _i(fid) + f"{direction:>10s}" + " " * 10 + _i(0, grnod) + " " * 10 + _i(0),
                 _f(1.0, scale, 0.0, 1e30))

    def fail_tensstrain(self, mid: int, eps_start: float, eps_rupture: float) -> None:
        """Element deleted when its max principal (engineering) strain goes from `eps_start` to `eps_rupture`:
        tearing of elastomers, which have no plastic strain for a ductile-damage law."""
        self.add(f"/FAIL/TENSSTRAIN/{mid}", _f(eps_start, eps_rupture) + _i(0) + _f(0.0, 0.0) + _i(0), _i(0))

    def fail_johnson(self, mid: int, D: tuple, eps_dot0: float = 1.0) -> None:
        """Johnson-Cook damage; solids are deleted when the damage reaches 1 (ISOLID = 1)."""
        self.add(f"/FAIL/JOHNSON/{mid}", _f(*D), _f(eps_dot0) + _i(0, 1) + _f(0.0, 0.0) + " " * 10 + _i(0),
                 _i(0))

    def surf_part_all(self, sid: int, title: str, parts) -> None:
        """Every face of solid parts, internal ones included: eroding contacts see material exposed by deletion."""
        self.add(f"/SURF/PART/ALL/{sid}", title, *_ids(parts))

    def inter25(self, iid: int, title: str, surf1: int, surf2: int, fric: float = 0.1, idel: int = 1,
                iedge: int = 1) -> None:
        """Symmetric surface-to-surface penalty contact (TYPE25): robust at sharp edges (`iedge` = 1) and with
        eroding solids (`idel`)."""
        self.add(f"/INTER/TYPE25/{iid}", title, _i(surf1, surf2, 0, 0, 0, 0) + " " * 10 + _i(idel, iedge, 0),
                 _i(0) + " " * 10 + _f(0.0, 0.0, 0.0, 0.0), _f(0.0, 0.0) + _i(0, 0) + _f(0.0, 0.0),
                 _f(0.0, fric, 0.0, 0.0, 0.0), " " * 7 + "000" + " " * 10 + _i(0, 0) + _f(0.0) + _i(0) + " " * 10 + _f(0.0),
                 _i(0, 0) + _f(0.0) + " " * 10 + _i(0) + _f(0.0) + " " * 10 + _i(0))

    def grnod(self, gid: int, title: str, ids) -> None:
        self.add(f"/GRNOD/NODE/{gid}", title, *_ids(ids))

    def grnod_part(self, gid: int, title: str, parts) -> None:
        self.add(f"/GRNOD/PART/{gid}", title, *_ids(parts))

    def surf_part(self, sid: int, title: str, parts) -> None:
        """External surface of solid parts."""
        self.add(f"/SURF/PART/EXT/{sid}", title, *_ids(parts))

    def rbody(self, rid: int, title: str, master: int, grnod: int, mass: float = 0.0, J=(0, 0, 0, 0, 0, 0)) -> None:
        """Rigid body: `grnod` nodes slaved to `master` (whose added mass/inertia are `mass`, `J`)."""
        self.add(f"/RBODY/{rid}", title, _i(master, 0, 0, 0) + _f(mass) + _i(grnod, 0, 0, 0),
                 _f(*J[:3]), _f(*J[3:]), _i(0, 0, 0))

    def bcs(self, bid: int, title: str, trans: str, rot: str, grnod: int) -> None:
        self.add(f"/BCS/{bid}", title, f"   {trans} {rot}" + _i(0, grnod))

    def inivel_axis(self, iid: int, title: str, grnod: int, omega: float, v=(0.0, 0.0, 0.0)) -> None:
        """Rigid rotation about global Z (rad/ms) plus a translation (mm/ms) for the nodes of `grnod`."""
        self.add(f"/INIVEL/AXIS/{iid}", title, f"{'Z':>10s}" + _i(0, grnod), _f(*v, omega), _f(0.0) + _i(0))

    def inivel_rot(self, iid: int, title: str, grnod: int, w) -> None:
        """Angular velocity (rad/ms) of nodes with rotational dofs (e.g. a rigid body's master node)."""
        self.add(f"/INIVEL/ROT/{iid}", title, _f(*w) + _i(grnod, 0), _f(0.0) + _i(0))

    def inivel_tra(self, iid: int, title: str, grnod: int, v) -> None:
        self.add(f"/INIVEL/TRA/{iid}", title, _f(*v) + _i(grnod, 0), _f(0.0) + _i(0))

    def inter7(self, iid: int, title: str, grnod: int, surf: int, gapmin: float = 0.1, fric: float = 0.1,
               istf: int = 4, stfac: float = 1.0, idel: int = 0) -> None:
        """Node-to-surface penalty contact (nodes of `grnod` against `surf`). `idel` = 2: a segment leaves the
        contact as soon as one of its elements is deleted (eroding/cut materials)."""
        self.add(f"/INTER/TYPE7/{iid}", title, _i(grnod, surf, istf, 0, 0) + " " * 10 + _i(0, idel, 0, 0),
                 _f(0.0, 0.0, 0.0) + " " * 20 + _i(0, 0), _f(0.0, 0.0, 0.0, 0.0) + _i(0, 0),
                 _f(stfac, fric, gapmin, 0.0, 0.0), " " * 7 + "000" + " " * 20 + _i(0) + _f(0.0, 0.0, 0.0),
                 _i(0, 0) + _f(0.0) + _i(0, 0, 0) + _f(0.0) + _i(0))

    def th(self, kind: str, tid: int, title: str, variables: list[str], ids) -> None:
        self.add(f"/TH/{kind}/{tid}", title, "".join(f"{v:<10s}" for v in variables), *_ids(ids))

    # --- engine -------------------------------------------------------------------------------------
    def run(self, tstop_ms: float, anim_dt_ms: float, th_dt_ms: float, anim_elem=("VONM", "EPSP"),
            dt_min_ms: float = 0.0, solid_cst_ms: float = 0.0, brick_delete_ms: float = 0.0) -> None:
        """`dt_min_ms` > 0: nodal mass scaling up to that time step (the added mass is in the time history).
        `solid_cst_ms` > 0 instead: a solid whose step falls below it switches to a small-strain formulation at
        that constant step (no added mass, no deletion) -- the robust choice with crushable soft materials."""
        dt = ["/DT/NODA/CST/0", f"0.67 {dt_min_ms:g}"] if dt_min_ms > 0 else []
        if solid_cst_ms > 0:
            # fixed columns (20 + 1 + 20): a free-format line is misread
            dt += ["/DT/BRICK/CST", f"{0.67:<20g} {solid_cst_ms:<20g}"]
        if brick_delete_ms > 0:  # bricks (hexas; not the tet10) whose step falls below it are deleted
            dt += ["/DT/BRICK/DEL", f"{0.67:<20g} {brick_delete_ms:<20g}"]
        self.engine = [f"/RUN/{self.name}/1/", f"{tstop_ms:g}", *dt, "/ANIM/DT", f"0 {anim_dt_ms:g}",
                       *[f"/ANIM/ELEM/{a}" for a in anim_elem], "/ANIM/VECT/VEL", "/ANIM/VECT/CONT", "/ANIM/NODA/DMAS",
                       "/TFILE/0", f"{th_dt_ms:g}", "/PRINT/-500", "/VERS/2026"]

    def write(self, folder: Path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        head = ["#RADIOSS STARTER", "/BEGIN", self.name, _i(2026, 0), f"{'kg':>20s}{'mm':>20s}{'ms':>20s}",
                f"{'kg':>20s}{'mm':>20s}{'ms':>20s}"]
        (folder / f"{self.name}_0000.rad").write_text("\n".join(head + self.cards + ["/END"]) + "\n")
        (folder / f"{self.name}_0001.rad").write_text("\n".join(self.engine + ["/END"]) + "\n")
        return folder


@dataclass
class RunResult:
    folder: Path
    ok: bool
    starter_log: str
    engine_log: str
    anim: list[Path]
    th: Path | None


def run(folder: Path, name: str, threads: int = 4, timeout_s: float = 3600) -> RunResult:
    def call(tool, *args):
        p = subprocess.run([str(WRAPPER), tool, *args], cwd=folder, capture_output=True, text=True,
                           timeout=timeout_s, env=None)
        return p.returncode, p.stdout + p.stderr

    rc, slog = call("starter", "-i", f"{name}_0000.rad", "-np", "1")
    ok = rc == 0 and "ERROR" not in slog.upper().replace("0 ERROR", "")
    elog = ""
    if ok:
        import os

        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        rc, elog = call("engine", "-i", f"{name}_0001.rad", "-nt", str(threads))
        ok = rc == 0 and "NORMAL TERMINATION" in (elog + _read(folder / f"{name}_0001.out")).upper()
    anim = sorted(folder.glob(f"{name}A[0-9][0-9][0-9]"))
    th = next(iter(sorted(folder.glob(f"{name}T01"))), None)
    return RunResult(folder, ok, slog, elog, anim, th)


def _read(p: Path) -> str:
    return p.read_text(errors="ignore") if p.exists() else ""

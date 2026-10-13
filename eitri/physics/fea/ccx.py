"""CalculiX: write .inp, run ccx, parse nodal results from the .frd.

Unit system: mm, N, tonne, s -> stress in MPa, density in tonne/mm^3.
"""

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from eitri.physics.fea.mesh import Mesh2D

CCX = shutil.which("ccx")


@dataclass
class LoadCase:
    name: str
    omega_rad_s: float = 0.0  # centrifugal about Z through the origin
    pressures: list[tuple[int, int, float]] = field(default_factory=list)  # (elem, face 3/4/5, MPa)
    cloads: list[tuple[int, int, float]] = field(default_factory=list)  # (node, dof 1/2, N)
    fixed: dict[int, tuple[int, ...]] = field(default_factory=dict)  # node -> constrained dofs


@dataclass
class Result:
    disp: np.ndarray  # (n, 2) mm
    stress: np.ndarray  # (n, 6) MPa: sxx syy szz sxy syz szx (nodal, extrapolated + averaged)

    @property
    def von_mises(self) -> np.ndarray:
        sx, sy, sz, txy, tyz, tzx = self.stress.T
        return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
                       + 3 * (txy**2 + tyz**2 + tzx**2))

    @property
    def max_principal(self) -> np.ndarray:
        sx, sy, txy = self.stress[:, 0], self.stress[:, 1], self.stress[:, 3]
        return (sx + sy) / 2 + np.sqrt(((sx - sy) / 2) ** 2 + txy**2)


def write_inp(path: Path, mesh: Mesh2D, thickness_mm: float, E_mpa: float, nu: float,
              density_kg_m3: float, case: LoadCase) -> None:
    L = ["*HEADING", f"eitri {case.name}", "*NODE, NSET=Nall"]
    L += [f"{i + 1},{x:.9g},{y:.9g},0" for i, (x, y) in enumerate(mesh.nodes)]
    L.append("*ELEMENT, TYPE=CPS6, ELSET=Eall")
    L += [f"{i + 1}," + ",".join(map(str, e)) for i, e in enumerate(mesh.elems)]
    L += ["*MATERIAL, NAME=STEEL", "*ELASTIC", f"{E_mpa},{nu}", "*DENSITY", f"{density_kg_m3 * 1e-12:.6e}",
          "*SOLID SECTION, ELSET=Eall, MATERIAL=STEEL", f"{thickness_mm}"]
    L.append("*BOUNDARY")
    for node, dofs in case.fixed.items():
        L += [f"{node},{d},{d},0." for d in dofs]
    L += ["*STEP", "*STATIC"]
    if case.omega_rad_s:
        L += ["*DLOAD", f"Eall,CENTRIF,{case.omega_rad_s**2:.9g},0.,0.,0.,0.,0.,1."]
    if case.pressures:
        L.append("*DLOAD")
        L += [f"{e},P{f},{p:.9g}" for e, f, p in case.pressures]
    if case.cloads:
        L.append("*CLOAD")
        L += [f"{n},{d},{v:.9g}" for n, d, v in case.cloads]
    L += ["*NODE FILE", "U", "*EL FILE", "S", "*END STEP"]
    path.write_text("\n".join(L) + "\n")


def parse_frd(path: Path, n_nodes: int) -> Result:
    """Nodal DISP and STRESS blocks. Fixed-width records: ' -1' + I10 node + E12.5 values."""
    blocks: dict[str, np.ndarray] = {}
    current, ncomp = None, 0
    for line in path.read_text().splitlines():
        if line.startswith(" -4"):
            parts = line.split()
            current = parts[1]
            ncomp = {"DISP": 3, "STRESS": 6}.get(current, 0)
            if ncomp:
                blocks[current] = np.zeros((n_nodes, ncomp))
        elif line.startswith(" -1") and current in blocks:
            node = int(line[3:13])
            if node <= n_nodes:
                blocks[current][node - 1] = [float(line[13 + 12 * i: 25 + 12 * i]) for i in range(ncomp)]
        elif line.startswith(" -3"):
            current = None
    if "DISP" not in blocks or "STRESS" not in blocks:
        raise RuntimeError(f"no DISP/STRESS results in {path}")
    return Result(disp=blocks["DISP"], stress=blocks["STRESS"])


def run_job(inp: str, n_nodes: int, name: str, threads: int = 1, keep_dir: Path | None = None) -> Result:
    """Run one ccx deck (text) and parse its nodal results."""
    if CCX is None:
        raise RuntimeError("CalculiX `ccx` not found on PATH (apt install calculix-ccx)")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "job.inp").write_text(inp)
        env = {**os.environ, "OMP_NUM_THREADS": str(threads), "CCX_NPROC_EQUATION_SOLVER": str(threads)}
        proc = subprocess.run([CCX, "-i", "job"], cwd=work, env=env, capture_output=True, text=True)
        if keep_dir is not None:
            keep_dir.mkdir(parents=True, exist_ok=True)
            for f in ("job.inp", "job.frd", "job.dat"):
                if (work / f).exists():
                    shutil.copy(work / f, keep_dir / f"{name}.{f.split('.')[1]}")
        if proc.returncode != 0 or "*ERROR" in proc.stdout:
            err = [l.strip() for l in proc.stdout.splitlines() if "ERROR" in l][:5]
            raise RuntimeError(f"ccx failed ({name}, exit code {proc.returncode}): "
                               f"{err or proc.stderr.strip()[-300:] or 'no error message (crash)'}")
        return parse_frd(work / "job.frd", n_nodes)


def solve(mesh: Mesh2D, thickness_mm: float, E_mpa: float, nu: float, density_kg_m3: float,
          case: LoadCase, threads: int = 1, keep_dir: Path | None = None) -> Result:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "job.inp"
        write_inp(path, mesh, thickness_mm, E_mpa, nu, density_kg_m3, case)
        inp = path.read_text()
    res = run_job(inp, mesh.n_nodes, case.name, threads, keep_dir)
    return Result(disp=res.disp[:, :2], stress=res.stress)

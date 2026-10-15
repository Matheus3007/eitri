"""Material database. Values live in materials.toml so they can be read and edited as text."""

import tomllib
from functools import cache
from pathlib import Path

from pydantic import BaseModel

DB_PATH = Path(__file__).with_name("materials.toml")


class Material(BaseModel):
    name: str
    description: str
    density: float  # kg/m^3
    E: float  # MPa
    nu: float
    yield_: float  # MPa
    uts: float  # MPa
    elongation_pct: float
    charpy_j: float
    hardness: str
    # explicit dynamics (eitri/physics/explicit): Johnson-Cook flow stress A + B eps^n (1 + C ln rate), and the
    # true fracture strain. Missing values are fitted from yield/UTS/elongation (`johnson_cook()`): estimates.
    jc_b: float | None = None  # MPa
    jc_n: float | None = None
    jc_c: float = 0.01
    fracture_strain: float | None = None

    def johnson_cook(self) -> dict:
        """A, B (MPa), n, C and fracture strain. Fit: A = yield; B, n so that the true stress at the uniform
        strain (0.6 x elongation) equals the true UTS and Considere's necking condition holds there
        (d sigma / d eps = sigma). Fracture strain default 2.5 x elongation (true strain): an estimate --
        measured values (tension tests, supplier data) go in materials.toml."""
        import math

        A = self.yield_
        if self.jc_b is not None and self.jc_n is not None:
            B, n = self.jc_b, self.jc_n
        else:
            eu = 0.6 * self.elongation_pct / 100
            eps_u, s_u = math.log(1 + eu), self.uts * (1 + eu)
            # A + B e^n = s_u and B n e^(n-1) = s_u  ->  (s_u - A) n / e = s_u
            n = max(min(s_u * eps_u / (s_u - A), 0.9), 0.02) if s_u > A else 0.05
            B = (s_u - A) / eps_u**n if s_u > A else 0.05 * A
        ef = self.fracture_strain if self.fracture_strain is not None else 2.5 * self.elongation_pct / 100
        return {"A": A, "B": B, "n": n, "C": self.jc_c, "eps_f": ef}


@cache
def all_materials() -> dict[str, Material]:
    raw = tomllib.loads(DB_PATH.read_text())
    return {
        name: Material(name=name, yield_=props.pop("yield"), **props)
        for name, props in raw.items()
    }


def get(name: str) -> Material:
    mats = all_materials()
    if name not in mats:
        raise KeyError(f"unknown material {name!r}; known: {sorted(mats)}")
    return mats[name]

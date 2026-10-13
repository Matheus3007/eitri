"""Project spec: the single text file (TOML) that drives every eitri command.

Units at the spec level are the ones builders think in: mm, kg, rpm, V, A, m/s, ms.
Design parameters are either a fixed number or a [min, max] range that the optimizer may vary.
"""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from eitri import materials

Param = float | tuple[float, float] | str  # str for style choices, e.g. pocket_style = "swept"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Weapon(Strict):
    type: Literal["fang", "disc"] = "fang"  # fang = asymmetric single tooth; "drum"/"eggbeater" later
    spin: Literal["ccw", "cw"] = "ccw"  # seen from +Z; teeth face the spin direction


class Material(Strict):
    name: str | None = None
    candidates: list[str] | None = None

    @model_validator(mode="after")
    def _check(self):
        names = ([self.name] if self.name else []) + (self.candidates or [])
        if not names:
            raise ValueError("material needs `name` or `candidates`")
        for n in names:
            materials.get(n)
        return self

    def names(self) -> list[str]:
        return self.candidates or [self.name]


class Motor(Strict):
    kv: float  # rpm/V
    voltage: float  # V
    resistance_ohm: float = 0.0  # phase-to-phase; 0 = ideal
    current_limit_a: float | None = None
    rotor_inertia_kgm2: float = 0.0
    speed_efficiency: float = 0.9  # loaded top speed / ideal Kv*V


class Drive(Strict):
    rpm: float | None = None  # weapon rpm, if known directly
    motor: Motor | None = None
    ratio: float = 1.0  # motor turns per weapon turn (reduction > 1)

    @model_validator(mode="after")
    def _check(self):
        if (self.rpm is None) == (self.motor is None):
            raise ValueError("drive needs exactly one of `rpm` or `motor`")
        return self


class Context(Strict):
    """The fight the weapon is designed for."""

    closing_speed_mps: float = 3.0
    opponent_mass_kg: float = 13.6
    restitution: float = 0.2
    # Hit pulse duration -> equivalent static peak force. 2 ms matches explicit impact sims the user's team ran;
    # it is the least certain input: impact results scale with it (see stress.min_elastic_contact_time_ms).
    contact_time_ms: float = 2.0


class Bound(Strict):
    min: float | None = None
    max: float | None = None


class Constraints(Strict):
    mass_kg: Bound = Field(default_factory=Bound)
    min_safety_factor: float = 2.0  # spinning, no hit: yield / peak von Mises in the body
    min_impact_safety_factor: float | None = 1.0  # body yield force / estimated peak hit force; None = report only
    max_spinup_s: float | None = None
    max_imbalance_gmm: float = 20.0
    min_ligament_mm: float = 5.0


class Style(Strict):
    """Looks are a hard requirement: weapons must read as aggressive and well made, never as optimizer blobs.
    These are enforced as constraints like any other; set a value to null/omit-by-override to relax it."""

    min_hook_deg: float | None = 5.0  # teeth lean into the spin and grab
    max_root_fillet_ratio: float | None = 0.4  # gullet fillet / tooth height: crisp gullets, not scoops
    min_pocket_sweep_deg: float | None = 15.0  # swept pockets must visibly sweep, not look like wheel spokes


class Analysis(Strict):
    fea: bool = True  # CalculiX plane-stress FEA; False = analytic Lame estimate only (fast, crude)
    mesh_mm: float | None = None  # max element size; default od/25 (optimizer), use --fine for od/75
    threads: int = 4


class Objective(Strict):
    maximize: Literal["energy", "moi", "effective_mass"] = "energy"


class Spec(Strict):
    name: str = "unnamed"
    weapon: Weapon = Field(default_factory=Weapon)
    material: Material
    drive: Drive
    context: Context = Field(default_factory=Context)
    constraints: Constraints = Field(default_factory=Constraints)
    objective: Objective = Field(default_factory=Objective)
    analysis: Analysis = Field(default_factory=Analysis)
    style: Style = Field(default_factory=Style)
    design: dict[str, Param]

    def free_params(self) -> dict[str, tuple[float, float]]:
        return {k: v for k, v in self.design.items() if isinstance(v, tuple)}

    def fixed_design(self) -> dict[str, float]:
        """Design with every range collapsed to its midpoint (what `eval` uses)."""
        return {k: (v[0] + v[1]) / 2 if isinstance(v, tuple) else v for k, v in self.design.items()}


def load(path: str | Path) -> Spec:
    return Spec.model_validate(tomllib.loads(Path(path).read_text()))

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
    type: Literal["hook", "fang", "disc", "step"] = "hook"  # hook/fang = asymmetric single tooth; step = imported
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
    # Hit pulse duration -> equivalent static peak force; the least certain input (impact results scale with it).
    # 0.71 ms is calibrated on Rebot V3's GanchoUnidente (Hardox 500, proven to survive): at this pulse its body
    # is exactly at first yield, so impact SF >= 1 means "at least as robust as a weapon known to survive".
    # The user's team's explicit sims used a 2 ms total window, so the real pulse is between 0.71 and 2 ms.
    contact_time_ms: float = 0.71


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
    min_energy_j: float | None = None  # stored energy at top speed
    # Re-entrant (concave) corners are stress raisers: anything kinking more than this must be filleted.
    # Near-tangent kinks are tolerated (Rebot V3's worst is 21 deg, between a pocket's end arc and its wall).
    max_sharp_corner_deg: float = 30.0


class Style(Strict):
    """Looks are a hard requirement: weapons must read as aggressive and well made, never as optimizer blobs.
    These are enforced as constraints like any other; set a value to null/omit-by-override to relax it."""

    min_hook_deg: float | None = 5.0  # teeth lean into the spin and grab
    max_root_fillet_ratio: float | None = 0.4  # gullet fillet / tooth height: crisp gullets, not scoops
    min_pocket_sweep_deg: float | None = 15.0  # swept pockets must visibly sweep, not look like wheel spokes
    min_wall_frac: float | None = 0.08  # pocket walls >= this x tip radius, no thin-rim hoops (Rebot: 0.096)


class Analysis(Strict):
    fea: bool = True  # CalculiX plane-stress FEA; False = analytic Lame estimate only (fast, crude)
    mesh_mm: float | None = None  # max element size; default od/25 (optimizer), use --fine for od/75
    # ccx 2.17's multithreaded solve is nondeterministic (~1% stress scatter between identical runs), so 1.
    # The optimizer parallelizes across candidates instead.
    threads: int = 1
    # The optimizer's coarse mesh reads safety factors a little high vs the fine mesh the winner is verified
    # on (~1-6%), so during optimization SF limits are raised by this fraction.
    coarse_margin: float = 0.05


class Reference(Strict):
    """A proven weapon the design must not lose against (see eitri.reference): every load case, every region
    at least as strong, tooth at least as wide, at least as much energy."""

    spec: str  # path to the reference weapon's spec (relative to where eitri runs)


class Objective(Strict):
    maximize: Literal["energy", "moi", "effective_mass"] = "energy"
    minimize: Literal["mass"] | None = None  # set to make the weapon as light as possible (overrides maximize);
    # pair it with [constraints] min_energy_j so it still hits hard

    @property
    def name(self) -> str:
        return self.minimize or self.maximize

    @property
    def sign(self) -> float:
        """+1 maximize, -1 minimize: the optimizer maximizes sign * value."""
        return -1.0 if self.minimize else 1.0


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
    reference: Reference | None = None
    design: dict[str, Param]

    def free_params(self) -> dict[str, tuple[float, float]]:
        return {k: v for k, v in self.design.items() if isinstance(v, tuple)}

    def fixed_design(self) -> dict[str, float]:
        """Design with every range collapsed to its midpoint (what `eval` uses)."""
        return {k: (v[0] + v[1]) / 2 if isinstance(v, tuple) else v for k, v in self.design.items()}


def load(path: str | Path) -> Spec:
    return Spec.model_validate(tomllib.loads(Path(path).read_text()))

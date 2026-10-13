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

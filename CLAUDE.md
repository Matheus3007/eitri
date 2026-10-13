# Eitri — spinner weapon design suite

Code-first tool to design combat-robot spinner weapons from a text spec, evaluate them (mass, MOI, energy,
bite, spin-up, impact loads, FEA stress), optimize them, and export STEP.
Weapon types: `fang` (asymmetric single-tooth bar, the default and preferred style) and `disc`.
Drums/eggbeaters come later as new `Weapon` subclasses.
Plan/roadmap: `/home/tths/.claude/plans/we-are-going-to-vectorized-dragonfly.md`.
`projects/` is the local, git-ignored workspace for real weapons (never pushed); `examples/` holds
reference specs that are tracked.

## Hard requirements (from the user)
- **Looks are a requirement, not a nice-to-have**: every output must read as an aggressive, well-made
  weapon. Asymmetric single-tooth designs are preferred over symmetric round discs. Enforce looks through
  the parametrization and `[style]` constraints, and always render + look at a design before presenting it.
- Everything must be drivable from text/CLI and inspectable (JSON + PNG).

## Running
The repo lives on /mnt/c; the venv lives at `~/.venvs/eitri` (Windows-drive venvs are ~10x slower).
Always go through the wrapper, which sets `UV_PROJECT_ENVIRONMENT`:
```
./eitri.sh eitri new projects/<name> [--type fang|disc]
./eitri.sh eitri eval projects/<name>/spec.toml [--json] [--no-files] [--fine] [--no-fea]
./eitri.sh eitri optimize projects/<name>/spec.toml [--trials 200] [--workers 8] [--seed 0]
./eitri.sh eitri check projects/<name>/spec.toml
./eitri.sh eitri materials
./eitri.sh eitri show projects/<name>/spec.toml   # OCP CAD Viewer panel in VS Code (needs `--extra viewer`)
./eitri.sh pytest -q
```
For `uv add`/`uv sync`, export `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/eitri` first.
System deps (apt, need the user's sudo password): `libglu1-mesa` (gmsh), `calculix-ccx` (FEA).

## Design loop
1. Edit `projects/<name>/spec.toml` (units: mm, deg, kg, rpm, m/s, ms). `[design]` values are a number
   (fixed) or `[min, max]` (free for the optimizer; `eval` uses the midpoint).
2. `eval` writes `runs/<timestamp>-eval/`: `report.json`, `profile_<mat>.png`, `stress_<mat>.png`,
   `spinup_<mat>.png`, `weapon.step`, `spec_fixed.toml`, `fea/` (ccx decks + results).
   Read the PNGs; read `report.json` for numbers, `checks[]`, `failed[]`. Exit code 1 = a check failed.
3. `optimize` writes `runs/<timestamp>-optimize/`: `history.jsonl` (every candidate), `pareto.png`,
   `summary.json`, and `best/` (eval outputs of the winner re-verified on the fine mesh, including
   `best/spec_fixed.toml` to continue iterating with `eval`).

## Physics model (know its limits)
- Mass/MOI: exact from the B-rep. Drive: BLDC with current limit, ODE spin-up.
- Hit: tooth = point mass I/R^2 at tip speed vs opponent at rest, restitution e -> impulse J;
  equivalent static peak force = (pi/2) J / contact_time (half-sine). `contact_time_ms` = 2 ms default
  (the user's team used 2 ms windows in explicit impact sims) and is the least certain input.
- FEA (CalculiX plane stress, CPS6): `centrifugal` (3 tangential bore constraints) and `impact`
  (contact force on the outer `bite` of the face as consistent nodal loads, torque reacted by d'Alembert
  inertial loads, net force by the bolt holes). Stress split into zones: `contact` (tooth-tip denting,
  excluded), `support`, `body`. `yield_force_kn` = hit force at first body yield while spinning
  (independent of contact time); `impact_safety_factor` = yield force / estimated peak force.
- ccx 2.17 segfaults on *DLOAD pressure at some CPS6 edges touching sharp tips: use nodal CLOADs.
- Center of percussion is intentionally not an objective: a balanced rotor's axle always reacts the full
  impulse. Imbalance is a hard constraint; we report impulse / peak force / inward force fraction.

## Layout
- `eitri/spec/` pydantic spec models (TOML). `eitri/materials/materials.toml` material data.
- `eitri/geometry/weapon.py` abstract `Weapon`; `fang.py` (asym bar, counterweight solved for balance);
  `disc.py` (n-tooth disc, round or swept pockets); `common.py` 2D helpers.
- `eitri/physics/` `mass.py`, `drive.py`, `impact.py`, `fea/{mesh,ccx,analysis}.py`.
- `eitri/evaluate.py` spec + design -> metrics + checks (shared by CLI and optimizer). Invalid parameter
  combos return `{"ok": False, "error": ...}` instead of raising.
- `eitri/optimize/` optuna TPE, ask/tell, process pool (gmsh is not thread-safe).
- `eitri/report/render.py` PNG renders.

## Conventions
- Spin axis Z, part centered at z=0, default spin CCW seen from +Z; impact faces lead the rotation;
  the (first) tooth tip sits at angle 0.
- Positive `hook_deg` = tooth tip leads its root (grabs). `ratio` = motor turns per weapon turn.
- Build123d faces must be oriented +Z before booleans (`common.facing_up`), or unions don't merge.
- Every physics formula gets a closed-form test (`tests/test_physics.py`, `tests/test_fea.py`); every
  geometry parameter gets a test (`tests/test_geometry.py`, `tests/test_fang.py`).

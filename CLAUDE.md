# Eitri — spinner weapon design suite

Code-first tool to design combat-robot spinner weapons from a text spec, evaluate them (mass, MOI, energy,
bite, spin-up, impact loads, FEA stress), optimize them, and export STEP.
Weapon types: `hook` (asymmetric single-tooth, modelled on the user's Rebot V3 weapon; the default and
preferred style), `fang` (older asymmetric bar), `disc`, and `step` (evaluate an existing STEP part).
Drums/eggbeaters come later as new `Weapon` subclasses.
Plan/roadmap: `/home/tths/.claude/plans/we-are-going-to-vectorized-dragonfly.md`.
`projects/` is the local, git-ignored workspace for real weapons (never pushed); `examples/` holds
reference specs that are tracked.

## Hard requirements (from the user)
- **Looks are a requirement, not a nice-to-have**: every output must read as an aggressive, well-made
  weapon. Asymmetric single-tooth designs are preferred over symmetric round discs. Enforce looks through
  the parametrization and `[style]` constraints, and always render + look at a design before presenting it.
  Reference look: Rebot V3's `GanchoUnidente` (in the user's CAD, see `projects/rebot/spec.toml`): one
  continuous S/hook outline, single hooked tooth, comma counterweight with a sharp point, pockets with
  constant walls that follow the outline. Thin-rim "hoop" counterweights are NOT acceptable.
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
  equivalent static peak force = (pi/2) J / contact_time (half-sine). `contact_time_ms` = 0.71 ms default,
  calibrated on Rebot V3 (Hardox 500, 6S 880/sqrt3 kV, 1.5:1, survives): at 0.71 ms its body is exactly at
  first yield, so impact SF >= 1 means "at least as robust as a proven weapon". (The user's team's explicit
  sims used a 2 ms total window, so the true pulse is between 0.71 and 2 ms.)
- Hook geometry: outline drawn first inside the single-tooth bite envelope r(phi) = R - bite*phi/360 (the
  counterweight can never be a second tooth), sized by `cw_size_frac`, then balanced by Newton iteration on
  the counterweight pocket's span + center (pockets = outline inset by a wall, masked to sectors).
- FEA (CalculiX plane stress, CPS6): `centrifugal` (3 tangential bore constraints) and `impact`
  (contact force on the outer `bite` of the face as consistent nodal loads, torque reacted by d'Alembert
  inertial loads, net force by the bolt holes). Stress split into zones: `contact` (tooth-tip denting,
  excluded), `support`, `body`. `yield_force_kn` = hit force at first body yield while spinning
  (independent of contact time); `impact_safety_factor` = yield force / estimated peak force.
- ccx 2.17 segfaults on *DLOAD pressure at some CPS6 edges touching sharp tips: use nodal CLOADs.
- Mirrored (CW) faces mesh clockwise; the mesher flips elements to CCW (ccx: "nonpositive jacobian").
- OCC offset curves are silently dropped by the STEP writer: anything built with `bd.offset` goes through
  `common.to_nurbs`, and every export is verified by re-import (`step_export` in report.json).
- build123d `fillet()` finds its target via the vertices' topo_parent: rebuild a face from its wires before
  filleting a boolean result, or it fillets the boolean's operand.
- Center of percussion is intentionally not an objective: a balanced rotor's axle always reacts the full
  impulse. Imbalance is a hard constraint; we report impulse / peak force / inward force fraction.

## Layout
- `eitri/spec/` pydantic spec models (TOML). `eitri/materials/materials.toml` material data.
- `eitri/geometry/weapon.py` abstract `Weapon`; `hook.py` (asym, bite envelope + pocket balance);
  `fang.py` (asym bar, counterweight arm solved for balance); `disc.py` (n-tooth disc, round or swept
  pockets); `step_part.py` (imported STEP: auto-detects tip, impact face, hook, bolts); `common.py` helpers.
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

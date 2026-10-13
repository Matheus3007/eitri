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
- **Manufacturable, like the real part**: every re-entrant (concave) corner is filleted -- sharp inside corners
  are crack starters (`sharp_corners` check, all weapon types); only convex points (tip, counterweight point)
  may be sharp. The hub carries the pulley lock: 29 mm cutout with a 10.5 mm boss around each of the 7 bolts,
  R1 fillets (Rebot's pattern, `lock_*` params). When copying a reference design, compare *every* feature
  (hub pattern, fillets), not just the outline.
- Everything must be drivable from text/CLI and inspectable (JSON + PNG).

## Running
The repo lives on /mnt/c; the venv lives at `~/.venvs/eitri` (Windows-drive venvs are ~10x slower).
Always go through the wrapper, which sets `UV_PROJECT_ENVIRONMENT`:
```
./eitri.sh eitri new projects/<name> [--type fang|disc]
./eitri.sh eitri eval projects/<name>/spec.toml [--json] [--no-files] [--fine] [--no-fea]
./eitri.sh eitri optimize projects/<name>/spec.toml [--trials 200] [--workers 8] [--seed 0]
./eitri.sh eitri refine projects/<name>/spec.toml --start <spec_fixed.toml | history.jsonl#<trial>>
./eitri.sh eitri compare projects/a/spec.toml [projects/b/spec.toml ...] [--ref projects/rebot/spec.toml]
./eitri.sh eitri topo projects/<name>/spec.toml [--no-eval]   # needs [topology]; one design per MOI target
./eitri.sh eitri faces projects/<name>/spec.toml               # numbered boundary edges + keep regions
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
3. `optimize` writes `runs/<timestamp>-optimize/`: `history.jsonl` (every candidate, with its `phase`),
   `pareto.png`, `summary.json`, and `best/` (eval outputs of the winner re-verified on the fine mesh, including
   `best/spec_fixed.toml` to continue iterating with `eval`).
4. With a `[reference]`, every eval / optimize best / `compare` also writes `compare_sidebyside.png` (filled
   silhouettes, same scale, canonical frame), `compare_overlay.png` and `compare.md` (every quantity side by
   side, anything worse than the reference flagged **WORSE**). This is the default review format the user wants.
- Long runs (optimize/refine in the background): run them from a frozen copy of the package, or don't edit
  `eitri/` while they run -- worker processes spawn lazily and pick up half-edited code (a run died that way).

## Physics model (know its limits)
- Mass/MOI: exact from the B-rep. Drive: BLDC with current limit, ODE spin-up.
- Hit: tooth = point mass I/R^2 at tip speed vs opponent at rest, restitution e -> impulse J;
  equivalent static peak force = (pi/2) J / contact_time (half-sine). `contact_time_ms` = 0.67 ms default,
  calibrated on Rebot V3 (Hardox 500, 6S 880/sqrt3 kV, 1.5:1, survives): at 0.67 ms its body is exactly at
  first yield, so impact SF >= 1 means "at least as robust as a proven weapon". (The user's team's explicit
  sims used a 2 ms total window, so the true pulse is between 0.67 and 2 ms. It was 0.71 ms with the older
  fully-fixed bolt holes; old project specs still say 0.71.)
- Hub (`[analysis] hub_support`, default "auto"): the hit's net force is reacted by rigid frictionless parts
  bearing on the hole walls that face the push (linear: nodes held along the wall normal via *EQUATION) --
  "lock" (the pulley's lobed boss in the center cutout; bolts in clearance holes only clamp), "bolts", "auto"
  (lock if the weapon has one). Hole stress is judged (mesh refined around holes) except a 1.5 mm layer at the
  rigid contact; reactions (RF minus applied loads) give `stress.joint` load shares and, for "bolts", the
  `bolt_shear` check (`[joint]`). "fixed" = the old fully-fixed holes. On Rebot: lock 22.6 kN vs fixed 21.4,
  same first-yield spot (hub neck). Which path really carries the hit depends on clearances: lock by design.
- Robot side (`report.robot`): gyro couple I*omega*tilt rate (`[context] tilt_rate_dps`), optional `[shaft]`
  bending + bearing loads (`shaft_bending` check), and a damage-tolerance number: K_IC from Charpy
  (Roberts-Newton lower bound) and the critical edge-crack depth at the hot spot under the estimated peak hit.
- Hook geometry: outline drawn first inside the single-tooth bite envelope r(phi) = R - bite*phi/360 (the
  counterweight can never be a second tooth), sized by `cw_size_frac`. Balance: the lobe is hollowed into a
  constant-wall shell; the wall is solved (regula falsi, >= `wall_mm`) to remove the excess moment, and the
  lobe is turned within the envelope (`lobe_turn_deg`, secant, <= `max_lobe_turn_deg`) to cancel the sideways
  remainder. Pockets are rounded by morphological opening (shrink/grow by `pocket_fillet_mm`), which is
  continuous in the parameters; corner-by-corner fillets are not (they flip as a cut sweeps past a corner)
  and broke the Newton. `BalanceError` messages say which way the outline is off (tooth-heavy / too heavy /
  lobe hits a waist). ~2 s per build. The arm slot has its own `arm_fillet_mm`; if a requested slot can't
  fit (walls + fillets wider than the arm) `style_arm_slot` fails -- without it the optimizer drifted into
  slot-less designs (thick arm walls + big fillets), with no energy gain to show for it.
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
- OCC integration: the default (fixed-order) `BRepGProp` misses B-spline faces by ~1e-4 relative (tens of
  mm^3 of first moment; booleans stop adding up), and 3D volume integration of B-spline side faces is off by
  ~9 g*mm of imbalance, while its epsilon variant returns wrong volumes. So: `common.face_props` (adaptive,
  eps 1e-9, checked against a 320k-point polygon) everywhere, and mass properties of plates come from the 2D
  profile extruded analytically (`Weapon.plate()`; imported parts only if they are exact prisms). STEP
  export is verified by the re-imported top-face area (1e-6), not by volume.
- OCC 2D offsets fail on B-splines with an internal C0 kink and on some open-curve distances: use
  `common.offset_pieces` (splits at kinks, returns all pieces or []) on closed faces.
- CalculiX 2.17 multithreaded runs are nondeterministic (~1% stress scatter between identical runs):
  `[analysis] threads = 1` (default); parallelism comes from evaluating candidates in parallel.
- Reference comparison (`[reference] spec = "projects/rebot/spec.toml"`, eitri/reference.py): the design must
  not lose against a proven weapon. A replacement gets the reference's mass budget (`match_mass`, default true:
  the user rejected a 399 g design meant to replace the 374 g Rebot). Cheap checks (energy, mass, tooth width)
  run on every candidate; the load cases only on candidates passing everything else (or `full_reference`).
  Same code/mesh on both, four load cases -- in-plane hit on the bite, on the
  outer 3 mm (tooth-to-tooth stand-in) and 8 mm of the face, and a 3D out-of-plane hit on the flat of the tooth
  (C3D10, solid3d.py, validated vs cantilever theory) -- each split into regions tip/tooth/arm/hub; every region
  must hold what the reference's same region holds (up to 1.25x the reference's first yield), plus tooth width
  at every depth and energy. Reference metrics are cached in `.eitri_reference.json` next to its spec (key:
  spec text + `METRICS_VERSION` -- bump it whenever the FEA model changes); the load cases cost ~20 s and
  ~650 MB per candidate (6 workers on 7 GB).
- Objectives: `[objective] maximize = energy|moi|effective_mass`, or `minimize = "mass"` (lightest weapon;
  pair with `[constraints] min_energy_j`). The optimizer maximizes `objective.score` = sign x value.
- Optimizer: global TPE, then a polish phase (optimize/local.py): Newton repair of the best near-misses --
  finite-difference Jacobian of every check margin (all load cases computed at every point), smallest step
  that fixes them (slack QP, trust region, Broyden updates) -- or an SQP climb of the best feasible design,
  then CMA-ES within +-`radius` of each range. `refine` runs the polish phase from any design. SF limits get
  `[analysis] coarse_margin` (5%) on the coarse mesh; at the end the top `--verify-top` candidates are re-run
  on the fine mesh and the best that still passes wins.
- Balance is always solved, never tuned: every generator balances itself by Newton/secant to < 0.5 mm^3 of
  first moment, and the default `max_imbalance_gmm` is 0.1 (CG on the axis).
- Topology (`[topology]`, eitri/topo/): domain = a base weapon's outline (or, `envelope = true`, the whole
  single-tooth bite envelope) with its hub holes and pockets filled; `keep` selectors (hub, rim, tooth, holes,
  edge:I:W from `eitri faces`, circle, box) stay solid. SIMP on linear triangles (own numpy/scipy FE, fem.py)
  minimizing the p-norm von Mises of the in-plane hits (bite, 3, 8 mm; inertia relief with alpha = T/I(x)) plus
  the out-of-plane hit (DKT plate bending, plate.py, validated vs cantilever), sqrt(x) stress relaxation,
  density filter (`rmin_mm` = smallest member), Heaviside projection (beta 1 -> 32: grey density is cheap
  mass that carries nothing). Mass, MOI and both first moments are exact linear constraints on the projected
  densities, re-linearized every iteration and enforced by a semismooth-Newton projection. Adjoint gradients
  are checked against finite differences (tests). Compliance objective exists but thins hub necks (blind to
  stress concentration). Extraction (extract.py): 0.5 iso-contour -> smoothed rings; voids reaching the
  outline run past it (clean booleans; the outline stays exact where material reaches it).
- `topo` weapon (geometry/topo.py): pockets offset/clipped/opened on shapely polygons (OCC 2D offsets of
  offset-built boundaries silently drop pieces), min web between pockets, one periodic spline per pocket;
  Newton on pocket offsets (common offset for the mass + two pockets' offsets, or a pocket turn when there are
  only two) holds mass and puts the CG on the axis; islands are dropped (largest face kept).
- Machining (`[machining]`, geometry/machining.py): blind pockets (depth, 1 or 2 sides), 45 deg face bevel
  (an explicit wedge: OCC's chamfer follows tangent edges around the root fillet), counterbores. Mass = 2D
  plate + exact signed parts; the features' first moment goes into the hook/topo Newton (`pocket_depth_frac`,
  `extra_*`; estimated, then once more exact); 2D FEA gets per-element thickness sections; 3D/STEP use the
  real solid. OCC's 3D volume of plates with B-spline pocket walls is ~60 mm^3 off by itself: never use it.
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

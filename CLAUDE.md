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
./eitri.sh eitri impact projects/<name>/spec.toml [--case bite|tip3|all] [--no-ref]   # explicit 3D hit
./eitri.sh eitri ui [--port 8765] [--no-reload]   # web app: runs, compare, simulations, STEP part, spec, jobs (hot reload)
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
  the out-of-plane hit (DKT plate bending, plate.py, validated vs cantilever), each judged spinning like the
  reference cases (von Mises of the density-following centrifugal stress + the hit at the reference's first-
  yield force; without it pockets crowd the hub and the 3D flat hit fails there), sqrt(x) stress relaxation,
  density filter (`rmin_mm` = smallest member), Heaviside projection (beta 1 -> 32: grey density is cheap
  mass that carries nothing). Mass, MOI and both first moments are exact linear constraints on the projected
  densities, re-linearized every iteration and enforced by a semismooth-Newton projection. Adjoint gradients
  are checked against finite differences (tests). Compliance objective exists but thins hub necks (blind to
  stress concentration). Extraction (extract.py): 0.5 iso-contour -> smoothed rings; voids reaching the
  outline run past it (clean booleans; the outline stays exact where material reaches it).
  Findings on Rebot (2026-10-13): inside Rebot's own outline at its mass, topology matches Rebot in-plane and
  gains ~3-4% energy at MOI x1.04, the flat hit at the hub being the binding case; in the free bite envelope
  it converges to a straight bar with a compact counterweight lump -- structurally efficient, not the user's
  look, so envelope mode is exploratory.
- `topo` weapon (geometry/topo.py): pockets offset/clipped/opened on shapely polygons (OCC 2D offsets of
  offset-built boundaries silently drop pieces), min web between pockets, one periodic spline per pocket;
  Newton on pocket offsets (common offset for the mass + two pockets' offsets, or a pocket turn when there are
  only two) holds mass and puts the CG on the axis; islands are dropped (largest face kept).
- Machining (`[machining]`, geometry/machining.py): blind pockets (depth, 1 or 2 sides), 45 deg face bevel
  (an explicit wedge: OCC's chamfer follows tangent edges around the root fillet), counterbores. Mass = 2D
  plate + exact signed parts; the features' first moment goes into the hook/topo Newton (`pocket_depth_frac`,
  `extra_*`; estimated, then once more exact); 2D FEA gets per-element thickness sections; 3D/STEP use the
  real solid. OCC's 3D volume of plates with B-spline pocket walls is ~60 mm^3 off by itself: never use it.
- Explicit impact (eitri/physics/explicit/, `eitri impact`): OpenCourant -- the community fork of OpenRadioss
  (Siemens shut OpenRadioss down in 2026; AGPL v3, linux64 zip extracted to ~/opt/OpenCourant, run only through
  `tools/opencourant.sh`, which the user allowed in .claude/settings.json; the sandbox blocks running
  downloaded binaries any other way). Block-format decks written by radioss.py (formats read from
  ~/opt/OpenCourant/hm_cfg_files/config/CFG/radiossYYYY/*.cfg: take the newest FORMAT block -- `ls` sorts
  radioss110 after radioss2025). Units mm/ms/kg -> kN, GPa, J, m/s. Model: weapon tet10 (gmsh, ccx node order
  = Radioss order), Johnson-Cook plasticity fitted from yield/UTS/elongation (materials.johnson_cook(),
  Considere at UTS) and JC damage shaped like 4340's triaxiality curve, scaled to `fracture_strain` in tension
  (a constant fracture strain erodes merely-crushed tooth material); rigid hub (RBODY on the hole walls) free
  about Z with the drive inertia -- it needs /INIVEL/ROT on its master node besides /INIVEL/AXIS on the nodes;
  opponent block of the opponent's mass, deformable skin (weapon's own steel by default) on a rigid backing,
  closing at the closing speed, engaging the bite or the outer 3 mm; TYPE7 contact; mass scaling floor 30 ns
  (tiny tets at the R1 lock fillets set ~5 ns; added mass ~0.2 %). Outputs: contact pulse (peak, duration,
  impulse, equivalent half-sine), energies, plastic strain/failed elements per region (tip excluded = denting,
  2 mm hub layer excluded = rigid-boundary artifact), every frame as VTK + .pvd under vtk/ (for the future
  visualizer). ~15 min per 1 ms run with 5 threads. Radioss writes interface forces to the time history as
  *cumulative impulses* (the force is their derivative); "CONTACT ENERGY" is a global column, not the interface.
  Rebot bite hit vs a hardened-steel opponent face: impulse 10.8 N s (momentum model 9.7), pulse ~0.3 ms with a
  ~90 kN peak (equivalent half-sine 0.19 ms -- the static 0.67 ms calibration is a stand-in, softer armor
  lengthens it), the weapon loses ~290 J; Rebot's peak hub plastic strain sits at the static model's first-yield
  spot (hub neck, ~(8, 30) mm): both models agree on where it gives.
- Explicit-hit gate: `optimize`/`refine`/`topo --impact-gate N` (default 1) run both explicit hits (bite, tip3,
  steel opponent) on up to N fine-mesh survivors; a design fails if any region (tooth/arm/hub) has > 2 % more
  plastic strain than the reference or more failed elements outside the tip. The reference's runs are cached in
  `.eitri_reference_impact/` next to its spec (key: spec text, case, setup, IMPACT_VERSION). Skipped (with a
  message) when OpenCourant isn't installed.
- TPU opponent (`eitri impact --opponent tpu`, physics/explicit/tpu.py): a 3D-printed TPU 95A wall -- solid skin
  (one-term Ogden = neo-Hookean, nu 0.45, + Prony series) over a homogenized 15 % gyroid core (LAW70
  rate-dependent foam), torn at 4.5-5.5 strain; the wall is held by the robot's mass along its far edges (a
  rigid back face made a cut-through tooth hit an infinitely stiff plate). What it took to make it run (each
  one killed runs with "TOTAL MASS ERROR LIMIT" / NaN / negative volume): TYPE25 contact, weapon surface vs
  /SURF/PART/ALL of the TPU (TYPE7 node-to-surface spiked at the sharp tooth corner); TPU bricks Isolid 24 +
  Icontrol 1 (fully integrated ones inverted); /DT/BRICK/DEL 10 ns for the TPU bricks (it ignores tet10) on top
  of the weapon's nodal mass scaling; near-cubic TPU elements; nu 0.45 (0.48+ goes unstable past ~80 %
  compression). Engine /DT cards are fixed-column (20+1+20): a free-format line is misread and deletes
  healthy elements. Single-element checks (projects/tpu_check): skin E ~27 MPa, core plateau 0.06 MPa; at
  5000/s ~1.8x / 2x.
  Trust: order of magnitude of the pulse and design-vs-design ranking; validation data listed in tpu.py.
- UI (eitri/ui/, `eitri ui`): stdlib HTTP server over the project files + a no-build ES-module front end
  (static/: app.js shell, overview.js, stage.js = three.js viewport + telemetry timeline + HUD, ui.js = Select and
  editable Combo dropdowns, styles.css tokens). Quality bar (user, 2026-10-18): every UI pass ends with "would
  SpaceX / Ferrari / Red Bull / Mercedes present this as theirs in an F1 engineering suite?" -- look at screenshots
  and iterate until yes; no off-the-shelf widgets (no native video bar/selects for values).
  Tabs: Overview (verdict, KPIs with delta vs the reference's latest eval, checks grouped with margin bars, rotating
  part preview, simulations, run-type sections, figures, files), Compare (two panes, each any run x any view --
  part / simulation / figure -- linked camera and time; A-vs-B table + B's compare.md), Simulation, Part (STEP
  tessellated server-side by ui/part.py: faces unshared for sharp creases, B-rep edges, face ids for hover info,
  section cut; cached in <run>/anim/part_<stem>), Spec (TOML highlighting), Jobs. Visual language: engineering
  drawing x race telemetry -- Barlow / Barlow Condensed / JetBrains Mono, a drawing title block per run, numbered
  sections, KPI delta bars vs the reference, a run-log timeline in the sidebar, figures grouped into categories
  (line plots inked onto the dark sheet with a CSS invert, color maps kept on paper), direction-aware tab slides.
  3D navigation is SolidWorks-like (static/controls.js, not OrbitControls, which felt inverted to the user):
  drag rotates the part with the cursor (screen-space trackball), right/Ctrl drag pans, the wheel zooms about the
  cursor with SolidWorks' direction (forward = out), F / double-click fit, Ctrl+1/3/5/7 views, all animated.
  Hot reload: `eitri ui` runs ui/dev.py, which restarts the server child when any eitri/**/*.py changes; the page
  holds /api/events (SSE): a new boot id reloads the page (state is in the hash), CSS changes swap in place, JS/HTML
  changes reload, and new/changed runs under projects/ (directory-mtime scan every 3 s) refresh the sidebar and
  toast. `--no-reload` serves directly. Jobs are detached shells writing `<log>.rc` and listed in
  projects/.eitri_jobs.json, so they survive restarts; a job runs on the selected run's spec_fixed.toml when it
  has one (a topology project's own spec has no single design -- evaluate one of its designs).
  Every simulation becomes the same animation bundle (ui/anim.py: manifest.json, mesh.bin, frames.bin, vis.bin),
  built on first view and cached in the run: static FEA (hit force ramped to first yield on the spinning stress),
  explicit hits (VTK series, cells eroded after t = 0 hidden; cells already "eroded" at t = 0 are elements Radioss deactivated inside a rigid body -- hub wall layer, opponent backing -- and are drawn), topology (density_history.npz). /api/timeline adds solver steps
  per frame (explicit: cycles interpolated from the engine .out; FEA: load increments; topo: iterations) and time
  curves (explicit: contact force from the cumulative-impulse TH, energies). The animation list order is stable
  (converting one doesn't move it). Colormaps are sRGB; vertex colors must be converted to linear (or every map
  washes out). Deep links `#project=..&run=..&tab=simulation&sim=N&frame=K`. The user runs IDM (Internet
  Download Manager): it hooks Chrome, grabs anything that looks like a download (".bin" URLs, octet-stream), pops a
  window on their PC and hands fetch() an empty 204. So arrays go through /api/data?dir=&part= as text/plain; never
  serve app data as a download type, and never drive the user's Windows Chrome (it's their desktop). Check pages
  with `./eitri.sh python tools/ui_shot.py out.png "project=..&run=..&tab=simulation&sim=0&frame=30" [--scroll SEL]
  [--click SEL --after S] [--size 1920x1080]`: Playwright's headless Chromium inside WSL (dev dependency; its system
  libs nss/nspr/gbm/asound unpacked without sudo via `apt-get download` + `dpkg -x` into ~/opt/chromium-libs/root).
  Never `pkill -f` the server (it matches your own shell); find its PID with `ss -ltnp`.
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

# Eitri

AI-drivable design and simulation suite for combat robot spinner weapons. Give it a text spec
(size, mass budget, motor/gearing, material, style rules) and it generates, simulates and optimizes the
weapon, and exports a STEP ready for CAD or waterjet.

- Weapon types: `fang` (asymmetric single-tooth bar, counterweight solved for balance) and `disc`.
- Physics: exact mass/MOI from the B-rep, BLDC spin-up, impact impulse/bite, CalculiX plane-stress FEA
  (spinning + hit), constrained optimization (optuna) in parallel.
- Every command is non-interactive and writes JSON + PNG, so Claude (or you) can loop on it.

```
./eitri.sh eitri new projects/myweapon          # template spec
./eitri.sh eitri eval projects/myweapon/spec.toml
./eitri.sh eitri optimize projects/myweapon/spec.toml --trials 300
./eitri.sh pytest -q
```

Requires `uv`, plus `calculix-ccx` and `libglu1-mesa` from apt. See `CLAUDE.md` for the full workflow and
physics assumptions, and `examples/` for reference specs.

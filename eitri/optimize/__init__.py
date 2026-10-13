"""Constrained design optimization over the spec's free parameters ([min, max] ranges in [design]).

Optuna TPE with constraints, driven ask/tell from the main process; each candidate is a full `evaluate`
(geometry + mass + drive + FEA) in a worker process (gmsh is not thread-safe, ccx is single-threaded per job).
The midpoint design is always evaluated first as the baseline. Every trial is appended to history.jsonl.
"""

import json
import math
import multiprocessing as mp
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable

import optuna

from eitri.evaluate import evaluate, public
from eitri.geometry.disc import DiscParams
from eitri.geometry.fang import FangParams
from eitri.spec import Spec

PARAM_CLASSES = {"disc": DiscParams, "fang": FangParams}


def int_params(weapon_type: str) -> set[str]:
    return {f.name for f in fields(PARAM_CLASSES[weapon_type]) if f.type is int}


def violations(result: dict) -> list[float]:
    """Optuna constraint values: <= 0 feasible; failing numeric checks give their relative miss."""
    if "error" in result:
        return [10.0]
    out = []
    for c in result["checks"]:
        if c["ok"]:
            out.append(0.0)
        elif c["value"] is not None and c["limit"] is not None:
            out.append(abs(c["value"] - c["limit"]) / max(abs(c["limit"]), 1e-9) + 1e-3)
        else:
            out.append(1.0)
    return out


def _worker(spec: Spec, design: dict, material: str) -> dict:
    try:
        return public(evaluate(spec, design=design, material=material))
    except Exception as e:  # never let one bad candidate kill the run
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "design": design, "material": material}


def _record(number: int, result: dict) -> dict:
    rec = {"trial": number, "ok": result.get("ok", False), "material": result.get("material"),
           "design": result.get("design")}
    if "error" in result:
        rec["error"] = result["error"]
        return rec
    s = result["stress"]
    rec |= {
        "objective": result["objective"]["value"],
        "failed": result["failed"],
        "mass_kg": result["mass"]["mass_kg"],
        "moi_kgm2": result["mass"]["moi_kgm2"],
        "energy_j": result["impact"]["energy_j"],
        "bite_mm": result["impact"]["bite_mm"],
        "sf_spinning": s.get("safety_factor_yield"),
        "sf_impact": s.get("impact_safety_factor"),
        "yield_force_kn": s.get("yield_force_kn"),
        "imbalance_gmm": result["mass"]["static_imbalance_gmm"],
        "spinup_s_90": result["drive"]["spinup_s_90"],
    }
    return rec


@dataclass
class OptimizeResult:
    history: list[dict]
    best: dict | None  # best feasible record, else the least-violating one
    best_feasible: bool
    baseline: dict


def optimize(spec: Spec, trials: int, workers: int, seed: int, history_path: Path,
             log: Callable[[str], None] = print) -> OptimizeResult:
    free = spec.free_params()
    mats = spec.material.names()
    if not free and len(mats) == 1:
        raise ValueError("nothing to optimize: give some [design] values as [min, max] ranges")
    ints = int_params(spec.weapon.type)
    fixed = spec.fixed_design()
    spec = spec.model_copy(deep=True)
    spec.analysis.threads = 1

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True,
                                         n_startup_trials=max(8, min(25, trials // 5)))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    base = {k: (round(fixed[k]) if k in ints else fixed[k]) for k in free}
    if len(mats) > 1:
        base["material"] = mats[0]
    study.enqueue_trial(base)

    def suggest(trial: optuna.Trial) -> tuple[dict, str]:
        design = dict(fixed)
        for name, (lo, hi) in free.items():
            design[name] = (trial.suggest_int(name, round(lo), round(hi)) if name in ints
                            else trial.suggest_float(name, lo, hi))
        mat = trial.suggest_categorical("material", mats) if len(mats) > 1 else mats[0]
        return design, mat

    history: list[dict] = []
    best_val = -math.inf
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool, history_path.open("w") as hist:
        pending, submitted = {}, 0
        while submitted < trials or pending:
            while len(pending) < workers and submitted < trials:
                trial = study.ask()
                design, mat = suggest(trial)
                pending[pool.submit(_worker, spec, design, mat)] = trial
                submitted += 1
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                trial = pending.pop(fut)
                result = fut.result()
                v = violations(result)
                trial.set_constraint("violation", sum(v))
                value = result["objective"]["value"] if "error" not in result else 0.0
                study.tell(trial, value)
                rec = _record(trial.number, result)
                rec["violation"] = sum(v)
                history.append(rec)
                hist.write(json.dumps(rec) + "\n")
                hist.flush()
                if rec["ok"] and value > best_val:
                    best_val = value
                    log(f"  trial {trial.number:4d}: new best {spec.objective.maximize} = {value:.4g} "
                        f"({rec['material']}, mass {rec['mass_kg']:.3f} kg, SF spin {rec['sf_spinning']:.2f}, "
                        f"SF impact {rec['sf_impact'] or float('nan'):.2f})")
                elif len(history) % 25 == 0:
                    n_ok = sum(h["ok"] for h in history)
                    log(f"  {len(history)}/{trials} done, {n_ok} feasible")

    baseline = next(h for h in history if h["trial"] == 0)
    feasible = [h for h in history if h["ok"]]
    if feasible:
        best = max(feasible, key=lambda h: h["objective"])
    else:
        valid = [h for h in history if "error" not in h] or history
        best = min(valid, key=lambda h: h["violation"])
    return OptimizeResult(history, best, bool(feasible), baseline)

"""Constrained design optimization over the spec's free parameters ([min, max] ranges in [design]).

Global phase: Optuna TPE with constraints, driven ask/tell from the main process; each candidate is a full
`evaluate` (geometry + mass + drive + FEA) in a worker process (gmsh is not thread-safe, ccx is single-threaded
per job). The midpoint design is always evaluated first as the baseline.
Polish phase (`local.py`): Newton repair of the best near-misses (or SQP climb of the best feasible design),
then CMA-ES around the winner. Every evaluation is appended to history.jsonl with its phase.
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
from eitri.reference import FORCE_RANKED, reference_metrics
from eitri.geometry.disc import DiscParams
from eitri.geometry.fang import FangParams
from eitri.geometry.hook import HookParams
from eitri.spec import Spec

PARAM_CLASSES = {"disc": DiscParams, "fang": FangParams, "hook": HookParams}


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


SF_CHECKS = ("safety_factor_spinning", "impact_safety_factor")


def apply_margin(result: dict, margin: float) -> dict:
    """Raise the safety-factor limits by `margin` (coarse mesh reads a few % optimistic vs the fine mesh the
    winner is verified on), re-judging ok/failed."""
    if "error" in result or margin <= 0:
        return result
    for c in result["checks"]:
        if c["name"] in SF_CHECKS + FORCE_RANKED and c["value"] is not None and c["limit"] is not None:
            c["limit"] *= 1 + margin
            c["ok"] = c["value"] >= c["limit"]
            c["detail"] = (c.get("detail") or "") + f" (limit +{margin:.0%} coarse-mesh margin)"
    result["failed"] = [c["name"] for c in result["checks"] if not c["ok"]]
    result["ok"] = not result["failed"]
    return result


def _worker(spec: Spec, design: dict, material: str, full_reference: bool = False) -> dict:
    """`full_reference`: run the reference load cases even when cheap checks fail (finite differences need
    every margin at every point)."""
    try:
        return apply_margin(public(evaluate(spec, design=design, material=material, full_reference=full_reference)),
                            spec.analysis.coarse_margin)
    except Exception as e:  # never let one bad candidate kill the run
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "design": design, "material": material}


def _record(number: int, result: dict, phase: str = "global") -> dict:
    rec = {"trial": number, "phase": phase, "ok": result.get("ok", False), "material": result.get("material"),
           "design": result.get("design")}
    if "error" in result:
        rec["error"] = result["error"]
        return rec
    s = result["stress"]
    rec |= {
        "objective": result["objective"]["value"],
        "score": result["objective"]["score"],
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


def warm_designs(history: Path, k: int) -> list[dict]:
    """The k best feasible designs of a previous run, then the least-violating ones, to seed a new run."""
    recs = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
    recs = [r for r in recs if "error" not in r]
    ok = sorted((r for r in recs if r["ok"]), key=lambda r: -r.get("score", r["objective"]))
    rest = sorted((r for r in recs if not r["ok"]), key=lambda r: r["violation"])
    return [r["design"] for r in (ok + rest)[:k]]


def optimize(spec: Spec, trials: int, workers: int, seed: int, history_path: Path,
             log: Callable[[str], None] = print, warm: list[dict] = (), polish: bool = True,
             local_trials: int | None = None, radius: float = 0.15) -> OptimizeResult:
    from eitri.optimize import local

    free = spec.free_params()
    mats = spec.material.names()
    if not free and len(mats) == 1:
        raise ValueError("nothing to optimize: give some [design] values as [min, max] ranges")
    ints = int_params(spec.weapon.type)
    fixed = spec.fixed_design()
    spec = spec.model_copy(deep=True)
    spec.analysis.threads = 1
    if spec.reference is not None:  # compute (and cache) the reference's profile once, before the workers start
        log(f"reference: {spec.reference.spec} (cached after the first run)")
        reference_metrics(spec.reference.spec, fine=False, hub_support=spec.analysis.hub_support)

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True,
                                         n_startup_trials=max(8, min(25, trials // 5)))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    base = {k: (round(fixed[k]) if k in ints else fixed[k]) for k in free}
    if len(mats) > 1:
        base["material"] = mats[0]
    study.enqueue_trial(base)
    for d in warm:  # earlier designs, clipped into this spec's ranges
        params = {k: (round(min(max(d[k], lo), hi)) if k in ints else min(max(d[k], lo), hi))
                  for k, (lo, hi) in free.items() if k in d}
        if len(mats) > 1:
            params["material"] = mats[0]
        study.enqueue_trial(params)
    if warm:
        log(f"warm start: {len(warm)} designs from a previous run")

    def suggest(trial: optuna.Trial) -> tuple[dict, str]:
        design = dict(fixed)
        for name, (lo, hi) in free.items():
            design[name] = (trial.suggest_int(name, round(lo), round(hi)) if name in ints
                            else trial.suggest_float(name, lo, hi))
        mat = trial.suggest_categorical("material", mats) if len(mats) > 1 else mats[0]
        return design, mat

    history: list[dict] = []
    full: dict[int, dict] = {}  # trial -> result (checks), for the polish phase
    best_val = -math.inf
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool, history_path.open("w") as hist:

        def record(result: dict, phase: str, number: int | None = None) -> dict:
            nonlocal best_val
            v = violations(result)
            rec = _record(len(history) if number is None else number, result, phase)
            rec["violation"] = sum(v)
            history.append(rec)
            full[rec["trial"]] = {k: result.get(k) for k in ("ok", "failed", "checks", "objective", "error",
                                                             "design", "material")}
            hist.write(json.dumps(rec) + "\n")
            hist.flush()
            value = result["objective"]["score"] if "error" not in result else -1e9
            if rec["ok"] and value > best_val:
                best_val = value
                log(f"  trial {rec['trial']:4d} ({phase}): new best {spec.objective.name} = {rec['objective']:.4g} "
                    f"({rec['material']}, mass {rec['mass_kg']:.3f} kg, SF spin {rec['sf_spinning']:.2f}, "
                    f"SF impact {rec['sf_impact'] or float('nan'):.2f})")
            elif len(history) % 25 == 0:
                log(f"  {len(history)} evaluated, {sum(h['ok'] for h in history)} feasible")
            return rec

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
                trial.set_constraint("violation", sum(violations(result)))
                study.tell(trial, result["objective"]["score"] if "error" not in result else -1e9)
                record(result, "global", trial.number)
        n_global = len(history)

        if polish:
            _polish(spec, history, full, pool, record, ints, workers, seed, log,
                    local_trials if local_trials is not None else max(workers, trials // 4), radius)
        log(f"evaluated {len(history)} designs ({n_global} global, {len(history) - n_global} local)")

    baseline = next(h for h in history if h["trial"] == 0)
    feasible = [h for h in history if h["ok"]]
    if feasible:
        best = max(feasible, key=lambda h: h["score"])
    else:
        valid = [h for h in history if "error" not in h] or history
        best = min(valid, key=lambda h: h["violation"])
    return OptimizeResult(history, best, bool(feasible), baseline)


def _polish(spec, history, full, pool, record, ints, workers, seed, log, local_trials, radius) -> None:
    """Newton repair of near-misses (or an SQP climb of the best feasible design), then CMA-ES around it."""
    from eitri.optimize import local

    valid = [h for h in history if "error" not in h]
    if not valid:
        return
    feasible = sorted((h for h in valid if h["ok"]), key=lambda h: -h["score"])
    starts = feasible[:1] or sorted(valid, key=lambda h: h["violation"])[:3]
    winner = None
    for h in starts:
        mat = h["material"]
        batch = local.Batch(pool, _worker, spec, mat, record)
        res = full[h["trial"]]
        space = local.Space.of(spec, res["design"], ints)
        if not space.names:
            return
        log(f"  polish: Newton {'climb' if h['ok'] else 'repair'} from trial {h['trial']} "
            f"({'feasible' if h['ok'] else 'fails ' + ','.join(h['failed'])})")
        lr = local.repair(space, res["design"], res, batch, log)
        if lr.ok:
            winner = (lr, space, batch)
            break
    if winner is None:
        log("  polish: no repair reached feasibility")
        return
    lr, space, batch = winner
    if local_trials > 0:
        log(f"  polish: CMA-ES, {local_trials} designs within +-{radius:.0%} of each range")
        local.refine(space, lr.design, lr.result, batch, workers, local_trials, radius, seed, log)

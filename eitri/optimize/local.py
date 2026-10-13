"""Local search around one design: Newton repair of near-misses, then CMA-ES refinement.

repair
    A near-miss fails a few checks by a few percent. Each check's margin (signed, relative to its limit) is
    linearized in the free parameters by finite differences (one evaluation per parameter, in parallel), and
    the smallest step that brings every near-active margin to +`goal` is solved as a small QP (SLSQP), inside
    the parameter ranges and a trust region. The step is evaluated; the Jacobian is updated with Broyden's
    rank-1 formula (rebuilt by finite differences when a step doesn't help). Once feasible, the same model
    takes SQP steps on the objective while it keeps every margin >= `goal` -- Newton's method, as used for the
    balance, applied to the whole check list.

refine
    CMA-ES (optuna) in a box around the design, `radius` x each parameter's range. Infeasible candidates are
    ranked below every feasible one (by violation), so the search stays on the feasible side.

Both work in normalized coordinates (each free parameter's [min, max] range -> [0, 1]); integer parameters
(bolt counts, on/off switches) stay at the seed's values.
"""

import math
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import optuna
from scipy.optimize import minimize

from eitri.spec import Spec

IGNORED = {"ref_profile"}  # bookkeeping checks (fail only because others fail first)


def margins(result: dict) -> dict[str, float]:
    """Signed relative margin of every numeric check (>= 0 passes). Non-numeric failures -> -1."""
    out = {}
    for c in result.get("checks", []):
        if c["name"] in IGNORED:
            continue
        v, lim = c["value"], c["limit"]
        if v is None or lim is None or not (math.isfinite(v) and math.isfinite(lim)):
            out[c["name"]] = 0.0 if c["ok"] else -1.0
            continue
        scale = max(abs(lim), 1e-9)
        # lower bound (value >= limit passes) unless the verdict says otherwise
        lower = (v >= lim) == c["ok"]
        out[c["name"]] = (v - lim) / scale if lower else (lim - v) / scale
    return out


@dataclass
class Space:
    """The free continuous parameters, normalized to [0, 1]."""
    names: list[str]
    lo: np.ndarray
    hi: np.ndarray
    base: dict  # every other design value (fixed + integers)

    @classmethod
    def of(cls, spec: Spec, design: dict, ints: set[str]) -> "Space":
        free = {k: v for k, v in spec.free_params().items() if k not in ints and v[1] > v[0]}
        names = sorted(free)
        return cls(names, np.array([free[k][0] for k in names], float), np.array([free[k][1] for k in names], float),
                   dict(design))

    def to_u(self, design: dict) -> np.ndarray:
        x = np.array([design[k] for k in self.names], float)
        return np.clip((x - self.lo) / (self.hi - self.lo), 0, 1)

    def to_design(self, u: np.ndarray) -> dict:
        x = self.lo + np.clip(u, 0, 1) * (self.hi - self.lo)
        return self.base | {k: float(v) for k, v in zip(self.names, x)}


@dataclass
class Batch:
    """Evaluate designs in parallel; every result is logged through `record`."""
    pool: ProcessPoolExecutor
    worker: Callable
    spec: Spec
    material: str
    record: Callable[[dict, str], dict]  # (result, phase) -> history record
    n_evals: int = 0

    def __call__(self, designs: list[dict], phase: str, full: bool = False) -> list[dict]:
        futs = [self.pool.submit(self.worker, self.spec, d, self.material, full) for d in designs]
        out = []
        for f in futs:
            r = f.result()
            self.n_evals += 1
            self.record(r, phase)
            out.append(r)
        return out


@dataclass
class LocalResult:
    design: dict
    result: dict
    ok: bool
    history: list[tuple[str, float, bool]] = field(default_factory=list)  # (phase, score, ok)


def _violation(r: dict) -> float:
    return sum(max(0.0, -m) for m in margins(r).values())


def _score(r: dict) -> float:
    return r["objective"]["score"] if "error" not in r else -math.inf


def repair(space: Space, design: dict, result: dict, batch: Batch, log: Callable[[str], None] = print,
           max_iter: int = 6, h: float = 0.04, trust: float = 0.12, goal: float = 0.005,
           climb: int = 2) -> LocalResult:
    """Newton/SQP on the check margins (see module doc). `climb`: objective steps after feasibility."""
    names = space.names
    u = space.to_u(design)
    if not result.get("ok"):  # re-run with every load case, so every margin is known at the start
        r_full = batch([space.to_design(u)], "repair", full=True)[0]
        result = r_full if "error" not in r_full else result
    r0 = result
    m0 = margins(r0)
    keys = sorted(m0)

    def vec(r):
        m = margins(r)
        return np.array([m.get(k, np.nan) for k in keys])

    def fd_jacobian(u, r):
        steps = np.where(u + h <= 1, h, -h)
        pts = [u + steps[i] * np.eye(len(u))[i] for i in range(len(u))]
        res = batch([space.to_design(p) for p in pts], "repair-fd", full=True)
        base_m, base_s = vec(r), _score(r)
        J = np.zeros((len(keys), len(u)))
        g = np.zeros(len(u))
        for i, ri in enumerate(res):
            if "error" in ri:
                continue  # leave this direction out of the model
            J[:, i] = np.nan_to_num((vec(ri) - base_m) / steps[i])  # a check missing at either point: no info
            g[i] = (_score(ri) - base_s) / steps[i]
        return J, g

    best = (u, r0)
    J, g = fd_jacobian(u, r0)
    feasible_steps, fails = 0, 0
    tr = trust
    for it in range(max_iter):
        m = np.nan_to_num(vec(r0), nan=1.0)  # unknown margins are left out of the model (act below)
        ok = bool(r0.get("ok"))
        # near-active margins are modelled; far ones are assumed to stay satisfied
        act = m < max(0.15, goal * 4)
        n, na = len(u), int(act.sum())
        # variables [step, slack]: m + J step + slack >= goal, slack >= 0, heavily penalized -- so a target out
        # of reach of the trust region still gives the step that gets closest (then the next iteration goes on)
        W = 1e4
        if ok:
            gn = g / (np.linalg.norm(g) + 1e-12)

            def fun(z):
                d, sl = z[:n], z[n:]
                return -(gn @ d) / tr + 0.5 * (d @ d) / tr**2 + W * (sl @ sl)

            def jac(z):
                d, sl = z[:n], z[n:]
                return np.concatenate([-gn / tr + d / tr**2, 2 * W * sl])
        else:
            def fun(z):
                d, sl = z[:n], z[n:]
                return 0.5 * (d @ d) + W * (sl @ sl)

            def jac(z):
                return np.concatenate([z[:n], 2 * W * z[n:]])
        Ja = J[act]
        cons = [{"type": "ineq", "fun": lambda z: m[act] + Ja @ z[:n] + z[n:] - goal,
                 "jac": lambda z: np.hstack([Ja, np.eye(na)])}]
        bounds = [(max(-ui, -tr), min(1 - ui, tr)) for ui in u] + [(0, None)] * na
        z0 = np.concatenate([np.zeros(n), np.maximum(0, goal - m[act])])
        sol = minimize(fun, z0, jac=jac, bounds=bounds, constraints=cons, method="SLSQP", options={"maxiter": 300})
        d = sol.x[:n]
        if np.linalg.norm(d) < 1e-4:
            log(f"    repair it {it}: no step left ({'feasible' if ok else 'stuck: ' + ','.join(r0['failed'])})")
            break
        u1 = np.clip(u + d, 0, 1)
        r1 = batch([space.to_design(u1)], "repair", full=True)[0]
        if "error" in r1:
            tr *= 0.5
            log(f"    repair it {it}: invalid step ({r1['error'][:80]}), trust -> {tr:.3f}")
            continue
        m1 = np.nan_to_num(vec(r1), nan=-1.0)
        worst0, worst1 = m[act].min(initial=0), m1[act].min(initial=0)
        # merit while infeasible: total shortfall over all margins (one check slightly worse is fine if the sum drops)
        improved = (r1["ok"] and (not ok or _score(r1) > _score(r0))) or (
            not ok and not r1["ok"] and _violation(r1) < _violation(r0) - 1e-4)
        log(f"    repair it {it}: |step| {np.linalg.norm(d):.3f} -> {'OK' if r1['ok'] else 'fail ' + ','.join(r1['failed'])}"
            f" score {_score(r1):.4g} (worst margin {worst0:+.3f} -> {worst1:+.3f}, shortfall "
            f"{_violation(r0):.3f} -> {_violation(r1):.3f})")
        # Broyden: make the model reproduce what the step actually did
        dm, ds = np.nan_to_num(vec(r1) - m), _score(r1) - _score(r0)
        J += np.outer(dm - J @ d, d) / (d @ d)
        g += (ds - g @ d) * d / (d @ d)
        if improved:
            u, r0 = u1, r1
            if r1["ok"] and (not best[1].get("ok") or _score(r1) > _score(best[1])):
                best = (u1, r1)
            if r1["ok"]:
                feasible_steps += 1
                if feasible_steps > climb:
                    break
            tr = min(trust, tr * 1.5)
        else:
            tr = 0.5 * min(tr, float(np.abs(d).max()))
            fails += 1
            if tr < 0.005:
                break
            if fails % 2 == 0:
                J, g = fd_jacobian(u, r0)  # the model keeps being wrong: rebuild it where we are
    u, r = best if best[1].get("ok") else (u, r0)
    return LocalResult(space.to_design(u), r, bool(r.get("ok")))


def refine(space: Space, design: dict, result: dict, batch: Batch, workers: int, trials: int,
           radius: float = 0.15, seed: int = 0, log: Callable[[str], None] = print) -> LocalResult:
    """CMA-ES in a box of +-radius (normalized) around `design`."""
    u0 = space.to_u(design)
    lo, hi = np.clip(u0 - radius, 0, 1), np.clip(u0 + radius, 0, 1)
    s0 = _score(result)
    ref = abs(s0) if math.isfinite(s0) and s0 != 0 else 1.0

    def value(r):  # feasible always above infeasible
        if "error" in r:
            return s0 - ref * 10
        return _score(r) if r["ok"] else s0 - ref * (0.5 + _violation(r))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = optuna.samplers.CmaEsSampler(x0={k: float(v) for k, v in zip(space.names, u0)},
                                               sigma0=radius / 2, seed=seed, warn_independent_sampling=False)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    best = (design, result) if result.get("ok") else (None, None)
    done = 0
    while done < trials:
        n = min(workers, trials - done)
        trs = [study.ask() for _ in range(n)]
        us = [np.array([t.suggest_float(k, lo[i], hi[i]) for i, k in enumerate(space.names)]) for t in trs]
        res = batch([space.to_design(u) for u in us], "refine")
        for t, u, r in zip(trs, us, res):
            study.tell(t, value(r))
            if r.get("ok") and (best[1] is None or _score(r) > _score(best[1])):
                best = (space.to_design(u), r)
                log(f"    refine: new best score {_score(r):.4g}")
        done += n
    if best[1] is None:
        return LocalResult(design, result, False)
    return LocalResult(best[0], best[1], True)

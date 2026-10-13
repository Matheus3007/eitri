"""Newton balance: put the CG on the spin axis by solving two geometric knobs.

Every generator must come out balanced (the user's rule: CG dead on the axis, solved, never tuned by hand).
The hook and fang have their own specialized solvers (monotone wall + lobe turn, club width + angle); anything
new -- topology pockets, keep/free designs, machined features -- uses this one: the residual is the profile's
first moment (mm^3, a 2-vector) as a function of two knobs (e.g. the offsets of two pockets on different sides),
solved by Newton with a finite-difference Jacobian, Broyden updates, step damping and bounds.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


class BalanceError(ValueError):
    pass


@dataclass
class BalanceResult:
    x: np.ndarray
    residual: np.ndarray
    evals: int

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.residual))


def newton(f: Callable[[np.ndarray], np.ndarray], x0, lo, hi, tol: float = 0.5, h: float = 0.05,
           max_iter: int = 30) -> BalanceResult:
    """Solve f(x) = 0 (x, f in R^n, square: e.g. first moment x/y, or x/y + mass with three knobs) within
    [lo, hi]. `h`: finite-difference step (units of x; scalar or per knob). Raises BalanceError if the knobs
    can't reach the target (residual stalls above 20 x tol)."""
    x = np.asarray(x0, float)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    hs = np.broadcast_to(np.asarray(h, float), x.shape)
    k = len(x)
    n = 0

    def F(z):
        nonlocal n
        n += 1
        return np.asarray(f(np.clip(z, lo, hi)), float)

    r = F(x)
    best = (np.linalg.norm(r), x.copy(), r.copy())

    def jac(x, r):
        J = np.zeros((len(r), k))
        for i in range(k):
            step = hs[i] if x[i] + hs[i] <= hi[i] else -hs[i]
            xi = x.copy()
            xi[i] += step
            J[:, i] = (F(xi) - r) / step
        return J

    J = jac(x, r)
    for _ in range(max_iter):
        if best[0] < tol:
            break
        try:
            d = -np.linalg.solve(J, r)
        except np.linalg.LinAlgError:
            d = -np.linalg.lstsq(J, r, rcond=None)[0]
        t = 1.0
        for _ in range(6):  # backtrack until the residual drops
            x1 = np.clip(x + t * d, lo, hi)
            r1 = F(x1)
            if np.linalg.norm(r1) < np.linalg.norm(r):
                break
            t *= 0.5
        else:
            J = jac(x, r)  # the model is off: rebuild it here
            continue
        s, y = x1 - x, r1 - r
        if s @ s > 0:
            J += np.outer(y - J @ s, s) / (s @ s)
        x, r = x1, r1
        if np.linalg.norm(r) < best[0]:
            best = (np.linalg.norm(r), x.copy(), r.copy())
    if best[0] > 20 * tol:
        raise BalanceError(f"balance knobs can't cancel the first moment (residual {best[0]:.1f} mm^3 at "
                           f"{np.round(best[1], 3).tolist()})")
    return BalanceResult(best[1], best[2], n)


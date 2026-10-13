"""The generic Newton balance solver."""

import numpy as np
import pytest

from eitri.geometry.balance import BalanceError, newton


def test_newton_solves_a_nonlinear_moment_balance():
    # two pockets of radius a, b at fixed positions remove first moment pi r^2 * position from a fixed excess
    excess = np.array([300.0, -120.0])
    p1, p2 = np.array([20.0, 5.0]), np.array([-3.0, -25.0])

    def f(x):
        return excess - np.pi * x[0] ** 2 * p1 / 20 - np.pi * x[1] ** 2 * p2 / 20

    res = newton(f, [1.0, 1.0], [0.0, 0.0], [20.0, 20.0], tol=1e-6)  # root at r = (10.16, 7.15)
    assert res.norm < 1e-6 and res.evals < 40


def test_newton_reports_unreachable_targets():
    with pytest.raises(BalanceError):
        newton(lambda x: np.array([100.0 + x[0], 50.0]), [0, 0], [0, 0], [1, 1])

import numpy as np
import pytest

from uengram.host.stepcal import innovation_inflation

LAM = 0.1


def _ar1_runs(rho: float, n_runs: int, length: int, d: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(n_runs):
        x = np.empty((length, d))
        x[0] = rng.normal(size=d)
        for t in range(1, length):  # unit marginal variance
            x[t] = rho * x[t - 1] + np.sqrt(1 - rho ** 2) * rng.normal(size=d)
        runs.append(x)
    return runs


def test_independent_innovations_need_no_correction():
    f, n = innovation_inflation(_ar1_runs(0.0, 40, 300, 4, seed=1), LAM)
    assert n == 40 * (300 - 19 + 1)
    assert np.all(np.abs(f - 1.0) < 0.15)


def test_correlated_innovations_are_measured_at_the_ewma_variance_inflation():
    # EWMA(lam) of a unit-variance AR(1): Var = lam/(2-lam) * (1+a)/(1-a), a = rho(1-lam)
    rho = 0.9
    a = rho * (1 - LAM)
    f, _ = innovation_inflation(_ar1_runs(rho, 60, 400, 3, seed=2), LAM)
    assert np.all(np.abs(f / ((1 + a) / (1 - a)) - 1.0) < 0.2)


def test_runs_shorter_than_the_ewma_memory_are_not_evidence():
    short = _ar1_runs(0.5, 50, 18, 2, seed=3)  # every run ends before the 19-innovation warm-up
    with pytest.raises(ValueError, match="Record longer"):
        innovation_inflation(short, LAM)

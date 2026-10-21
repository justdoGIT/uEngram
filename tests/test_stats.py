import math

import numpy as np
import pytest

from uengram import stats


def test_chi2_d2_exact():
    x = np.array([0.0, 1.0, 5.5, 40.0])
    np.testing.assert_allclose(stats.chi2_logsf(x, 2), -x / 2, atol=1e-12)


def test_chi2_odd_raises():
    with pytest.raises(ValueError):
        stats.chi2_logsf(1.0, 3)


def test_chi2_isf_table():
    assert abs(stats.chi2_isf(0.001, 16) - 39.252) < 1e-3
    assert abs(stats.chi2_isf(0.01, 16) - 32.000) < 1e-3


def test_surprisal_capped_and_monotone():
    assert stats.surprisal(0.0, 16) == 0.0
    assert stats.surprisal(1e6, 16) == 300.0
    assert stats.surprisal(60.0, 16) > stats.surprisal(40.0, 16)


def test_ew_update_matches_batch():
    rng = np.random.default_rng(0)
    x = rng.normal(2.0, 1.5, size=(200, 3))
    mu, S = np.zeros(3), np.zeros(3)
    for n, xi in enumerate(x, start=1):
        mu, S = stats.ew_update(mu, S, xi, 1.0 / n)
    np.testing.assert_allclose(mu, x.mean(0), atol=1e-10)
    np.testing.assert_allclose(S, x.var(0), atol=1e-10)


def test_merge_moments_matches_concat():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, size=(30, 4))
    b = rng.normal(3, 2, size=(70, 4))
    n, mu, var = stats.merge_moments(30, a.mean(0), a.var(0), 70, b.mean(0), b.var(0))
    c = np.vstack([a, b])
    assert n == 100
    np.testing.assert_allclose(mu, c.mean(0), atol=1e-10)
    np.testing.assert_allclose(var, c.var(0), atol=1e-10)


def test_bhattacharyya_zero_for_identical():
    mu, var = np.array([1.0, 2.0]), np.array([0.5, 2.0])
    assert abs(stats.bhattacharyya_diag(mu, var, mu, var)) < 1e-12
    pw = stats.bhattacharyya_pairwise(np.stack([mu, mu + 1]), np.stack([var, var]))
    assert np.isinf(pw[0, 0]) and pw[0, 1] > 0
    assert abs(pw[0, 1] - stats.bhattacharyya_diag(mu, var, mu + 1, var)) < 1e-12


def test_robust_noise_sigma_recovers():
    rng = np.random.default_rng(2)
    T = 5000
    level = np.repeat(rng.normal(0, 5, size=T // 50), 50)
    # 5% of steps carry a persistent level jump (one contaminated difference
    # each); isolated spikes would contaminate two differences apiece.
    jump = rng.random(T) < 0.05
    z = level + rng.normal(0, 0.5, size=T) + np.cumsum(jump * rng.normal(0, 10, size=T))
    s = stats.robust_noise_sigma(z[:, None], 1, None)
    assert abs(s[0] - 0.5) / 0.5 < 0.1


def test_robust_noise_sigma_groups_and_few_pairs():
    z = np.zeros((5, 1))
    with pytest.raises(ValueError):
        stats.robust_noise_sigma(z, 1, None)
    rng = np.random.default_rng(3)
    z = rng.normal(size=(200, 1))
    g = np.repeat(np.arange(2), 100)
    z[100:] += 1000.0  # level shift between groups must be ignored
    s = stats.robust_noise_sigma(z, 1, g)
    assert abs(s[0] - 1.0) < 0.2


def test_ward_cost():
    n = np.array([2, 3])
    mu = np.array([[0.0, 0.0], [2.0, 0.0]])
    c = stats.ward_cost_pairwise(n, mu, np.array([1.0, 4.0]))
    assert math.isclose(c[0, 1], 2 * 3 / 5 * 4.0, rel_tol=1e-9)
    assert np.isinf(c[0, 0])

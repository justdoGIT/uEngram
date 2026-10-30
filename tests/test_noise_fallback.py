import warnings

import numpy as np
import pytest

from uengram import stats
from uengram.encoders import DSPEncoder, Whitener, fit_projection
from uengram.noise_fallback import (
    soft_ewma_inflation, soft_noise_cov, soft_noise_sigma, worst_quality,
)


def _regime_switching(rng, T, d, regimes=6, dwell_p=1 / 3):
    """Unit noise around one of several discrete levels that changes with probability dwell_p per
    sample (about 3 samples per regime): the shape of CMAPSS FD004's operating conditions."""
    levels = rng.normal(0, 20.0, size=(regimes, d))
    idx = np.zeros(T, dtype=int)
    for t in range(1, T):
        idx[t] = rng.integers(regimes) if rng.random() < dwell_p else idx[t - 1]
    return levels[idx] + rng.normal(size=(T, d))


def test_healthy_data_gives_the_strict_answers_and_no_warning():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(600, 4))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sigma, sq = soft_noise_sigma(z, 1, None)
        cov, cq = soft_noise_cov(z, 1, None)
        infl, iq = soft_ewma_inflation(rng.normal(size=(4000, 4)), 0.1, None)
    np.testing.assert_array_equal(sigma, stats.robust_noise_sigma(z, 1, None))
    np.testing.assert_array_equal(cov, stats.robust_noise_cov(z, 1, None))
    assert (sq, cq, iq) == ("robust", "robust", "exact")
    assert infl.shape == (4,) and np.isfinite(infl).all()


def test_noise_sigma_ignores_groups_that_leave_too_few_pairs():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(300, 2))
    singletons = np.arange(300)  # every window its own group: no within-group pairs at all
    with pytest.raises(stats.NoiseEstimateError):
        stats.robust_noise_sigma(z, 1, singletons)
    with pytest.warns(RuntimeWarning, match="ungrouped"):
        sigma, quality = soft_noise_sigma(z, 1, singletons)
    assert quality == "ungrouped" and np.all(np.abs(sigma - 1.0) < 0.2)


def test_noise_sigma_still_refuses_when_there_is_no_data_to_estimate_from():
    with pytest.raises(ValueError, match="fewer than 10"):
        soft_noise_sigma(np.random.default_rng(2).normal(size=(8, 2)), 1, None)


def test_noise_cov_recovers_correlated_noise_when_groups_fragment():
    rng = np.random.default_rng(3)
    true = np.array([[1.0, 0.8], [0.8, 1.0]])
    z = rng.multivariate_normal(np.zeros(2), true, size=2000)
    singletons = np.arange(2000)
    with pytest.raises(stats.NoiseEstimateError):
        stats.robust_noise_cov(z, 1, singletons)
    with pytest.warns(RuntimeWarning, match="ungrouped"):
        cov, quality = soft_noise_cov(z, 1, singletons)
    assert quality == "ungrouped" and np.abs(cov - true).max() < 0.15


def test_noise_cov_shrinks_toward_the_diagonal_when_pairs_are_fewer_than_2d():
    rng = np.random.default_rng(4)
    d = 20
    z = rng.normal(size=(30, d))  # 29 pairs < 2d = 40: the full covariance is not estimable
    with pytest.raises(stats.NoiseEstimateError):
        stats.robust_noise_cov(z, 1, None)
    with pytest.warns(RuntimeWarning, match="shrunk"):
        cov, quality = soft_noise_cov(z, 1, None)
    assert quality == "shrunk"
    assert np.allclose(cov, cov.T) and np.linalg.eigvalsh(cov).min() > 0
    assert 0.5 < np.trace(cov) / d < 1.5  # true covariance is the identity


def test_noiseless_ramp_is_refused_not_whitened_by_the_numerical_floor():
    # Mirrors tests/test_stats.py: no within-state noise exists. A fallback that "succeeds" would
    # divide by the 1e-6 floor and hand downstream code a whitener amplifying by 1e6.
    z = np.arange(100, dtype=float)[:, None] * np.array([[1.0, 2.0]])
    with pytest.raises(stats.NoiseEstimateError, match="numerical floor"):
        soft_noise_cov(z, 1, None)


def test_an_unsupported_dimension_is_not_mistaken_for_a_data_shortage():
    # chi2 tails are exact only for even d (stats.chi2_logsf). That is a limit of the method, not a
    # lack of data, so the ladder must let it through instead of degrading to a covariance rung.
    z = np.random.default_rng(11).normal(size=(500, 3))
    with pytest.raises(ValueError, match="must be even"):
        soft_noise_cov(z, 1, None)


def test_inflation_falls_back_to_group_segments_when_jump_segmentation_fragments():
    z = _regime_switching(np.random.default_rng(5), 4000, 4)
    with pytest.raises(stats.NoiseEstimateError):
        stats.ewma_inflation(z, 0.1, None)
    with pytest.warns(RuntimeWarning, match="group_segments"):
        infl, quality = soft_ewma_inflation(z, 0.1, None)
    assert quality == "group_segments" and infl.shape == (4,) and (infl > 0).all() and np.isfinite(infl).all()


def test_inflation_assumes_white_noise_only_when_nothing_is_estimable():
    z = np.random.default_rng(6).normal(size=(30, 4))  # EWMA memory 19 and lag 38 exceed the data
    with pytest.raises(stats.NoiseEstimateError):
        stats.ewma_inflation(z, 0.1, None)
    with pytest.warns(RuntimeWarning, match="assumed_white"):
        infl, quality = soft_ewma_inflation(z, 0.1, None)
    assert quality == "assumed_white"
    np.testing.assert_array_equal(infl, np.ones(4))


def test_worst_quality_orders_the_rungs():
    assert worst_quality("robust", "shrunk") == "shrunk"
    assert worst_quality("ungrouped", "robust") == "ungrouped"
    assert worst_quality("shrunk", "ungrouped") == "shrunk"


def test_fit_projection_is_numerically_unchanged_when_nothing_degrades():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(800, 10)) @ rng.normal(size=(10, 10))
    _, noise, sigma0 = fit_projection(X, 6, 1, None)
    Zp = Whitener().fit(X, 6).transform(X)
    lam, V = np.linalg.eigh(stats.robust_noise_cov(Zp, 1, None))
    lam = np.maximum(lam, 1e-12 * max(float(lam.max()), 1e-12))
    np.testing.assert_array_equal(noise.W, (V / np.sqrt(lam)) @ V.T)
    np.testing.assert_array_equal(sigma0, stats.robust_noise_sigma(noise.transform(Zp), 1, None))
    assert noise.quality == "robust"


def test_fit_projection_degrades_loudly_instead_of_raising_when_groups_fragment():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(400, 6))
    with pytest.warns(RuntimeWarning, match="ungrouped"):
        _, noise, sigma0 = fit_projection(X, 4, 1, np.arange(400))
    assert noise.quality == "ungrouped" and np.isfinite(sigma0).all() and (sigma0 > 0).all()


def test_dsp_encoder_reports_the_dimension_it_actually_has():
    rng = np.random.default_rng(9)
    W = rng.normal(size=(600, 2, 64)).astype(np.float32)  # 2 channels x 6 features = 12 < d = 16
    with pytest.warns(RuntimeWarning, match="exceeds"):
        enc = DSPEncoder(d=16).fit(W, 1, None)
    assert enc.d == 12 and enc.encode(W).shape == (600, 12) and enc.sigma0.shape == (12,)


def test_dsp_encoder_healthy_fit_keeps_its_dimension_and_reports_robust_quality():
    rng = np.random.default_rng(10)
    W = rng.normal(size=(800, 8, 64)).astype(np.float32)
    enc = DSPEncoder(d=16).fit(W, 1, None)
    assert enc.d == 16 and enc.whitening_quality == "robust"

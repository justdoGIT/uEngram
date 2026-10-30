import numpy as np
import pytest

from uengram.health import HealthAgeRUL, HealthRUL


def test_recovers_monotone_health_rul_relation():
    # RUL falls linearly in log health; noise averages out within bins
    rng = np.random.default_rng(0)
    lh = rng.uniform(0.0, 6.0, 20_000)
    rul = 150.0 - 20.0 * lh + rng.normal(0, 10.0, lh.size)
    tab = HealthRUL(bins=20).fit(np.exp(lh), rul)
    q = np.array([1.0, 3.0, 5.0])
    np.testing.assert_allclose(tab.predict(np.exp(q)), 150.0 - 20.0 * q, atol=2.0)
    # outside the fitted range the end values are held, not extrapolated
    assert tab.predict([np.exp(-5.0)])[0] == pytest.approx(tab.value[0])
    assert tab.predict([np.exp(20.0)])[0] == pytest.approx(tab.value[-1])


def test_tied_health_values_and_nan():
    # a healthy plateau: half the windows share one health value
    h = np.r_[np.full(500, 2.0), np.linspace(3.0, 50.0, 500)]
    rul = np.r_[np.full(500, 120.0), np.linspace(100.0, 0.0, 500)]
    tab = HealthRUL(bins=20).fit(np.r_[h, np.nan], np.r_[rul, 5.0])
    assert np.all(np.isfinite(tab.value)) and np.all(np.diff(tab.centre) > 0)
    assert tab.predict([2.0])[0] == pytest.approx(120.0)
    assert np.isnan(tab.predict([np.nan])[0])


def test_age_resolves_what_health_alone_cannot():
    # two cohorts reach the same health at different ages with different remaining life
    rng = np.random.default_rng(1)
    age = rng.uniform(30.0, 300.0, 6000)
    lh = rng.uniform(0.0, 3.0, age.size)
    rul = np.clip(200.0 - 0.5 * age - 25.0 * lh, 0.0, 125.0)
    alone = HealthRUL().fit(np.exp(lh), rul).predict(np.exp(lh))
    both = HealthAgeRUL().fit(np.exp(lh), age, rul).predict(np.exp(lh), age)
    assert np.sqrt(np.mean((both - rul) ** 2)) < 0.5 * np.sqrt(np.mean((alone - rul) ** 2))
    assert both.min() >= 0.0 and both.max() <= 125.0
    assert HealthAgeRUL().fit(np.exp(lh), age, rul).table_bytes() == 24


def test_age_regression_ignores_nan_rows_and_propagates_nan_health():
    h = np.r_[np.linspace(1.0, 20.0, 400), np.nan]
    age = np.r_[np.linspace(20.0, 200.0, 400), 50.0]
    model = HealthAgeRUL().fit(h, age, np.r_[100.0 - 4.0 * np.log(h[:400]), 5.0])
    assert np.isfinite(model.coef).all() and np.isnan(model.predict([np.nan], [50.0])[0])

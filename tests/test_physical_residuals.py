from types import SimpleNamespace

import numpy as np

from bench.physical_residuals import PhysicalResiduals, physical_features
from uengram.metropt import ANALOG


def test_rates_use_observed_intervals_and_exclude_existing_alarm_signals():
    n, k = 40, 24
    seconds = np.cumsum(np.tile([1, 2, 3], 8))
    times = np.tile(seconds, (n, 1)).astype('datetime64[s]')
    w = np.ones((n, len(ANALOG), k))
    w[:, ANALOG.index('Reservoirs')] = 8 + seconds * .002
    w[:, ANALOG.index('Oil_temperature')] = 50 + seconds * .001
    d = np.zeros((n, 4, k))
    d[:, 1, :12] = 1
    stream = SimpleNamespace(W=w, cond=d, segment=np.zeros(n), extra={})
    context, target = physical_features(stream, times)
    np.testing.assert_allclose(target[:, 1:3], .12)
    np.testing.assert_allclose(target[:, 3], .06)
    stream.cond[:, 3] = 1  # Existing LPS alarm must not become predictive evidence.
    changed_context, changed_target = physical_features(stream, times)
    np.testing.assert_array_equal(changed_context, context)
    np.testing.assert_array_equal(changed_target, target)


def test_fit_does_not_consume_future_targets_or_context():
    rng = np.random.default_rng(21)
    context = rng.normal(size=(250, 5))
    target = rng.normal(size=(250, 6))
    fit = np.arange(250) < 200
    before = PhysicalResiduals().fit(context, target, fit)
    altered_context, altered_target = context.copy(), target.copy()
    altered_context[~fit] += 100000
    altered_target[~fit] -= 100000
    after = PhysicalResiduals().fit(altered_context, altered_target, fit)
    np.testing.assert_allclose(before.score(context, target)[0], after.score(context, target)[0])


def test_abnormally_slow_recovery_is_not_normalized_away():
    rng = np.random.default_rng(3)
    context = rng.normal(size=(240, 5))
    target = rng.normal(0, .02, (240, 6))
    target[:, 1] += 2 * context[:, 0]
    fit = np.arange(240) < 200
    model = PhysicalResiduals().fit(context, target, fit)
    normal, _ = model.score(context[~fit], target[~fit])
    degraded = target[~fit].copy()
    degraded[:, 1] -= 10
    score, parts = model.score(context[~fit], degraded)
    assert np.all(score > normal.max())
    assert np.all(parts[:, 1] > normal.max())


def test_gap_does_not_reuse_previous_segment_duty():
    n, k = 40, 24
    w = np.ones((n, len(ANALOG), k))
    d = np.zeros((n, 4, k))
    d[:20, 1] = 1
    s = SimpleNamespace(W=w, cond=d, segment=np.repeat([0, 1], 20), extra={})
    times = np.tile(np.arange(k), (n, 1)).astype('datetime64[s]')
    context, target = physical_features(s, times)
    assert np.isnan(context[20, 4])
    assert np.isnan(target[0, 2])  # No unloaded samples: no invented decay slope.
    assert np.isnan(target[20, 1])  # No loaded samples: no invented recovery slope.

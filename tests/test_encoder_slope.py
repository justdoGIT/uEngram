import numpy as np

from uengram.encoder_slope import DSPSlopeEncoder
from uengram.metrics import auroc


def test_training_embeddings_whitened_like_dsp_encoder():
    rng = np.random.default_rng(0)
    windows = rng.normal(0, 1, (400, 3, 16))
    segment = np.zeros(400, dtype=np.int64)
    enc = DSPSlopeEncoder(d=4, lags=(2, 5)).fit(windows, 1, segment)
    z = enc.encode(windows, segment=segment)
    assert z.shape == (400, 4)
    assert np.abs(z.mean(0)).max() < 1e-6


def test_slope_is_zero_across_a_segment_boundary_and_at_the_start():
    rng = np.random.default_rng(1)
    windows = rng.normal(0, 0.1, (20, 2, 8))
    windows[10:, 0] += 5.0  # a level shift at the segment boundary
    segment = np.r_[np.zeros(10, dtype=np.int64), np.ones(10, dtype=np.int64)]
    enc = DSPSlopeEncoder(d=2, lags=(3,)).fit(windows, 1, segment)
    F = enc._slopes(windows, segment)
    n_channels = windows.shape[1]
    slope_col = n_channels * 6  # first slope column, channel 0, lag 3
    # windows 0-2 have no window 3 lags back at all -> zero
    assert np.allclose(F[:3, slope_col], 0.0)
    # window 10 (first of the new segment) has no same-segment predecessor -> zero
    assert F[10, slope_col] == 0.0
    # window 5 is mid-segment-0 and has a valid 3-lag predecessor -> nonzero in general
    assert F[5, slope_col] != 0.0


def test_slope_separates_a_sustained_ramp():
    """A ramp held in one channel for many consecutive windows, each step
    small relative to the within-window noise, accumulates into a large
    windowed slope; the latent norm should separate it from steady windows,
    matching tests/test_encoders.py's convention for DSPEncoder."""
    rng = np.random.default_rng(2)
    n, C, K = 600, 2, 10
    windows = rng.normal(0, 1.0, (n, C, K))
    segment = np.zeros(n, dtype=np.int64)
    drift = np.zeros(n, dtype=bool)
    ramp = np.linspace(0, 6.0, 100)
    for i, level in enumerate(ramp):
        windows[n - 100 + i, 0] += level
        drift[n - 100 + i] = True

    fit_windows, fit_segment = windows[: n - 100], segment[: n - 100]
    enc = DSPSlopeEncoder(d=8, lags=(5, 20)).fit(fit_windows, 1, fit_segment)
    z = enc.encode(windows, segment=segment)
    score = np.linalg.norm(z, axis=1)
    assert auroc(score, drift) > 0.8


def test_segment_none_treats_the_whole_array_as_one_segment():
    """Matches bench.run_synth's convention (`fit(train.windows, 1, None)`,
    `enc.encode(stream.windows)`): no real multi-file segments, so slopes
    run over the full array uninterrupted."""
    rng = np.random.default_rng(3)
    windows = rng.normal(0, 1, (50, 2, 8))
    enc = DSPSlopeEncoder(d=4, lags=(4,)).fit(windows, 1, None)
    z_default = enc.encode(windows)
    z_explicit = enc.encode(windows, segment=np.zeros(50, dtype=np.int64))
    np.testing.assert_allclose(z_default, z_explicit)
    assert np.isfinite(z_default).all()

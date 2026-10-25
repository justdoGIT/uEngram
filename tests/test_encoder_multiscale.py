import numpy as np
import pytest

from uengram.encoder_multiscale import MultiScaleEncoder


def _training_windows(seed=7, n=80, channels=2, length=32):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, channels, length))
    drift = rng.normal(size=(n, channels, 1))
    return base + drift


def test_multiscale_is_deterministic_and_finite():
    windows = _training_windows()
    first = MultiScaleEncoder(d=6).fit(windows, noise_lag=1, groups=None)
    second = MultiScaleEncoder(d=6).fit(windows, noise_lag=1, groups=None)
    components = first.encode_components(windows)
    repeated = second.encode_components(windows)
    assert set(components) == {"level", "dynamics", "combined"}
    assert components["combined"].shape == (len(windows), first.d)
    assert components["level"].shape[1] == first.level_d
    assert components["dynamics"].shape[1] == first.dynamics_d
    for name in components:
        assert np.isfinite(components[name]).all()
        np.testing.assert_array_equal(components[name], repeated[name])
    np.testing.assert_array_equal(
        components["combined"], np.concatenate((components["level"], components["dynamics"]), axis=1))
    np.testing.assert_array_equal(first.encode(windows), components["combined"])
    np.testing.assert_array_equal(first.encode(windows, sequential=False), components["combined"])
    assert first.d == first.level_d + first.dynamics_d
    assert first.sigma0.shape == (first.d,)
    assert first.sigma0_level.shape == (first.level_d,)
    assert first.sigma0_dynamics.shape == (first.dynamics_d,)
    assert np.isfinite(first.sigma0).all()


def test_level_shift_is_represented_in_level_block():
    windows = _training_windows(seed=21)
    encoder = MultiScaleEncoder(d=6).fit(windows, noise_lag=1, groups=None)
    shifted = windows.copy()
    shifted[:, 0, :] += 5.0
    before = encoder.encode_components(windows)
    after = encoder.encode_components(shifted)
    level_change = np.linalg.norm(after["level"] - before["level"], axis=1)
    dynamics_change = np.linalg.norm(after["dynamics"] - before["dynamics"], axis=1)
    assert np.all(level_change > 1.0)
    np.testing.assert_allclose(dynamics_change, 0.0, atol=1e-10)


def test_impulse_is_represented_in_dynamics_block():
    windows = _training_windows(seed=13)
    encoder = MultiScaleEncoder(d=6).fit(windows, noise_lag=1, groups=None)
    impulse = windows.copy()
    impulse[:, 0, 12] += 4.0
    before = encoder.encode_components(windows)
    after = encoder.encode_components(impulse)
    assert np.all(np.linalg.norm(after["dynamics"] - before["dynamics"], axis=1) > 0.1)
    # A localized impulse may change the raw window mean; level remains an
    # explicit independent block rather than being folded into dynamics.
    assert np.isfinite(after["level"]).all()



def test_unequal_latent_blocks_keep_per_dimension_noise_scales():
    windows = _training_windows(seed=12)
    encoder = MultiScaleEncoder(d=7).fit(windows, noise_lag=1, groups=None)
    assert encoder.level_d != encoder.dynamics_d
    assert encoder.d == encoder.level_d + encoder.dynamics_d
    assert encoder.level_d % 2 == encoder.dynamics_d % 2 == 0
    assert encoder.sigma0.shape == (encoder.d,)
    assert np.isfinite(encoder.sigma0).all()

def test_small_channel_count_and_dimension_clamping():
    windows = _training_windows(seed=4, channels=1, length=6)
    encoder = MultiScaleEncoder(d=100, lags=(1, 2, 4)).fit(windows, 1, None)
    result = encoder.encode(windows)
    assert 2 <= encoder.d < 100
    assert result.shape == (len(windows), encoder.d)
    assert np.isfinite(result).all()


@pytest.mark.parametrize("kwargs", [{"d": 1}, {"d": 2}, {"d": 0}, {"lags": ()}, {"lags": (0,)}, {"lags": (1, 1)}])
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        MultiScaleEncoder(**kwargs)


def test_invalid_windows_and_unfitted_encode_are_rejected():
    encoder = MultiScaleEncoder()
    with pytest.raises(RuntimeError):
        encoder.encode(np.zeros((2, 1, 4)))
    with pytest.raises(ValueError):
        encoder.fit(np.zeros((1, 1, 4)), 1, None)
    with pytest.raises(ValueError):
        encoder.fit(np.zeros((4, 1, 1)), 1, None)
    with pytest.raises(ValueError):
        encoder.fit(np.full((4, 1, 4), np.nan), 1, None)
    with pytest.raises(ValueError):
        encoder.fit(np.zeros((4, 1, 4)), 0, None)

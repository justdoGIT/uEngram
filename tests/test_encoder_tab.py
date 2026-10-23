import numpy as np
import pytest

from uengram.encoder_tab import TabularEncoder


def _rows(n=600, f=24, seed=0):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 4))
    X = latent @ rng.normal(size=(4, f)) + 0.3 * rng.normal(size=(n, f))
    return X


def test_encoding_is_invariant_to_per_sensor_units_and_offsets():
    X = _rows()
    scale = np.linspace(0.01, 100, X.shape[1])
    offset = np.linspace(-500, 500, X.shape[1])
    a = TabularEncoder(d=8).fit(X, 1, None).encode(X)
    b = TabularEncoder(d=8).fit(X * scale + offset, 1, None).encode(X * scale + offset)
    np.testing.assert_allclose(np.abs(a), np.abs(b), rtol=1e-6, atol=1e-8)


def test_missing_reading_encodes_like_a_typical_value_and_constants_are_dropped():
    X = _rows()
    X = np.column_stack([X, np.full(len(X), 3.0)])  # constant sensor
    enc = TabularEncoder(d=8).fit(X, 1, None)
    assert len(enc.keep) == X.shape[1] - 1
    row = X[:1].copy()
    typical = row.copy()
    typical[0, 5] = np.median(X[:, 5])
    row[0, 5] = np.nan
    np.testing.assert_allclose(enc.encode(row), enc.encode(typical))
    assert np.isfinite(enc.encode(np.where(np.isnan(X), 0, X))).all()


def test_rejects_data_with_too_few_informative_features():
    X = np.column_stack([_rows(f=4), np.zeros((600, 10))])
    with pytest.raises(ValueError, match="informative"):
        TabularEncoder(d=8).fit(X, 1, None)

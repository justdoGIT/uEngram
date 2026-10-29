import numpy as np

from bench import feature_group_ablation as fga
from uengram import encoders


def test_patched_band_features_keep_only_requested_columns():
    windows = np.random.default_rng(0).normal(size=(5, 3, 16))
    full = encoders.band_features(windows).reshape(5, 3, 6)
    kept = fga._patched((0, 1))(windows).reshape(5, 3, 2)
    np.testing.assert_array_equal(kept, full[:, :, [0, 1]])


def test_ablation_patch_reaches_the_encoder_and_is_restored():
    """Guards against a silent no-op: DSPEncoder must see the patched feature width."""
    windows = np.random.default_rng(1).normal(size=(40, 4, 16))
    encoders.band_features = fga._patched((0, 1))
    try:
        width = encoders.DSPEncoder()._raw(windows, True).shape[1]
    finally:
        encoders.band_features = fga._orig
    assert width == 4 * 2
    assert encoders.DSPEncoder()._raw(windows, True).shape[1] == 4 * 6

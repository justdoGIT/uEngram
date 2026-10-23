import numpy as np
import pytest

torch = pytest.importorskip("torch")

from uengram.encoder_cnn3 import CNN3Encoder


def windows(n=100, k=24):
    rng = np.random.default_rng(8)
    x = rng.normal(0, 0.01, (n, 2, k)).astype(np.float32)
    # Blocks, not alternation: lag-1 noise estimation needs same-state neighbours.
    state = (np.arange(n) // 20) % 2
    x[state == 1, 0] += 3.0
    # Rare loaded condition: one sample within the fit prefix has substantial
    # operating range although the common state's scale is nearly constant.
    x[3, 1] += np.linspace(-2, 2, k)
    cond = np.zeros((n, 3, k), np.float32)
    cond[:, 1] = state[:, None]
    return x, cond


def test_range_scaling_has_healthy_range_floor_and_preserves_level():
    x, cond = windows()
    enc = CNN3Encoder(mode="range", seed=2, d=4, epochs=1, members=1,
                      batch=32, patience=1).fit(x, 1, None, cond=cond)
    assert enc.scale[0, 1, 0] >= enc.scale_floor[0, 1, 0]
    assert enc.scale[0, 1, 0] > 0.1
    altered = x[:1].copy()
    altered[:, 0] += 2.0
    assert np.linalg.norm(enc.encode(altered, cond=cond[:1]) - enc.encode(x[:1], cond=cond[:1])) > 1e-5
    assert enc.normalization_diagnostics["fit_prefix_windows"] == int(len(x) * .85)


def test_state_scales_floor_rare_states_and_reject_unseen_conditions():
    x, cond = windows()
    enc = CNN3Encoder(mode="state", seed=3, d=4, epochs=1, members=1,
                      batch=32, patience=1).fit(x, 1, None, cond=cond)
    assert enc.state_scales.shape == (2, 2)
    assert np.all(enc.state_scales >= enc.scale_floor[0, :, 0])
    assert enc.encode(x[:4], cond=cond[:4]).shape == (4, 4)
    unseen = cond[:1].copy()
    unseen[:, 1] = 7
    with pytest.raises(ValueError, match="unsupported DV_eletric"):
        enc.encode(x[:1], cond=unseen)


def test_state_mode_requires_conditioning_and_does_not_fit_validation_tail():
    x, cond = windows()
    with pytest.raises(ValueError, match="DV_eletric"):
        CNN3Encoder(mode="state", d=4, epochs=1, members=1).fit(x, 1, None)
    a = CNN3Encoder(mode="range", d=4, epochs=1, members=1, seed=9).fit(x, 1, None, cond=cond)
    changed = x.copy()
    changed[int(len(x) * .85):] += 10000
    b = CNN3Encoder(mode="range", d=4, epochs=1, members=1, seed=9).fit(changed, 1, None, cond=cond)
    np.testing.assert_array_equal(a.center, b.center)
    np.testing.assert_array_equal(a.scale, b.scale)
    np.testing.assert_array_equal(a.scale_floor, b.scale_floor)


def test_scaling_is_unit_invariant_and_state_ablation_is_distinct():
    x, cond = windows()
    split = int(len(x) * .85)
    joined = np.concatenate((x, cond), axis=1)
    normalizers = []
    for mode in ('range', 'state'):
        enc = CNN3Encoder(mode=mode)
        enc.analog_channels = 2
        enc._fit_normalizer(joined, split)
        normalizers.append(enc)
    assert not np.allclose(normalizers[0]._scaled(joined), normalizers[1]._scaled(joined))
    converted = joined.copy()
    converted[:, 0] = converted[:, 0] * 100 + 32
    converted[:, 1] = converted[:, 1] * 10 - 4
    for enc in normalizers:
        other = CNN3Encoder(mode=enc.mode)
        other.analog_channels = 2
        other._fit_normalizer(converted, split)
        np.testing.assert_allclose(enc._scaled(joined), other._scaled(converted), atol=2e-5, rtol=2e-5)


def test_valid_unseen_duty_bin_uses_global_scale_without_claiming_support():
    x, cond = windows()
    joined = np.concatenate((x, cond), axis=1)
    enc = CNN3Encoder(mode='state')
    enc.analog_channels = 2
    enc._fit_normalizer(joined, int(len(x) * .85))
    cycling = joined[:1].copy()
    cycling[:, 3] = 0
    cycling[:, 3, :5] = 1
    assert enc._state_ids(cycling)[0] not in enc.state_values
    np.testing.assert_allclose(enc._scaled(cycling)[:, :2],
                               (cycling[:, :2] - enc.center) / enc.scale)

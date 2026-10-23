import numpy as np
import pytest

torch = pytest.importorskip("torch")

from uengram.encoder_cnn2 import CNN2Encoder


def duty_windows(n=72, k=48, seed=4):
    rng = np.random.default_rng(seed)
    out = np.empty((n, 2, k), np.float32)
    state = np.arange(n) % 3
    for i, s in enumerate(state):
        duty, level = ((.25, 0.), (.5, 1.5), (.75, -.8))[s]
        phase = rng.integers(k)
        t = (np.arange(k) + phase) % k
        out[i, 0] = level + ((t % 12) < round(duty * 12)) * 1.2
        out[i, 1] = .4 * level + ((t % 12) < round(duty * 12)) * .7
        out[i] += rng.normal(0, .025, out[i].shape)
    return out


def test_phase_invariance_and_level_sensitivity():
    x = duty_windows()
    enc = CNN2Encoder(seed=8, d=4, epochs=4, batch=24, members=2, patience=2).fit(x, 1, None)
    # Construct clean identical cycles with distinct starting phases and a distinct
    # duty/level state; phase shifts should be smaller than state changes.
    t = np.arange(48)
    a = np.stack([((t % 12) < 3).astype(np.float32), ((t % 12) < 3).astype(np.float32) * .5])[None]
    phase = np.roll(a, 5, axis=2)
    other = np.stack([1.5 + ((t % 12) < 6), .6 + ((t % 12) < 6) * .5]).astype(np.float32)[None]
    za, zp, zo = enc.encode(a), enc.encode(phase), enc.encode(other)
    assert np.linalg.norm(za - zp) < np.linalg.norm(za - zo)
    level_shift = a.copy(); level_shift[:, 0] += 1.0
    assert np.linalg.norm(za - enc.encode(level_shift)) > 1e-3


def test_determinism_cond_and_member_diagnostics():
    x = duty_windows(n=64)
    cond = (np.arange(64)[:, None, None] % 2 * np.ones((64, 1, 48))).astype(np.float32)
    kwargs = dict(seed=13, d=4, epochs=3, batch=32, members=3, patience=2)
    a = CNN2Encoder(**kwargs).fit(x, 1, None)
    b = CNN2Encoder(**kwargs).fit(x, 1, None)
    np.testing.assert_array_equal(a.encode(x), b.encode(x))
    conditioned = CNN2Encoder(**kwargs).fit(x, 1, None, cond=cond)
    assert conditioned.encode(x, cond=cond).shape == (64, 4)
    assert a.encode(x).shape == (64, 4)
    assert a.member_cka.shape == (3, 3)
    np.testing.assert_allclose(np.diag(a.member_cka), 1.)
    means = (a.member_cka.sum(1) - 1) / 2
    assert a.medoid == int(np.argmax(means))

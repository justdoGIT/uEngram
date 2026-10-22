import numpy as np
import pytest

from uengram import synth
from uengram.encoders import DSPEncoder
from uengram.stats import robust_noise_cov
from uengram.metrics import auroc


def test_dsp_noise_whitened_and_separates_anomalies():
    s = synth.generate(3000, seed=1)
    enc = DSPEncoder().fit(s.windows, 1, None)
    z = enc.encode(s.windows)
    assert z.shape == (3000, 16)
    assert np.abs(z.mean(0)).max() < 1e-6
    # within-state noise is uncorrelated with unit variance: each correlated
    # fluctuation is counted once by the per-dimension slot statistics
    assert np.abs(robust_noise_cov(z, 1, None) - np.eye(16)).max() < 0.01
    assert enc.sigma0.shape == (16,) and (enc.sigma0 > 0).all()
    score = np.linalg.norm(z, axis=1)
    assert auroc(score[200:], s.anomaly[200:] > 0) > 0.8


def test_cnn_shapes_finite():
    pytest.importorskip("torch")
    from uengram.encoder_cnn import CNNEncoder

    s = synth.generate(512, seed=2)
    enc = CNNEncoder(seed=0, epochs=2).fit(s.windows, 1, None)
    z = enc.encode(s.windows)
    assert z.shape == (512, 16) and np.isfinite(z).all()
    assert enc.sigma0.shape == (16,)


def test_context_encoder_learns_lag_and_explains_transitions():
    from uengram.encoder_ctx import ContextEncoder

    train = synth.generate(4000, seed=11)
    s = synth.generate(4000, seed=12)
    t = np.arange(len(s.anomaly))
    chg = np.zeros(len(t), bool)
    for c in np.flatnonzero(np.diff(s.regime) != 0) + 1:
        chg[c : c + 4] = True
    nom = (s.anomaly == 0) & (t >= 200)

    def frac_far(enc):
        """Fraction of (near-change, steady) nominal windows beyond the gate
        radius from their state's steady centroid (500-window blocks)."""
        z = enc.encode(s.windows)
        far, ok = np.zeros(len(t), bool), np.zeros(len(t), bool)
        for a in range(0, len(t), 500):
            for r in range(3):
                m = nom & (s.regime == r) & (t >= a) & (t < a + 500)
                if (m & ~chg).sum() < 20:
                    continue
                c = z[m & ~chg].mean(0)
                far[m] = np.linalg.norm((z[m] - c) / enc.sigma0, axis=1) > 6.8
                ok[m] = True
        return far[ok & chg].mean(), far[ok & ~chg].mean()

    ctx = ContextEncoder(lag_channels=(2,), drive_channel=0).fit(train.windows, 1, None)
    assert abs(ctx.lag_coefficient() - 0.5) < 0.05  # synthetic coolant lag is 0.5
    dsp = DSPEncoder().fit(train.windows, 1, None)
    near, steady = frac_far(dsp)
    assert near > steady + 0.25  # transitions look novel to plain spectral features
    near, steady = frac_far(ctx)
    assert near < steady + 0.02  # ... but not once the lag is predicted
    assert ctx.encode(s.windows[:10], sequential=False).shape == (10, 16)

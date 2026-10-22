import numpy as np
import pytest

from uengram import synth
from uengram.encoders import DSPEncoder
from uengram.metrics import auroc


def test_dsp_whitened_and_separates_anomalies():
    s = synth.generate(3000, seed=1)
    enc = DSPEncoder().fit(s.windows, 1, None)
    z = enc.encode(s.windows)
    assert z.shape == (3000, 16)
    assert np.abs(z.mean(0)).max() < 1e-6
    assert np.abs(np.cov(z, rowvar=False) - np.eye(16)).max() < 1e-6
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
        z = enc.encode(s.windows)
        far = np.zeros(len(t), bool)
        for r in range(3):
            for e in range(2):
                m = nom & (s.regime == r) & (s.era == e)
                if (m & ~chg).sum() < 20:
                    continue
                c = z[m & ~chg].mean(0)
                far[m] = np.linalg.norm((z[m] - c) / enc.sigma0, axis=1) > 6.8
        return far[nom & chg].mean()

    ctx = ContextEncoder(lag_channels=(2,), drive_channel=0).fit(train.windows, 1, None)
    assert abs(ctx.lag_coefficient() - 0.5) < 0.05  # synthetic coolant lag is 0.5
    dsp = DSPEncoder().fit(train.windows, 1, None)
    assert frac_far(dsp) > 0.3 and frac_far(ctx) < 0.1
    assert ctx.encode(s.windows[:10], sequential=False).shape == (10, 16)

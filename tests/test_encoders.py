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

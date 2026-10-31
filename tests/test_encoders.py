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

def test_dsp_zero_iqr_features_use_observed_spread_instead_of_amplifying_outliers():
    from uengram.encoders import band_features

    rng = np.random.default_rng(5)
    windows = rng.normal(0, 0.1, (100, 4, 16))
    windows[:, 0] = 0
    windows[0, 0] = np.sin(np.arange(16))
    features = band_features(windows)
    encoder = DSPEncoder(d=4).fit(windows, 1, None)
    # The exceptional channel's mean has zero IQR but nonzero spread.
    assert encoder.scale[0] == pytest.approx(features.std(axis=0)[0])
    assert encoder.scale[0] > 1e-6


def test_whitener_handles_constant_features():
    from uengram.encoders import Whitener

    z = Whitener().fit(np.ones((8, 4)), 4).transform(np.ones((8, 4)))
    assert np.isfinite(z).all()


def test_cnn_restores_torch_global_settings():
    pytest.importorskip("torch")
    import torch
    from uengram.encoder_cnn import CNNEncoder

    previous_threads = torch.get_num_threads()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    x = synth.generate(64, seed=17).windows
    CNNEncoder(seed=0, d=4, epochs=1, batch=32).fit(x, 1, None)
    assert torch.get_num_threads() == previous_threads
    assert torch.are_deterministic_algorithms_enabled() == previous_determinism


def test_cnn_shapes_finite():
    pytest.importorskip("torch")
    from uengram.encoder_cnn import CNNEncoder

    s = synth.generate(512, seed=2)
    enc = CNNEncoder(seed=0, epochs=2).fit(s.windows, 1, None)
    z = enc.encode(s.windows)
    assert z.shape == (512, 16) and np.isfinite(z).all()
    assert enc.sigma0.shape == (16,)



def test_cnn_maxavg_ortho_pool_finite():
    pytest.importorskip("torch")
    from uengram.encoder_cnn import CNNEncoder

    s = synth.generate(512, seed=3)
    enc = CNNEncoder(seed=0, d=8, epochs=2, pool="maxavg", ortho_beta=0.05).fit(s.windows, 1, None)
    z = enc.encode(s.windows)
    assert z.shape == (512, 8) and np.isfinite(z).all()
    assert enc.sigma0.shape == (8,)
    with pytest.raises(ValueError, match="pool"):
        CNNEncoder(pool="invalid")


def test_cnn_multiscale_readout_is_finite_and_changes_representation():
    pytest.importorskip("torch")
    from uengram.encoder_cnn import CNNEncoder

    s = synth.generate(256, seed=23)
    common = {"seed": 0, "d": 8, "epochs": 2, "batch": 64, "pool": "maxavg"}
    bottleneck = CNNEncoder(**common).fit(s.windows, 1, None).encode(s.windows)
    multiscale = CNNEncoder(
        **common, representation="multiscale"
    ).fit(s.windows, 1, None).encode(s.windows)

    assert multiscale.shape == (256, 8)
    assert np.isfinite(multiscale).all()
    assert not np.allclose(multiscale, bottleneck)
    with pytest.raises(ValueError, match="representation"):
        CNNEncoder(representation="invalid")

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

def test_context_rejected_window_updates_drive_but_propagates_ramp():
    from uengram.encoder_ctx import ContextEncoder

    enc = ContextEncoder(reject_k=1)
    enc.b = np.array([[0.5, 1., 0.]])
    enc.rscale = np.array([0.1])
    drive = np.array([0., 1., 2.])
    ramps = np.array([[0.], [10.], [0.]])
    residual = enc._residuals(drive, ramps, sequential=True)
    # The rejected ramp at t=1 is not accepted; at t=2 the ramp prediction
    # uses propagated ramp 0 and the immediately previous drive u=1.
    assert residual[2, 0] == pytest.approx(-1.)


def test_context_encoder_resets_at_a_segment_boundary():
    from uengram.encoder_ctx import ContextEncoder

    enc = ContextEncoder(reject_k=1)
    enc.b = np.array([[0.5, 1., 0.]])
    enc.rscale = np.array([0.1])
    drive = np.array([0., 1., 2., 5.])
    ramps = np.array([[0.], [1.], [0.], [3.]])
    segment = np.array([0, 0, 0, 1])  # window 3 starts a new file/unit
    residual = enc._residuals(drive, ramps, sequential=True, segment=segment)
    # window 3 has no valid same-segment predecessor: residual falls back to
    # the raw ramp (the sequential=False convention), not a prediction built
    # from window 2's unrelated drive/ramp.
    assert residual[3, 0] == pytest.approx(3.0)


def test_context_encoder_same_segment_mask():
    from uengram.encoder_ctx import ContextEncoder

    assert ContextEncoder._same_segment(None, 5).tolist() == [False, True, True, True, True]
    seg = np.array([0, 0, 1, 1, 1])
    assert ContextEncoder._same_segment(seg, 5).tolist() == [False, True, False, True, True]
    assert ContextEncoder._same_segment(seg, 1).tolist() == [False]

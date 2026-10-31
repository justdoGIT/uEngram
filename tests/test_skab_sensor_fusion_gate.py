import numpy as np

from bench.skab_sensor_fusion_gate import _fit_representation


def test_all_cnn_representations_produce_finite_latents():
    rng = np.random.default_rng(31)
    windows = rng.normal(size=(80, 8, 10)).astype(np.float32)
    widths = {}
    for representation in ("bottleneck", "middle", "multiscale"):
        encoder, _ = _fit_representation(windows, representation, epochs=1)
        latent = encoder.encode(windows[:8])
        assert np.isfinite(latent).all()
        widths[representation] = encoder.readout_dim
    assert widths["middle"] > widths["bottleneck"]
    assert widths["multiscale"] > widths["middle"]

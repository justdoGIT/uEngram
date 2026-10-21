import numpy as np

from uengram import synth


def test_shapes_and_determinism():
    a = synth.generate(1000, seed=3)
    b = synth.generate(1000, seed=3)
    assert a.windows.shape == (1000, 6, 64) and a.windows.dtype == np.float32
    np.testing.assert_array_equal(a.windows, b.windows)
    np.testing.assert_array_equal(a.anomaly, b.anomaly)
    c = synth.generate(1000, seed=4)
    assert not np.array_equal(a.windows, c.windows)


def test_seizures_and_quiet_start():
    s = synth.generate(4000, seed=3)
    assert len(set(s.event_id[s.anomaly == 4])) == 3
    assert (s.anomaly[:200] == 0).all()
    assert ((s.event_id >= 0) == (s.anomaly > 0)).all()


def test_maintenance_step_visible():
    T = 4000
    s = synth.generate(T, seed=3)
    t_m = int(0.6 * T)
    psi = s.windows[:, 1, :].mean(1)
    nom = (s.anomaly == 0) & (s.regime == 1)
    idx = np.arange(T)
    before = psi[nom & (idx >= t_m - 500) & (idx < t_m)].mean()
    after = psi[nom & (idx >= t_m) & (idx < t_m + 500)].mean()
    assert after - before > 0.5


def test_probe_composition():
    p = synth.make_probe(0, 4000)
    assert p.windows.shape == (210, 6, 64)
    assert (p.anomaly == 0).sum() == 90
    assert all((p.anomaly == k).sum() == 30 for k in (1, 2, 3, 4))
    assert (p.era == 1).all()

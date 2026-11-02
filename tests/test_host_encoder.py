import numpy as np
import pytest

from uengram.host import schema
from uengram.host.encoder import HostEncoder, ThermalModel
from uengram.host.features import WindowBatch, windows
from uengram.host.synth import make_ticks


def _thermal_batch(n=400, K=32, tau=60.0, R=0.7, T_amb=28.0, seed=0):
    """Exact one-pole RC system with power held constant per window."""
    rng = np.random.default_rng(seed)
    P = rng.choice([5.0, 20.0, 35.0], size=n) + rng.normal(0, 0.5, n)
    T, Tm = T_amb + R * P[0], np.empty(n)
    sub = 64
    for w in range(n):
        acc = 0.0
        for _ in range(sub):  # integrate exactly within the window
            T = T_amb + R * P[w] + (T - T_amb - R * P[w]) * np.exp(-K / sub / tau)
            acc += T
        Tm[w] = acc / sub + rng.normal(0, 0.02)
    F = np.zeros((n, len(schema.feature_names())))
    return WindowBatch(np.arange(n) * float(K), F, np.zeros((n, 4)), Tm, P, {}, {})


def test_thermal_model_recovers_rc_parameters():
    p = ThermalModel(32).fit([_thermal_batch()]).params()
    assert p["tau_s"] == pytest.approx(60.0, rel=0.05)
    assert p["R_c_per_w"] == pytest.approx(0.7, rel=0.02)
    assert p["T_amb_c"] == pytest.approx(28.0, abs=0.3)


def test_thermal_residual_sees_a_cooling_fault_not_a_power_step():
    m = ThermalModel(32).fit([_thermal_batch(seed=1)])
    b = _thermal_batch(seed=2)
    r, _ = m.residuals(b, None)
    assert np.nanmax(np.abs(r)) < 6 * m.rscale  # power steps are predicted
    b.temp_mean[200:] += 8.0  # fan failure: same power, hotter
    r, _ = m.residuals(b, None)
    assert np.all(r[200:205] > 5.0)


@pytest.fixture(scope="module")
def fitted():
    ticks = make_ticks(3 * 3600, seed=4)
    seqs = [windows(ticks, 32, off) for off in (0, 8, 16, 24)]
    return HostEncoder(16, 32).fit(seqs), ticks


def test_change_in_a_channel_quiet_during_commissioning_is_visible(fitted):
    enc, ticks = fitted
    Z0, *_ = enc.encode(windows(ticks[-640:], 32))
    # page scanning never happened during commissioning: zero variance there,
    # so the change lies outside the PCA subspace and only Q can carry it
    hot = [dict(r, vm_pgscan=400) for r in ticks[-640:]]
    Z1, *_ = enc.encode(windows(hot, 32))
    assert np.median(np.linalg.norm(Z1 - Z0, axis=1)) > 5.0


def test_attribution_sums_to_d2_and_names_the_changed_modality(fitted):
    enc, ticks = fitted
    b = windows(ticks[-320:], 32)
    Z, X, *_ = enc.encode(b)
    mu, var = Z[0].copy(), np.full(enc.d, 2.0)
    thrash = [dict(r, psi_mem_us=4e5, vm_pgscan=9000, vm_pswpout=3000) for r in ticks[-320:]]
    Zt, Xt, *_ = enc.encode(windows(thrash, 32))
    c = enc.attribution(Xt[0], Zt[0], mu, var)
    d2 = float(np.sum((Zt[0] - mu) ** 2 / var))
    assert c.sum() == pytest.approx(d2, rel=1e-8)
    mods = {m: c[enc.modality == m].sum() for m in schema.MODALITIES}
    assert max(mods, key=mods.get) == "memory"

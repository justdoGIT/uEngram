import numpy as np
import pytest

from bench.model_selection import (FAULT_FAMILIES, generate_synthetic_faults, rank_candidates,
                                   select_by_synthetic_challenges)


class FlattenEncoder:
    def encode(self, windows, sequential=True):
        return np.asarray(windows, dtype=np.float64).reshape(len(windows), -1)


class MeanEncoder:
    def encode(self, windows, sequential=True):
        return np.asarray(windows, dtype=np.float64).mean(axis=2)


def selection_data(n=24, channels=2, length=32, seed=7):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, channels, length)).astype(np.float32)


def test_fault_generation_is_deterministic_and_does_not_mutate_input():
    x = selection_data()
    before = x.copy()
    a = generate_synthetic_faults(x, seed=11)
    b = generate_synthetic_faults(x, seed=11)
    assert tuple(a) == FAULT_FAMILIES
    for family in FAULT_FAMILIES:
        np.testing.assert_array_equal(a[family], b[family])
        assert a[family].shape == x.shape
        assert not np.array_equal(a[family], x)
    np.testing.assert_array_equal(x, before)


def test_selector_reports_each_fault_and_ranks_fault_sensitive_candidate():
    candidates = {"mean": MeanEncoder(), "flatten": FlattenEncoder()}
    rows = rank_candidates(candidates, selection_data(), seed=4)
    selected = select_by_synthetic_challenges(candidates, selection_data(), alpha=0.1)
    assert rows[0]["name"] == "flatten"
    assert selected["winner"] == "flatten"
    assert selected["ranking"][0] == selected["winner"]
    assert set(selected["per_candidate"]["flatten"]["fault_metrics"]) == set(FAULT_FAMILIES)
    assert rows[0]["score"] == rows[0]["worst_family_score"]
    assert all(np.isfinite(v) and v >= 0 for row in rows for v in row["fault_metrics"].values())


def test_ssl_cnn_is_torch_optional_and_deterministic():
    torch = pytest.importorskip("torch")
    from uengram.encoder_cnn_ssl import SSLTemporalEncoder

    with pytest.raises(ValueError):
        SSLTemporalEncoder(d=3)
    torch.manual_seed(71)
    rng_before = torch.random.get_rng_state().clone()
    threads_before = torch.get_num_threads()
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    x = selection_data(n=20, length=16)
    kwargs = dict(seed=19, d=4, epochs=2, batch=10, width=8)
    first = SSLTemporalEncoder(**kwargs).fit(x, 1, None)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    second = SSLTemporalEncoder(**kwargs).fit(x, 1, None)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert torch.get_num_threads() == threads_before
    assert torch.are_deterministic_algorithms_enabled() == deterministic_before
    z1, z2 = first.encode(x), second.encode(x)
    assert z1.shape == (len(x), 4)
    assert first.sigma0.shape == (4,)
    np.testing.assert_array_equal(z1, z2)

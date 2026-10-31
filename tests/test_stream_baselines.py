import numpy as np
import pytest

from uengram.stream_baselines import HSTDetector, MemStream


def test_hst_far_record_scores_above_normal_after_warmup():
    pytest.importorskip("river")
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(500, 4))
    det = HSTDetector(Z, seed=0)
    for t, z in enumerate(Z[:det.warmup_count]):
        det.ingest(z, t, t)
    normals = [det.ingest(z, t, t).score for t, z in enumerate(Z[250:350], start=250)]
    far = det.ingest(Z.mean(0) + 10.0 * Z.std(0), 350, 350).score
    assert all(np.isfinite(score) and score >= 0.0 for score in normals)
    assert far > np.mean(normals)


def _memstream(data, *, beta=float("-inf")):
    pytest.importorskip("torch")
    return MemStream(data, seed=0, memory_len=len(data), epochs=40, lr=1e-2, beta=beta)


def test_memstream_accepts_score_at_beta_boundary():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(16, 4))
    detector = _memstream(data)
    sample = data[0] + np.array([0.2, -0.1, 0.15, 0.05])
    threshold_detector = detector.clone(beta=float("inf"))
    score = threshold_detector.ingest(sample, 0, 0).score
    threshold_detector = detector.clone(beta=score)
    threshold_detector.ingest(sample, 0, 0)
    assert threshold_detector.count == 1


def test_memstream_clone_updates_are_independent_and_change_scores():
    rng = np.random.default_rng(2)
    data = rng.normal(size=(16, 3))
    base = _memstream(data)
    updating = base.clone(beta=float("inf"))
    frozen = base.clone(beta=float("-inf"))
    replacement = np.array([8.0, -6.0, 5.0])
    updating.ingest(replacement, 0, 0)
    assert updating.count == 1
    assert frozen.count == 0
    assert not np.array_equal(updating.memory.numpy(), frozen.memory.numpy())
    probe = data[1]
    assert updating.ingest(probe, 1, 1).score != frozen.ingest(probe, 1, 1).score


def test_memstream_zero_variance_dimensions_score_finitely():
    pytest.importorskip("torch")
    data = np.column_stack((np.arange(16, dtype=float), np.ones(16)))
    detector = MemStream(data, seed=1, memory_len=16, epochs=40)
    assert detector.std[1].item() == 0.0
    assert np.isfinite(detector.ingest(np.array([3.0, 1.0]), 0, 0).score)


def test_memstream_rejects_non_vector_or_wrong_width_input():
    data = np.arange(48, dtype=float).reshape(16, 3)
    detector = _memstream(data)
    with pytest.raises(ValueError, match="expected one raw record"):
        detector.ingest(np.zeros((1, 3)), 0, 0)
    with pytest.raises(ValueError, match="expected one raw record"):
        detector.ingest(np.zeros(2), 0, 0)

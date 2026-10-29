import numpy as np
import pytest

from bench.smd_feature_relations import permutation_p, spearman
from bench.stream_smd import smd_stream
from uengram import smd


def test_label_runs_returns_inclusive_maximal_runs():
    label = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1], dtype=bool)
    assert smd.label_runs(label) == [(1, 2), (4, 4), (7, 9)]


@pytest.mark.parametrize("label, expected", [
    (np.zeros(6, dtype=bool), []),
    (np.ones(6, dtype=bool), [(0, 5)]),
    (np.array([1, 0, 0, 0, 0, 1], dtype=bool), [(0, 0), (5, 5)]),
])
def test_label_runs_edges(label, expected):
    assert smd.label_runs(label) == expected


def _write_machine(root, name, n_train=100, n_test=60, anomalous=(25, 26, 27, 40)):
    rng = np.random.default_rng(0)
    for kind, n in (("train", n_train), ("test", n_test)):
        (root / kind).mkdir(parents=True, exist_ok=True)
        np.savetxt(root / kind / f"{name}.txt", rng.random((n, 38)), delimiter=",", fmt="%.6f")
    (root / "test_label").mkdir(parents=True, exist_ok=True)
    label = np.zeros(n_test, dtype=int)
    label[list(anomalous)] = 1
    np.savetxt(root / "test_label" / f"{name}.txt", label, fmt="%d")


def test_stream_windows_never_span_a_split_and_events_use_the_test_grid(tmp_path):
    _write_machine(tmp_path, "machine-x-1")
    s = smd_stream("machine-x-1", root=tmp_path, K=10, fit_frac=0.7)
    # 100 train rows -> 70 fit + 30 cal; 60 test rows; K = 10 -> 7 / 3 / 6 windows, none shared
    assert (int(s.fit.sum()), int(s.cal.sum()), int(s.test.sum())) == (7, 3, 6)
    assert (s.fit.astype(int) + s.cal.astype(int) + s.test.astype(int) == 1).all()
    # 1-minute grid continues from train into test: test row r is at train_len + r minutes
    first_test = s.start[s.test][0]
    assert first_test == smd.EPOCH + np.timedelta64(100, "m")
    # runs (25-27) and (40) become two events at rows 125-127 and 140
    expected = ((str((smd.EPOCH + np.timedelta64(125, "m")).astype("datetime64[s]")),
                 str((smd.EPOCH + np.timedelta64(127, "m")).astype("datetime64[s]"))),
                (str((smd.EPOCH + np.timedelta64(140, "m")).astype("datetime64[s]")),) * 2)
    assert s.events == expected
    # a window is an event window iff it overlaps an event interval
    eid = s.event_id()
    assert set(np.flatnonzero(eid[s.test] >= 0)) == {2, 4}  # test windows 20-29 and 40-49


def test_spearman_handles_ties_and_direction():
    x = np.array([1.0, 2.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    expected = np.corrcoef([1.0, 2.5, 2.5, 4.0], [1.0, 2.0, 3.0, 4.0])[0, 1]
    assert spearman(x, y) == pytest.approx(expected)
    assert spearman(y, -y) == pytest.approx(-1.0)


def test_permutation_p_separates_a_real_relation_from_noise():
    rng = np.random.default_rng(3)
    x = rng.normal(size=40)
    assert permutation_p(x, x + 0.1 * rng.normal(size=40), n=2000) < 0.01
    assert permutation_p(x, rng.normal(size=40), n=2000) > 0.05

import numpy as np

from uengram import metrics


def test_auroc_matches_bruteforce_with_ties():
    rng = np.random.default_rng(0)
    s = rng.integers(0, 6, size=80).astype(float)
    y = rng.random(80) < 0.3
    pos, neg = s[y], s[~y]
    tot = 0.0
    for p in pos:
        tot += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    assert abs(metrics.auroc(s, y) - tot / (len(pos) * len(neg))) < 1e-12


def test_auroc_single_class_nan():
    assert np.isnan(metrics.auroc([1.0, 2.0], [True, True]))


def test_average_precision_hand_case():
    ap = metrics.average_precision([0.9, 0.8, 0.7], [True, False, True])
    assert abs(ap - (1 + 2 / 3) / 2) < 1e-12


def test_mmd_identical_zero_and_pointset_vstat():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(40, 3))
    b = rng.normal(0.5, 1.0, size=(30, 3))
    z3, z4 = np.zeros_like(a), np.zeros_like(b)
    w3, w4 = np.ones(40), np.ones(30)
    assert abs(metrics.gauss_mixture_mmd2(w3, a, z3, w3, a, z3, 1.3)) < 1e-10
    h = 1.3

    def k(x, y):
        d2 = ((x[:, None] - y[None]) ** 2).sum(-1)
        return np.exp(-d2 / (2 * h * h))

    direct = k(a, a).mean() + k(b, b).mean() - 2 * k(a, b).mean()
    got = metrics.gauss_mixture_mmd2(w3, a, z3, w4, b, z4, h)
    assert abs(got - direct) < 1e-10


def test_mmd_gaussian_component_vs_samples():
    # a single Gaussian component ~ its large sample cloud
    rng = np.random.default_rng(2)
    s = rng.normal(0.0, 0.7, size=(4000, 2))
    got = metrics.gauss_mixture_mmd2(
        np.ones(1), np.zeros((1, 2)), np.full((1, 2), 0.49),
        np.ones(4000), s, np.zeros((4000, 2)), 1.0,
    )
    assert abs(got) < 5e-3


def test_bootstrap_ci_contains_mean_and_paired():
    v = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    m, lo, hi = metrics.bootstrap_ci(v)
    assert m == 2.5 and lo <= m <= hi
    m, lo, hi = metrics.paired_bootstrap_diff([2, 3, 4, 5], [1, 2, 3, 4])
    assert m == 1.0 and lo == 1.0 and hi == 1.0


def test_median_heuristic_simple():
    x = np.array([[0.0], [1.0], [3.0]])
    assert metrics.median_heuristic(x) == 2.0

"""Evaluation metrics: AUROC, AP, mixture MMD, bootstrap CIs."""
from __future__ import annotations

import numpy as np


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties share the mean rank."""
    order = np.argsort(a, kind="stable")
    sa = a[order]
    ranks = np.empty(len(a), dtype=np.float64)
    n = len(a)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def auroc(scores, labels_bool) -> float:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels_bool, dtype=bool)
    ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _rankdata(s)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(scores, labels_bool) -> float:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels_bool, dtype=bool)
    ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    yy = y[order]
    tp = np.cumsum(yy)
    # A threshold selects the whole equal-score group, never an arbitrary
    # prefix ordered by time or labels.
    ends = np.r_[np.flatnonzero(s[order][1:] != s[order][:-1]), len(s) - 1]
    gains = np.diff(np.r_[0, tp[ends]])
    return float(np.sum(gains * tp[ends] / (ends + 1)) / n_pos)


def _norm_w(w) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    return w / w.sum()


def _cross_kernel_points(w1, mu1, w2, mu2, h, chunk=2048) -> float:
    """Zero-variance fast path: RBF Gram blocks via BLAS."""
    sq1, sq2 = np.sum(mu1 * mu1, axis=1), np.sum(mu2 * mu2, axis=1)
    total = 0.0
    for i in range(0, mu1.shape[0], chunk):
        for j in range(0, mu2.shape[0], chunk):
            d2 = sq1[i : i + chunk, None] + sq2[None, j : j + chunk] - 2.0 * mu1[i : i + chunk] @ mu2[j : j + chunk].T
            total += float(w1[i : i + chunk] @ np.exp(-np.maximum(d2, 0.0) / (2.0 * h * h)) @ w2[j : j + chunk])
    return total


def _cross_kernel(w1, mu1, v1, w2, mu2, v2, h, chunk=512) -> float:
    h2 = h * h
    if not v1.any() and not v2.any():
        return _cross_kernel_points(w1, mu1, w2, mu2, h)
    total = 0.0
    for i in range(0, mu1.shape[0], chunk):
        m1, a1, w1c = mu1[i : i + chunk], v1[i : i + chunk], w1[i : i + chunk]
        for j in range(0, mu2.shape[0], chunk):
            m2, a2, w2c = mu2[j : j + chunk], v2[j : j + chunk], w2[j : j + chunk]
            den = h2 + a1[:, None, :] + a2[None, :, :]
            diff = m1[:, None, :] - m2[None, :, :]
            logk = 0.5 * np.sum(np.log(h2 / den), axis=2) - np.sum(diff * diff / (2.0 * den), axis=2)
            total += float(w1c @ np.exp(logk) @ w2c)
    return total


def gauss_mixture_mmd2(a_w, a_mu, a_var, b_w, b_mu, b_var, h) -> float:
    """Exact squared MMD between weighted diagonal-Gaussian mixtures, RBF kernel."""
    a_w, b_w = _norm_w(a_w), _norm_w(b_w)
    a_mu, b_mu = np.asarray(a_mu, np.float64), np.asarray(b_mu, np.float64)
    a_var, b_var = np.asarray(a_var, np.float64), np.asarray(b_var, np.float64)
    # chunk so the [c,c,d] temporary stays small (d may be large)
    d = a_mu.shape[1]
    chunk = max(8, min(512, int(1.2e7 // (512 * max(d, 1)) * 8)))
    kaa = _cross_kernel(a_w, a_mu, a_var, a_w, a_mu, a_var, h, chunk)
    kbb = _cross_kernel(b_w, b_mu, b_var, b_w, b_mu, b_var, h, chunk)
    kab = _cross_kernel(a_w, a_mu, a_var, b_w, b_mu, b_var, h, chunk)
    return float(kaa + kbb - 2.0 * kab)


def median_heuristic(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] > 2000:
        idx = np.random.default_rng(0).choice(x.shape[0], 2000, replace=False)
        x = x[idx]
    sq = np.sum(x * x, axis=1)
    dists = []
    for i in range(0, x.shape[0], 512):
        d2 = sq[i : i + 512, None] + sq[None, :] - 2.0 * x[i : i + 512] @ x.T
        dists.append(np.sqrt(np.maximum(d2, 0.0)))
    dm = np.concatenate(dists, axis=0)
    iu = np.triu_indices(dm.shape[0], k=1)
    return float(np.median(dm[iu]))


def bootstrap_ci(values, n_boot=2000, seed=12345, level=0.95):
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    q = (1.0 - level) / 2.0
    return float(v.mean()), float(np.quantile(means, q)), float(np.quantile(means, 1.0 - q))


def paired_bootstrap_diff(a, b, n_boot=2000, seed=12345, level=0.95):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = (a - b)[ok]
    return bootstrap_ci(d, n_boot=n_boot, seed=seed, level=level)

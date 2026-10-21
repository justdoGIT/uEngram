"""Statistics primitives: chi-square tail, EW moments, merges, divergences."""
from __future__ import annotations

import math

import numpy as np

_LN10 = math.log(10.0)


def _logsumexp(a: np.ndarray, axis: int = 0) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def chi2_logsf(x: float | np.ndarray, d: int) -> np.ndarray:
    """log P(chi2_d > x), exact for even d.

    Q(d/2, x/2) = exp(-x/2) * sum_{k<d/2} (x/2)^k / k!
    """
    if d % 2 != 0 or d <= 0:
        raise ValueError("d must be even for exact chi2 tail")
    x = np.asarray(x, dtype=np.float64)
    h = np.maximum(x, 0.0) / 2.0
    ks = np.arange(d // 2, dtype=np.float64)
    lgam = np.array([math.lgamma(k + 1.0) for k in ks])
    with np.errstate(divide="ignore", invalid="ignore"):
        logh = np.log(h)
        # k*log(h): define 0*log(0)=0 so the k=0 term is 0 at h=0
        terms = np.where(
            ks.reshape((-1,) + (1,) * h.ndim) == 0,
            0.0,
            ks.reshape((-1,) + (1,) * h.ndim) * logh[None, ...],
        ) - lgam.reshape((-1,) + (1,) * h.ndim)
    out = -h + _logsumexp(terms, axis=0)
    return np.minimum(out, 0.0)


def chi2_isf(alpha: float, d: int) -> float:
    """x such that P(chi2_d > x) = alpha, by bisection."""
    target = math.log(alpha)
    lo, hi = 0.0, 10.0 * d + 200.0
    while hi - lo > 1e-10:
        mid = 0.5 * (lo + hi)
        if float(chi2_logsf(mid, d)) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def surprisal(x: float | np.ndarray, d: int) -> float:
    """-log10 chi2 tail probability (decades), capped at 300."""
    v = -float(chi2_logsf(x, d)) / _LN10
    return min(v, 300.0)


def robust_noise_sigma(z: np.ndarray, lag: int, groups: np.ndarray | None) -> np.ndarray:
    """Per-dimension noise std from MAD of lag-`lag` successive differences."""
    z = np.asarray(z, dtype=np.float64)
    delta = z[lag:] - z[:-lag]
    if groups is not None:
        keep = groups[lag:] == groups[:-lag]
        delta = delta[keep]
    if delta.shape[0] < 10:
        raise ValueError("fewer than 10 difference pairs for noise estimate")
    med = np.median(delta, axis=0)
    mad = np.median(np.abs(delta - med), axis=0)
    return np.maximum(1.4826 * mad / math.sqrt(2.0), 1e-6)


def ew_update(mu, S, x, alpha):
    """Finch exponentially weighted mean/variance update."""
    delta = x - mu
    mu_new = mu + alpha * delta
    S_new = (1.0 - alpha) * (S + alpha * delta * delta)
    return mu_new, S_new


def shrunk_var(S, n_eff, sigma0_sq, kappa0):
    return (kappa0 * sigma0_sq + n_eff * S) / (kappa0 + n_eff)


def merge_moments(n1, mu1, var1, n2, mu2, var2):
    n = n1 + n2
    mu = (n1 * mu1 + n2 * mu2) / n
    var = (n1 * (var1 + mu1 * mu1) + n2 * (var2 + mu2 * mu2)) / n - mu * mu
    return n, mu, np.maximum(var, 0.0)


def bhattacharyya_diag(mu1, var1, mu2, var2) -> float:
    vb = 0.5 * (var1 + var2)
    return float(
        0.125 * np.sum((mu1 - mu2) ** 2 / vb)
        + 0.5 * np.sum(np.log(vb / np.sqrt(var1 * var2)))
    )


def bhattacharyya_pairwise(mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    n = mu.shape[0]
    out = np.empty((n, n))
    for i in range(n):
        vb = 0.5 * (var[i][None, :] + var)
        t1 = 0.125 * np.sum((mu[i][None, :] - mu) ** 2 / vb, axis=1)
        t2 = 0.5 * np.sum(np.log(vb / np.sqrt(var[i][None, :] * var)), axis=1)
        out[i] = t1 + t2
    np.fill_diagonal(out, np.inf)
    return out


def ward_cost_pairwise(n: np.ndarray, mu: np.ndarray, sigma0_sq: np.ndarray) -> np.ndarray:
    n = np.asarray(n, dtype=np.float64)
    w = mu / np.sqrt(sigma0_sq)
    sq = np.sum(w * w, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (w @ w.T), 0.0)
    f = n[:, None] * n[None, :] / (n[:, None] + n[None, :])
    out = f * d2
    np.fill_diagonal(out, np.inf)
    return out

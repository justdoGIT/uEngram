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


def conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    """Score threshold from a calibration set of normal scores.

    With m calibration scores, alarming when a new score exceeds the
    ceil((1-alpha)(m+1))-th smallest one gives P(false alarm) <= alpha for an
    exchangeable new normal score (split-conformal p-value <= alpha).
    Returns +inf when m is too small for the requested alpha.
    """
    s = np.sort(np.asarray(scores, dtype=np.float64)[~np.isnan(scores)])
    m = len(s)
    k = int(math.ceil((1.0 - alpha) * (m + 1)))
    return float(s[k - 1]) if k <= m else float("inf")


def _diff_pairs(z: np.ndarray, lag: int, groups: np.ndarray | None) -> np.ndarray:
    """Lag-`lag` differences, keeping only pairs inside the same group."""
    z = np.asarray(z, dtype=np.float64)
    delta = z[lag:] - z[:-lag]
    if groups is not None:
        delta = delta[groups[lag:] == groups[:-lag]]
    return delta


def robust_noise_sigma(z: np.ndarray, lag: int, groups: np.ndarray | None) -> np.ndarray:
    """Per-dimension noise std from MAD of lag-`lag` successive differences."""
    delta = _diff_pairs(z, lag, groups)
    if delta.shape[0] < 10:
        raise ValueError("fewer than 10 difference pairs for noise estimate")
    med = np.median(delta, axis=0)
    mad = np.median(np.abs(delta - med), axis=0)
    return np.maximum(1.4826 * mad / math.sqrt(2.0), 1e-6)


def _segments(z: np.ndarray, groups: np.ndarray | None, alpha: float = 1e-3) -> np.ndarray:
    """Group ids that also break at abrupt changes (state switches, steps,
    anomaly onsets): lag-1 differences whose Mahalanobis distance under the
    robust noise covariance exceeds the chi2_d quantile at alpha."""
    g = np.zeros(len(z), dtype=np.int64) if groups is None else np.asarray(groups)
    delta = (z[1:] - z[:-1]) / math.sqrt(2.0)
    d2 = np.einsum("ij,jk,ik->i", delta, np.linalg.inv(robust_noise_cov(z, 1, groups)), delta)
    brk = (d2 > chi2_isf(alpha, z.shape[1])) | (g[1:] != g[:-1])
    return np.r_[0, np.cumsum(brk)]


def ewma_inflation(z: np.ndarray, lam: float, groups: np.ndarray | None) -> np.ndarray:
    """Per-dimension variance inflation of an EWMA(lam) of z over independent
    samples: Var(EWMA) / (lam / (2 - lam) Var(z)); 1 for white noise, larger
    when slow within-state variation correlates successive samples.

    Measured within segments free of abrupt changes (_segments), from robust
    lag-L differences with L twice the EWMA memory b = (2 - lam) / lam. The
    EWMA restarts in each segment and its first b samples are skipped."""
    z = np.asarray(z, dtype=np.float64)
    b = (2.0 - lam) / lam
    lag = int(math.ceil(2.0 * b))
    g = _segments(z, groups)
    y = np.empty_like(z)
    age = np.empty(len(z), dtype=np.int64)
    for t in range(len(z)):
        if t == 0 or g[t] != g[t - 1]:
            y[t], age[t] = z[t], 0
        else:
            y[t], age[t] = lam * z[t] + (1.0 - lam) * y[t - 1], age[t - 1] + 1
    warm = age >= b
    var_y = robust_noise_sigma(y[warm], lag, g[warm]) ** 2
    var_z = robust_noise_sigma(z, lag, g) ** 2
    return var_y * b / var_z


def robust_noise_cov(z: np.ndarray, lag: int, groups: np.ndarray | None,
                     trim: float = 0.975, iters: int = 5) -> np.ndarray:
    """Full within-state noise covariance from lag differences, robust to the
    jumps caused by regime switches, steps and anomalies.

    Delta = (z[t+lag] - z[t]) / sqrt(2) has the noise covariance when the state
    is unchanged. Reweighted trimming: keep differences with Mahalanobis
    distance <= the chi2_d `trim` quantile, re-estimate, repeat; then rescale by
    trim / P(chi2_{d+2} <= q), the Gaussian consistency factor for trimming.
    """
    D = _diff_pairs(z, lag, groups) / math.sqrt(2.0)
    n, d = D.shape
    if n < max(10, 2 * d):
        raise ValueError("too few difference pairs for noise covariance")
    q = chi2_isf(1.0 - trim, d)
    cons = trim / (1.0 - math.exp(float(chi2_logsf(q, d + 2))))
    s = robust_noise_sigma(z, lag, groups)
    C = np.diag(s * s)
    for _ in range(iters):
        d2 = np.einsum("ij,jk,ik->i", D, np.linalg.inv(C), D)
        C = np.cov(D[d2 <= q], rowvar=False) * cons
    return C


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

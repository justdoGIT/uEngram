"""Fail-soft noise estimation: a labeled ladder of fallbacks in place of a hard error.

The strict estimators in ``stats`` raise ``NoiseEstimateError`` (a ``ValueError``) when the data
cannot support them: too few within-group difference pairs, or a jump detector that fragments the
stream so that no segment is long enough. That is the right behaviour for a pre-fit check, but it
turns a usable-but-degraded encoder into no encoder. Every function here first calls the strict
estimator and returns its answer
unchanged when it succeeds (quality ``robust`` / ``exact``), so nothing that worked before changes.
Only when the strict call raises does it step down a rung, emit a ``RuntimeWarning`` naming the rung
and the reason, and return the rung's label so the caller can record it. Run with
``-W error::RuntimeWarning`` to get the old strict behaviour back.

Noise covariance / sigma rungs, best to worst (``QUALITY``):
  robust     the strict estimate, differences taken inside groups only
  ungrouped  the strict estimate with group boundaries ignored; the trimmed estimator is built to
             tolerate the regime jumps that a boundary difference contributes
  shrunk     winsorised sample covariance of all differences, shrunk toward the diagonal MAD variance
             with weight d / (n + d); used when there are fewer than 2d pairs or trimming rejects them

Step-inflation rungs (``INFLATION_QUALITY``):
  exact           the strict estimate, segments from jump detection
  group_segments  segments are the caller's groups; used when jump detection fragments the stream
  assumed_white   inflation 1 (innovations assumed independent); used only when even the groups hold
                  too few samples. This can over-trigger the step detector on autocorrelated latents.

Refused, not degraded: fewer than 10 difference pairs even ignoring groups, and a noise scale at the
numerical floor (a noiseless signal). Whitening by that floor would amplify by 1e6.
"""
from __future__ import annotations

import math
import warnings

import numpy as np

from .stats import NoiseEstimateError, _diff_pairs, ewma_inflation, robust_noise_cov, robust_noise_sigma

QUALITY = ("robust", "ungrouped", "shrunk")
INFLATION_QUALITY = ("exact", "group_segments", "assumed_white")
_SIGMA_FLOOR = 1e-6  # the floor robust_noise_sigma applies


def worst_quality(*qualities: str) -> str:
    return max(qualities, key=QUALITY.index)


def _degraded(what: str, rung: str, reason: str) -> None:
    warnings.warn(f"{what} degraded to '{rung}': {reason}", RuntimeWarning, stacklevel=3)


def soft_noise_sigma(z: np.ndarray, lag: int, groups: np.ndarray | None):
    """(per-dimension noise std, quality) with the ``robust`` then ``ungrouped`` rungs."""
    try:
        return robust_noise_sigma(z, lag, groups), "robust"
    except NoiseEstimateError as strict:
        if groups is None:
            raise
        sigma = robust_noise_sigma(z, lag, None)  # raises if there is no data at all
        _degraded("noise sigma", "ungrouped", str(strict))
        return sigma, "ungrouped"


def soft_noise_cov(z: np.ndarray, lag: int, groups: np.ndarray | None,
                   trim: float = 0.975, iters: int = 5):
    """(noise covariance, quality) with the ``robust``, ``ungrouped`` and ``shrunk`` rungs."""
    first_reason = None
    for rung, g in (("robust", groups), ("ungrouped", None)):
        if rung == "ungrouped" and groups is None:
            break
        try:
            cov = robust_noise_cov(z, lag, g, trim, iters)
        except NoiseEstimateError as exc:
            first_reason = first_reason or str(exc)
            continue
        if rung != "robust":
            _degraded("noise covariance", rung, first_reason)
        return cov, rung
    D = _diff_pairs(z, lag, None) / math.sqrt(2.0)
    n, d = D.shape
    if n < 10:
        raise NoiseEstimateError("fewer than 10 difference pairs for noise estimate even ignoring groups")
    s = robust_noise_sigma(z, lag, None)
    if np.any(s <= _SIGMA_FLOOR * (1.0 + 1e-9)):
        raise NoiseEstimateError(
            "noise scale at the numerical floor: the signal has no within-state noise "
            "to estimate, and whitening by the floor would amplify by 1e6")
    centre = np.median(D, axis=0)
    winsorised = np.clip(D, centre - 4.0 * s, centre + 4.0 * s)  # jumps and anomalies capped at 4 sigma
    shrink = d / (n + d)
    cov = (1.0 - shrink) * np.atleast_2d(np.cov(winsorised, rowvar=False)) + shrink * np.diag(s * s)
    _degraded("noise covariance", "shrunk", first_reason or "too few difference pairs")
    return cov, "shrunk"


def soft_ewma_inflation(z: np.ndarray, lam: float, groups: np.ndarray | None):
    """(per-dimension EWMA variance inflation, quality) with all three step-inflation rungs."""
    try:
        return ewma_inflation(z, lam, groups), "exact"
    except NoiseEstimateError as strict:
        reason = str(strict)
    z = np.asarray(z, dtype=np.float64)
    if groups is None:
        segments = np.zeros(len(z), dtype=np.int64)
    else:
        g = np.asarray(groups)
        segments = np.r_[0, np.cumsum(g[1:] != g[:-1])]  # contiguous run ids
    try:
        inflation = ewma_inflation(z, lam, groups, segments=segments)
    except NoiseEstimateError as exc:
        _degraded("step inflation", "assumed_white", f"{reason}; groups too short as well ({exc})")
        return np.ones(z.shape[1]), "assumed_white"
    _degraded("step inflation", "group_segments", reason)
    return inflation, "group_segments"

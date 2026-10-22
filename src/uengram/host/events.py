"""Exact tail test for rare error counters (numpy + math).

An error counter (OOM kills, disk I/O errors, NIC drops/errors, thermal
throttle events) is zero in almost every window. Under a constant-rate model
the count per window is Poisson(lambda); with the Jeffreys prior
Gamma(1/2, 0) and n commissioning windows holding s events, the predictive
distribution of the next window's count is negative binomial with
r = s + 1/2 and success probability q = n / (n + 1). The upper-tail mid-p
value  P(X > k) + P(X = k)/2  (Lancaster 1961) is the score; P(X > k) is the
regularised incomplete beta I_{1-q}(k+1, r).

The posterior is frozen after commissioning: an error counter that starts
firing should keep alarming, not become normal.
"""
from __future__ import annotations

import math

import numpy as np

from . import schema


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (modified Lentz)."""
    tiny, eps = 1e-300, 3e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 10000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def nb_midp(k: float, r: float, q: float) -> float:
    """Upper-tail mid-p of NegBin(r, q) at k: P(X > k) + P(X = k) / 2."""
    k = int(round(k))
    log_pmf = (math.lgamma(k + r) - math.lgamma(k + 1) - math.lgamma(r)
               + r * math.log(q) + k * math.log1p(-q))
    return betainc(k + 1.0, r, 1.0 - q) + 0.5 * math.exp(log_pmf)


class EventModel:
    def __init__(self):
        self.names = [a.name for a in schema.event_attributes()]

    def fit(self, E: np.ndarray) -> "EventModel":
        E = np.asarray(E, dtype=np.float64)
        ok = np.isfinite(E).all(axis=1)
        self.n = int(ok.sum())
        self.r = 0.5 + E[ok].sum(axis=0)
        self.q = self.n / (self.n + 1.0)
        return self

    def pvalues(self, counts: np.ndarray) -> np.ndarray:
        """Mid-p per counter; NaN where the count is missing (counter reset)."""
        return np.array([nb_midp(k, r, self.q) if np.isfinite(k) else np.nan
                         for k, r in zip(counts, self.r)])

    def report(self) -> dict:
        return {"windows": self.n, "rate_per_window": {n: float((r - 0.5) / self.n)
                                                       for n, r in zip(self.names, self.r)}}

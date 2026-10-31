"""Remaining useful life from the health score: a binned lookup table.

The table is fitted offline on run-to-failure data (each window's health score
and its RUL, known once the unit has failed) and shipped to the device as at
most `bins` (centre, value) pairs. Bins are quantiles of log health, so each
holds the same number of fitting windows; prediction interpolates linearly
between bin centres and holds the end values outside them."""
from __future__ import annotations

import numpy as np


class HealthRUL:
    def __init__(self, bins: int = 20):
        self.bins = bins

    @staticmethod
    def _log(health) -> np.ndarray:
        return np.log(np.maximum(np.asarray(health, dtype=np.float64), 1e-12))

    def fit(self, health, rul) -> "HealthRUL":
        lh, rul = self._log(health), np.asarray(rul, dtype=np.float64)
        ok = np.isfinite(lh) & np.isfinite(rul)
        lh, rul = lh[ok], rul[ok]
        if len(lh) == 0:
            raise ValueError("no finite (health, rul) pairs to fit")
        # tied health values can merge quantile edges; empty bins are dropped
        edges = np.unique(np.quantile(lh, np.linspace(0.0, 1.0, self.bins + 1)))
        nb = max(len(edges) - 1, 1)
        idx = np.clip(np.searchsorted(edges, lh, side="right") - 1, 0, nb - 1)
        used = np.unique(idx)
        self.centre = np.array([lh[idx == b].mean() for b in used])
        self.value = np.array([rul[idx == b].mean() for b in used])
        return self

    def predict(self, health) -> np.ndarray:
        """RUL for each health score; NaN where the score is NaN."""
        lh = self._log(health)
        out = np.interp(lh, self.centre, self.value)
        return np.where(np.isnan(lh), np.nan, out)

    def table_bytes(self) -> int:
        return len(self.centre) * 2 * 4


class HealthAgeRUL:
    """RUL from the health score and the unit's age (cycles observed so far).

    Health alone saturates: units at the same health can be early or late in life. A quadratic
    ridge regression on (log health, age) fitted offline on run-to-failure windows keeps six
    float32 coefficients (24 bytes). Predictions are clipped to [0, cap]."""
    AGE_SCALE = 100.0

    def __init__(self, ridge: float = 1e-3, cap: float = 125.0):
        self.ridge, self.cap = ridge, cap

    @classmethod
    def _design(cls, health, age) -> np.ndarray:
        lh = HealthRUL._log(health)
        a = np.asarray(age, dtype=np.float64) / cls.AGE_SCALE
        return np.column_stack([np.ones(len(lh)), lh, lh ** 2, a, lh * a, a ** 2])

    def fit(self, health, age, rul) -> "HealthAgeRUL":
        A, rul = self._design(health, age), np.asarray(rul, dtype=np.float64)
        ok = np.isfinite(A).all(axis=1) & np.isfinite(rul)
        if not ok.any():
            raise ValueError("no finite (health, age, rul) rows to fit")
        A, rul = A[ok], rul[ok]
        self.coef = np.linalg.solve(A.T @ A + self.ridge * np.eye(A.shape[1]), A.T @ rul)
        return self

    def predict(self, health, age) -> np.ndarray:
        """Clipped RUL for each (health, age); NaN where the health score is NaN."""
        out = np.clip(self._design(health, age) @ self.coef, 0.0, self.cap)
        return np.where(np.isnan(np.asarray(health, dtype=np.float64)), np.nan, out)

    def table_bytes(self) -> int:
        return len(self.coef) * 4

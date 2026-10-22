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

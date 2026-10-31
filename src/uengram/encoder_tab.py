"""Encoder for rows that are already window summaries (e.g. 10-minute SCADA
averages in CARE to Compare): robust per-feature standardisation, then the
same PCA and noise whitening as the DSP encoder (encoders.fit_projection).

- Location and scale are the commissioning median and IQR/1.349 (standard
  deviation where the IQR is zero); features constant during commissioning
  carry no information about normal variation and are dropped.
- Missing values are imputed with the commissioning median, so a missing
  sensor reads as typical rather than extreme.
- Standardised values are clipped to +-50: operators log missing values as 0
  in some farms, and a single impossible reading must not dominate PCA.
"""
from __future__ import annotations

import numpy as np

from .encoders import fit_projection

CLIP = 50.0


class TabularEncoder:
    def __init__(self, d: int = 16, rcond: float = 1e-4):
        self.d = d
        self.rcond = rcond

    def _std(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)[:, self.keep]
        X = np.where(np.isfinite(X), X, self.med)
        return np.clip((X - self.med) / self.scale, -CLIP, CLIP)

    def fit(self, train_rows, noise_lag: int, groups: np.ndarray | None) -> "TabularEncoder":
        X = np.asarray(train_rows, dtype=np.float64)
        med = np.nanmedian(X, axis=0)
        q75, q25 = np.nanpercentile(X, [75, 25], axis=0)
        scale = (q75 - q25) / 1.349
        sd = np.nanstd(X, axis=0)
        scale = np.where(scale > 1e-9, scale, sd)
        self.keep = np.flatnonzero(np.isfinite(med) & (scale > 1e-9))
        if len(self.keep) < self.d:
            raise ValueError(f"only {len(self.keep)} informative features for d={self.d}")
        self.med, self.scale = med[self.keep], scale[self.keep]
        self.whitener, self.noise, self.sigma0 = fit_projection(
            self._std(X), self.d, noise_lag, groups, rcond=self.rcond)
        return self

    def encode(self, rows, sequential: bool = True) -> np.ndarray:
        """`sequential` is accepted for protocol compatibility; rows are independent."""
        return self.noise.transform(self.whitener.transform(self._std(rows)))

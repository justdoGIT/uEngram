"""DSP encoder with first-order-lag residuals for slowly responding channels.

A channel that follows its operating point with a first-order lag (coolant
temperature after a speed change) ramps for several windows after every regime
switch. Raw spectral features make each ramp look novel. This encoder replaces
such a channel's six DSP features by

* the residual of its within-window ramp against a lag model that uses the
  previous window:  ramp_t = b0 ramp_{t-1} + b1 du + b2 d(u^2),
  where u is the window mean of the drive channel (rpm). With
  ramp_t = (1-a) ramp_{t-1} + a [T_eq(u_t) - T_eq(u_{t-1})] and quadratic T_eq,
  b0 = 1-a and (b1, b2) are a times the T_eq coefficients; slow wear cancels
  between consecutive windows;
* the log std of the channel after removing its within-window linear trend.

Encoding is sequential. A window whose residual exceeds `reject_k` robust
scales is treated as anomalous: it does not update the context, which is
propagated by the model instead (Kalman-style innovation gating), so one fault
window does not corrupt the prediction for the next one.
"""
from __future__ import annotations

import numpy as np

from .encoders import DSPEncoder, band_features

TRIM_ITERS = 6
TRIM_K = 4.0


class ContextEncoder(DSPEncoder):
    def __init__(self, lag_channels=(2,), drive_channel: int = 0, d: int = 16, reject_k: float = 5.0):
        super().__init__(d)
        self.lag_channels = tuple(lag_channels)
        self.drive_channel = drive_channel
        self.reject_k = reject_k

    def _parts(self, windows):
        x = np.asarray(windows, dtype=np.float64)
        K = x.shape[2]
        A = np.stack([np.ones(K), np.arange(K) / K], 1)
        drive = x[:, self.drive_channel, :].mean(1)
        ramps, logstd = [], []
        for c in self.lag_channels:
            coef, *_ = np.linalg.lstsq(A, x[:, c, :].T, rcond=None)
            ramps.append(coef[1] * (K - 1) / K)
            logstd.append(np.log((x[:, c, :] - (A @ coef).T).std(1) + 1e-6))
        return band_features(windows), drive, np.stack(ramps, 1), np.stack(logstd, 1)

    @staticmethod
    def _design(prev_ramp, prev_u, u):
        return np.stack([prev_ramp, np.broadcast_to(u - prev_u, prev_ramp.shape),
                         np.broadcast_to(u * u - prev_u * prev_u, prev_ramp.shape)], -1)

    def _residuals(self, drive, ramps, sequential: bool) -> np.ndarray:
        if not sequential:  # independent windows: assume steady state (predicted ramp 0)
            return ramps.copy()
        r = np.empty_like(ramps)
        prev_ramp, prev_u = ramps[0].copy(), drive[0]
        for t in range(len(drive)):
            pred = np.einsum("ck,ck->c", self._design(prev_ramp, prev_u, drive[t]), self.b)
            r[t] = ramps[t] - pred
            if np.any(np.abs(r[t]) > self.reject_k * self.rscale):
                prev_ramp = self.b[:, 0] * prev_ramp
                prev_u = drive[t]
            else:
                prev_ramp, prev_u = ramps[t].copy(), drive[t]
        return r

    def _raw(self, windows, sequential: bool) -> np.ndarray:
        F, drive, ramps, logstd = self._parts(windows)
        drop = set(self.lag_channels)
        keep = [i for i in range(F.shape[1]) if i // 6 not in drop]
        return np.column_stack([F[:, keep], self._residuals(drive, ramps, sequential), logstd])

    def fit(self, train_windows, noise_lag: int, groups: np.ndarray | None) -> "ContextEncoder":
        _, drive, ramps, _ = self._parts(train_windows)
        prev_ramp = np.vstack([ramps[:1], ramps[:-1]])
        prev_u = np.r_[drive[:1], drive[:-1]]
        b, rscale = [], []
        for j in range(ramps.shape[1]):
            X = self._design(prev_ramp[:, j], prev_u, drive)
            y = ramps[:, j]
            keep = np.ones(len(y), bool)
            for _ in range(TRIM_ITERS):  # trimmed least squares: drop anomalies
                coef, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
                res = y - X @ coef
                scale = 1.4826 * np.median(np.abs(res[keep]))
                keep = np.abs(res) < TRIM_K * scale
            b.append(coef)
            rscale.append(max(scale, 1e-9))
        self.b, self.rscale = np.array(b), np.array(rscale)
        return super().fit(train_windows, noise_lag, groups)  # standardise, PCA, noise-whiten

    def lag_coefficient(self, j: int = 0) -> float:
        """Estimated first-order lag a for lag channel j (b0 = 1 - a)."""
        return float(1.0 - self.b[j, 0])

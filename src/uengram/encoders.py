"""Window encoders: PCA whitening, noise whitening, and the DSP+PCA encoder."""
from __future__ import annotations

import warnings

import numpy as np

from .noise_fallback import soft_noise_cov, soft_noise_sigma, worst_quality


class Whitener:
    """Centre, rotate onto the top-d principal axes, scale to unit variance."""

    def fit(self, F: np.ndarray, d: int) -> "Whitener":
        F = np.asarray(F, dtype=np.float64)
        self.m = F.mean(axis=0)
        lam, vec = np.linalg.eigh(np.cov(F, rowvar=False))
        order = np.argsort(lam)[::-1][:d]
        lam, vec = lam[order], vec[:, order]
        lam = np.maximum(lam, 1e-8 * max(float(lam.max()), 1e-12))
        self.V = vec
        self.lam = lam
        return self

    def transform(self, F: np.ndarray) -> np.ndarray:
        return (np.asarray(F, dtype=np.float64) - self.m) @ self.V / np.sqrt(self.lam)


class NoiseWhitener:
    """Make within-state noise uncorrelated with unit variance.

    PCA whitening decorrelates the *total* spread, which is dominated by
    differences between operating states. Noise inside a state can still be
    correlated across dimensions (one physical fluctuation moving several
    features), and a per-dimension distance would then count it several
    times. With N the robust noise covariance from successive differences,
    z -> z N^{-1/2} (symmetric root, minimal rotation) counts each correlated
    fluctuation once and amplifies combinations that are normally consistent.
    """
    def __init__(self, rcond: float = 1e-12):
        self.rcond = rcond
        self.quality = "robust"  # noise_fallback.QUALITY rung of the fitted estimate

    def fit(self, z: np.ndarray, noise_lag: int, groups: np.ndarray | None) -> "NoiseWhitener":
        cov, self.quality = soft_noise_cov(z, noise_lag, groups)
        lam, V = np.linalg.eigh(cov)
        lam = np.maximum(lam, self.rcond * max(float(lam.max()), 1e-12))
        self.W = (V / np.sqrt(lam)) @ V.T
        return self

    def transform(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=np.float64) @ self.W


def fit_projection(X: np.ndarray, d: int, noise_lag: int, groups: np.ndarray | None, rcond: float = 1e-12):
    """PCA-whiten X to d dims, then noise-whiten. Returns (whitener, noise, sigma0).

    ``noise.quality`` is the worst noise_fallback rung used; a degraded fit also warns."""
    whitener = Whitener().fit(X, d)
    Zp = whitener.transform(X)
    noise = NoiseWhitener(rcond=rcond).fit(Zp, noise_lag, groups)
    sigma0, sigma_quality = soft_noise_sigma(noise.transform(Zp), noise_lag, groups)
    noise.quality = worst_quality(noise.quality, sigma_quality)
    return whitener, noise, sigma0


def band_features(windows: np.ndarray) -> np.ndarray:
    """Per channel: mean, log std, 4 log band energies -> [N, 6*C]."""
    x = np.asarray(windows, dtype=np.float64)
    N, C, K = x.shape
    mean = x.mean(axis=2)
    xc = x - mean[:, :, None]
    log_std = np.log(xc.std(axis=2) + 1e-6)
    power = np.abs(np.fft.rfft(xc, axis=2)[:, :, 1 : K // 2 + 1]) ** 2 / K
    bands = [np.log(b.sum(axis=2) + 1e-8) for b in np.array_split(power, 4, axis=2)]
    feats = np.stack([mean, log_std, *bands], axis=2)  # [N,C,6]
    return feats.reshape(N, C * 6)


def windowed_slopes(means: np.ndarray, segment, lags: tuple[int, ...]) -> np.ndarray:
    """For each lag, each column's value minus its value `lag` rows earlier,
    zero when that earlier row is missing or in a different segment.
    `segment=None` means a single implicit segment. Shared by
    `DSPSlopeEncoder` and `CtxDSPEncoder` (encoder_slope.py, encoder_ctx.py)."""
    n = len(means)
    segment = np.zeros(n, dtype=np.int64) if segment is None else np.asarray(segment)
    parts = []
    for lag in lags:
        shifted = np.zeros_like(means)
        same_seg = np.zeros(n, bool)
        if lag < n:
            shifted[lag:] = means[:-lag]
            same_seg[lag:] = segment[lag:] == segment[:-lag]
        parts.append(np.where(same_seg[:, None], means - shifted, 0.0))
    return np.column_stack(parts) if parts else np.empty((n, 0))


class DSPEncoder:
    """Hand-built spectral features, robust standardisation, PCA + noise whitening."""

    def __init__(self, d: int = 16, gate_lag: int = 0):
        """``gate_lag > 0`` widens each latent dimension's gate scale ``sigma0`` by the ratio of its
        lag-``gate_lag`` difference noise to its lag-1 noise (never below 1). Whitening by lag-1
        noise treats slow within-state variation as novelty; at ``d = 36`` that is most dimensions.
        0 (default) keeps ``sigma0`` as the lag-1 noise std."""
        self.d = d
        self._d_requested = d
        self.gate_lag = gate_lag

    def _raw(self, windows, sequential: bool) -> np.ndarray:
        return band_features(windows)

    def _fit_projection(self, X: np.ndarray, noise_lag: int, groups: np.ndarray | None) -> None:
        """PCA + noise whitening at min(requested d, number of features); ``self.d`` is what the
        latent really has, so callers sizing a memory from ``enc.d`` never see a mismatch."""
        d = min(self._d_requested, X.shape[1])
        if d < self._d_requested:
            warnings.warn(f"requested d={self._d_requested} exceeds the {X.shape[1]} available "
                          f"features; using d={d}", RuntimeWarning, stacklevel=3)
        self.d = d
        self.whitener, self.noise, self.sigma0 = fit_projection(X, d, noise_lag, groups)
        if self.gate_lag:
            Z = self.noise.transform(self.whitener.transform(X))
            slow, _ = soft_noise_sigma(Z, self.gate_lag, groups)
            self.sigma0 = self.sigma0 * np.maximum(slow / self.sigma0, 1.0)

    @property
    def whitening_quality(self) -> str:
        """noise_fallback.QUALITY rung of the fitted noise estimate ('robust' = nothing degraded)."""
        return self.noise.quality

    def fit(self, train_windows, noise_lag: int, groups: np.ndarray | None) -> "DSPEncoder":
        F = self._raw(train_windows, True)
        self.med = np.median(F, axis=0)
        q75, q25 = np.percentile(F, [75, 25], axis=0)
        robust = (q75 - q25) / 1.349
        spread = F.std(axis=0)
        self.scale = np.where(robust > 1e-6, robust, np.maximum(spread, 1e-6))
        self._fit_projection((F - self.med) / self.scale, noise_lag, groups)
        return self

    def encode(self, windows, sequential: bool = True) -> np.ndarray:
        """`sequential=False` marks independent windows (only matters for
        encoders with temporal context)."""
        F = (self._raw(windows, sequential) - self.med) / self.scale
        return self.noise.transform(self.whitener.transform(F))

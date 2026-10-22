"""Window encoders: whitening and the DSP+PCA encoder."""
from __future__ import annotations

import numpy as np

from .stats import robust_noise_sigma


class Whitener:
    """Centre, rotate onto the top-d principal axes, scale to unit variance."""

    def fit(self, F: np.ndarray, d: int) -> "Whitener":
        F = np.asarray(F, dtype=np.float64)
        self.m = F.mean(axis=0)
        lam, vec = np.linalg.eigh(np.cov(F, rowvar=False))
        order = np.argsort(lam)[::-1][:d]
        lam, vec = lam[order], vec[:, order]
        lam = np.maximum(lam, 1e-8 * lam.max())
        self.V = vec
        self.lam = lam
        return self

    def transform(self, F: np.ndarray) -> np.ndarray:
        return (np.asarray(F, dtype=np.float64) - self.m) @ self.V / np.sqrt(self.lam)


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


class DSPEncoder:
    """Hand-built spectral features, robust standardisation, PCA whitening."""

    def __init__(self, d: int = 16):
        self.d = d

    def fit(self, train_windows, noise_lag: int, groups: np.ndarray | None) -> "DSPEncoder":
        F = band_features(train_windows)
        self.med = np.median(F, axis=0)
        q75, q25 = np.percentile(F, [75, 25], axis=0)
        self.scale = np.maximum((q75 - q25) / 1.349, 1e-6)
        self.whitener = Whitener().fit((F - self.med) / self.scale, self.d)
        self.sigma0 = robust_noise_sigma(self.encode(train_windows), noise_lag, groups)
        return self

    def encode(self, windows, sequential: bool = True) -> np.ndarray:
        """`sequential` is accepted for protocol compatibility; windows are independent here."""
        F = (band_features(windows) - self.med) / self.scale
        return self.whitener.transform(F)

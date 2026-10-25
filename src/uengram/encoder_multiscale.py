"""Deterministic multi-scale temporal filter-bank encoder (NumPy only)."""
from __future__ import annotations

import numpy as np

from .encoders import fit_projection


class MultiScaleEncoder:
    """Project channel levels and multi-lag dynamics into separate latent blocks.

    ``d`` is the requested total output dimension. The fitted ``d`` may be
    smaller when the feature blocks cannot support it; the level and dynamics
    allocations are exposed as ``level_d`` and ``dynamics_d``.
    """

    def __init__(self, d: int = 16, lags: tuple[int, ...] = (1, 2, 4, 8)):
        if not isinstance(d, (int, np.integer)) or isinstance(d, bool) or d < 4:
            raise ValueError("d must be an integer >= 4")
        if not lags or any(not isinstance(lag, (int, np.integer)) or isinstance(lag, bool) or lag < 1 for lag in lags):
            raise ValueError("lags must be a non-empty sequence of positive integers")
        if len(set(lags)) != len(lags):
            raise ValueError("lags must not contain duplicates")
        self.requested_d = int(d)
        self.lags = tuple(int(lag) for lag in lags)
        self.d = int(d)

    @staticmethod
    def _windows(windows) -> np.ndarray:
        x = np.asarray(windows, dtype=np.float64)
        if x.ndim != 3:
            raise ValueError("windows must have shape (n_windows, n_channels, window_length)")
        if x.shape[0] < 2 or x.shape[1] < 1 or x.shape[2] < 2:
            raise ValueError("need at least two windows, one channel, and two samples per window")
        if not np.isfinite(x).all():
            raise ValueError("windows must contain only finite values")
        return x

    def _features(self, windows) -> tuple[np.ndarray, np.ndarray]:
        x = self._windows(windows)
        # The level block is deliberately the unfiltered raw per-channel mean.
        level = x.mean(axis=2)
        dynamics = []
        length = x.shape[2]
        for lag in self.lags:
            if lag >= length:
                continue
            # Haar-like low/high pair: adjacent interval means at scale `lag`.
            left = x[:, :, :-lag]
            right = x[:, :, lag:]
            difference = right - left
            dynamics.append(difference.mean(axis=2))
            dynamics.append(np.sqrt(np.mean(difference * difference, axis=2) + 1e-12))
        if not dynamics:
            # Valid windows always have lag 1 available, retained for clarity.
            raise ValueError("no configured lag is shorter than the window")
        return level, np.concatenate(dynamics, axis=1)

    @staticmethod
    def _scale(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        med = np.median(features, axis=0)
        q25, q75 = np.percentile(features, [25, 75], axis=0)
        robust = (q75 - q25) / 1.349
        spread = features.std(axis=0)
        scale = np.where(robust > 1e-6, robust, np.maximum(spread, 1e-6))
        return med, scale, (features - med) / scale

    def fit(self, train_windows, noise_lag: int, groups: np.ndarray | None) -> "MultiScaleEncoder":
        if not isinstance(noise_lag, (int, np.integer)) or noise_lag < 1:
            raise ValueError("noise_lag must be a positive integer")
        level, dynamics = self._features(train_windows)
        # The exact chi-square noise scale estimator requires an even latent
        # dimension. Keep both semantic branches non-empty and even.
        if level.shape[1] == 1:
            level = np.column_stack((level, np.zeros(level.shape[0])))
        self.level_med, self.level_scale, level_std = self._scale(level)
        self.dynamics_med, self.dynamics_scale, dynamics_std = self._scale(dynamics)
        level_capacity = level_std.shape[1] - level_std.shape[1] % 2
        dynamics_capacity = dynamics_std.shape[1] - dynamics_std.shape[1] % 2
        requested = min(self.requested_d, level_capacity + dynamics_capacity)
        requested -= requested % 2
        level_target = max(2, (requested // 4) * 2)
        ld = min(level_target, level_capacity)
        dd = min(max(2, requested - ld), dynamics_capacity)
        remaining = requested - ld - dd
        if remaining:
            take = min(remaining, level_capacity - ld)
            ld += take
            remaining -= take
        if remaining:
            dd += remaining
        self.level_d, self.dynamics_d = ld, dd
        self.d = ld + dd
        self.level_whitener, self.level_noise, level_sigma = fit_projection(
            level_std, ld, noise_lag, groups)
        self.dynamics_whitener, self.dynamics_noise, dynamics_sigma = fit_projection(
            dynamics_std, dd, noise_lag, groups)
        self.sigma0_level = level_sigma
        self.sigma0_dynamics = dynamics_sigma
        self.sigma0 = np.concatenate((level_sigma, dynamics_sigma))
        self.n_channels = int(self._windows(train_windows).shape[1])
        self.window_length = int(self._windows(train_windows).shape[2])
        return self

    def encode_components(self, windows) -> dict[str, np.ndarray]:
        if not hasattr(self, "level_whitener"):
            raise RuntimeError("encoder must be fitted before encode")
        level, dynamics = self._features(windows)
        if level.shape[1] == 1:
            level = np.column_stack((level, np.zeros(level.shape[0])))
        level_std = (level - self.level_med) / self.level_scale
        dynamics_std = (dynamics - self.dynamics_med) / self.dynamics_scale
        level_z = self.level_noise.transform(self.level_whitener.transform(level_std))
        dynamics_z = self.dynamics_noise.transform(self.dynamics_whitener.transform(dynamics_std))
        return {"level": level_z, "dynamics": dynamics_z,
                "combined": np.concatenate((level_z, dynamics_z), axis=1)}

    def encode(self, windows, sequential: bool = True) -> np.ndarray:
        """Encode windows; ``sequential`` is accepted for interface compatibility."""
        del sequential
        return self.encode_components(windows)["combined"]

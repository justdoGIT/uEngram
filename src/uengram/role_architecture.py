"""Role-separated sensor representation and evidence-level fusion.

The architecture keeps operating mode, absolute level, dynamics, and health
separate. Each alarmable branch is calibrated on healthy data before fusion;
mode is context only. Memory admission is conservative so an alarmed or
unconfirmed sample cannot silently become semantic normal memory.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .encoder_multiscale import MultiScaleEncoder


@dataclass(frozen=True)
class RoleScores:
    mode: np.ndarray
    level: np.ndarray
    dynamic: np.ndarray
    health: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "mode": self.mode,
            "level": self.level,
            "dynamic": self.dynamic,
            "health": self.health,
        }


class EmpiricalCalibrator:
    """Upper-tail empirical p-values from a label-free calibration stream."""

    def __init__(self):
        self.sorted_scores: np.ndarray | None = None

    def fit(self, scores) -> "EmpiricalCalibrator":
        values = np.asarray(scores, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if len(values) < 2:
            raise ValueError("need at least two finite calibration scores")
        self.sorted_scores = np.sort(values)
        return self

    def p_values(self, scores) -> np.ndarray:
        if self.sorted_scores is None:
            raise RuntimeError("calibrator must be fitted before p_values")
        values = np.asarray(scores, dtype=np.float64)
        out = np.full(values.shape, np.nan, dtype=np.float64)
        finite = np.isfinite(values)
        ranks = np.searchsorted(self.sorted_scores, values[finite], side="left")
        out[finite] = (len(self.sorted_scores) - ranks + 1) / (len(self.sorted_scores) + 1)
        return out


class RoleBasedArchitecture:
    """Small role-separated representation for edge anomaly experiments.

    ``fit`` is restricted to commissioning/healthy data. ``fit_calibration``
    must receive a separate healthy period. The deterministic multiscale
    encoder supplies level and dynamic branches; health is distance from the
    frozen commissioning reference. Mode distance is diagnostic context and is
    intentionally excluded from alarm fusion.
    """

    _ALARM_BRANCHES = ("level", "dynamic", "health")
    _POLICIES = ("any", "at_least_2", "dynamic_health", "all")

    def __init__(self, d: int = 16, lags: tuple[int, ...] = (1, 2, 4, 8)):
        self.encoder = MultiScaleEncoder(d=d, lags=lags)
        self.calibrators: dict[str, EmpiricalCalibrator] = {}
        self._fitted = False

    @staticmethod
    def _validate_windows(windows) -> np.ndarray:
        x = np.asarray(windows, dtype=np.float64)
        if x.ndim != 3 or x.shape[1] < 1 or x.shape[2] < 2:
            raise ValueError("windows must have shape (n_windows, channels, length)")
        if len(x) < 2 or not np.isfinite(x).all():
            raise ValueError("windows must contain at least two finite rows")
        return x

    @staticmethod
    def _robust_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        center = np.median(values, axis=0)
        q25, q75 = np.percentile(values, [25, 75], axis=0)
        scale = np.maximum((q75 - q25) / 1.349, 1e-6)
        return center, scale

    def fit(self, windows, noise_lag: int = 1, groups=None, mode_ids=None):
        x = self._validate_windows(windows)
        self.encoder.fit(x, noise_lag=noise_lag, groups=groups)
        components = self.encoder.encode_components(x)
        self._health_center, self._health_scale = self._robust_scale(
            components["combined"]
        )
        mode_features = np.column_stack((x.mean(axis=2), x.std(axis=2)))
        self._mode_center, self._mode_scale = self._robust_scale(mode_features)
        mode_z = (mode_features - self._mode_center) / self._mode_scale
        ids = np.zeros(len(x), dtype=np.int64) if mode_ids is None else np.asarray(mode_ids)
        if ids.shape != (len(x),):
            raise ValueError("mode_ids must have one value per window")
        self._mode_centers = {
            key: np.median(mode_z[ids == key], axis=0)
            for key in np.unique(ids)
            if np.any(ids == key)
        }
        self._fitted = True
        return self

    def score(self, windows, mode_ids=None) -> RoleScores:
        if not self._fitted:
            raise RuntimeError("architecture must be fitted before score")
        x = self._validate_windows(windows)
        components = self.encoder.encode_components(x)
        level = np.sum(components["level"] ** 2, axis=1)
        dynamic = np.sum(components["dynamics"] ** 2, axis=1)
        health_z = (components["combined"] - self._health_center) / self._health_scale
        health = np.sum(health_z ** 2, axis=1)
        mode_features = np.column_stack((x.mean(axis=2), x.std(axis=2)))
        mode_z = (mode_features - self._mode_center) / self._mode_scale
        if mode_ids is not None:
            ids = np.asarray(mode_ids)
            if ids.shape != (len(x),):
                raise ValueError("mode_ids must have one value per window")
            centers = np.stack([
                self._mode_centers.get(key, np.zeros(mode_z.shape[1])) for key in ids
            ])
            mode = np.sum((mode_z - centers) ** 2, axis=1)
        else:
            centers = np.stack(tuple(self._mode_centers.values()))
            mode = np.min(
                np.sum((mode_z[:, None, :] - centers[None, :, :]) ** 2, axis=2),
                axis=1,
            )
        return RoleScores(mode=mode, level=level, dynamic=dynamic, health=health)

    def fit_calibration(self, windows, mode_ids=None) -> "RoleBasedArchitecture":
        scores = self.score(windows, mode_ids=mode_ids)
        self.calibrators = {
            name: EmpiricalCalibrator().fit(getattr(scores, name))
            for name in self._ALARM_BRANCHES
        }
        return self

    def p_values(self, scores: RoleScores) -> dict[str, np.ndarray]:
        if set(self.calibrators) != set(self._ALARM_BRANCHES):
            raise RuntimeError("fit_calibration must precede p_values")
        values = scores.as_dict()
        return {
            name: self.calibrators[name].p_values(values[name])
            for name in (*self._ALARM_BRANCHES, "mode")
            if name != "mode"
        } | {"mode": values["mode"]}

    @classmethod
    def fuse(cls, p_values: dict[str, np.ndarray], alpha: float = 0.01,
             policy: str = "dynamic_health") -> np.ndarray:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if policy not in cls._POLICIES:
            raise ValueError(f"policy must be one of {cls._POLICIES}")
        missing = [name for name in cls._ALARM_BRANCHES if name not in p_values]
        if missing:
            raise ValueError(f"missing calibrated branches: {missing}")
        flags = np.stack([np.asarray(p_values[name]) <= alpha for name in cls._ALARM_BRANCHES], axis=1)
        if policy == "any":
            return flags.any(axis=1)
        if policy == "at_least_2":
            return flags.sum(axis=1) >= 2
        if policy == "dynamic_health":
            return flags[:, 1] & flags[:, 2]
        return flags.all(axis=1)

    @staticmethod
    def memory_route(p_values: dict[str, float], alpha: float, *, alarm_open: bool,
                     data_quality_ok: bool, operator_confirmed_normal: bool = False) -> str:
        """Route evidence without allowing unconfirmed data to poison normal memory."""
        if not data_quality_ok:
            return "QUARANTINE"
        if alarm_open or any(float(p_values.get(name, np.nan)) <= alpha
                             for name in RoleBasedArchitecture._ALARM_BRANCHES):
            return "EPISODIC"
        if operator_confirmed_normal:
            return "SEMANTIC"
        return "QUARANTINE"


def persistent_starts(flags, k: int) -> np.ndarray:
    """Return one start flag for each complete persistent run."""
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1:
        raise ValueError("persistence k must be positive")
    values = np.asarray(flags, dtype=bool).reshape(-1)
    starts = np.zeros(len(values), dtype=bool)
    for i in range(len(values) - k + 1):
        starts[i] = bool(np.all(values[i:i + k]) and (i == 0 or not values[i - 1]))
    return starts

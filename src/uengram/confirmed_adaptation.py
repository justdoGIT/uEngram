"""Opt-in adaptation utilities gated exclusively by operator-confirmed normal data."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .stats import conformal_threshold


def _finite_matrix(values, name: str, width: int | None = None) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 2 or (width is not None and a.shape[1] != width):
        expected = "a 2D array" if width is None else f"a 2D array with {width} columns"
        raise ValueError(f"{name} must be {expected}")
    if a.shape[0] == 0 or a.shape[1] == 0 or not np.isfinite(a).all():
        raise ValueError(f"{name} must be non-empty and finite")
    return a


def _states(states, n: int) -> np.ndarray:
    a = np.asarray(states)
    if a.ndim == 0:
        a = np.full(n, a.item())
    if a.shape != (n,) or not np.issubdtype(a.dtype, np.integer):
        raise ValueError("states must be an integer scalar or an integer vector matching samples")
    return a.astype(np.int64, copy=False)


def _gate(value: bool, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean")
    return bool(value)


@dataclass(frozen=True)
class AdapterSnapshot:
    """Independent adapter state suitable for later rollback."""

    offsets: Mapping[int, np.ndarray]
    quarantine: Mapping[int, tuple[tuple[float, ...], ...]]


class ConfirmedLatentAdapter:
    """Zero-start, bounded state-specific latent offsets.

    Only rows submitted in a call explicitly marked operator-confirmed-normal,
    alarm-closed, and commissioned enter quarantine. Once a state's quarantine
    reaches ``min_quarantine_count``, a bounded update is applied. ``apply``
    always returns a new array and never mutates the caller's latent values.
    """

    def __init__(self, d: int, max_offset: float, learning_rate: float = 1.0,
                 min_quarantine_count: int = 20):
        if not isinstance(d, int) or isinstance(d, bool) or d <= 0:
            raise ValueError("d must be a positive integer")
        if not np.isfinite(max_offset) or max_offset <= 0:
            raise ValueError("max_offset must be finite and positive")
        if not np.isfinite(learning_rate) or not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if not isinstance(min_quarantine_count, int) or isinstance(min_quarantine_count, bool) or min_quarantine_count <= 0:
            raise ValueError("min_quarantine_count must be a positive integer")
        self.d = d
        self.max_offset = float(max_offset)
        self.learning_rate = float(learning_rate)
        self.min_quarantine_count = min_quarantine_count
        self._offsets: dict[int, np.ndarray] = {}
        self._quarantine: dict[int, list[np.ndarray]] = {}

    @property
    def state(self) -> dict[int, np.ndarray]:
        """Current offsets, copied so callers cannot mutate adapter state."""
        return {key: value.copy() for key, value in self._offsets.items()}

    def apply(self, latent, states) -> np.ndarray:
        z = _finite_matrix(latent, "latent", self.d)
        g = _states(states, len(z))
        corrected = z.copy()
        for key in np.unique(g):
            offset = self._offsets.get(int(key))
            if offset is not None:
                corrected[g == key] -= offset
        return corrected

    def update(self, latent, states, *, operator_confirmed_normal: bool,
               alarm_closed: bool, commissioned) -> int:
        z = _finite_matrix(latent, "latent", self.d)
        g = _states(states, len(z))
        confirmed = _gate(operator_confirmed_normal, "operator_confirmed_normal")
        closed = _gate(alarm_closed, "alarm_closed")
        ready = np.asarray(commissioned)
        if ready.ndim == 0:
            if not isinstance(ready.item(), (bool, np.bool_)):
                raise ValueError("commissioned must be boolean or a boolean vector")
            ready = np.full(len(z), bool(ready), dtype=bool)
        elif ready.shape == (len(z),) and ready.dtype == np.bool_:
            ready = ready.astype(bool, copy=False)
        else:
            raise ValueError("commissioned must be boolean or a boolean vector matching samples")
        if not (confirmed and closed):
            return 0
        changed = 0
        for key in np.unique(g):
            sid = int(key)
            rows = z[(g == key) & ready]
            if not len(rows):
                continue
            pending = self._quarantine.setdefault(sid, [])
            pending.extend(row.copy() for row in rows)
            if len(pending) < self.min_quarantine_count:
                continue
            mean = np.mean(np.asarray(pending, dtype=np.float64), axis=0)
            old = self._offsets.get(sid, np.zeros(self.d, dtype=np.float64))
            candidate = old + self.learning_rate * (mean - old)
            norm = float(np.linalg.norm(candidate))
            if norm > self.max_offset:
                candidate *= self.max_offset / norm
            self._offsets[sid] = candidate
            changed += len(pending)
            pending.clear()
        return changed

    def snapshot(self) -> AdapterSnapshot:
        return AdapterSnapshot(
            offsets={key: value.copy() for key, value in self._offsets.items()},
            quarantine={key: tuple(tuple(float(v) for v in row) for row in rows)
                        for key, rows in self._quarantine.items()},
        )

    def rollback(self, snapshot: AdapterSnapshot) -> None:
        if not isinstance(snapshot, AdapterSnapshot):
            raise ValueError("snapshot must be an AdapterSnapshot")
        offsets: dict[int, np.ndarray] = {}
        for key, value in snapshot.offsets.items():
            a = np.asarray(value, dtype=np.float64)
            if not isinstance(key, (int, np.integer)) or a.shape != (self.d,) or not np.isfinite(a).all():
                raise ValueError("snapshot contains an invalid offset")
            if np.linalg.norm(a) > self.max_offset + 1e-12:
                raise ValueError("snapshot offset exceeds configured bound")
            offsets[int(key)] = a.copy()
        quarantine: dict[int, list[np.ndarray]] = {}
        for key, rows in snapshot.quarantine.items():
            if not isinstance(key, (int, np.integer)):
                raise ValueError("snapshot contains an invalid state")
            a = np.asarray(rows, dtype=np.float64)
            if a.size and (a.ndim != 2 or a.shape[1] != self.d or not np.isfinite(a).all()):
                raise ValueError("snapshot contains invalid quarantine samples")
            quarantine[int(key)] = [row.copy() for row in a.reshape((-1, self.d))]
        self._offsets, self._quarantine = offsets, quarantine


    def rollback_if_disagreement(
        self, snapshot: AdapterSnapshot, frozen_scores, adapted_scores,
        *, max_abs_difference: float,
    ) -> bool:
        """Restore a saved adapter when any shadow score exceeds the bound."""
        frozen = np.asarray(frozen_scores, dtype=np.float64)
        adapted = np.asarray(adapted_scores, dtype=np.float64)
        limit = float(max_abs_difference)
        if (
            frozen.ndim != 1 or frozen.size == 0 or adapted.shape != frozen.shape
            or not np.isfinite(frozen).all() or not np.isfinite(adapted).all()
        ):
            raise ValueError("shadow score vectors must be non-empty, finite, and aligned")
        if not np.isfinite(limit) or limit < 0:
            raise ValueError("max_abs_difference must be finite and non-negative")
        if float(np.max(np.abs(frozen - adapted))) <= limit:
            return False
        self.rollback(snapshot)
        return True


class AdaptiveConformalCalibrator:
    """Rolling state/season score calibration with confirmed-normal updates.

    A supported state-season stratum is preferred, then its state, then the
    global calibration pool. Online updates require explicit operator
    confirmation and a closed alarm. This gate does not create a guarantee
    under drift or protect against incorrect human labels.
    """

    def __init__(self, calibration_scores, calibration_states, alpha: float = 0.01,
                 min_support: int = 100, window_size: int = 1000,
                 calibration_seasons=None):
        scores = np.asarray(calibration_scores, dtype=np.float64)
        if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
            raise ValueError("calibration_scores must be a non-empty finite vector")
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        for value, name in ((min_support, "min_support"), (window_size, "window_size")):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        states = _states(calibration_states, len(scores))
        seasons = self._season_values(calibration_seasons, len(scores))
        if window_size < len(scores):
            scores, states, seasons = (scores[-window_size:], states[-window_size:],
                                       seasons[-window_size:])
        self.alpha = float(alpha)
        self.min_support = min_support
        self.window_size = window_size
        self._records: deque[tuple[float, int, object]] = deque()
        self._global: deque[float] = deque()
        self._by_state: dict[int, deque[float]] = {}
        self._by_stratum: dict[tuple[int, object], deque[float]] = {}
        for score, state, season in zip(scores, states, seasons):
            self._append(float(score), int(state), season)

    @staticmethod
    def _season_values(seasons, n: int) -> np.ndarray:
        if seasons is None:
            return np.full(n, None, dtype=object)
        values = np.asarray(seasons, dtype=object)
        if values.ndim == 0:
            values = np.full(n, values.item(), dtype=object)
        if values.shape != (n,):
            raise ValueError("seasons must be a scalar or vector matching samples")
        for value in values:
            key = value.item() if isinstance(value, np.generic) else value
            try:
                hash(key)
            except TypeError as error:
                raise ValueError("season values must be hashable scalars") from error
            if isinstance(key, float) and not np.isfinite(key):
                raise ValueError("season values must be finite")
        return np.asarray(
            [value.item() if isinstance(value, np.generic) else value for value in values],
            dtype=object,
        )

    def _append(self, score: float, state: int, season) -> None:
        if len(self._records) == self.window_size:
            _, old_state, old_season = self._records.popleft()
            self._global.popleft()
            state_values = self._by_state[old_state]
            state_values.popleft()
            if not state_values:
                del self._by_state[old_state]
            stratum = (old_state, old_season)
            stratum_values = self._by_stratum[stratum]
            stratum_values.popleft()
            if not stratum_values:
                del self._by_stratum[stratum]
        self._records.append((score, state, season))
        self._global.append(score)
        self._by_state.setdefault(state, deque()).append(score)
        self._by_stratum.setdefault((state, season), deque()).append(score)

    @property
    def state(self) -> dict[int, tuple[float, ...]]:
        """Current per-state rolling samples as immutable tuples."""
        return {key: tuple(values) for key, values in self._by_state.items()}

    @property
    def strata(self) -> dict[tuple[int, object], tuple[float, ...]]:
        """Current per-state-season rolling samples as immutable tuples."""
        return {key: tuple(values) for key, values in self._by_stratum.items()}

    @property
    def global_scores(self) -> tuple[float, ...]:
        return tuple(self._global)

    def threshold(self, state: int, season=None) -> float:
        if not isinstance(state, (int, np.integer)) or isinstance(state, (bool, np.bool_)):
            raise ValueError("state must be an integer")
        if season is not None:
            key = self._season_values(season, 1)[0]
            own_season = self._by_stratum.get((int(state), key))
            if own_season is not None and len(own_season) >= self.min_support:
                return conformal_threshold(np.asarray(own_season), self.alpha)
        own_state = self._by_state.get(int(state))
        if own_state is not None and len(own_state) >= self.min_support:
            return conformal_threshold(np.asarray(own_state), self.alpha)
        return conformal_threshold(np.asarray(self._global), self.alpha)

    def thresholds(self, states, seasons=None) -> dict:
        raw_states = np.asarray(states)
        if raw_states.ndim != 1:
            raise ValueError("states must be an integer vector")
        groups = _states(raw_states, len(raw_states))
        season_values = self._season_values(seasons, len(groups))
        if seasons is None:
            return {int(key): self.threshold(int(key)) for key in np.unique(groups)}
        return {
            (int(state), season): self.threshold(int(state), season)
            for state, season in zip(groups, season_values)
        }

    def update(self, scores, states, *, operator_confirmed_normal: bool,
               alarm_closed: bool, commissioned, seasons=None) -> int:
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim == 0:
            values = values.reshape(1)
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("scores must be a non-empty finite vector")
        groups = _states(states, len(values))
        season_values = self._season_values(seasons, len(values))
        confirmed = _gate(operator_confirmed_normal, "operator_confirmed_normal")
        closed = _gate(alarm_closed, "alarm_closed")
        ready = np.asarray(commissioned)
        if ready.ndim == 0:
            if not isinstance(ready.item(), (bool, np.bool_)):
                raise ValueError("commissioned must be boolean or a boolean vector")
            ready = np.full(len(values), bool(ready), dtype=bool)
        elif ready.shape == (len(values),) and ready.dtype == np.bool_:
            ready = ready.astype(bool, copy=False)
        else:
            raise ValueError("commissioned must be boolean or a boolean vector matching samples")
        if not (confirmed and closed):
            return 0
        accepted = 0
        for value, state, season, eligible in zip(values, groups, season_values, ready):
            if eligible:
                self._append(float(value), int(state), season)
                accepted += 1
        return accepted

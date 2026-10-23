"""Exploratory compressor residuals; demand proxies are not measured demand.

Fit only commissioning data. Inputs are observed at or before window end;
LPS, oil-level warnings, report labels and future observations are excluded.
Pressure slopes use the supplied observation times, not an invented 1 Hz grid.
"""
from __future__ import annotations

import numpy as np

from uengram import metropt

TARGETS = ('load_share', 'loaded_recovery_bar_min', 'unloaded_decay_bar_min',
           'oil_rate_c_min', 'motor_current_a', 'compressor_panel_gap_bar')
DIRECTIONS = np.array([1, -1, -1, 0, 0, 0])


def observation_times(stream):
    """Recover the actual decimated observation times underlying model windows."""
    if stream.name == 'metropt3':
        t, _, _ = metropt.load(metropt.fetch())
    else:
        from uengram.metropt2 import load
        t, _, _ = load(stream.name)
    indices = np.searchsorted(t, stream.start)[:, None] + np.arange(stream.W.shape[2])
    times = t[indices]
    if not np.array_equal(times[:, 0], stream.start) or not np.array_equal(times[:, -1], stream.end):
        raise ValueError('source observation times do not match model windows')
    return times


def _phase_rate(pressure, dt, phase):
    pair = phase[:, 1:] & phase[:, :-1] & (dt > 0)
    elapsed = np.sum(np.where(pair, dt, 0), axis=1)
    delta = np.sum(np.where(pair, np.diff(pressure, axis=1), 0), axis=1)
    good = (pair.sum(1) >= 3) & (elapsed > 0)
    out = np.full(len(pressure), np.nan)
    np.divide(delta, elapsed, out=out, where=good)
    return out


def physical_features(stream, times):
    if stream.cond is None or stream.W.ndim != 3:
        raise ValueError('physical residuals require compressor waveforms and controls')
    w, control = np.asarray(stream.W, float), np.asarray(stream.cond, float)
    if times.shape != (len(w), w.shape[2]):
        raise ValueError('observation times must align with every waveform sample')
    a = {name: w[:, i] for i, name in enumerate(metropt.ANALOG)}
    dt = np.diff(times.astype('datetime64[s]').astype(np.int64), axis=1) / 60
    if np.any(dt <= 0):
        raise ValueError('observation timestamps must increase within each window')
    loaded = control[:, 1] > .5
    duration = dt.sum(1)
    duty = loaded.mean(1)
    y = np.column_stack((duty, _phase_rate(a['Reservoirs'], dt, loaded),
                         _phase_rate(a['Reservoirs'], dt, ~loaded),
                         (a['Oil_temperature'][:, -1] - a['Oil_temperature'][:, 0]) / duration,
                         a['Motor_current'].mean(1), (a['TP2'] - a['TP3']).mean(1)))
    previous_duty = np.r_[np.nan, duty[:-1]]
    previous_duty[np.r_[True, stream.segment[1:] != stream.segment[:-1]]] = np.nan
    # No direct air-demand sensor is available. Starting pressure, prior duty
    # and observed controls are proxies; a leak itself can change these proxies.
    context = np.column_stack((duty, control[:, 0].mean(1), a['Oil_temperature'][:, 0],
                               a['Reservoirs'][:, 0], previous_duty))
    if 'parked_share' in stream.extra:
        context = np.column_stack((context, stream.extra['parked_share']))
    return context, y


def _scale(x):
    q25, q75 = np.nanpercentile(x, [25, 75], axis=0)
    full = np.nanmax(x, axis=0) - np.nanmin(x, axis=0)
    return np.maximum(np.maximum((q75 - q25) / 1.349, full / 6), 1e-8)


class PhysicalResiduals:
    """Commissioning-only conditional ridge or marginal physical baseline.

    Missing phase slopes remain missing. A component without enough training
    support is absent, not replaced with zero evidence. Overall scores are the
    maximum available standardized directional residual, calibrated separately.
    """
    def __init__(self, conditional=True, ridge=.01, min_rows=30):
        self.conditional, self.ridge, self.min_rows = conditional, ridge, min_rows

    def fit(self, context, targets, fit):
        context, targets, fit = np.asarray(context, float), np.asarray(targets, float), np.asarray(fit, bool)
        self.center = np.nanmedian(context[fit], axis=0)
        self.context_scale = _scale(context[fit])
        x = self._design(context)
        self.components = []
        for j in range(targets.shape[1]):
            good = fit & np.isfinite(targets[:, j])
            if good.sum() < self.min_rows:
                self.components.append(None)
                continue
            # Do not condition duty on itself or its complementary valve signal.
            cols = np.arange(x.shape[1]) if self.conditional else np.array([0])
            if j == 0 and self.conditional:
                cols = cols[~np.isin(cols, [1, 2])]
            xx, yy = x[good][:, cols], targets[good, j]
            penalty = np.eye(len(cols)) * self.ridge * len(yy)
            penalty[0, 0] = 0
            coef = np.linalg.solve(xx.T @ xx + penalty, xx.T @ yy)
            residual = yy - xx @ coef
            location = float(np.median(residual))
            scale = float(_scale(residual[:, None])[0])
            self.components.append((cols, coef, location, scale))
        if not any(c is not None for c in self.components):
            raise ValueError('no physically supported residual component')
        self.fit_rows = int(fit.sum())
        return self

    def _design(self, context):
        values = np.where(np.isfinite(context), context, self.center)
        return np.column_stack((np.ones(len(values)), (values - self.center) / self.context_scale))

    def score(self, context, targets):
        x = self._design(np.asarray(context, float))
        residuals = np.full_like(targets, np.nan, dtype=float)
        for j, component in enumerate(self.components):
            if component is None:
                continue
            cols, coef, location, scale = component
            r = (targets[:, j] - x[:, cols] @ coef - location) / scale
            residuals[:, j] = np.abs(r) if DIRECTIONS[j] == 0 else np.maximum(DIRECTIONS[j] * r, 0)
        finite = np.isfinite(residuals)
        score = np.max(np.where(finite, residuals, -np.inf), axis=1)
        score[~finite.any(1)] = np.nan
        return score, residuals

    def metadata(self):
        return {'conditional': self.conditional, 'ridge': self.ridge, 'fit_rows': self.fit_rows,
                'targets': list(TARGETS), 'supported': [c is not None for c in self.components],
                'demand_caveat': 'pressure/prior-duty/parked proxies, not measured customer air demand',
                'coefficients': [None if c is None else {'columns': c[0].tolist(), 'coef': c[1].tolist(),
                                 'residual_center': c[2], 'residual_scale': c[3]} for c in self.components]}

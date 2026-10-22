"""MetroPT step calibration from commissioning semantic-slot innovations.

The raw-latent lag-38 estimator cannot operate when duty cycles leave only
short uninterrupted state runs. Instead, collect the standardized innovations
that the actual engine would feed to its step detector, in each slot's visit
order. Commissioning is declared healthy: step splits are held off while
estimating this null distribution. Merges and logging gaps start new runs.
This is an empirical scale calibration, not a sequential false-alarm theorem.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import numpy as np

from uengram.engine import Engram, EngramConfig


def innovation_inflation(runs, lam: float):
    """Robust zero-centred EWMA scale relative to unit independent innovations.

    For independent unit noise, Var(EWMA) = lam/(2-lam). After one effective
    memory of warmup, pool the filtered innovations and estimate their scale
    as median(abs(EWMA))/Phi^-1(0.75). No long uninterrupted raw-state run is
    needed. A null bias is deliberately not subtracted. Missing evidence is
    an error, never an implicit independence assumption.
    """
    b = (2.0 - lam) / lam
    warmup = int(np.ceil(b))
    samples = []
    for run in runs:
        y = np.zeros(run.shape[1])
        for n, e in enumerate(run, 1):
            y = lam * e + (1.0 - lam) * y
            if n >= warmup:
                samples.append(y.copy())
    if len(samples) < 10:
        raise ValueError(f"step calibration needs 10 warmed-up innovations; got {len(samples)}")
    scale = 1.4826 * np.median(np.abs(samples), axis=0)
    return b * scale * scale, len(samples)


class _StepInputs(Engram):
    """Observe accepted semantic innovations without splitting a healthy state."""

    def __init__(self, *args):
        super().__init__(*args)
        self.runs = defaultdict(list)
        self.generation = defaultdict(int)

    def _step_detected(self, k, e):
        iid = int(self.item_id[k])
        self.runs[(iid, self.generation[iid])].append(e.copy())
        return False  # estimating the null distribution, not monitoring failures

    def restart(self, ids):
        for iid in ids:
            self.generation[int(iid)] += 1


def calibrate_steps(Z, groups, cfg: EngramConfig, sigma0, raw_bytes: int):
    """Fit only on pre-test commissioning windows; return inflation and evidence."""
    recorder = _StepInputs(replace(cfg, use_step_detect=True), Z.shape[1], sigma0, raw_bytes)
    for t, z in enumerate(Z):
        if t and groups[t] != groups[t - 1]:
            recorder.restart(recorder.item_id[recorder.item_id >= 0])
        recorder.ingest(z, t, t)
        for event in recorder.drain_events():
            if event[0] == "merge":
                recorder.restart([event[2]])
    runs = [np.asarray(run) for run in recorder.runs.values()]
    inflation, samples = innovation_inflation(runs, cfg.step_lambda)
    return inflation, {"estimator": "commissioning_slot_innovations", "warm_samples": samples,
                       "runs": len(runs), "inflation": inflation.tolist()}

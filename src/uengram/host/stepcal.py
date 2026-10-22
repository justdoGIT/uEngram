"""Step-detector calibration from commissioning innovations (uEngram v3).

The engine's MEWMA step test (docs/EXPLAINED.md 10b) averages about
b = (2 - lam) / lam = 19 standardised innovations of a semantic state and its
chi-square threshold assumes them independent. A host stays in one state for
runs of consecutive 32 s windows while its load wanders slowly, so successive
innovations of a state are correlated, their average varies more than the
threshold assumes, and ordinary wander reads as a repair or a new regime: the
state is superseded (EXPLAINED 10c). The engine divides each innovation
dimension by sqrt(max(f_j, 1)), with f_j the measured variance inflation of
the EWMA over the independent-noise value lam / (2 - lam).

Estimator: the commissioning-innovation estimator of the MetroPT amendment
(EXPLAINED 16; upstream bench/metropt_calibration.py), not the raw-latent
estimator of 10c. The latter needs uninterrupted runs longer than 2b = 38
windows (20 minutes of one host state without a job start or stop) between
abrupt changes; this one measures exactly what the detector sees. A recorder
copy of the commissioning engine runs over the declared-healthy fit windows
with step splits held off; each semantic slot's standardised innovations are
collected in visit order; a merge or a gap in the window sequence restarts
the run (a gap restarts every run). EWMA(lam) of each run after b warm-up
innovations; f_j = b * (1.4826 median |EWMA_j|)^2. This matches a scale, it
does not prove a sequential false-alarm rate under arbitrary dependence.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import numpy as np

from ..engine import Engram, EngramConfig

MIN_SAMPLES = 10


def innovation_inflation(runs: list[np.ndarray], lam: float) -> tuple[np.ndarray, int]:
    """Per-dimension EWMA variance inflation of innovation runs; (f, samples used).
    Missing evidence is an error, never an implicit independence assumption."""
    b = (2.0 - lam) / lam
    warm = int(np.ceil(b))
    samples = []
    for run in runs:
        y = np.zeros(run.shape[1])
        for n, e in enumerate(run, 1):
            y = lam * e + (1.0 - lam) * y
            if n >= warm:
                samples.append(y.copy())
    if len(samples) < MIN_SAMPLES:
        raise ValueError(f"step calibration needs {MIN_SAMPLES} warmed-up innovations of semantic states; "
                         f"got {len(samples)}. Record longer.")
    scale = 1.4826 * np.median(np.abs(np.asarray(samples)), axis=0)
    return b * scale * scale, len(samples)


class _Recorder(Engram):
    """Records accepted semantic innovations instead of testing them."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.runs: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
        self.generation: dict[int, int] = defaultdict(int)

    def _step_detected(self, k: int, e: np.ndarray) -> bool:
        iid = int(self.item_id[k])
        self.runs[(iid, self.generation[iid])].append(e.copy())
        return False

    def restart(self, ids) -> None:
        for iid in ids:
            self.generation[int(iid)] += 1


def calibrate(Z: np.ndarray, t_step: np.ndarray, cfg: EngramConfig, sigma0: np.ndarray,
              raw_bytes: int) -> tuple[np.ndarray, dict]:
    """Inflation from the commissioning latents Z (one non-overlapping sequence,
    window steps t_step), fed through an engine configured like the real
    commissioning (`cfg` incl. its declared-normal promotion rule)."""
    rec = _Recorder(replace(cfg, use_step_detect=True), Z.shape[1], sigma0, raw_bytes)
    for i, z in enumerate(Z):
        if i and t_step[i] != t_step[i - 1] + 1:
            rec.restart(rec.item_id[rec.item_id >= 0])
        rec.ingest(z, int(t_step[i]), int(t_step[i]))
        for ev in rec.drain_events():
            if ev[0] == "merge":
                rec.restart([ev[2]])
    runs = [np.asarray(r) for r in rec.runs.values()]
    f, n = innovation_inflation(runs, cfg.step_lambda)
    return f, {"estimator": "commissioning_slot_innovations", "warm_samples": n, "runs": len(runs),
               "inflation_median": float(np.median(f)), "inflation_max": float(np.max(f)),
               "inflation": [float(x) for x in f]}

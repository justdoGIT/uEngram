"""SKAB (Skoltech Anomaly Benchmark) stream: an independent, previously
unused water-circulation testbed with verified pointwise fault timestamps.
Protocol in docs/validation-skab-protocol.md; it fills the gap identified in
docs/research-remediation-plan.md (no compressor-matched asset-disjoint
cohort with pointwise labels exists). SKAB is a different physical system
(centrifugal pump / water loop, not an air compressor) and not a multi-asset
cohort; see the protocol for exactly what this stream can and cannot show.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from bench.streams import Stream
from uengram import skab

DEFAULT_ROOT = skab.DEFAULT_ROOT


def skab_stream(root: str | Path | None = None, K: int = 10, fit_frac: float = 0.3) -> Stream:
    raw = skab.load(root or DEFAULT_ROOT, fit_frac=fit_frac, K=K)
    W, S, E, G, split, experiment = skab.windows(
        raw["t"], raw["X"], raw["split"], raw["experiment"], K)
    fit = split == "fit"
    cal = split == "cal"
    test = split == "test"
    monitorable = np.r_[False, G[1:] == G[:-1]]
    events = tuple(
        (str(raw["onset"][i]), str(raw["offset"][i])) for i in range(len(skab.FAULT_FILES))
    )
    feats = W.mean(axis=2).astype(np.float64)  # interpretable per-channel window mean
    return Stream(
        name="skab", W=W, start=S, end=E, segment=G, state=np.zeros(len(W), np.int64),
        monitorable=monitorable, fit=fit, cal=cal, test=test, events=events,
        raw_bytes=W.shape[1] * K * 4, features=feats, feature_names=skab.CHANNELS,
        before_h=0, after_h=0,
        extra={"experiment": experiment, "experiment_names": skab.FAULT_FILES},
        point_labels=False,
    )

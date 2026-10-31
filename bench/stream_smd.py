"""SMD (Server Machine Dataset) stream: one machine at a time, an asset-disjoint confirmation set.

Protocol: the normal-only train file is split 70/30 into commissioning (fit) and label-free
calibration (cal); the labelled test file is the scored stream. Windows are K non-overlapping
rows that never span a split. Time is a synthetic 1-minute grid (train then test, contiguous),
so events and false-alarm rates use that grid, not wall-clock time. An event is a maximal run
of anomalous test rows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from bench.streams import Stream
from uengram import smd


def smd_stream(machine: str, root: str | Path | None = None, K: int = 10, fit_frac: float = 0.7) -> Stream:
    train, test, label = smd.load(machine, root or smd.DEFAULT_ROOT)
    n_tr = len(train)
    X = np.concatenate([train, test])
    split = np.array(["fit"] * int(n_tr * fit_frac) + ["cal"] * (n_tr - int(n_tr * fit_frac))
                     + ["test"] * len(test), dtype=object)
    t = smd.EPOCH + np.arange(len(X)).astype("timedelta64[m]")
    W, S, E, SPL = [], [], [], []
    for s in range(0, len(X) - K + 1, K):
        block = split[s:s + K]
        if (block == block[0]).all():
            W.append(X[s:s + K].T)
            S.append(t[s])
            E.append(t[s + K - 1])
            SPL.append(block[0])
    W = np.stack(W).astype(np.float32)
    S, E, SPL = np.array(S, "datetime64[s]"), np.array(E, "datetime64[s]"), np.array(SPL, dtype=object)
    G = np.zeros(len(W), dtype=np.int64)
    events = tuple((str(t[n_tr + a].astype("datetime64[s]")), str(t[n_tr + b].astype("datetime64[s]")))
                   for a, b in smd.label_runs(label))
    return Stream(
        name=machine, W=W, start=S, end=E, segment=G, state=np.zeros(len(W), np.int64),
        monitorable=np.r_[False, G[1:] == G[:-1]], fit=SPL == "fit", cal=SPL == "cal", test=SPL == "test",
        events=events, raw_bytes=W.shape[1] * K * 4, features=W.mean(axis=2).astype(np.float64),
        feature_names=smd.CHANNELS, before_h=0, after_h=0, extra={"machine": machine}, point_labels=False,
    )

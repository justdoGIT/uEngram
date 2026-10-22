"""MetroPT-3 loader and windowing: air compressor (APU) of a Porto metro train,
February to September 2020, with the operator's air-leak failure reports.

The published CSV holds 1,516,948 rows at about one row per 10 s (the dataset
description says 1 Hz and 15,169,480 points; the file is a 1-in-10 subsample,
its index column steps by 10). Logging has gaps: 331 longer than 60 s, 909 h in
total, the longest 48 h. Windows never span a gap.
"""
from __future__ import annotations

import csv
import os
import urllib.request
import zipfile

import numpy as np

URL = "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"
CSV = "MetroPT3(AirCompressor).csv"
CACHE = "metropt3.npz"
ANALOG = ("TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current")
# Operator failure reports (all "air leak, high stress"). The published table
# numbers the second report "#1" too, and its maintenance note ("30 Apr 12:00")
# precedes that failure, so maintenance times are not used for scoring.
FAILURES = (
    ("2020-04-18T00:00", "2020-04-18T23:59"),
    ("2020-05-29T23:30", "2020-05-30T06:00"),
    ("2020-06-05T10:00", "2020-06-07T14:30"),
    ("2020-07-15T14:30", "2020-07-15T19:00"),
)
TRAIN_END = "2020-03-01T00:00"  # dataset recommendation: first month for training


def _failure_bounds() -> tuple[np.ndarray, np.ndarray]:
    fs = np.array([a for a, _ in FAILURES], dtype="datetime64[s]")
    fe = np.array([b for _, b in FAILURES], dtype="datetime64[s]")
    return fs, fe


def _parse_csv(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ts, rows = [], []
    with open(path, newline="") as fh:
        r = csv.reader(fh)
        cols = next(r)[2:]  # leading unnamed index column, then timestamp
        for row in r:
            ts.append(row[1])
            rows.append(row[2:])
    return np.array(ts, dtype="datetime64[s]"), np.array(rows, dtype=np.float64), cols


def fetch(dest: str = "data/metropt3") -> str:
    """Download and extract the CSV if needed; cache it as compressed npz."""
    os.makedirs(dest, exist_ok=True)
    cache = os.path.join(dest, CACHE)
    if os.path.exists(cache):
        return cache
    csv_path = os.path.join(dest, CSV)
    if not os.path.exists(csv_path):
        zpath = os.path.join(dest, "metropt3.zip")
        if not os.path.exists(zpath):
            urllib.request.urlretrieve(URL, zpath)
        with zipfile.ZipFile(zpath) as zf:
            if CSV not in zf.namelist():
                raise FileNotFoundError(f"{CSV} not in archive; members: {zf.namelist()}")
            zf.extract(CSV, dest)
    t, X, cols = _parse_csv(csv_path)
    np.savez_compressed(cache, t=t.astype(np.int64), X=X, cols=np.array(cols))
    return cache


def load(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(t datetime64[s] [n], X [n, 15], column names)."""
    z = np.load(path)
    return z["t"].astype("datetime64[s]"), z["X"], [str(c) for c in z["cols"]]


def windows(t: np.ndarray, X: np.ndarray, cols: list[str], K: int,
            channels=ANALOG, max_gap_s: int = 60):
    """Non-overlapping windows of K samples that do not span a logging gap.

    Returns (W[N, C, K] float32, start[N], end[N] (datetime64[s]), segment[N]):
    segment ids change at every gap longer than max_gap_s, so successive
    windows with different ids are not adjacent in time.
    """
    idx = [cols.index(c) for c in channels]
    brk = np.flatnonzero(np.diff(t).astype(np.int64) > max_gap_s) + 1
    bounds = np.r_[0, brk, len(t)]
    W, S, E, G = [], [], [], []
    for g, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        for s in range(a, b - K + 1, K):
            W.append(X[s : s + K, idx].T)
            S.append(t[s])
            E.append(t[s + K - 1])
            G.append(g)
    if not W:
        return (np.zeros((0, len(idx), K), np.float32), np.zeros(0, "datetime64[s]"),
                np.zeros(0, "datetime64[s]"), np.zeros(0, np.int64))
    return (np.stack(W).astype(np.float32), np.array(S, dtype="datetime64[s]"),
            np.array(E, dtype="datetime64[s]"), np.array(G, dtype=np.int64))


def failure_id(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Index of the reported failure a window overlaps, else -1."""
    fs, fe = _failure_bounds()
    out = np.full(len(start), -1, dtype=np.int64)
    for i in range(len(fs)):
        out[(start <= fe[i]) & (end >= fs[i])] = i
    return out


def clean_negative(start: np.ndarray, end: np.ndarray, before_h: int = 24,
                   after_h: int = 48) -> np.ndarray:
    """Windows outside every [failure start - before_h, failure end + after_h]:
    a leak develops before it is reported and persists until repaired, so
    windows next to a failure are neither clean positives nor negatives."""
    fs, fe = _failure_bounds()
    ok = np.ones(len(start), dtype=bool)
    for a, b in zip(fs - np.timedelta64(before_h, "h"), fe + np.timedelta64(after_h, "h")):
        ok &= (end < a) | (start > b)
    return ok

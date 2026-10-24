"""SKAB (Skoltech Anomaly Benchmark) loader and windowing: a water-circulation
testbed (centrifugal pump, inlet/outlet valves) with pointwise-labelled
injected faults. Source: https://github.com/waico/SKAB, pinned commit
b2c0d46c2971dcbfe71e26087b6d231998bb91c2, GPL-3.0.

This is a different physical system from this project's compressor data
(MetroPT-2/3): a water loop, not an air compressor, and all 35 files come
from one testbed, not multiple assets. It fills one specific gap
docs/research-remediation-plan.md identified: exact, independently verified
pointwise fault-onset and fault-end timestamps, never used to develop or
select this project's frozen detector or alarm policy. Protocol in
docs/validation-skab-protocol.md.

Each of the 34 fault files contains exactly one contiguous labelled anomaly
(verified by run-length analysis before any score was generated); the 2-4
`changepoint` marks inside that run are internal phase boundaries of that
one anomaly, not separate faults. `anomaly-free.csv` is pure healthy data
used only for commissioning and calibration.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np

CHANNELS = ("Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure",
            "Temperature", "Thermocouple", "Voltage", "Volume Flow RateRMS")
DEFAULT_ROOT = Path("data/skab")
# The 34 labelled fault experiments, fixed order (never reordered after this
# file is written, so experiment indices are stable across runs).
FAULT_FILES: tuple[str, ...] = (
    tuple(f"valve1/{i}.csv" for i in range(16))
    + tuple(f"valve2/{i}.csv" for i in range(4))
    + tuple(f"other/{i}.csv" for i in range(1, 15))
)
# Synthetic seconds inserted between files' internal clocks: far larger than
# evaluate_alarms' 24h early-alarm search window, so no alarm from one
# experiment (or from commissioning/calibration) can be misattributed to
# another's lead time.
FILE_SEP_S = 30 * 86400
COMMIT = "b2c0d46c2971dcbfe71e26087b6d231998bb91c2"
ALL_FILES: tuple[str, ...] = ("anomaly-free/anomaly-free.csv", *FAULT_FILES)


def _blob_sha1(data: bytes) -> str:
    h = hashlib.sha1()
    h.update(f"blob {len(data)}\0".encode())
    h.update(data)
    return h.hexdigest()


def fetch(dest: str | Path = DEFAULT_ROOT, commit: str = COMMIT) -> Path:
    """Download the 35 CSVs at the pinned commit if missing, verified against
    the git blob SHA-1 GitHub's tree API reports for that commit (there is no
    vendor-bundled checksum manifest). Raises on any mismatch."""
    dest = Path(dest)
    tree_url = f"https://api.github.com/repos/waico/SKAB/git/trees/{commit}?recursive=1"
    with urllib.request.urlopen(tree_url, timeout=30) as r:
        tree = json.load(r)
    shas = {t["path"][len("data/"):]: t["sha"] for t in tree["tree"]
            if t["path"].startswith("data/") and t["path"].endswith(".csv")}
    for rel in ALL_FILES:
        out = dest / rel
        if out.exists() and _blob_sha1(out.read_bytes()) == shas[rel]:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/waico/SKAB/{commit}/data/{rel}"
        with urllib.request.urlopen(url, timeout=30) as r:
            data = r.read()
        got = _blob_sha1(data)
        if got != shas[rel]:
            raise ValueError(f"{rel}: git blob sha1 mismatch (got {got}, expected {shas[rel]})")
        out.write_bytes(data)
    return dest


def _read_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(seconds_since_file_start [n] int64, X [n,8] float64, anomaly [n] int8,
    changepoint [n] int8); label columns are zero when absent (anomaly-free)."""
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.reader(fh, delimiter=";"))
    header, rows = rows[0], rows[1:]
    idx = [header.index(c) for c in CHANNELS]
    t0 = dt.datetime.strptime(rows[0][0], "%Y-%m-%d %H:%M:%S")
    secs = np.array([
        (dt.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S") - t0).total_seconds()
        for row in rows
    ], dtype=np.int64)
    X = np.array([[float(row[i]) for i in idx] for row in rows], dtype=np.float64)
    if "anomaly" in header:
        ai, ci = header.index("anomaly"), header.index("changepoint")
        anomaly = np.array([int(float(row[ai])) for row in rows], dtype=np.int8)
        changepoint = np.array([int(float(row[ci])) for row in rows], dtype=np.int8)
    else:
        anomaly = np.zeros(len(rows), dtype=np.int8)
        changepoint = np.zeros(len(rows), dtype=np.int8)
    return secs, X, anomaly, changepoint


def load(root: str | Path = DEFAULT_ROOT, fit_frac: float = 0.3, K: int = 10) -> dict:
    """Build the concatenated row stream: commissioning (fit split of
    anomaly-free), calibration (the rest of anomaly-free), then the 34 fault
    experiments in FAULT_FILES order. Every chunk is separated by FILE_SEP_S
    in the returned synthetic clock, so windows never bridge two experiments
    or bridge fit into cal, and no 24h lead-time search can cross a boundary.

    `fit_frac=0.3` and the default `K=10`: at alpha=0.003 the conformal
    threshold needs about ceil((1-alpha)/alpha) ~= 333 calibration windows
    to be finite (uengram.alarm._commission's own rule); this split gives
    commissioning ~280 windows (enough for the 48->16 PCA/noise fit) and
    calibration ~650 windows, comfortably above that minimum.

    Returns a dict: t (datetime64[s], strictly increasing), X [n,8], split
    (object array of 'fit'|'cal'|'test' per row), experiment (int64, -1
    outside test, else index into FAULT_FILES), onset/offset (dict
    experiment_index -> datetime64[s], the first/last anomalous row's
    synthetic time).
    """
    root = Path(root)
    secs, X, anomaly, _ = _read_file(root / "anomaly-free/anomaly-free.csv")
    n_af = len(secs)
    n_fit = (int(n_af * fit_frac) // K) * K

    t_parts, X_parts, split_parts, exp_parts = [], [], [], []
    onset: dict[int, int] = {}
    offset: dict[int, int] = {}
    clock = 0

    def _append(local_secs: np.ndarray, Xc: np.ndarray, anomalyc: np.ndarray,
                split: str, exp_idx: int) -> None:
        nonlocal clock
        n = len(local_secs)
        t_parts.append(clock + local_secs)
        X_parts.append(Xc)
        split_parts.append(np.full(n, split, dtype=object))
        exp_parts.append(np.full(n, exp_idx, dtype=np.int64))
        if exp_idx >= 0:
            hits = np.flatnonzero(anomalyc)
            if len(hits):
                onset[exp_idx] = int(clock + local_secs[hits[0]])
                offset[exp_idx] = int(clock + local_secs[hits[-1]])
        clock = int(clock + local_secs[-1]) + FILE_SEP_S

    _append(secs[:n_fit], X[:n_fit], anomaly[:n_fit], "fit", -1)
    rest0 = secs[n_fit]
    _append(secs[n_fit:] - rest0, X[n_fit:], anomaly[n_fit:], "cal", -1)
    for i, rel in enumerate(FAULT_FILES):
        fsecs, fX, fanom, _ = _read_file(root / rel)
        _append(fsecs, fX, fanom, "test", i)

    t = np.concatenate(t_parts).astype("datetime64[s]")
    X = np.concatenate(X_parts).astype(np.float64)
    split = np.concatenate(split_parts)
    experiment = np.concatenate(exp_parts)
    missing = set(range(len(FAULT_FILES))) - set(onset)
    if missing:
        raise ValueError(f"fault files with no labelled anomaly: {sorted(missing)}")
    epoch = np.datetime64("1970-01-01T00:00:00")
    onset_t = {i: epoch + np.timedelta64(v, "s") for i, v in onset.items()}
    offset_t = {i: epoch + np.timedelta64(v, "s") for i, v in offset.items()}
    return dict(t=t, X=X, split=split, experiment=experiment,
                onset=onset_t, offset=offset_t, n_fit=n_fit)


def windows(t: np.ndarray, X: np.ndarray, split: np.ndarray, experiment: np.ndarray,
            K: int, max_gap_s: int = 60):
    """Non-overlapping K-sample windows that never span a >max_gap_s logging
    gap or a fit/cal/test boundary. Returns W[N,8,K] float32, start/end
    (datetime64[s]), segment[N] int64 (monotonic block id, for the
    'monitorable' convention), split[N] object (the single split all K rows
    share), experiment[N] int64."""
    brk = np.flatnonzero(np.diff(t).astype(np.int64) > max_gap_s) + 1
    bounds = np.r_[0, brk, len(t)]
    W, S, E, G, SPL, EXP = [], [], [], [], [], []
    for g, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        for s in range(a, b - K + 1, K):
            seg_split, seg_exp = split[s:s + K], experiment[s:s + K]
            if not (seg_split == seg_split[0]).all() or not (seg_exp == seg_exp[0]).all():
                continue
            W.append(X[s:s + K].T)
            S.append(t[s])
            E.append(t[s + K - 1])
            G.append(g)
            SPL.append(seg_split[0])
            EXP.append(int(seg_exp[0]))
    if not W:
        return (np.zeros((0, len(CHANNELS), K), np.float32), np.zeros(0, "datetime64[s]"),
                np.zeros(0, "datetime64[s]"), np.zeros(0, np.int64), np.zeros(0, object),
                np.zeros(0, np.int64))
    return (np.stack(W).astype(np.float32), np.array(S, dtype="datetime64[s]"),
            np.array(E, dtype="datetime64[s]"), np.array(G, dtype=np.int64),
            np.array(SPL, dtype=object), np.array(EXP, dtype=np.int64))

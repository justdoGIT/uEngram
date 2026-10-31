"""NASA CMAPSS loader and windowing (subsets FD001-FD004; FD001 is the default)."""
from __future__ import annotations

import io
import os
import urllib.request
import zipfile

import numpy as np

URL = ("https://phm-datasets.s3.amazonaws.com/NASA/"
       "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip")
URL_BYTES = 12_429_152
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
FILES = ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt")
# columns: unit, cycle, op1-3, s1-s21; keep the 14 informative sensors
SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
SENSOR_COLS = [4 + s for s in SENSORS]  # s1 is column 5 (0-based index 5)
RUL_CAP = 125


def files(subset: str = "FD001") -> tuple[str, ...]:
    if subset not in SUBSETS:
        raise ValueError(f"unknown CMAPSS subset {subset!r}; expected one of {SUBSETS}")
    return (f"train_{subset}.txt", f"test_{subset}.txt", f"RUL_{subset}.txt")


def _find(zf: zipfile.ZipFile, seen: list[str], wanted: tuple[str, ...]) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    for name in zf.namelist():
        seen.append(name)
        base = os.path.basename(name)
        if base in wanted:
            found[base] = zf.read(name)
    if len(found) < len(wanted):
        for name in zf.namelist():
            if name.lower().endswith(".zip"):
                inner = zipfile.ZipFile(io.BytesIO(zf.read(name)))
                for k, v in _find(inner, seen, wanted).items():
                    found.setdefault(k, v)
    return found


def fetch(dest: str = "data/cmapss", subset: str = "FD001") -> str:
    wanted = files(subset)
    os.makedirs(dest, exist_ok=True)
    if all(os.path.exists(os.path.join(dest, f)) for f in wanted):
        return dest
    zpath = os.path.join(dest, "cmapss.zip")
    if not (os.path.exists(zpath) and os.path.getsize(zpath) == URL_BYTES):
        urllib.request.urlretrieve(URL, zpath)
    seen: list[str] = []
    found = _find(zipfile.ZipFile(zpath), seen, wanted)
    missing = [f for f in wanted if f not in found]
    if missing:
        raise FileNotFoundError(f"{missing} not in archive; members seen: {seen}")
    for f in wanted:
        with open(os.path.join(dest, f), "wb") as fh:
            fh.write(found[f])
    return dest


def load(path: str) -> np.ndarray:
    return np.loadtxt(path)


def condition_ids(data: np.ndarray) -> np.ndarray:
    """Operating condition per row: the first operating setting rounded to an integer.

    FD002 and FD004 take six values (0, 10, 20, 25, 35, 42, each with 0.01 jitter);
    FD001 and FD003 are single-condition (always 0).
    """
    return np.round(data[:, 2]).astype(int)


def fit_condition_scaler(data: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per-condition mean and standard deviation of the 14 sensors, from ``data`` only."""
    cond, x = condition_ids(data), data[:, SENSOR_COLS]
    scaler = {}
    for c in np.unique(cond):
        block = x[cond == c]
        scaler[int(c)] = (block.mean(axis=0), np.maximum(block.std(axis=0), 1e-6))
    return scaler


def windows(data: np.ndarray, K: int = 16, last_only: bool = False, scaler=None):
    """Sliding windows (stride 1) inside each unit.

    Returns (W[N,14,K], unit[N], rul[N]); rul is for the window's last cycle,
    max_cycle - cycle, capped at RUL_CAP. With last_only, only each unit's final
    window is returned (test protocol). With ``scaler`` (from ``fit_condition_scaler``),
    each row's sensors are standardized by the statistics of its own operating condition
    before windowing; the scaler must be fitted on training units only.
    """
    W, U, R = [], [], []
    for u in np.unique(data[:, 0]):
        rows = data[data[:, 0] == u]
        rows = rows[np.argsort(rows[:, 1])]
        x = rows[:, SENSOR_COLS]  # [n,14]
        if scaler is not None:
            cond = condition_ids(rows)
            x = np.stack([(x[i] - scaler[int(c)][0]) / scaler[int(c)][1] for i, c in enumerate(cond)])
        n = len(rows)
        max_cycle = rows[-1, 1]
        starts = [n - K] if last_only else range(0, n - K + 1)
        for s in starts:
            W.append(x[s : s + K].T)
            U.append(int(u))
            R.append(min(max_cycle - rows[s + K - 1, 1], RUL_CAP))
    return np.stack(W).astype(np.float32), np.array(U), np.array(R, dtype=np.float64)

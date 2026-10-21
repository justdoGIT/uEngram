"""NASA CMAPSS FD001 loader and windowing."""
from __future__ import annotations

import io
import os
import urllib.request
import zipfile

import numpy as np

URL = ("https://phm-datasets.s3.amazonaws.com/NASA/"
       "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip")
URL_BYTES = 12_429_152
FILES = ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt")
# columns: unit, cycle, op1-3, s1-s21; keep the 14 informative sensors
SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
SENSOR_COLS = [4 + s for s in SENSORS]  # s1 is column 5 (0-based index 5)
RUL_CAP = 125


def _find(zf: zipfile.ZipFile, seen: list[str]) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    for name in zf.namelist():
        seen.append(name)
        base = os.path.basename(name)
        if base in FILES:
            found[base] = zf.read(name)
    if len(found) < len(FILES):
        for name in zf.namelist():
            if name.lower().endswith(".zip"):
                inner = zipfile.ZipFile(io.BytesIO(zf.read(name)))
                for k, v in _find(inner, seen).items():
                    found.setdefault(k, v)
    return found


def fetch(dest: str = "data/cmapss") -> str:
    os.makedirs(dest, exist_ok=True)
    if all(os.path.exists(os.path.join(dest, f)) for f in FILES):
        return dest
    zpath = os.path.join(dest, "cmapss.zip")
    if not (os.path.exists(zpath) and os.path.getsize(zpath) == URL_BYTES):
        urllib.request.urlretrieve(URL, zpath)
    seen: list[str] = []
    found = _find(zipfile.ZipFile(zpath), seen)
    missing = [f for f in FILES if f not in found]
    if missing:
        raise FileNotFoundError(f"{missing} not in archive; members seen: {seen}")
    for f in FILES:
        with open(os.path.join(dest, f), "wb") as fh:
            fh.write(found[f])
    return dest


def load(path: str) -> np.ndarray:
    return np.loadtxt(path)


def windows(data: np.ndarray, K: int = 16, last_only: bool = False):
    """Sliding windows (stride 1) inside each unit.

    Returns (W[N,14,K], unit[N], rul[N]); rul is for the window's last cycle,
    max_cycle - cycle, capped at RUL_CAP. With last_only, only each unit's final
    window is returned (test protocol).
    """
    W, U, R = [], [], []
    for u in np.unique(data[:, 0]):
        rows = data[data[:, 0] == u]
        rows = rows[np.argsort(rows[:, 1])]
        x = rows[:, SENSOR_COLS]  # [n,14]
        n = len(rows)
        max_cycle = rows[-1, 1]
        starts = [n - K] if last_only else range(0, n - K + 1)
        for s in starts:
            W.append(x[s : s + K].T)
            U.append(int(u))
            R.append(min(max_cycle - rows[s + K - 1, 1], RUL_CAP))
    return np.stack(W).astype(np.float32), np.array(U), np.array(R, dtype=np.float64)

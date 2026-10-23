"""SCANIA Component X stream for frozen, point-label validation.

The dataset has relative time steps and coarse per-vehicle classes, not failure
intervals. This module therefore builds a Stream for causal scoring and keeps
point labels in ``extra`` for the SCANIA-specific evaluator.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from bench.streams import Stream

DEFAULT_ROOT = Path("/home/kk/Backup/data/uengram/scania/2024-34-3")


def _root(root: str | Path | None) -> Path:
    return Path(root or os.environ.get("UENGRAM_SCANIA_ROOT", DEFAULT_ROOT))


def _iter_numeric_csv(path: Path, chunk_rows: int = 8192):
    """Yield bounded numeric chunks; empty CSV fields become NaN."""
    with path.open(encoding="utf-8") as handle:
        next(handle)
        chunk: list[str] = []
        for line in handle:
            chunk.append(line)
            if len(chunk) == chunk_rows:
                yield np.atleast_2d(np.genfromtxt(
                    chunk, delimiter=",", dtype=np.float32,
                    missing_values="", filling_values=np.nan,
                ))
                chunk.clear()
        if chunk:
            yield np.atleast_2d(np.genfromtxt(
                chunk, delimiter=",", dtype=np.float32,
                missing_values="", filling_values=np.nan,
            ))


def _selected_rows(path: Path, keep) -> np.ndarray:
    parts = []
    for chunk in _iter_numeric_csv(path):
        mask = keep(chunk[:, 0].astype(np.int64))
        if mask.any():
            parts.append(chunk[mask])
    if not parts:
        raise ValueError(f"no rows selected from {path}")
    return np.concatenate(parts, axis=0)


def _labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int64))
    return rows[:, 0], rows[:, 1]


def _healthy_ids(path: Path) -> np.ndarray:
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return np.asarray(rows["vehicle_id"][rows["in_study_repair"] == 0], dtype=np.int64)


def _feature_names(path: Path) -> tuple[str, ...]:
    with path.open(encoding="utf-8") as handle:
        return tuple(handle.readline().strip().split(",")[2:])


def load_scania(root: str | Path | None = None) -> Stream:
    """Load train commissioning/calibration and held-out validation rows.

    Healthy training vehicles are split by ``vehicle_id % 5``. Validation rows
    remain untouched until score evaluation. The source's vehicle-grouped row
    order is retained so a vehicle boundary is a segment boundary.
    """
    base = _root(root)
    train_path = base / "data/train_operational_readouts.csv"
    val_path = base / "data/validation_operational_readouts.csv"
    healthy = _healthy_ids(base / "data/train_tte.csv")
    fit_parts, cal_parts = [], []
    for chunk in _iter_numeric_csv(train_path):
        ids = chunk[:, 0].astype(np.int64)
        good = np.isin(ids, healthy)
        if np.any(good & (ids % 5 != 0)):
            fit_parts.append(chunk[good & (ids % 5 != 0)])
        if np.any(good & (ids % 5 == 0)):
            cal_parts.append(chunk[good & (ids % 5 == 0)])
    width = fit_parts[0].shape[1] if fit_parts else cal_parts[0].shape[1]
    fit_rows = np.concatenate(fit_parts, axis=0) if fit_parts else np.empty((0, width), np.float32)
    cal_rows = np.concatenate(cal_parts, axis=0) if cal_parts else np.empty((0, width), np.float32)
    validation = _selected_rows(val_path, lambda ids: np.ones(len(ids), dtype=bool))

    W = np.concatenate((fit_rows[:, 2:], cal_rows[:, 2:], validation[:, 2:])).astype(
        np.float32, copy=False
    )
    vehicle_id = np.concatenate((
        fit_rows[:, 0], cal_rows[:, 0], validation[:, 0]
    )).astype(np.int64, copy=False)
    time_step = np.concatenate((
        fit_rows[:, 1], cal_rows[:, 1], validation[:, 1]
    )).astype(np.float32, copy=False)
    n_fit, n_cal = len(fit_rows), len(cal_rows)
    del fit_rows, cal_rows, validation

    n = len(W)
    end = np.arange(n, dtype=np.int64).astype("datetime64[s]")
    start = end - np.timedelta64(1, "s")
    segment = vehicle_id.copy()
    monitorable = np.r_[False, vehicle_id[1:] == vehicle_id[:-1]]
    fit = np.zeros(n, bool)
    fit[:n_fit] = True
    cal = np.zeros(n, bool)
    cal[n_fit:n_fit + n_cal] = True
    test = ~(fit | cal)

    label_ids, label_values = _labels(base / "data/validation_labels.csv")
    label_map = dict(zip(label_ids.tolist(), label_values.tolist()))
    classes = np.array([label_map.get(int(v), -1) for v in vehicle_id], dtype=np.int8)
    names = _feature_names(train_path)
    missing_any = np.isnan(W).any(axis=1)
    features = W  # a view; freeze frames omit the >20-column raw snapshot
    return Stream(
        name="scania_validation",
        W=W,
        start=start,
        end=end,
        segment=segment,
        state=np.zeros(n, dtype=np.int64),
        monitorable=monitorable,
        fit=fit,
        cal=cal,
        test=test,
        events=(),
        raw_bytes=W.shape[1] * 4,
        features=features,
        feature_names=names,
        extra={
            "vehicle_id": vehicle_id,
            "time_step": time_step,
            "class_label": classes,
            "missing_any": missing_any,
            "fit_rows": n_fit,
            "cal_rows": n_cal,
            "test_rows": int(test.sum()),
        },
        point_labels=True,
    )

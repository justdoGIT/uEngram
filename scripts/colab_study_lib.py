"""Shared helpers for the EngineAD CNN seed study; driven by colab_run_study.py."""
from __future__ import annotations

import hashlib
import math
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_SALT = "uengram-enginead-2.2-asset-split-v1"
TRAIN_COUNT, CAL_COUNT = 15, 5
SEEDS = tuple(range(10))
DIMENSIONS = (8, 16, 32)
EPOCHS, BATCH, NOISE_LAG, CAL_ALPHA = 30, 512, 1, 0.05
CHUNK = 8 * 1024 * 1024
FEATURES = tuple(f"PC{i}" for i in range(1, 9))

ALLOWED_GLOBALS = {
    ("builtins", "slice"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("pandas._libs.internals", "_unpickle_block"),
    ("pandas.core.frame", "DataFrame"),
    ("pandas.core.internals.managers", "BlockManager"),
    ("pandas.core.indexes.base", "_new_Index"),
    ("pandas.core.indexes.base", "Index"),
    ("pandas.core.indexes.range", "RangeIndex"),
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) not in ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")
        return super().find_class(module, name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _split(names: list[str]) -> dict[str, list[str]]:
    def key(name: str) -> str:
        return hashlib.sha256(f"{SPLIT_SALT}\0{name}".encode()).hexdigest()

    ordered = sorted(names, key=key)
    return {
        "train": ordered[:TRAIN_COUNT],
        "calibration": ordered[TRAIN_COUNT:TRAIN_COUNT + CAL_COUNT],
        "test": ordered[TRAIN_COUNT + CAL_COUNT:],
    }


def load_segments(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    with path.open("rb") as source:
        segments = RestrictedUnpickler(source).load()
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"{path.name}: expected non-empty segment list")
    match = re.fullmatch(r"truck_(\d+)-1\.pickle", path.name)
    if match is None:
        raise ValueError(f"unexpected EngineAD filename: {path.name}")
    expected_vehicle = f"truck {match.group(1)}"

    x_parts: list[np.ndarray] = []
    first_labels: list[int] = []
    any_positive: list[int] = []
    all_normal: list[bool] = []
    mixed_count = 0
    for index, frame in enumerate(segments):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{path.name}: segment {index} is not a DataFrame")
        if tuple(c for c in frame.columns if c not in ("label", "vehicle_id")) != FEATURES:
            raise ValueError(f"{path.name}: unexpected PCA feature schema at segment {index}")
        if len(frame) != 300:
            raise ValueError(f"{path.name}: segment {index} has {len(frame)} rows, expected 300")
        labels = frame["label"].to_numpy()
        if not np.isin(labels, (0, 1)).all():
            raise ValueError(f"{path.name}: invalid point label at segment {index}")
        vehicle_ids = frame["vehicle_id"].drop_duplicates().tolist()
        if len(vehicle_ids) != 1 or str(vehicle_ids[0]) != expected_vehicle:
            raise ValueError(f"{path.name}: vehicle ID mismatch at segment {index}")
        values = frame.loc[:, list(FEATURES)].to_numpy(dtype=np.float32, copy=False)
        if not np.isfinite(values).all():
            raise ValueError(f"{path.name}: non-finite feature at segment {index}")

        first_labels.append(int(labels[0]))
        any_positive.append(int(np.any(labels == 1)))
        all_normal.append(bool(np.all(labels == 0)))
        mixed_count += int(np.any(labels != labels[0]))
        x_parts.append(values.T)

    first = np.asarray(first_labels, dtype=np.int8)
    any_label = np.asarray(any_positive, dtype=np.int8)
    clean = np.asarray(all_normal, dtype=bool)
    audit = {
        "file": path.name,
        "segment_count": len(first),
        "all_point_labels_normal": int(clean.sum()),
        "first_row_anomaly_segments": int(first.sum()),
        "any_positive_segments": int(any_label.sum()),
        "mixed_label_segments": mixed_count,
        "first_vs_any_disagreement_segments": int(np.count_nonzero(first != any_label)),
    }
    return np.stack(x_parts), first, any_label, clean, audit


def normal_training_groups(clean_normal: np.ndarray, first_group: int) -> tuple[np.ndarray, np.ndarray, int]:
    indices = np.flatnonzero(clean_normal)
    if not len(indices):
        raise ValueError("training vehicle has no all-normal segments")
    groups = np.empty(len(indices), dtype=np.int64)
    group_id, previous = first_group, None
    for position, index in enumerate(indices):
        if previous is None or index != previous + 1:
            group_id += 1
        groups[position] = group_id
        previous = int(index)
    return indices, groups, group_id


# Metrics & Stats
def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="stable")
    sa = a[order]
    ranks = np.empty(len(a), dtype=np.float64)
    n = len(a)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def auroc(scores, labels_bool) -> float:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels_bool, dtype=bool)
    ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _rankdata(s)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(scores, labels_bool) -> float:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels_bool, dtype=bool)
    ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    ends = np.flatnonzero(y_sorted)
    gains = np.ones(len(ends))
    return float(np.sum(gains * tp[ends] / (ends + 1)) / n_pos)


def conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    s = np.sort(np.asarray(scores, dtype=np.float64)[~np.isnan(scores)])
    m = len(s)
    k = int(math.ceil((1.0 - alpha) * (m + 1)))
    return float(s[k - 1]) if k <= m else float("inf")


def _diff_pairs(z: np.ndarray, lag: int, groups: np.ndarray | None) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    delta = z[lag:] - z[:-lag]
    if groups is not None:
        delta = delta[groups[lag:] == groups[:-lag]]
    return delta


def robust_noise_sigma(z: np.ndarray, lag: int, groups: np.ndarray | None) -> np.ndarray:
    delta = _diff_pairs(z, lag, groups)
    if delta.shape[0] < 10:
        raise ValueError("fewer than 10 difference pairs for noise estimate")
    med = np.median(delta, axis=0)
    mad = np.median(np.abs(delta - med), axis=0)
    return np.maximum(1.4826 * mad / math.sqrt(2.0), 1e-6)


def robust_noise_cov(z: np.ndarray, lag: int, groups: np.ndarray | None,
                     trim: float = 0.975, iters: int = 5) -> np.ndarray:
    delta = _diff_pairs(z, lag, groups) / math.sqrt(2.0)
    if delta.shape[0] < 10:
        raise ValueError("fewer than 10 difference pairs for noise estimate")
    n, d = delta.shape
    C = np.cov(delta, rowvar=False)
    if d == 1:
        C = np.array([[float(C)]])
    for _ in range(iters):
        try:
            invC = np.linalg.pinv(C)
        except np.linalg.LinAlgError:
            break
        d2 = np.einsum("ij,jk,ik->i", delta, invC, delta)
        cutoff = float(np.percentile(d2, trim * 100.0))
        keep = d2 <= cutoff
        if keep.sum() < max(d + 1, 10):
            break
        C = np.cov(delta[keep], rowvar=False)
        if d == 1:
            C = np.array([[float(C)]])
    return C


class Whitener:
    def fit(self, F: np.ndarray, d: int) -> "Whitener":
        F = np.asarray(F, dtype=np.float64)
        self.m = F.mean(axis=0)
        lam, vec = np.linalg.eigh(np.cov(F, rowvar=False))
        order = np.argsort(lam)[::-1][:d]
        lam, vec = lam[order], vec[:, order]
        lam = np.maximum(lam, 1e-8 * max(float(lam.max()), 1e-12))
        self.V = vec
        self.lam = lam
        return self

    def transform(self, F: np.ndarray) -> np.ndarray:
        return (np.asarray(F, dtype=np.float64) - self.m) @ self.V / np.sqrt(self.lam)


class NoiseWhitener:
    def __init__(self, rcond: float = 1e-12):
        self.rcond = rcond

    def fit(self, z: np.ndarray, noise_lag: int, groups: np.ndarray | None) -> "NoiseWhitener":
        lam, V = np.linalg.eigh(robust_noise_cov(z, noise_lag, groups))
        lam = np.maximum(lam, self.rcond * max(float(lam.max()), 1e-12))
        self.W = (V / np.sqrt(lam)) @ V.T
        return self

    def transform(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=np.float64) @ self.W


def fit_projection(X: np.ndarray, d: int, noise_lag: int, groups: np.ndarray | None, rcond: float = 1e-12):
    whitener = Whitener().fit(X, d)
    Zp = whitener.transform(X)
    noise = NoiseWhitener(rcond=rcond).fit(Zp, noise_lag, groups)
    return whitener, noise, robust_noise_sigma(noise.transform(Zp), noise_lag, groups)
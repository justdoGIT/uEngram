"""Ticks -> window features, applying each attribute's window rule (numpy).

Windows hold K consecutive 1 Hz ticks aligned to multiples of K seconds of
wall time; only complete windows are used. Every level is a function of
window *sums* (a ratio of sums, never a mean of per-tick ratios: per-tick
ratios with small denominators are noisy and their mean is biased by
Jensen's inequality). A missing input (counter reset, PMU group not
scheduled for half the window, zero denominator) gives NaN; the encoder
imputes the training median for it and counts the imputation.

The thermal_residual column is left NaN here: it needs the fitted thermal
model and the previous window, which the encoder owns. The window means of
package temperature and power it needs are returned as context.
"""
from __future__ import annotations

import gzip
import json
from array import array
from dataclasses import dataclass

import numpy as np

from . import schema

Columns = dict[str, np.ndarray]

PMU_FIELDS = ("pmu_inst", "pmu_cyc", "pmu_ref", "pmu_brmiss", "pmu_llcmiss")
# Scaled counts are an estimate when the group shares the PMU (two system-wide
# groups that both need fixed counter 2 alternate, each running ~50 %); below a
# quarter of the time the window's PMU attributes count as missing.
MIN_PMU_RUNNING = 0.25
RATIO_EPS = 1e-6
BURST_EPS = 1e-2  # transformed units: a tick-to-tick scale below 0.01 is indistinguishable


@dataclass
class WindowBatch:
    t0: np.ndarray  # [W] window start, wall seconds
    F: np.ndarray  # [W, n_features], schema.feature_names() order (Gaussian model)
    events: np.ndarray  # [W, n_event], raw counts of schema.event_attributes() (count model)
    temp_mean: np.ndarray  # [W] degC
    power_mean: np.ndarray  # [W] W
    outcomes: dict[str, np.ndarray]  # schema.OUTCOMES, per window
    quality: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.t0)


def tick_columns(ticks: list[dict]) -> Columns:
    cols = {"t": np.array([r["t"] for r in ticks], dtype=np.float64)}
    for f in schema.TICK_FIELDS:
        cols[f] = np.array([np.nan if r.get(f) is None else r[f] for r in ticks], dtype=np.float64)
    return cols


def read_columns(paths: list[str]) -> tuple[dict, Columns]:
    """Stream tick files (JSONL, optionally .gz) into float64 columns, 8 bytes
    per value: three days of 1 Hz ticks are ~90 MB instead of ~600 MB of dicts.
    Truncated lines are skipped; rows are returned in time order."""
    meta: dict = {}
    buf = {f: array("d") for f in ("t", *schema.TICK_FIELDS)}
    nan = float("nan")
    for p in paths:
        with (gzip.open(p, "rt") if p.endswith(".gz") else open(p)) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "meta" in rec:
                    meta = meta or rec["meta"]
                    continue
                for f, a in buf.items():
                    v = rec.get(f)
                    a.append(nan if v is None else float(v))
    cols = {f: np.frombuffer(a, dtype=np.float64) for f, a in buf.items()}
    if len(cols["t"]) > 1 and np.any(np.diff(cols["t"]) < 0):
        order = np.argsort(cols["t"], kind="stable")
        cols = {f: v[order] for f, v in cols.items()}
    return meta, cols


def select(cols: Columns, mask: np.ndarray) -> Columns:
    return {f: v[mask] for f, v in cols.items()}


def window_index(t: np.ndarray, K: int, offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Rows of complete windows [offset + wK, offset + (w+1)K): (window ids [W], row index [W, K])."""
    wid = np.floor((t - offset) / K).astype(np.int64)
    ids, start, count = np.unique(wid, return_index=True, return_counts=True)
    keep = count == K
    # ticks are time-ordered, so a complete window is K contiguous rows
    rows = start[keep][:, None] + np.arange(K)[None, :]
    return ids[keep], rows


def _angular(p: np.ndarray) -> np.ndarray:
    return np.arcsin(np.sqrt(np.clip(p, 0.0, 1.0)))


def _div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _level(a: schema.Attribute, S: dict[str, np.ndarray], M: dict[str, np.ndarray]) -> np.ndarray:
    num = sum(S[f] for f in a.num) * a.scale
    if a.kind == "fraction":
        return _angular(_div(num, S[a.den]))
    if a.kind == "gauge_fraction":
        return _angular(M[a.num[0]])
    if a.kind == "log_ratio":
        return np.log(_div(num, S[a.den]) + RATIO_EPS)
    if a.kind == "log1p_rate":
        return np.log1p(_div(num, S["dt_s"]))
    if a.kind == "log1p_count":
        return np.log1p(num)
    if a.kind == "log_rate":
        r = _div(num, S[a.den])
        return np.where(r > 0, np.log(np.where(r > 0, r, 1.0)), np.nan)
    if a.kind == "residual":
        return np.full(num.shape, np.nan)
    raise ValueError(a.kind)


def _burst(b: schema.BurstBlock, X: dict[str, np.ndarray]) -> np.ndarray:
    """[W]: log of the robust tick-to-tick noise scale of the activity series."""
    num = sum(X[f] for f in b.num) * b.scale
    r = _div(num, X[b.den])
    s = _angular(r) if b.transform == "angular" else np.log1p(r)
    d = np.diff(s, axis=1)
    mad = np.median(np.abs(d - np.median(d, axis=1, keepdims=True)), axis=1)
    out = np.log(1.4826 * mad / np.sqrt(2.0) + BURST_EPS)
    out[~np.isfinite(s).all(axis=1)] = np.nan
    return out


def windows(ticks: list[dict], K: int, offset: float = 0.0) -> WindowBatch:
    return windows_from_columns(tick_columns(ticks), K, offset)


def windows_from_columns(cols: Columns, K: int, offset: float = 0.0) -> WindowBatch:
    ids, rows = window_index(cols["t"], K, offset)
    X = {f: cols[f][rows] for f in schema.TICK_FIELDS}  # [W, K]
    S = {f: X[f].sum(axis=1) for f in X}  # NaN if any tick is missing the field
    M = {f: X[f].mean(axis=1) for f in X}
    pmu_bad = np.nanmin(X["pmu_running"], axis=1) < MIN_PMU_RUNNING
    for f in PMU_FIELDS:
        S[f] = np.where(pmu_bad, np.nan, S[f])
    F = np.column_stack([_level(a, S, M) for a in schema.gaussian_attributes()]
                        + [_burst(b, X) for b in schema.BURST])
    E = np.column_stack([sum(S[f] for f in a.num) for a in schema.event_attributes()])
    dt = S["dt_s"]
    outcomes = {"ips": _div(S["pmu_inst"], dt), "pkg_watts": _div(S["pkg_energy_j"], dt),
                "nj_per_inst": _div(S["pkg_energy_j"] * 1e9, S["pmu_inst"])}
    quality = {"dt": dt, "pmu_running": np.nanmin(X["pmu_running"], axis=1),
               "read_us_max": np.nanmax(X["read_us"], axis=1),
               "overhead_frac": _div(S["self_cpu_us"] * 1e-6, dt),
               "n_missing": np.isnan(F).sum(axis=1) - 1}  # thermal_residual is filled later
    return WindowBatch(ids.astype(np.float64) * K + offset, F, E, M["pkg_temp_c"],
                       _div(S["pkg_energy_j"], dt), outcomes, quality)

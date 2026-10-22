"""Ablations and baselines for the validation report (numpy).

Ablations re-commission the node on the same commissioning data with one part
switched off (node.Options) and replay the same test windows. Baselines use
the deployed model's encoder (same features, same commissioning split, same
two-window on-delay and the same alpha/2) so only the detector differs:

* saturation_only: a window fires iff a pressure exceeds its commissioning maximum;
* max_z: the largest robust z of any single feature (univariate thresholds);
* single_gaussian: one Gaussian over all commissioning latents (Hotelling T2
  without uEngram's memory of separate states).
"""
from __future__ import annotations

import numpy as np

from ..stats import conformal_threshold
from .features import select, windows_from_columns
from .node import PHASES, NodeMemory, Options
from .validate import block_ci, fired, in_fault, near_fault, replay, touches
from .validate_criteria import wilson

ABLATIONS = {"no_thermal": Options(use_thermal=False), "no_spe": Options(use_spe=False),
             "no_ondelay": Options(on_delay=1), "no_events": Options(use_events=False),
             "no_saturation": Options(use_saturation=False)}
BASELINES = ("saturation_only", "max_z", "single_gaussian")
ALL = ("full", *ABLATIONS, *BASELINES)


def _persistent(t0: np.ndarray, s: np.ndarray, K: int, n: int = 2) -> np.ndarray:
    """Smallest score of the last n consecutive windows, NaN where they are not consecutive."""
    out = np.full(len(s), np.nan)
    for i in range(n - 1, len(s)):
        if t0[i] - t0[i - n + 1] == (n - 1) * K:
            out[i] = np.min(s[i - n + 1: i + 1])
    return out


def _baselines(node: NodeMemory, comm, test) -> dict[str, dict[float, bool]]:
    split = node.commission_report["t_split"]  # the deployed model's fit/calibration cut
    A, B = select(comm, comm["t"] < split), select(comm, comm["t"] >= split)

    def enc(cols, offset=0.0):
        seq = windows_from_columns(cols, node.K, offset)
        Z, X, _, _ = node.enc.encode(seq, None)
        return seq.t0, Z, X

    ZA = enc(A)[1]
    mu, prec = ZA.mean(axis=0), np.linalg.pinv(np.cov(ZA, rowvar=False))
    score = {"max_z": lambda Z, X: np.max(np.abs(X), axis=1),
             "single_gaussian": lambda Z, X: np.einsum("ij,jk,ik->i", Z - mu, prec, Z - mu)}
    # calibrated like the node: persistent scores of the four phase-shifted sequences
    calib = [enc(B, k * node.K / PHASES) for k in range(PHASES)]
    t0T, ZT, XT = enc(test)
    out = {}
    for name, f in score.items():
        sB = np.concatenate([_persistent(t0, f(Z, X), node.K, node.opt.on_delay) for t0, Z, X in calib])
        thr = conformal_threshold(sB, node.alpha / 2)
        sT = _persistent(t0T, f(ZT, XT), node.K, node.opt.on_delay)
        out[name] = {float(t): bool(s > thr) for t, s in zip(t0T, sT)}
    return out


def run_variants(node: NodeMemory, meta: dict, comm, test, recs_full: list[dict], alpha: float,
                 which: tuple[str, ...] | None = None) -> dict[str, dict[float, bool]]:
    """variant -> {window start: fired}."""
    which = which or ALL
    out = {}
    if "full" in which:
        out["full"] = {r["t0"]: fired(r) for r in recs_full}
    if "saturation_only" in which:
        out["saturation_only"] = {r["t0"]: bool(r["saturation"]) for r in recs_full}
    for name, opt in ABLATIONS.items():
        if name in which:
            n = NodeMemory.commission(comm, meta, window=node.K, alpha=alpha, options=opt)
            out[name] = {r["t0"]: fired(r) for r in replay(n, test)}
    if any(b in which for b in ("max_z", "single_gaussian")):
        out.update({k: v for k, v in _baselines(node, comm, test).items() if k in which})
    return out


def variant_summary(var_rows: dict[str, dict[str, dict[float, bool]]], faults: list[dict], K: int) -> dict:
    """variant -> {"fa": false-alarm share of normal windows with CI, "det": detected
    share of executed faults touching the node with CI, "per_type": {type: (share, n)}}."""
    types = sorted({f["type"] for f in faults if f.get("status") == "done"})
    out = {}
    for v in [v for v in ALL if any(v in rows for rows in var_rows.values())]:
        normal, hits = [], {t: [] for t in types}
        for name, rows in var_rows.items():
            series = rows.get(v)
            if series is None:
                continue
            for t0 in sorted(series):
                if not near_fault(t0, faults, name, K):
                    normal.append(series[t0])
            for f in faults:
                if f.get("status") != "done" or touches(f, name) is None:
                    continue
                hits[f["type"]].append(any(ok for t0, ok in series.items() if in_fault(t0, f, K)))
        allhits = [h for t in types for h in hits[t]]
        out[v] = {"fa": block_ci(normal), "det": wilson(sum(allhits), len(allhits)),
                  "per_type": {t: (float(np.mean(h)) if h else float("nan"), len(h)) for t, h in hits.items()}}
    return out


def variant_table(summary: dict) -> list[str]:
    types = sorted({t for s in summary.values() for t in s["per_type"]})
    lines = ["| variant | false alarms (normal windows) | detected, all faults | " + " | ".join(types) + " |",
             "|" + "---|" * (3 + len(types))]
    for v, s in summary.items():
        (m, lo, hi), (p, plo, phi) = s["fa"], s["det"]
        per = " | ".join(f"{s['per_type'][t][0]:.2f} ({s['per_type'][t][1]})" if s["per_type"][t][1] else "-"
                         for t in types)
        lines.append(f"| {v} | {m:.4f} [{lo:.4f}, {hi:.4f}] | {p:.2f} [{plo:.2f}, {phi:.2f}] | {per} |")
    return lines

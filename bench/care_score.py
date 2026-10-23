"""CARE score (Coverage, Accuracy, Reliability, Earliness) for CARE to Compare.

Adapted from EnergyFaultDetector's CAREScore and calculate_criticality algorithms
(Fraunhofer IEE, MIT license; see ``src/uengram/EnergyFaultDetector-LICENSE.txt``).
Changes: NumPy replaces pandas/scikit-learn for array metrics, and persistence
helpers are omitted; this module exposes a stateless event/final-score API.
Defaults as published: coverage and reliability F_0.5, weights 1/2/1/1
(coverage/accuracy/earliness/reliability), criticality threshold 72, and earliness
weights that start to decay at 1/4 of the event. Normal behaviour is status True.

Per event: ``predicted`` and ``normal`` cover the event dataset's prediction
period in time order; ``in_event`` marks the labelled event window (all False
for normal events); ``end_pos`` is the inclusive last row index used to assess
criticality.
"""

from __future__ import annotations

import numpy as np

CRITICALITY_THRESHOLD = 72
BETA = 0.5
WEIGHTS = {"coverage": 1.0, "accuracy": 2.0, "earliness": 1.0, "reliability": 1.0}


def f_beta(tp: int, fp: int, fn: int, beta: float = BETA) -> float:
    b2 = beta * beta
    den = (1 + b2) * tp + b2 * fn + fp
    return (1 + b2) * tp / den if den else 0.0  # sklearn zero_division -> 0


def criticality(predicted, normal, cap: int = 1000) -> np.ndarray:
    """Algorithm 1: +1 on a detected anomaly during normal operation, -1 (not
    below 0) on a normal prediction during normal operation, unchanged otherwise."""
    out, c = np.zeros(len(predicted), dtype=np.int64), 0
    for i, (p, n) in enumerate(zip(predicted, normal)):
        if n:
            c = min(c + 1, cap) if p else max(c - 1, 0)
        out[i] = c
    return out


def weighted_score(event_prediction, start_of_descend=(1, 4)) -> float:
    """Earliness: weight 1 for the first quarter of the event, then linear to 0."""
    n = len(event_prediction)
    num, den = start_of_descend
    sod, scale = num / den, int(den)
    m = n * scale
    cp = int(m * sod)
    x = np.linspace(0, scale, m)
    w = np.zeros(m)
    w[:cp] = scale
    w[cp:] = scale / (1 - sod) - x[cp:] / (1 - sod)
    w = w[::scale] / scale
    return float(np.sum(np.asarray(event_prediction, float) * w) / np.sum(w))


def evaluate_event(predicted, normal, in_event, end_pos: int, label: str) -> dict:
    predicted = np.asarray(predicted, bool)
    normal = np.asarray(normal, bool)
    in_event = np.asarray(in_event, bool)
    if not normal.any():
        raise ValueError("CARE event has no normal-status rows to evaluate")
    truth = in_event | ~normal  # non-normal points are anomalies, then filtered out
    g, p = truth[normal], predicted[normal]
    tp, fp = int(np.sum(g & p)), int(np.sum(~g & p))
    fn, tn = int(np.sum(g & ~p)), int(np.sum(~g & ~p))
    crit = criticality(predicted[:end_pos + 1], normal[:end_pos + 1])
    rec = {"label": label, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
           "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
           "max_criticality": int(crit.max(initial=0)),
           "f_beta": f_beta(tp, fp, fn) if label == "anomaly" else float("nan"),
           "weighted_score": (weighted_score(predicted[in_event]) if label == "anomaly"
                              else float("nan"))}
    rec["detected"] = rec["max_criticality"] >= CRITICALITY_THRESHOLD
    return rec


def final_score(events: list[dict]) -> dict:
    """CARE score and its four sub-scores over evaluated events."""
    anom = [e for e in events if e["label"] == "anomaly"]
    norm = [e for e in events if e["label"] == "normal"]
    if not anom or not norm:
        raise ValueError("CARE needs at least one anomaly and one normal event")
    tp = sum(e["detected"] for e in anom)
    fp = sum(e["detected"] for e in norm)
    out = {"coverage": float(np.mean([e["f_beta"] for e in anom])),
           "accuracy": float(np.mean([e["accuracy"] for e in norm])),
           "earliness": float(np.mean([e["weighted_score"] for e in anom])),
           "reliability": f_beta(tp, fp, len(anom) - tp),
           "events_detected_by_criticality": tp, "normal_events_flagged": fp}
    if tp + fp == 0:
        out["care"] = 0.0
    elif out["accuracy"] <= 0.5:
        out["care"] = out["accuracy"]
    else:
        out["care"] = sum(WEIGHTS[k] * out[k] for k in WEIGHTS) / sum(WEIGHTS.values())
    return out

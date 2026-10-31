"""Backend-agnostic second-stage advisory evaluation over frozen baseline traces.

Labels, event intervals and metric masks are read here only to score finished timelines;
they never reach request construction (see bench.laya_advisory / bench.jev_advisory).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from bench.alarm_eval import evaluate_alarms
from bench.metropt_eval import episodes

CANDIDATE_STATES = (2, 3)  # decision_state unresolved, alarm
REASSESS_SECONDS = 3600
FRESH_SECONDS = 3600
# Metric keys that do not depend on the policy's alert list (absent from traces).
REPRODUCED_KEYS = ("false_alarm_per_week", "events_detected", "alarm_time_share_clean",
                   "lead_hours", "window_balanced_accuracy", "window_f1",
                   "window_sensitivity", "window_specificity")


def sha256_file(path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_trace(record) -> dict:
    """Frozen trace arrays after size + SHA-256 verification against the freeze record."""
    meta = record["trace"]
    path = Path(meta["path"])
    if path.stat().st_size != meta["bytes"] or sha256_file(path) != meta["sha256"]:
        raise ValueError(f"baseline trace integrity mismatch: {path}")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def assessment_schedule(decision_state, end) -> np.ndarray:
    """Assess on candidate entry, then at most once per observed hour while it persists."""
    due = np.zeros(len(end), dtype=bool)
    last = None
    for i, (state, timestamp) in enumerate(zip(decision_state, end)):
        if int(state) not in CANDIDATE_STATES:
            last = None
            continue
        now = float(timestamp)
        if last is None or now - last >= REASSESS_SECONDS:
            due[i] = True
            last = now
    return due


def fresh_timeline(decision_state, end, outcomes) -> tuple[np.ndarray, np.ndarray]:
    """(elevated, unavailable) per window from {index: True/False/None} assessment outcomes.

    A window inherits the latest assessment of its ongoing candidate run only while that
    assessment is at most FRESH_SECONDS old; None (unavailable) is carried separately.
    """
    elevated = np.zeros(len(end), dtype=bool)
    unavailable = np.zeros(len(end), dtype=bool)
    latest_time, latest = None, None
    for i, (state, timestamp) in enumerate(zip(decision_state, end)):
        if int(state) not in CANDIDATE_STATES:
            latest_time, latest = None, None
            continue
        now = float(timestamp)
        if i in outcomes:
            latest_time, latest = now, outcomes[i]
        fresh = latest_time is not None and now - latest_time <= FRESH_SECONDS
        elevated[i] = fresh and latest is True
        unavailable[i] = not fresh or latest is None
    return elevated, unavailable


def opened_from(active) -> np.ndarray:
    active = np.asarray(active, bool)
    return active & ~np.r_[False, active[:-1]]


def episode_openings(z, active) -> np.ndarray:
    """Rising edges of `active`, except where the frozen policy itself continued an alarm
    across an inactive window (its rising edge without an `opened` flag); exactly the
    trace's `opened` for the baseline timeline."""
    active = np.asarray(active, bool)
    base_rise = opened_from(np.asarray(z["active"], bool))
    continued = base_rise & ~np.asarray(z["opened"], bool)
    return (opened_from(active) & ~continued) | (np.asarray(z["opened"], bool) & active)


def interpretations(baseline, elevated, unavailable) -> dict:
    """The two preregistered readings on the same timeline; an outage never silences a
    baseline alarm and never elevates."""
    base = np.asarray(baseline, bool)
    return {"baseline": base, "conservative_preserve_baseline": base | elevated,
            "advisory_assessment_only": elevated | (base & unavailable)}


def metric_valid(stream, z, test) -> np.ndarray:
    """Exactly apply_policy's metric eligibility: monitorable, enabled channel scored, mask."""
    cfg = json.loads(str(z["config"]))
    scored = np.zeros(len(test), dtype=bool)
    if cfg["k_novelty"] > 0:
        scored |= np.isfinite(z["novelty"])
    if cfg["k_health"] > 0:
        scored |= np.isfinite(z["health"])
    valid = stream.monitorable[test] & scored
    return valid if stream.metric_mask is None else valid & stream.metric_mask[test]


def paired_metrics(stream, z, test, active) -> tuple[dict, dict]:
    """Baseline-comparable alarm metrics plus window confusion for one active timeline."""
    active = np.asarray(active, bool)
    opened = episode_openings(z, active)
    valid = metric_valid(stream, z, test)
    start = np.asarray(z["start"], dtype="datetime64[s]")
    end = np.asarray(z["end"], dtype="datetime64[s]")
    event_id = stream.event_id()[test]
    negative = stream.clean_negative()[test]
    m, d = evaluate_alarms(start, end, event_id, negative, valid, stream.events, active, opened,
                           z["novelty"], z["health"], (), stream.lps)
    event = event_id >= 0
    labeled = valid & (event | negative)
    tp, fp = int((active & event & labeled).sum()), int((active & ~event & labeled).sum())
    fn, tn = int((~active & event & labeled).sum()), int((~active & ~event & labeled).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    seconds = float((end[valid] - start[valid]).astype(np.int64).sum())
    weeks = seconds / (7 * 86400) if seconds else float("nan")
    m.update({"window_sensitivity": sens, "window_specificity": spec,
              "window_balanced_accuracy": (sens + spec) / 2,
              "window_precision": tp / (tp + fp) if tp + fp else float("nan"),
              "window_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan"),
              "notification_episodes_per_week_all": episodes(end[opened & valid]) / weeks})
    d["window_confusion"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return m, d


def baseline_reproduction(frozen: dict, recomputed: dict) -> dict:
    """Differences between frozen and recomputed baseline metrics (must all be ~0)."""
    out = {}
    for key in REPRODUCED_KEYS:
        a, b = frozen.get(key), recomputed.get(key)
        if a is None or b is None:
            continue
        same = (np.isnan(a) and np.isnan(b)) or abs(a - b) <= 1e-9 * max(1.0, abs(a))
        out[key] = {"frozen": a, "recomputed": b, "match": bool(same)}
    return out


def coarse_labels(stream, z, test) -> np.ndarray:
    """1 labeled event window, 0 clean negative, -1 neither/ineligible (evaluation only)."""
    valid = metric_valid(stream, z, test)
    y = np.full(len(test), -1, dtype=np.int64)
    y[valid & stream.clean_negative()[test]] = 0
    y[valid & (stream.event_id()[test] >= 0)] = 1
    return y


def calibration(p, y, bins: int = 10) -> dict:
    """Brier score and equal-width ECE of probabilities p against binary labels y."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    if not len(p):
        return {"n": 0, "positives": 0, "brier": None, "ece": None}
    idx = np.minimum((p * bins).astype(int), bins - 1)
    ece = sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
              for b in range(bins) if (idx == b).any())
    return {"n": int(len(p)), "positives": int(y.sum()),
            "brier": float(np.mean((p - y) ** 2)), "ece": float(ece)}

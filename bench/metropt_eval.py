"""MetroPT report-window scoring; no point adjustment or alarm retuning."""
from __future__ import annotations

import numpy as np

from uengram import metropt
from uengram.metrics import auroc, average_precision


def episodes(alert_times, gap_s=3600):
    """Merge alerts strictly less than gap_s apart; an exact hour starts anew."""
    t = np.sort(np.asarray(alert_times, dtype="datetime64[s]"))
    return 0 if not len(t) else 1 + int(np.count_nonzero(np.diff(t).astype(np.int64) >= gap_s))


def evaluate(scores, health, start, end, fid, negative, threshold, warmup=0):
    """Use window end as alert availability; exclude unavailable detector scores.

    Outside buffered report intervals means 'unreported', not verified normal.
    Early alerts are descriptive associations, not proven fault predictions.
    """
    valid = np.isfinite(scores)
    valid[:warmup] = False
    keep = valid & ((fid >= 0) | negative)
    y = fid[keep] >= 0
    alarm = valid & (scores > threshold)
    m = {"auroc": auroc(scores[keep], y), "ap": average_precision(scores[keep], y),
         "health_auroc": auroc(health[keep], y), "health_ap": average_precision(health[keep], y)}
    observed_negative = valid & negative
    seconds = float((end[observed_negative] - start[observed_negative]).astype(np.int64).sum())
    false_episodes = episodes(end[alarm & observed_negative])
    m["false_alarm_per_week"] = false_episodes * (7 * 86400) / seconds if seconds else float("nan")
    failures, leads = [], []
    for i, (a, b) in enumerate(metropt.FAILURES):
        fs, fe = np.datetime64(a, "s"), np.datetime64(b, "s")
        detected = bool(np.any(alarm & (fid == i)))
        early = alarm & (end >= fs - np.timedelta64(24, "h")) & (end <= fe)
        first = np.min(end[early]) if early.any() else None
        lead = float((first - fs).astype(np.int64)) / 3600 if first is not None else None
        if lead is not None:
            leads.append(lead)
        failures.append({"failure_id": i, "detected": detected,
                         "first_alert": str(first) if first is not None else None,
                         "lead_hours": lead, "reported_windows": int(np.count_nonzero(fid == i)),
                         "alarm_windows": int(np.count_nonzero(alarm & (fid == i)))})
    m["failures_detected"] = float(sum(f["detected"] for f in failures))
    m["lead_hours"] = float(np.mean(leads)) if leads else float("nan")
    m["alarm_threshold"] = float(threshold)
    details = {"failures": failures, "false_alarm_episodes": false_episodes,
               "clean_observed_hours": seconds / 3600,
               "scored_windows": int(keep.sum()), "positive_windows": int(np.count_nonzero(y)),
               "missing_score_windows": int(np.count_nonzero(~valid))}
    return m, details

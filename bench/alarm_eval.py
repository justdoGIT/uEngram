"""Event-level evaluation of eligible alarm episodes (protocol v2).

Alarm availability uses eligible window ends. Detection and lead time exclude
unmonitorable or unscored windows. False-alarm episodes count alarm openings
on clean negatives, merged when strictly less than 1 h apart (the v1 rule),
per week of observed eligible clean time; a continuing alarm does not create
another false-alarm episode merely by extending into clean time. Clean
alarm-time share is duration-weighted observed seconds in alarm divided by
duration-weighted observed clean seconds.
"""
from __future__ import annotations

import numpy as np

from bench.metropt_eval import episodes
from uengram.alarm import AlarmConfig, AlarmLogic, AlarmPolicy
from uengram.metrics import auroc, average_precision

H = np.timedelta64(1, "h")
ACTIONS = (
    "none", "score", "absorb_semantic", "step_change", "promote",
    "absorb_candidate", "absorb_episodic", "new_candidate", "discard",
)


class CalibrationUnavailable(ValueError):
    """No enabled alarm channel has an estimable finite calibration quantile."""


def eligible(stream, sc: dict, cfg: AlarmConfig, idx: np.ndarray) -> np.ndarray:
    """Windows idx the policy may use: monitorable and scored by an enabled channel."""
    scored = ((np.isfinite(sc["novelty"][idx]) if cfg.k_novelty > 0 else False) |
              (np.isfinite(sc["health"][idx]) if cfg.k_health > 0 else False))
    return stream.monitorable[idx] & scored


def apply_policy(stream, sc: dict, cfg: AlarmConfig, evidence=None, logic: AlarmLogic | None = None):
    """Calibrate the policy on the calibration windows, stream the test
    windows, and evaluate. Returns (metrics, details, policy).

    `evidence` (optional): one True/False/None per test window, the policy's
    corroboration input (src/uengram/alarm.py); None corroborates every window.
    `logic` (optional): AlarmLogic; None is the v2 alarm."""
    nov, health, act = sc["novelty"], sc["health"], sc["action"]
    cal, test = stream.cal, np.flatnonzero(stream.test)
    novelty_enabled, health_enabled = cfg.k_novelty > 0, cfg.k_health > 0
    cal_valid = eligible(stream, sc, cfg, np.flatnonzero(cal))
    valid = eligible(stream, sc, cfg, test)
    pol = AlarmPolicy(cfg, stream.feature_names, logic or AlarmLogic()).calibrate(
        nov[cal], health[cal], stream.state[cal], cal_valid, stream.features[cal])
    supported = any((novelty_enabled and b["novelty_threshold_supported"]) or
                    (health_enabled and b["health_threshold_supported"])
                    for b in pol.baseline_counts.values())
    if not supported:
        raise CalibrationUnavailable(
            f"no finite calibrated threshold for an enabled alarm channel (alpha={cfg.alpha})")
    active, opened = pol.run(stream.end[test], nov[test], health[test],
                             [ACTIONS[a] for a in act[test]], stream.state[test],
                             valid, stream.features[test], evidence)
    metric_valid = valid if stream.metric_mask is None else valid & stream.metric_mask[test]
    m, d = evaluate_alarms(stream.start[test], stream.end[test], stream.event_id()[test],
                           stream.clean_negative()[test], metric_valid, stream.events, active,
                           opened, nov[test], health[test], [a["index"] for a in pol.alerts],
                           stream.lps)
    d["active"], d["opened"] = active, opened
    d["valid"] = metric_valid  # eligible test windows the metrics above use
    d["baseline_counts"] = pol.baseline_counts
    alerts = np.zeros(len(test), bool)
    alerts[[a["index"] for a in pol.alerts]] = True
    seconds = float((stream.end[test][valid] - stream.start[test][valid]).astype(np.int64).sum())
    weeks = seconds / (7 * 86400) if seconds else float("nan")
    m["notifications_per_week_all"] = episodes(stream.end[test][(alerts | opened) & valid]) / weeks
    d["runtime_observed_hours"] = seconds / 3600
    d["runtime_alarm_episodes"] = int(opened.sum())
    return m, d, pol

H = np.timedelta64(1, "h")


def evaluate_alarms(start, end, event_id, negative, valid, events, active, opened,
                    novelty, health, alert_index=(), lps=None):
    """All arrays cover test windows only; alert_index indexes into them.

    AUROC/AP use separately scored eligible subsets. Alarm availability,
    detection, clean uptime and episode rates use eligible windows only.
    """
    valid = np.asarray(valid, bool)
    active, opened = np.asarray(active, bool) & valid, np.asarray(opened, bool) & valid
    event = event_id >= 0
    nov_keep = valid & np.isfinite(novelty) & (event | negative)
    health_keep = valid & np.isfinite(health) & (event | negative)
    m = {"auroc": auroc(novelty[nov_keep], event[nov_keep]),
         "ap": average_precision(novelty[nov_keep], event[nov_keep]),
         "health_auroc": auroc(health[health_keep], event[health_keep]),
         "health_ap": average_precision(health[health_keep], event[health_keep])}
    clean = valid & negative
    clean_seconds = (end[clean] - start[clean]).astype(np.int64)
    seconds = float(clean_seconds.sum())
    weeks = seconds / (7 * 86400) if seconds else float("nan")
    fa = episodes(end[opened & clean])
    alerts = np.zeros(len(end), bool)
    alerts[np.asarray(alert_index, dtype=np.int64)] = True
    m["false_alarm_per_week"] = fa / weeks
    m["alerts_per_week"] = episodes(end[alerts & clean]) / weeks
    m["operator_events_per_week"] = episodes(end[(alerts | opened) & clean]) / weeks
    active_clean_seconds = float((end[clean & active] - start[clean & active]).astype(np.int64).sum())
    m["alarm_time_share_clean"] = active_clean_seconds / seconds if seconds else float("nan")
    per_event, leads, lps_leads = [], [], []
    for i, (a, b) in enumerate(events):
        fs, fe = np.datetime64(a, "s"), np.datetime64(b, "s")
        event_valid = valid & (event_id == i)
        detected = bool(np.any(active & (event_id == i) & valid))
        early = active & valid & (end >= fs - 24 * H) & (end <= fe)
        first = end[early].min() if early.any() else None
        lead = float((first - fs).astype(np.int64)) / 3600 if first is not None else None
        if lead is not None:
            leads.append(lead)
        rec = {"event": i, "detected": detected, "first_alarm": None if first is None else str(first),
               "lead_hours": lead, "windows": int(event_valid.sum()),
               "alarm_windows": int((active & event_valid).sum())}
        if lps is not None and lps[i] is not None:
            w = np.datetime64(lps[i], "s")
            before = active & valid & (end >= fs - 24 * H) & (end <= w)
            rec["hours_before_lps"] = (float((w - end[before].min()).astype(np.int64)) / 3600
                                       if before.any() else None)
            if rec["hours_before_lps"] is not None:
                lps_leads.append(rec["hours_before_lps"])
        per_event.append(rec)
    m["events_detected"] = float(sum(r["detected"] for r in per_event))
    m["lead_hours"] = float(np.mean(leads)) if leads else float("nan")
    if lps is not None:
        m["events_warned_before_lps"] = float(len(lps_leads))
        m["hours_before_lps"] = float(np.mean(lps_leads)) if lps_leads else float("nan")
    details = {"events": per_event, "false_alarm_episodes": fa, "clean_observed_hours": seconds / 3600,
               "alarm_episodes": int(opened.sum()), "test_windows": int(len(end)),
               "unscored_windows": int((~valid).sum()), "eligible_windows": int(valid.sum()),
               "clean_observed_seconds": seconds,
               "clean_alarm_seconds": active_clean_seconds,
               "novelty_auroc_windows": int(nov_keep.sum()),
               "health_auroc_windows": int(health_keep.sum())}
    return m, details

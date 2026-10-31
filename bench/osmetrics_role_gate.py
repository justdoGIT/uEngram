"""Exploratory DSP-versus-role gate on the public OS-Metrics dataset.

This is not the preregistered fresh-field validation.  The dataset has causal
interval annotations, but it represents one published operational workload,
not independently adjudicated physical assets.  The split is chronological:
one day commissioning, one day calibration, and the remaining days test.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bench.osmetrics import (
    ARCHIVE_MD5,
    ARCHIVE_URL,
    SAMPLE_SECONDS,
    SOURCE_URL,
    Event,
    causal_windows,
    clean_values,
    event_overlap_mask,
    event_window_mask,
    fit_fill_values,
    load,
)
from uengram.encoders import DSPEncoder
from uengram.metrics import average_precision, auroc
from uengram.role_architecture import EmpiricalCalibrator, RoleBasedArchitecture, persistent_starts


FIT_POINTS = 24 * 60 * 60 // SAMPLE_SECONDS
CAL_POINTS = 24 * 60 * 60 // SAMPLE_SECONDS
WINDOW_LENGTH = 16
ALPHA = 0.01
PERSISTENCE = 3
LEAD_HORIZON_HOURS = 2.0


def _event_results(starts: np.ndarray, start_points: np.ndarray, events: tuple[Event, ...]):
    horizon = int(LEAD_HORIZON_HOURS * 3600 / SAMPLE_SECONDS)
    detected_strict = 0
    detected_horizon = 0
    leads: list[float] = []
    rows = []
    for event in events:
        candidates = start_points[starts & (start_points <= event.end_point)]
        strict = candidates[candidates >= event.start_point]
        early = candidates[candidates >= event.start_point - horizon]
        if len(strict):
            detected_strict += 1
        if len(early):
            detected_horizon += 1
            leads.append((event.start_point - int(early[0])) * SAMPLE_SECONDS / 3600.0)
        rows.append({
            "event_id": event.event_id,
            "metric": event.metric,
            "mode": event.mode,
            "intensity": event.intensity,
            "strict_detected": bool(len(strict)),
            "two_hour_detected": bool(len(early)),
            "lead_hours": None if not len(early) else float(leads[-1]),
        })
    return detected_strict, detected_horizon, leads, rows


def _false_alarm_episodes(starts: np.ndarray, start_points: np.ndarray,
                          events: tuple[Event, ...]) -> tuple[int, float]:
    horizon = int(LEAD_HORIZON_HOURS * 3600 / SAMPLE_SECONDS)
    protected = np.zeros(len(start_points), dtype=bool)
    for event in events:
        protected |= (
            (start_points >= event.start_point - horizon)
            & (start_points <= event.end_point)
        )
    clean_starts = starts & ~protected
    episodes = int(np.count_nonzero(clean_starts))
    weeks = len(start_points) * SAMPLE_SECONDS / 604800.0
    return episodes, episodes / max(weeks, 1e-12)


def _metrics(flags, score, start_points, events, labels):
    starts = persistent_starts(flags, PERSISTENCE)
    strict, horizon, leads, event_rows = _event_results(starts, start_points, events)
    false_episodes, false_per_week = _false_alarm_episodes(starts, start_points, events)
    return {
        "window_auroc": float(auroc(score, labels)),
        "window_average_precision": float(average_precision(score, labels)),
        "strict_event_recall": float(strict / len(events)),
        "two_hour_event_recall": float(horizon / len(events)),
        "events_detected_strict": strict,
        "events_detected_within_two_hours": horizon,
        "events_total": len(events),
        "median_lead_hours": None if not leads else float(np.median(leads)),
        "false_alarm_episodes": false_episodes,
        "false_alarm_episodes_per_week": float(false_per_week),
        "alarm_window_share": float(np.mean(flags)),
        "event_rows": event_rows,
    }


def _role_score(p_values: dict[str, np.ndarray]) -> np.ndarray:
    return -np.log(np.maximum(np.min(np.stack(tuple(p_values[name] for name in ("level", "dynamic", "health")), axis=0), axis=0), 1e-12))


def run(root: str | Path, *, alpha: float = ALPHA, persistence: int = PERSISTENCE,
        window_length: int = WINDOW_LENGTH) -> dict:
    data = load(root)
    fill_values = fit_fill_values(data, FIT_POINTS)
    values = clean_values(data, fill_values)
    W, starts, ends = causal_windows(values, length=window_length)
    fit = ends < FIT_POINTS
    cal = (ends >= FIT_POINTS) & (ends < FIT_POINTS + CAL_POINTS)
    test = ends >= FIT_POINTS + CAL_POINTS
    event_mask = event_overlap_mask(starts, ends, data.events)
    fit &= ~event_mask
    cal &= ~event_mask
    if fit.sum() < 100 or cal.sum() < 100 or test.sum() < 100:
        raise ValueError("OS-Metrics split has too few usable windows")
    mode_fit = np.zeros(int(fit.sum()), dtype=np.int64)
    mode_cal = np.zeros(int(cal.sum()), dtype=np.int64)
    mode_test = np.zeros(int(test.sum()), dtype=np.int64)
    # The published sixth metric is exactly a derived response-time
    # difference. Keeping it makes the strict within-state covariance
    # singular, so the gate uses five primitive channels and records this.
    model_columns = np.arange(5, dtype=np.int64)
    W_fit = W[fit][:, list(model_columns), :]
    W_cal = W[cal][:, list(model_columns), :]
    W_test = W[test][:, list(model_columns), :]

    role = RoleBasedArchitecture(d=16).fit(W_fit, groups=np.zeros(len(W_fit)), mode_ids=mode_fit)
    role.fit_calibration(W_cal, mode_ids=mode_cal)
    role_test = role.p_values(role.score(W_test, mode_ids=mode_test))
    role_results = {}
    test_points = ends[test]
    labels = event_overlap_mask(starts[test], ends[test], data.events)
    for policy in ("any", "at_least_2", "dynamic_health", "all"):
        flags = role.fuse(role_test, alpha=alpha, policy=policy)
        role_results[policy] = _metrics(flags, _role_score(role_test), test_points, data.events, labels)

    dsp = DSPEncoder(d=16).fit(W_fit, noise_lag=1, groups=np.zeros(len(W_fit)))
    dsp_cal_scores = np.sum(dsp.encode(W_cal) ** 2, axis=1)
    dsp_test_scores = np.sum(dsp.encode(W_test) ** 2, axis=1)
    dsp_calibrator = EmpiricalCalibrator().fit(dsp_cal_scores)
    dsp_p = dsp_calibrator.p_values(dsp_test_scores)
    dsp_flags = dsp_p <= alpha
    dsp_result = _metrics(dsp_flags, -np.log(np.maximum(dsp_p, 1e-12)), test_points, data.events, labels)

    latency_start = time.perf_counter()
    role.score(W_test, mode_ids=mode_test)
    role_latency_us = (time.perf_counter() - latency_start) * 1e6 / len(W_test)
    route_flags = role.fuse(role_test, alpha=alpha, policy="dynamic_health")
    route_counts = {"EPISODIC": int(route_flags.sum()), "QUARANTINE": int((~route_flags).sum()), "SEMANTIC": 0}
    return {
        "protocol": {
            "dataset": "OS-Metrics v1.0, Zenodo 17981352",
            "source_url": SOURCE_URL,
            "archive_url": ARCHIVE_URL,
            "archive_md5": ARCHIVE_MD5,
            "metric_names": list(data.metric_names),
            "model_metric_names": [data.metric_names[int(i)] for i in model_columns],
            "excluded_derived_metric": data.metric_names[5],
            "sample_seconds": SAMPLE_SECONDS,
            "rows": int(len(values)),
            "events": int(len(data.events)),
            "fit_points": FIT_POINTS,
            "calibration_points": CAL_POINTS,
            "test_points": int(test.sum()),
            "window_length": window_length,
            "alpha": alpha,
            "persistence": persistence,
            "lead_horizon_hours": LEAD_HORIZON_HOURS,
            "fit_calibration_event_rows_removed": True,
            "labels_used_for_fit_cal_filtering": True,
            "independent_field_validation": False,
            "limitation": "one published operational workload; no independent asset-held-out physical field cohort",
        },
        "preprocessing": {"invalid_values_filled_from_fit_medians": int(np.sum(~np.isfinite(data.values) | (data.values <= -90000.0)))},
        "baseline_dsp_d16": dsp_result,
        "role_architecture": {
            "encoder": "MultiScaleEncoder",
            "policies": role_results,
            "memory_routes_dynamic_health": route_counts,
            "score_latency_us_per_window": float(role_latency_us),
            "semantic_memory_admission": "not used; no operator-confirmed-normal labels",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Reproducible OS-Metrics gate for domain-specialist role architecture.

Evaluates subsystem specialists (CPU, memory, network) against all-channel
role pooling and the frozen DSP baseline.  Measures event recall, false alarms,
subsystem root-cause attribution, and numerical conditioning diagnostics.
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
    fit_fill_values,
    load,
)
from bench.osmetrics_role_gate import (
    CAL_POINTS,
    FIT_POINTS,
    LEAD_HORIZON_HOURS,
    PERSISTENCE,
    WINDOW_LENGTH,
    _metrics,
    _role_score,
)
from uengram.diagnostics import diagnose_channels
from uengram.domain_specialists import DomainSpecialistArchitecture
from uengram.encoders import DSPEncoder
from uengram.role_architecture import EmpiricalCalibrator, RoleBasedArchitecture

ALPHA = 0.01


def _check_attribution(test_points: np.ndarray, events: tuple[Event, ...],
                       domain_flags: dict[str, np.ndarray]) -> dict:
    results = []
    for event in events:
        event_windows = (test_points >= event.start_point) & (test_points <= event.end_point)
        if not np.any(event_windows):
            continue
        triggers = [d for d, flags in domain_flags.items() if bool(np.any(flags[event_windows]))]
        results.append({
            "event_id": event.event_id,
            "metric": event.metric,
            "detected": len(triggers) > 0,
            "triggers": triggers,
            "correct_subsystem": event.metric in triggers,
        })
    detected = [r for r in results if r["detected"]]
    correct = [r for r in detected if r["correct_subsystem"]]
    return {
        "test_events_total": len(results),
        "events_detected": len(detected),
        "correct_subsystem_attributions": len(correct),
        "attribution_accuracy": float(len(correct) / max(len(detected), 1)),
        "event_attributions": results,
    }


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

    # 1. Pre-fit diagnostics
    diag_all = diagnose_channels(data.values, list(data.metric_names)).as_dict()
    diag_primitive = diagnose_channels(data.values[:, :5], list(data.metric_names[:5])).as_dict()

    # Model channels: 5 primitive channels
    W_fit = W[fit][:, :5, :]
    W_cal = W[cal][:, :5, :]
    W_test = W[test][:, :5, :]

    test_points = ends[test]
    labels = event_overlap_mask(starts[test], ends[test], data.events)

    # 2. Domain Specialist Architecture
    domains = {
        "cpu": (0,),
        "memory": (1, 2),
        "network": (3, 4),
    }
    spec_arch = DomainSpecialistArchitecture(
        domains,
        channel_names=data.metric_names[:5],
        domain_d=4,
    )
    spec_arch.fit(W_fit, groups=np.zeros(len(W_fit)))
    spec_arch.fit_calibration(W_cal)

    latency_start = time.perf_counter()
    scores = spec_arch.score(W_test)
    spec_latency_us = (time.perf_counter() - latency_start) * 1e6 / len(W_test)

    # Individual specialist detections
    subsystem_breakdown = {}
    domain_flags_dyn_health = {}
    for d_name in ("cpu", "memory", "network"):
        p_dict = scores.domain_p_values[d_name]
        d_flags = spec_arch.specialists[d_name].fuse(p_dict, alpha=alpha, policy="dynamic_health")
        domain_flags_dyn_health[d_name] = d_flags
        d_events = tuple(e for e in data.events if e.metric == d_name)
        d_labels = event_overlap_mask(starts[test], ends[test], d_events)
        subsystem_breakdown[d_name] = _metrics(
            d_flags, _role_score(p_dict), test_points, d_events, d_labels
        )

    # Multi-expert fusion policies
    specialist_policies = {}
    for policy, rule in [
        ("or_dynamic_health", ("dynamic_health", "or")),
        ("or_any", ("any", "or")),
        ("at_least_2", ("dynamic_health", "at_least_2")),
        ("all", ("dynamic_health", "all")),
    ]:
        b_pol, s_rule = rule
        flags, _ = spec_arch.fuse(scores, alpha=alpha, branch_policy=b_pol, specialist_rule=s_rule)
        specialist_policies[policy] = _metrics(flags, scores.fused_score, test_points, data.events, labels)

    attribution = _check_attribution(test_points, data.events, domain_flags_dyn_health)

    # 3. All-channel Role Baseline
    all_role = RoleBasedArchitecture(d=16).fit(W_fit, groups=np.zeros(len(W_fit)))
    all_role.fit_calibration(W_cal)
    all_role_test = all_role.p_values(all_role.score(W_test))
    all_role_policies = {}
    for policy in ("any", "at_least_2", "dynamic_health", "all"):
        flags = all_role.fuse(all_role_test, alpha=alpha, policy=policy)
        all_role_policies[policy] = _metrics(flags, _role_score(all_role_test), test_points, data.events, labels)

    # 4. DSP Baseline
    dsp = DSPEncoder(d=16).fit(W_fit, noise_lag=1, groups=np.zeros(len(W_fit)))
    dsp_cal = np.sum(dsp.encode(W_cal) ** 2, axis=1)
    dsp_test = np.sum(dsp.encode(W_test) ** 2, axis=1)
    dsp_calibrator = EmpiricalCalibrator().fit(dsp_cal)
    dsp_p = dsp_calibrator.p_values(dsp_test)
    dsp_flags = dsp_p <= alpha
    dsp_result = _metrics(dsp_flags, -np.log(np.maximum(dsp_p, 1e-12)), test_points, data.events, labels)

    primary_flags, primary_attr = spec_arch.fuse(scores, alpha=alpha, branch_policy="dynamic_health", specialist_rule="or")
    route_counts = {"EPISODIC": int(primary_flags.sum()), "QUARANTINE": int((~primary_flags).sum()), "SEMANTIC": 0}

    return {
        "protocol": {
            "dataset": "OS-Metrics v1.0, Zenodo 17981352",
            "source_url": SOURCE_URL,
            "archive_url": ARCHIVE_URL,
            "archive_md5": ARCHIVE_MD5,
            "metric_names": list(data.metric_names),
            "model_metric_names": list(data.metric_names[:5]),
            "domains": {k: [data.metric_names[i] for i in v] for k, v in domains.items()},
            "excluded_derived_metric": data.metric_names[5],
            "sample_seconds": SAMPLE_SECONDS,
            "rows": int(len(values)),
            "events_total": int(len(data.events)),
            "fit_points": FIT_POINTS,
            "calibration_points": CAL_POINTS,
            "test_points": int(test.sum()),
            "window_length": window_length,
            "alpha": alpha,
            "persistence": persistence,
            "lead_horizon_hours": LEAD_HORIZON_HOURS,
            "independent_field_validation": False,
        },
        "diagnostics": {
            "all_channels_6": diag_all,
            "primitive_channels_5": diag_primitive,
        },
        "baseline_dsp_d16": dsp_result,
        "baseline_all_channel_role": all_role_policies,
        "domain_specialists": {
            "policies": specialist_policies,
            "subsystem_performance": subsystem_breakdown,
            "root_cause_attribution": attribution,
            "memory_routes_primary": route_counts,
            "score_latency_us_per_window": float(spec_latency_us),
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
    print(json.dumps({
        "status": "complete",
        "dsp_strict_recall": result["baseline_dsp_d16"]["strict_event_recall"],
        "all_channel_role_strict_recall": result["baseline_all_channel_role"]["dynamic_health"]["strict_event_recall"],
        "domain_specialists_strict_recall": result["domain_specialists"]["policies"]["or_dynamic_health"]["strict_event_recall"],
        "domain_specialists_false_per_week": result["domain_specialists"]["policies"]["or_dynamic_health"]["false_alarm_episodes_per_week"],
        "attribution_accuracy": result["domain_specialists"]["root_cause_attribution"]["attribution_accuracy"],
    }, indent=2))


if __name__ == "__main__":
    main()

"""Reconstruct notifications omitted by the frozen advisory scorer, without new inference.

The frozen scorer counts alarm-opening episodes under a notification name. Baseline
alerts are absent from its trace. Replay the saved numerical scores, require exact
baseline reproduction, and restore actual baseline-alert + alarm-opening burden.
Model inputs, answers, thresholds and preregistrations remain unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench import advisory_eval as ev
from bench import advisory_experiment as ax
from bench import laya_advisory as la
from bench import laya_hosts as hosts
from bench.alarm_eval import evaluate_alarms
from bench.followup_policy import evaluate_followup
from bench.metropt_eval import episodes
from bench.streams import load_stream
from uengram.alarm import AlarmConfig


BURDEN_KEYS = ("alerts_per_week", "operator_events_per_week", "notifications_per_week_all")


def notification_rate(start, end, valid, alerts, opened):
    """Original apply_policy definition, including alerts and observed-time denominator."""
    seconds = float((end[valid] - start[valid]).astype(np.int64).sum())
    return episodes(end[(alerts | opened) & valid]) * 604800 / seconds if seconds else None


def verify_baseline(frozen, recomputed):
    checks = ev.baseline_reproduction(frozen, recomputed)
    for key in BURDEN_KEYS:
        if key in frozen:
            a, b = frozen[key], recomputed[key]
            checks[key] = {"frozen": a, "recomputed": b,
                           "match": bool(np.isclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True))}
    bad = [key for key, value in checks.items() if not value["match"]]
    if bad:
        raise ValueError(f"baseline reproduction failed: {bad}")
    return checks


def replay(stream, record, z):
    """Read existing score caches only; never fall back to recomputing a detector."""
    if (record["method"], record["encoder"], record["seed"]) != ("uengram", "dsp", 0):
        raise ValueError("notification replay supports only the selected frozen DSP baseline")
    root = (Path.home() / ".local/state/uengram/v2scores" if stream.name == "metropt3"
            else Path("data/metropt2/v2_cddfbf1"))
    path = root / f"{stream.name}_uengram_dsp_0_final.npz"
    with np.load(path, allow_pickle=False) as cache:
        scores = {key: cache[key] for key in ("novelty", "health", "action")}
    test = np.flatnonzero(stream.test)
    source = {"start": stream.start[test].astype(np.int64),
              "end": stream.end[test].astype(np.int64), "state": stream.state[test],
              "features": stream.features[test], "monitorable": stream.monitorable[test],
              "novelty": scores["novelty"][test], "health": scores["health"][test]}
    for key, values in source.items():
        if not np.array_equal(values, z[key], equal_nan=True):
            raise ValueError(f"frozen trace/source mismatch: {stream.name}/{key}")
    metrics, details, policy = evaluate_followup(stream, scores, AlarmConfig(**record["config"]))
    for key in ("active", "opened", "decision_state"):
        if not np.array_equal(details[key], z[key]):
            raise ValueError(f"baseline replay mismatch: {stream.name}/{key}")
    thresholds = np.asarray([policy.thresholds(int(s)) for s in stream.state[test]])
    if not (np.array_equal(thresholds[:, 0], z["novelty_threshold"], equal_nan=True)
            and np.array_equal(thresholds[:, 1], z["health_threshold"], equal_nan=True)):
        raise ValueError(f"baseline threshold replay mismatch: {stream.name}")
    verify_baseline(record["metrics"], metrics)
    alerts = np.zeros(len(test), bool)
    alerts[[item["index"] for item in policy.alerts]] = True
    return alerts, {"path": str(path), "sha256": ev.sha256_file(path)}, metrics


def corrected_metrics(stream, z, test, active, alerts):
    metrics, details = ev.paired_metrics(stream, z, test, active)
    opened = ev.episode_openings(z, active)
    valid = ev.metric_valid(stream, z, test)
    start, end = (np.asarray(z[key], dtype="datetime64[s]") for key in ("start", "end"))
    alarm_metrics, alarm_details = evaluate_alarms(
        start, end, stream.event_id()[test], stream.clean_negative()[test], valid,
        stream.events, active, opened, z["novelty"], z["health"], np.flatnonzero(alerts), stream.lps)
    metrics.update(alarm_metrics)
    # Preserve the exact notification denominator used by the original apply_policy,
    # which precedes its evaluation-only metric mask.
    cfg = json.loads(str(z["config"]))
    runtime_valid = stream.monitorable[test] & (
        ((cfg["k_novelty"] > 0) & np.isfinite(z["novelty"]))
        | ((cfg["k_health"] > 0) & np.isfinite(z["health"])))
    metrics.pop("notification_episodes_per_week_all")
    metrics["notifications_per_week_all"] = notification_rate(start, end, runtime_valid, alerts, opened)
    details.update(alarm_details)
    return metrics, details


def build(args):
    prereg = hosts._check_drift(args)
    _, local_prov, streams = hosts._setup(args)
    hp = prereg["hosts"]["hp"]
    hp_prov = {"revision": hp["health_at_preregistration"]["revisions"][hosts.lh.ROUTE],
               "laya_version": hp["remote"]["versions"]["laya"]}
    raw_rows = {row["stream"]: row for row in map(json.loads, Path(args.raw).read_text().splitlines())}
    rows = []
    for name, record, z, due in streams:
        row = raw_rows.pop(name)
        results = hosts.merge(due, local_prov, hp_prov, hosts.LOCAL_CACHE, hosts.HP_CACHE)
        outcomes = {i: r["elevated"] if r["available"] else None for i, r in results.items()}
        elevated, unavailable = ev.fresh_timeline(z["decision_state"], z["end"], outcomes)
        timelines = ev.interpretations(z["active"], elevated, unavailable)
        stream = load_stream(name)
        test = np.flatnonzero(stream.test)
        alerts, cache, baseline = replay(stream, record, z)
        metrics, details = {}, {}
        for key, active in timelines.items():
            metrics[key], details[key] = corrected_metrics(stream, z, test, active, alerts)
        row.update({"metrics": metrics,
                    "events": {key: value["events"] for key, value in details.items()},
                    "baseline_reproduction": verify_baseline(baseline, metrics["baseline"]),
                    "notification_reconstruction": {"score_cache": cache,
                        "baseline_alert_windows": int(alerts.sum()),
                        "definition": "baseline numerical alerts remain visible; add alarm openings; "
                                      "merge notification times less than one hour apart",
                        "raw_metric_correction": "notification_episodes_per_week_all omitted baseline "
                                                 "alerts; replaced by notifications_per_week_all"}})
        rows.append(row)
        print(f"{name}: reconstructed notifications={metrics['baseline']['notifications_per_week_all']:.6f}/week; "
              f"baseline matches; coverage={row['coverage']}", flush=True)
    if raw_rows:
        raise ValueError(f"unexpected raw streams: {sorted(raw_rows)}")
    Path(args.out).write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = {"protocol": ax.PROTOCOL, "backend": la.BACKEND, "results": args.out,
               "raw_results": {"path": args.raw, "sha256": ev.sha256_file(args.raw)},
               "evaluation_correction": "post-answer reporting correction only; no model rerun or retuning",
               "report_code_sha256": ev.sha256_file(__file__), "caveats": ax.CAVEATS,
               "streams": {row["stream"]: {key: row[key] for key in (
                   "coverage", "elevations", "hosts", "metrics", "burden", "events",
                   "baseline_reproduction", "notification_reconstruction",
                   "calibration_vs_coarse_labels", "truncated_requests", "token_budget")}
                           for row in rows}}
    Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", default="results/v3_normal_freeze.json")
    parser.add_argument("--local-prereg", default="results/v3_laya_preregistration.json")
    parser.add_argument("--prereg-path", default="results/v3_laya_hp_preregistration.json")
    parser.add_argument("--endpoint", default=hosts.ENDPOINT)
    parser.add_argument("--raw", default="results/v3_laya.jsonl")
    parser.add_argument("--out", default="results/v3_laya_evaluated.jsonl")
    parser.add_argument("--summary", default="results/v3_laya_evaluated_summary.json")
    build(parser.parse_args())


if __name__ == "__main__":
    main()

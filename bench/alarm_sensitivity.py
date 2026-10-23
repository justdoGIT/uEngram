"""Sensitivity analysis of the MetroPT3 recall-vs-clean-alarm-time tradeoff.

Error under study (results/v4_error_definition.json): no explored AlarmConfig jointly
satisfies events_detected==4, alarm_time_share_clean<=0.02 and false_alarm_per_week<=1 on
MetroPT3 (DSP encoder, seed 0, frozen cached scores). This module replays the frozen
novelty/health scores through AlarmPolicy under many configurations -- no retraining, no
new inference, read-only on the cached scores -- to characterize how each AlarmConfig
parameter moves the tradeoff metrics, one at a time (OFAT) and pairwise (interaction).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from bench.alarm_eval import CalibrationUnavailable
from bench.followup_policy import evaluate_followup
from bench.streams import load_stream
from bench.v2_scoring import cached_scores
from uengram.alarm import AlarmConfig

STREAM_NAME = "metropt3"
SCORES_DIR = Path.home() / ".local/state/uengram/v2scores"
BASELINE = AlarmConfig(alpha=0.003, k_novelty=3, k_health=0, per_state=True,
                       min_state_windows=100, new_state_alert=True, aci_gamma=0.0,
                       health_needs_baseline=True)
MISSED_EVENT = 3
TARGET_METRICS = ("events_detected", "false_alarm_per_week", "alarm_time_share_clean",
                  "lead_hours", "window_balanced_accuracy", "window_f1",
                  "notifications_per_week_all")
# One-factor-at-a-time grids: every other field stays at BASELINE.
OFAT_GRIDS = {
    "alpha": (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1),
    "k_novelty": (1, 2, 3, 4, 6, 10),
    "k_health": (0, 1, 3, 6, 10),
    "per_state": (False, True),
    "new_state_alert": (False, True),
    "health_needs_baseline": (False, True),
    "aci_gamma": (0.0, 1e-4, 1e-3, 1e-2),
}


def _record(stream, scores, cfg: AlarmConfig, varied: dict) -> dict:
    """One sweep row: config, full metrics, event-3 detail and policy alert counts.

    A too-tight alpha for the calibration population leaves every enabled channel's
    quantile index past the sample count (conformal_threshold returns +inf for all of
    them): apply_policy refuses to proceed rather than silently evaluate a +inf
    threshold. That refusal is itself evidence about alpha's range, not a sweep failure.
    """
    try:
        metrics, details, policy = evaluate_followup(stream, scores, cfg)
    except CalibrationUnavailable as exc:
        return {"varied": varied, "config": cfg.__dict__, "calibration_unavailable": str(exc),
                "metrics": None, "event3_detected": None, "event3_alarm_windows": None,
                "event3_eligible_windows": None, "alarm_episodes": None, "alert_kinds": {},
                "satisfies_recall_and_budget": False}
    event3 = next(e for e in details["events"] if e["event"] == MISSED_EVENT)
    alert_kinds = {}
    for alert in policy.alerts:
        alert_kinds[alert["kind"]] = alert_kinds.get(alert["kind"], 0) + 1
    budget_ok = (metrics["events_detected"] == 4.0 and metrics["alarm_time_share_clean"] <= 0.02
                and metrics["false_alarm_per_week"] <= 1.0)
    return {"varied": varied, "config": cfg.__dict__, "calibration_unavailable": None,
            "metrics": {k: metrics[k] for k in TARGET_METRICS},
            "event3_detected": event3["detected"], "event3_alarm_windows": event3["alarm_windows"],
            "event3_eligible_windows": event3["windows"], "alarm_episodes": details["alarm_episodes"],
            "alert_kinds": alert_kinds, "satisfies_recall_and_budget": budget_ok}


def ofat(stream, scores, out_path: Path, log) -> list[dict]:
    """Vary one AlarmConfig field at a time around BASELINE; always include the baseline row."""
    rows = [_record(stream, scores, BASELINE, {"param": None, "value": None, "is_baseline": True})]
    for name, grid in OFAT_GRIDS.items():
        for value in grid:
            if value == getattr(BASELINE, name):
                continue  # baseline row already recorded once
            cfg = replace(BASELINE, **{name: value})
            rows.append(_record(stream, scores, cfg, {"param": name, "value": value, "is_baseline": False}))
            summary = ("calibration_unavailable" if rows[-1]["metrics"] is None else
                      f"events_detected={rows[-1]['metrics']['events_detected']} "
                      f"fa_per_week={rows[-1]['metrics']['false_alarm_per_week']:.3f} "
                      f"clean_share={rows[-1]['metrics']['alarm_time_share_clean']:.4f}")
            print(f"[ofat] {name}={value!r} {summary}", flush=True, file=log)
    out_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def pairwise(stream, scores, params: tuple[str, str], out_path: Path, log, fixed: dict | None = None) -> list[dict]:
    """Full grid over two named AlarmConfig fields, holding the rest at BASELINE except `fixed`."""
    a_name, b_name = params
    base = replace(BASELINE, **(fixed or {}))
    rows = []
    for a_val, b_val in itertools.product(OFAT_GRIDS[a_name], OFAT_GRIDS[b_name]):
        cfg = replace(base, **{a_name: a_val, b_name: b_val})
        rows.append(_record(stream, scores, cfg, {a_name: a_val, b_name: b_val, "fixed": fixed or {}}))
        summary = ("calibration_unavailable" if rows[-1]["metrics"] is None else
                  f"events_detected={rows[-1]['metrics']['events_detected']} "
                  f"clean_share={rows[-1]['metrics']['alarm_time_share_clean']:.4f}")
        print(f"[pairwise] {a_name}={a_val!r} {b_name}={b_val!r} {summary}", flush=True, file=log)
    out_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def summarize_ofat(rows: list[dict]) -> dict:
    """Per-parameter monotonicity and elasticity of each target metric around the baseline."""
    baseline = next(r for r in rows if r["varied"]["is_baseline"])
    by_param: dict[str, list[dict]] = {}
    for row in rows:
        if row["varied"]["is_baseline"]:
            continue
        by_param.setdefault(row["varied"]["param"], []).append(row)
    summary = {}
    for name, param_rows in by_param.items():
        numeric = isinstance(OFAT_GRIDS[name][0], (int, float)) and not isinstance(OFAT_GRIDS[name][0], bool)
        unavailable = [r["varied"]["value"] for r in param_rows if r["metrics"] is None]
        available = [r for r in param_rows if r["metrics"] is not None]
        entry = {"any_recall_and_budget": any(r["satisfies_recall_and_budget"] for r in available),
                "calibration_unavailable_at": unavailable,
                "rows": sorted(({"value": r["varied"]["value"], **r["metrics"],
                                 "event3_detected": r["event3_detected"]} for r in available),
                               key=lambda d: str(d["value"]))}
        if numeric:
            base_value = getattr(BASELINE, name)
            for metric in TARGET_METRICS:
                base_metric = baseline["metrics"][metric]
                pairs = [(r["varied"]["value"], r["metrics"][metric]) for r in available
                        if np.isfinite(r["metrics"][metric]) and r["varied"]["value"] != base_value
                        and base_value != 0 and np.isfinite(base_metric) and base_metric != 0]
                if pairs:
                    elasticities = [((m - base_metric) / base_metric) / ((v - base_value) / base_value)
                                    for v, m in pairs]
                    entry.setdefault("elasticity_vs_baseline", {})[metric] = {
                        "mean": float(np.mean(elasticities)), "min": float(np.min(elasticities)),
                        "max": float(np.max(elasticities)),
                        "monotonic": bool(np.all(np.diff([m for _, m in sorted(pairs)]) >= -1e-9)
                                         or np.all(np.diff([m for _, m in sorted(pairs)]) <= 1e-9))}
        summary[name] = entry
    return {"baseline": baseline, "by_parameter": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("ofat", "pairwise", "summarize"))
    parser.add_argument("--scores-dir", type=Path, default=SCORES_DIR)
    parser.add_argument("--ofat-out", type=Path, default=Path("results/v4_alarm_sensitivity_ofat.jsonl"))
    parser.add_argument("--pairwise-out", type=Path, default=Path("results/v4_alarm_sensitivity_pairwise.jsonl"))
    parser.add_argument("--summary-out", type=Path, default=Path("results/v4_alarm_sensitivity_summary.json"))
    parser.add_argument("--pair", nargs=2, default=("new_state_alert", "k_health"))
    parser.add_argument("--fixed", nargs="*", default=(),
                        help="NAME=VALUE overrides held fixed for pairwise (e.g. health_needs_baseline=false)")
    args = parser.parse_args()
    stream = load_stream(STREAM_NAME)
    scores = cached_scores(stream, "uengram", "dsp", 0, str(args.scores_dir), "[alarm_sensitivity]")
    fixed = {}
    for item in args.fixed:
        name, _, raw = item.partition("=")
        fixed[name] = {"true": True, "false": False}.get(raw.lower(), json.loads(raw))
    if args.mode == "ofat":
        ofat(stream, scores, args.ofat_out, sys.stderr)
    elif args.mode == "pairwise":
        pairwise(stream, scores, tuple(args.pair), args.pairwise_out, sys.stderr, fixed)
    else:
        rows = [json.loads(line) for line in args.ofat_out.read_text().splitlines()]
        args.summary_out.write_text(json.dumps(summarize_ofat(rows), indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.summary_out}")


if __name__ == "__main__":
    main()

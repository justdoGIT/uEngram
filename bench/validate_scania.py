"""Run the frozen SCANIA point-label validation protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench.v2_scoring import score_stream
from bench.stream_scania import load_scania
from uengram.alarm import AlarmConfig, AlarmLogic, AlarmPolicy
from uengram.metrics import auroc, average_precision

BASELINE = AlarmConfig(
    alpha=0.003,
    k_novelty=3,
    k_health=0,
    per_state=True,
    min_state_windows=100,
    new_state_alert=True,
    health_needs_baseline=True,
)


def _final_rows(vehicle_id: np.ndarray) -> np.ndarray:
    return np.r_[vehicle_id[1:] != vehicle_id[:-1], True]


def _alarm_metrics(stream, scores, active, opened) -> dict:
    test = stream.test
    ids = stream.extra["vehicle_id"][test]
    classes = stream.extra["class_label"][test]
    finals = _final_rows(ids)
    c = classes[finals]
    alarm = (active | opened)[finals]
    out = {
        "vehicles": int(len(c)),
        "alarm_vehicles": int(alarm.sum()),
        "healthy_alarm_rate": float(alarm[c == 0].mean()),
        "near_failure_recall": float(alarm[c > 0].mean()),
        "final_row_novelty_auroc": auroc(scores["novelty"][test][finals], c > 0),
        "final_row_novelty_ap": average_precision(scores["novelty"][test][finals], c > 0),
        "recall_by_class": {
            str(k): float(alarm[c == k].mean()) for k in range(5) if np.any(c == k)
        },
    }
    return out


def _run_policy(stream, scores, logic, evidence=None):
    cal = stream.cal
    test = np.flatnonzero(stream.test)
    valid = stream.monitorable[test]
    policy = AlarmPolicy(BASELINE, stream.feature_names, logic).calibrate(
        scores["novelty"][cal], scores["health"][cal], stream.state[cal],
        stream.monitorable[cal], stream.features[cal]
    )
    active, opened = policy.run(
        stream.end[test], scores["novelty"][test], scores["health"][test],
        ["score"] * len(test), stream.state[test], valid, stream.features[test],
        evidence=evidence,
    )
    return active, opened, policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--out", default="results/scania_validation.json")
    parser.add_argument("--scores", default="results/scania_scores.npz")
    parser.add_argument("--reuse-scores", action="store_true")
    args = parser.parse_args()

    stream = load_scania(args.root)
    score_path = Path(args.scores)
    if score_path.exists() and args.reuse_scores:
        with np.load(score_path, allow_pickle=False) as z:
            scores = {
                "novelty": z["novelty"], "health": z["health"], "action": z["action"],
                "meta": json.loads(str(z["meta"])),
            }
    else:
        scores = score_stream(stream, "uengram", "tab", 0, "scania", encoding_dir=None)
        np.savez_compressed(
            args.scores,
            novelty=scores["novelty"], health=scores["health"], action=scores["action"],
            meta=json.dumps(scores["meta"]),
        )

    and_active, and_opened, and_policy = _run_policy(stream, scores, AlarmLogic())
    or_control_active, or_control_opened, or_control_policy = _run_policy(
        stream, scores, AlarmLogic(combine="or", error_k=3)
    )
    if not (np.array_equal(and_active, or_control_active)
            and np.array_equal(and_opened, or_control_opened)):
        raise AssertionError("OR without evidence diverged from AND control")

    test = stream.test
    missing = stream.extra["missing_any"][test]
    or_active, or_opened, or_policy = _run_policy(
        stream, scores, AlarmLogic(combine="or", error_k=3), evidence=missing
    )
    test_ids = stream.extra["vehicle_id"][test]
    final = _final_rows(test_ids)
    final_classes = stream.extra["class_label"][test][final]
    out = {
        "protocol": "docs/validation-scania-protocol.md",
        "stream": {
            "W_shape": list(stream.W.shape),
            "fit_rows": int(stream.fit.sum()),
            "cal_rows": int(stream.cal.sum()),
            "test_rows": int(stream.test.sum()),
            "test_vehicles": int(len(final_classes)),
            "class_counts": {
                str(k): int((final_classes == k).sum()) for k in range(5)
            },
        },
        "scores": scores["meta"],
        "and_control": _alarm_metrics(stream, scores, and_active, and_opened),
        "or_no_evidence_control": _alarm_metrics(stream, scores, or_control_active, or_control_opened),
        "or_missingness_tripwire": _alarm_metrics(stream, scores, or_active, or_opened),
        "control_equal": True,
        "missing_rows": int(missing.sum()),
        "episodes": {
            "and": len(and_policy.episodes),
            "or_missingness": len(or_policy.episodes),
        },
        "trigger_kinds": {
            "and": {k: sum(e["trigger"] == k for e in and_policy.episodes)
                    for k in ("novelty", "health", "both", "error")},
            "or_missingness": {k: sum(e["trigger"] == k for e in or_policy.episodes)
                               for k in ("novelty", "health", "both", "error")},
        },
    }
    Path(args.out).write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    for key in ("and_control", "or_missingness_tripwire"):
        print(key, json.dumps(out[key], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

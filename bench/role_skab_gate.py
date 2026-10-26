"""Same-split SKAB screen for the role-based architecture.

This is a comparison gate, not a replacement for the frozen SKAB protocol:
it uses the same data split and labels but evaluates calibrated role scores and
explicit branch fusion before integration with AlarmPolicy v2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench.stream_skab import skab_stream
from uengram.metrics import average_precision, auroc
from uengram.role_architecture import RoleBasedArchitecture, persistent_starts


def _segmented_starts(flags, segments, k):
    flags = np.asarray(flags, dtype=bool)
    segments = np.asarray(segments)
    out = np.zeros(len(flags), dtype=bool)
    for segment in np.unique(segments):
        idx = np.flatnonzero(segments == segment)
        out[idx] = persistent_starts(flags[idx], k)
    return out



def _event_metrics(stream, starts, score):
    event_id = stream.event_id()
    events = np.arange(len(stream.events), dtype=np.int64)
    test_starts = starts & stream.test
    detected = int(sum(np.any(test_starts & (event_id == event)) for event in events))
    warned = int(sum(np.any(test_starts & (stream.start <= np.datetime64(end)))
                     for _, end in stream.events))
    clean = stream.test & stream.clean_negative()
    clean_hours = float(np.sum((stream.end[clean] - stream.start[clean]).astype("timedelta64[s]").astype(float)) / 3600.0)
    false_starts = int(np.sum(test_starts & clean))
    labels = event_id >= 0
    return {
        "window_auroc": float(auroc(score[stream.test], labels[stream.test])),
        "window_average_precision": float(average_precision(score[stream.test], labels[stream.test])),
        "strict_event_recall": detected / len(events),
        "events_detected": detected,
        "events_total": len(events),
        "events_warned_before_offset": warned,
        "healthy_false_alarm_starts": false_starts,
        "healthy_hours": clean_hours,
        "false_alarms_per_hour": false_starts / clean_hours if clean_hours else float("nan"),
        "persistent_starts": int(starts.sum()),
    }


def run(root=None, alpha=0.01, persistence=3):
    stream = skab_stream(root, K=10)
    fit, cal, test = stream.fit, stream.cal, stream.test
    model = RoleBasedArchitecture(d=16).fit(
        stream.W[fit], noise_lag=1, groups=stream.segment[fit], mode_ids=stream.state[fit]
    )
    model.fit_calibration(stream.W[cal], mode_ids=stream.state[cal])
    scores = model.score(stream.W, mode_ids=stream.state)
    p = model.p_values(scores)
    score = -np.log(np.maximum(np.min(np.stack([p[n] for n in ("level", "dynamic", "health")], axis=0), axis=0), 1e-12))
    result = {
        "protocol": "docs/validation-skab-protocol.md split; role-score screen",
        "same_split": {"fit_windows": int(fit.sum()), "calibration_windows": int(cal.sum()), "test_windows": int(test.sum())},
        "configuration": {
            "encoder": "MultiScaleEncoder",
            "dimension": int(model.encoder.d),
            "level_dimension": int(model.encoder.level_d),
            "dynamic_dimension": int(model.encoder.dynamics_d),
            "alpha": alpha,
            "persistence": persistence,
            "channels": int(stream.W.shape[1]),
        },
        "policies": {},
    }
    for policy in ("any", "at_least_2", "dynamic_health", "all"):
        flags = np.zeros(len(stream.W), dtype=bool)
        flags[test] = model.fuse({name: p[name][test] for name in p if name != "mode"}, alpha=alpha, policy=policy)
        starts = _segmented_starts(flags, stream.segment, persistence)
        result["policies"][policy] = _event_metrics(stream, starts, score)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--persistence", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root, args.alpha, args.persistence)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

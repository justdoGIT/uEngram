"""Bounded synthetic gate for the role-based edge architecture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from uengram.encoders import DSPEncoder
from uengram.metrics import average_precision, auroc
from uengram.role_architecture import (
    EmpiricalCalibrator,
    RoleBasedArchitecture,
    persistent_starts,
)
from uengram.synth import generate


def _event_recall(starts: np.ndarray, event_id: np.ndarray) -> tuple[int, int]:
    events = np.unique(event_id[event_id >= 0])
    detected = sum(bool(np.any(starts & (event_id == event))) for event in events)
    return int(detected), int(len(events))


def _summary(flags, starts, test, score):
    normal = test.anomaly == 0
    detected, events = _event_recall(starts, test.event_id)
    false_starts = int(np.sum(starts & normal))
    return {
        "window_auroc": float(auroc(score, test.anomaly > 0)),
        "window_average_precision": float(average_precision(score, test.anomaly > 0)),
        "event_recall": detected / events if events else float("nan"),
        "events_detected": detected,
        "events_total": events,
        "persistent_alarm_starts": int(starts.sum()),
        "normal_alarm_starts_per_1000_windows": float(false_starts / max(normal.sum(), 1) * 1000),
        "alarm_window_share": float(np.mean(flags)),
    }


def run(seed: int = 1, alpha: float = 0.01, persistence: int = 3) -> dict:
    fit = generate(1200, seed=seed + 10, maint_frac=1.0, anomalies=False)
    calibration = generate(1200, seed=seed + 20, maint_frac=1.0, anomalies=False)
    test = generate(3000, seed=seed, maint_frac=0.6, anomalies=True)
    fit_groups = fit.regime

    architecture = RoleBasedArchitecture(d=16).fit(
        fit.windows, groups=fit_groups, mode_ids=fit.regime
    )
    architecture.fit_calibration(calibration.windows, mode_ids=calibration.regime)
    test_scores = architecture.score(test.windows, mode_ids=test.regime)
    cal_scores = architecture.score(calibration.windows, mode_ids=calibration.regime)
    p_values = architecture.p_values(test_scores)
    calibrated = {
        name: EmpiricalCalibrator().fit(getattr(cal_scores, name)).p_values(
            getattr(test_scores, name)
        )
        for name in ("level", "dynamic", "health")
    }

    dsp = DSPEncoder(d=16).fit(fit.windows, noise_lag=1, groups=fit_groups)
    dsp_cal = np.sum(dsp.encode(calibration.windows) ** 2, axis=1)
    dsp_test = np.sum(dsp.encode(test.windows) ** 2, axis=1)
    dsp_p = EmpiricalCalibrator().fit(dsp_cal).p_values(dsp_test)
    dsp_flags = dsp_p <= alpha
    dsp_starts = persistent_starts(dsp_flags, persistence)

    result = {
        "protocol": {
            "dataset": "uEngram synthetic stream",
            "fit_windows": len(fit.windows),
            "calibration_windows": len(calibration.windows),
            "test_windows": len(test.windows),
            "alpha": alpha,
            "persistence": persistence,
            "labels_used_for_selection": False,
            "note": "bounded development gate; not an independent field validation",
        },
        "architecture": {
            "encoder": "MultiScaleEncoder",
            "total_dimension": architecture.encoder.d,
            "level_dimension": architecture.encoder.level_d,
            "dynamic_dimension": architecture.encoder.dynamics_d,
            "policies": {},
        },
        "baseline_dsp_d16": _summary(
            dsp_flags, dsp_starts, test, -np.log(np.maximum(dsp_p, 1e-12))
        ),
    }
    for policy in ("any", "at_least_2", "dynamic_health", "all"):
        flags = architecture.fuse(calibrated, alpha=alpha, policy=policy)
        starts = persistent_starts(flags, persistence)
        score = -np.log(np.maximum(np.min(np.stack(tuple(calibrated.values()), axis=0), axis=0), 1e-12))
        result["architecture"]["policies"][policy] = _summary(flags, starts, test, score)
    result["architecture"]["memory_routes"] = {
        "unconfirmed_normal": "QUARANTINE",
        "operator_confirmed_normal": "SEMANTIC",
        "alarm_or_branch_exceedance": "EPISODIC",
        "bad_data": "QUARANTINE",
    }
    del p_values
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--persistence", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.seed, args.alpha, args.persistence)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

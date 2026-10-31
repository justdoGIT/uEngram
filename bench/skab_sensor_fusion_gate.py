"""Compare multichannel CNN representations on the frozen SKAB timing gate.

This is a compact student-only ablation. It reuses the wall-clock event and
healthy-hour metrics from ``skab_teacher_time_gate`` and deliberately keeps
large teachers out of the run so all eight sensors can be evaluated within a
bounded CPU budget.
"""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

from bench.skab_teacher_time_gate import (
    _edge_bytes,
    _event_metrics,
    _normalise,
    _prepare,
)
from uengram.encoder_cnn import CNNEncoder


def _fit_representation(train, representation: str, epochs: int):
    student = CNNEncoder(
        seed=12, d=16, epochs=epochs, batch=32, representation=representation,
    )
    started = time.perf_counter()
    student.fit(train, noise_lag=1, groups=None)
    train_latent = student.encode(train)
    student._fit_score_center = train_latent.mean(axis=0)
    student._fit_score_scale = (train_latent.std(axis=0)).clip(min=1e-6)
    return student, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/skab"))
    parser.add_argument("--out", type=Path, default=Path("results/skab_sensor_fusion_gate.json"))
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    train, calibration, records, channel_names = _prepare(
        args.root, list(range(8))
    )
    train, calibration = _normalise(train, calibration)
    results = {}
    for representation in ("bottleneck", "middle", "multiscale"):
        student, fit_seconds = _fit_representation(train, representation, args.epochs)
        started = time.perf_counter()
        result = _event_metrics(student, train, calibration, records)
        result.update({
            "representation": representation,
            "fit_seconds": float(fit_seconds),
            "edge_model_bytes": _edge_bytes(student),
            "encode_test_ms_per_window": float(
                1000 * (time.perf_counter() - started)
                / sum(len(record["windows"]) for record in records)
            ),
        })
        results[representation] = result
    report = {
        "role": "SKAB all-channel CNN sensor-fusion ablation",
        "protocol": {
            "source": "docs/validation-skab-protocol.md",
            "channels": list(channel_names),
            "representations": ["bottleneck", "middle", "multiscale"],
            "threshold": "99th percentile of fit healthy latent score",
            "persistence_windows": 3,
            "false_alarm_unit": "persistent episodes per healthy calibration hour",
            "warning_unit": "seconds before labelled anomaly onset",
        },
        "dataset": {
            "healthy_fit_windows": len(train),
            "healthy_calibration_windows": len(calibration),
            "fault_experiments": len(records),
            "fault_windows": sum(len(record["windows"]) for record in records),
        },
        "results": results,
        "resources": {
            "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },
        "limitations": [
            "All files come from one water-loop testbed, not a fleet of assets.",
            "Faults are injected experiments rather than naturally occurring failures.",
            "This ablation changes the sensor input and is not a same-input teacher comparison.",
            "The healthy calibration exposure is short, so hourly rates are noisy.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        key: {name: value for name, value in row.items() if name != "events"}
        for key, row in results.items()
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

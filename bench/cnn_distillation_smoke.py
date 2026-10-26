#!/usr/bin/env python3
"""Smoke-test teacher-to-student transfer on the consumed MetroPT-3 stream.

This is a development check, not independent confirmation. The teacher is a
small local CNN adapter standing in for an exported high-capacity teacher;
real foundation-model adapters must declare the same causal schema contract.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bench.streams import load_stream
from uengram.encoder_cnn import CNNEncoder
from uengram.metrics import auroc, average_precision
from uengram.teacher_transfer import TeacherSpec, WindowSchema


def evaluate(encoder, stream):
    scores = np.linalg.norm(encoder.encode(stream.W), axis=1)
    test = stream.test
    labels = stream.event_id()[test] >= 0
    return {
        "auroc": auroc(scores[test], labels),
        "average_precision": average_precision(scores[test], labels),
        "test_windows": int(test.sum()),
        "positive_windows": int(labels.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")

    stream = load_stream("metropt3")
    train = stream.W[stream.fit]
    spec = TeacherSpec(
        model_name="local-cnn-teacher-smoke",
        model_version="0.1",
        layer="whitened-embedding",
        feature_dim=16,
        input_schema=WindowSchema(
            tuple(f"sensor-{i}" for i in range(train.shape[1])),
            sample_rate_hz=1.0,
            window_size=train.shape[2],
        ),
    )
    started = time.perf_counter()
    teacher = CNNEncoder(seed=11, d=16, epochs=args.epochs, representation="multiscale")
    teacher.fit(train, noise_lag=1, groups=stream.segment[stream.fit])
    teacher_seconds = time.perf_counter() - started

    baseline = CNNEncoder(seed=12, d=16, epochs=args.epochs)
    started = time.perf_counter()
    baseline.fit(train, noise_lag=1, groups=stream.segment[stream.fit])
    baseline_seconds = time.perf_counter() - started

    distilled = CNNEncoder(seed=12, d=16, epochs=args.epochs)
    started = time.perf_counter()
    distilled.fit_distilled(
        train,
        teacher,
        spec,
        noise_lag=1,
        groups=stream.segment[stream.fit],
        teacher_weight=0.25,
    )
    distilled_seconds = time.perf_counter() - started

    result = {
        "role": "development smoke only; MetroPT-3 was previously consumed",
        "teacher": {
            "model_name": spec.model_name,
            "model_version": spec.model_version,
            "layer": spec.layer,
            "causal": spec.causal,
            "alignment": spec.alignment,
            "fit_seconds": teacher_seconds,
        },
        "student": {
            "baseline": {**evaluate(baseline, stream), "fit_seconds": baseline_seconds},
            "distilled": {**evaluate(distilled, stream), "fit_seconds": distilled_seconds},
        },
        "contract": {
            "channels": train.shape[1],
            "window_size": train.shape[2],
            "teacher_feature_dim": spec.feature_dim,
            "teacher_weight": 0.25,
        },
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    main()

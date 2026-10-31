#!/usr/bin/env python3
"""Distil a public MOMENT checkpoint into the embedded CNN encoder.

This benchmark uses the existing, consumed MetroPT-3 protocol. It is an
engineering transfer check, not an independent confirmation set. MOMENT-1-
large is the default teacher contract; ``--model-id`` can select a smaller
checkpoint when local memory or storage is limited.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bench.streams import metropt3_stream
from bench.v2_scoring import score_embeddings
from uengram import synth
from uengram.encoder_cnn import CNNEncoder
from uengram.metrics import auroc, average_precision
from uengram.moment_teacher import MOMENTTeacherAdapter
from uengram.teacher_transfer import TeacherSpec, WindowSchema


def _evaluate(stream, encoder, name: str):
    encoded = encoder.encode(stream.W)
    scored = score_embeddings(
        stream, "uengram", name, 0, encoder, encoded,
        label=name, record_encoding=False,
    )
    labels = stream.event_id()[stream.test] >= 0
    novelty = scored["novelty"][stream.test]
    return {
        "auroc": auroc(novelty, labels),
        "average_precision": average_precision(novelty, labels),
        "test_windows": int(labels.size),
        "positive_windows": int(labels.sum()),
        "events_retained": scored["meta"].get("events_retained", []),
        "writes": scored["meta"].get("writes"),
    }


def _evaluate_synthetic(teacher, epochs: int, teacher_weight: float):
    train = synth.generate(384, seed=901, maint_frac=1.0, anomalies=False).windows
    probe = synth.make_probe(seed=902, T=384, maint_frac=1.0)
    spec = TeacherSpec(
        model_name=teacher.spec.model_name,
        model_version=teacher.spec.model_version,
        layer=teacher.spec.layer,
        feature_dim=teacher.spec.feature_dim,
        input_schema=WindowSchema(tuple(synth.CHANNELS), 1.0, synth.K),
    )
    adapter = MOMENTTeacherAdapter(
        model=teacher.model,
        spec=spec,
        context_length=teacher.context_length,
        device=teacher.device,
        context_policy=teacher.context_policy,
        include_instance_stats=teacher.include_instance_stats,
        base_feature_dim=teacher.base_feature_dim,
    )
    baseline = CNNEncoder(seed=12, d=16, epochs=epochs, batch=64)
    baseline.fit(train, noise_lag=1, groups=None)
    distilled = CNNEncoder(seed=12, d=16, epochs=epochs, batch=64)
    distilled.fit_distilled(
        train, adapter, spec, noise_lag=1, groups=None,
        teacher_weight=teacher_weight, teacher_batch=8,
    )
    labels = probe.anomaly > 0
    values = {}
    for name, encoder in (("baseline", baseline), ("distilled", distilled)):
        scores = np.linalg.norm(encoder.encode(probe.windows), axis=1)
        values[name] = {
            "auroc": auroc(scores, labels),
            "average_precision": average_precision(scores, labels),
            "fault_families": {},
        }
        for fault in range(1, 5):
            subset = (probe.anomaly == 0) | (probe.anomaly == fault)
            fault_labels = probe.anomaly[subset] == fault
            values[name]["fault_families"][str(fault)] = {
                "auroc": auroc(scores[subset], fault_labels),
                "average_precision": average_precision(scores[subset], fault_labels),
            }
    return {
        "role": "controlled synthetic probe; four injected fault families",
        "train_windows": int(len(train)),
        "probe_windows": int(len(probe.windows)),
        "positive_windows": int(labels.sum()),
        "student": values,
        "teacher_schema": {
            "channels": len(synth.CHANNELS),
            "window_size": synth.K,
            "context_length": teacher.context_length,
        },
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-id", default="AutonLab/MOMENT-1-large")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--teacher-weight", type=float, default=0.25)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--context-policy", choices=("edge", "linear"), default="edge")
    parser.add_argument("--include-instance-stats", action="store_true")
    parser.add_argument("--include-synthetic", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0 or args.teacher_weight < 0:
        raise ValueError("epochs must be positive and teacher weight non-negative")

    # 512 samples is MOMENT's native context and leaves only 301 commissioning
    # windows, making the real-checkpoint run feasible on a constrained host.
    stream = metropt3_stream(K=512)
    train = stream.W[stream.fit]
    teacher = MOMENTTeacherAdapter.from_pretrained(
        model_id=args.model_id,
        channel_names=tuple(f"sensor-{i}" for i in range(train.shape[1])),
        sample_rate_hz=0.1,
        window_size=train.shape[2],
        device=args.device,
        local_files_only=args.local_files_only,
        revision=args.revision,
        context_policy=args.context_policy,
        include_instance_stats=args.include_instance_stats,
    )
    started = time.perf_counter()
    baseline = CNNEncoder(seed=12, d=16, epochs=args.epochs, batch=64)
    baseline.fit(train, noise_lag=1, groups=stream.segment[stream.fit])
    baseline_seconds = time.perf_counter() - started

    started = time.perf_counter()
    distilled = CNNEncoder(seed=12, d=16, epochs=args.epochs, batch=64)
    distilled.fit_distilled(
        train,
        teacher,
        teacher.spec,
        noise_lag=1,
        groups=stream.segment[stream.fit],
        teacher_weight=args.teacher_weight,
        teacher_batch=8,
    )
    distilled_seconds = time.perf_counter() - started

    result = {
        "role": "development transfer check; MetroPT-3 was previously consumed",
        "teacher": {
            "model_id": args.model_id,
            "model_version": teacher.spec.model_version,
            "license": "MIT (verify upstream revision before deployment)",
            "layer": teacher.spec.layer,
            "feature_dim": teacher.spec.feature_dim,
            "context_length": teacher.context_length,
            "context_policy": teacher.context_policy,
            "include_instance_stats": teacher.include_instance_stats,
            "causal": teacher.spec.causal,
            "alignment": teacher.spec.alignment,
        },
        "student": {
            "baseline": {**_evaluate(stream, baseline, "cnn-moment-baseline"),
                         "fit_seconds": baseline_seconds},
            "distilled": {**_evaluate(stream, distilled, "cnn-moment-distilled"),
                          "fit_seconds": distilled_seconds},
        },
        "contract": {
            "channels": int(train.shape[1]),
            "window_size": int(train.shape[2]),
            "epochs": args.epochs,
            "teacher_weight": args.teacher_weight,
            "student_dimension": 16,
        },
        "synthetic": (
            _evaluate_synthetic(teacher, args.epochs, args.teacher_weight)
            if args.include_synthetic else None
        ),
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare context adaptation and instance-statistics recovery for MOMENT.

This is a controlled synthetic probe.  It isolates two changes motivated by
recent time-series-foundation-model research: preserving a short window by
interpolation instead of edge padding, and appending per-channel mean/std
that instance normalization can otherwise hide.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from uengram import synth
from uengram.encoder_cnn import CNNEncoder
from uengram.metrics import auroc, average_precision
from uengram.moment_teacher import MOMENTTeacherAdapter


MODEL_REVISION = "411e288267f82cce86296dbe4d6c8bc533cc162f"


def _score(encoder: CNNEncoder, probe) -> dict[str, object]:
    scores = np.linalg.norm(encoder.encode(probe.windows), axis=1)
    labels = probe.anomaly > 0
    families: dict[str, dict[str, float]] = {}
    for fault in range(1, 5):
        subset = (probe.anomaly == 0) | (probe.anomaly == fault)
        fault_labels = probe.anomaly[subset] == fault
        families[str(fault)] = {
            "auroc": auroc(scores[subset], fault_labels),
            "average_precision": average_precision(scores[subset], fault_labels),
        }
    return {
        "auroc": auroc(scores, labels),
        "average_precision": average_precision(scores, labels),
        "fault_families": families,
    }


def _run_variant(
    model_id: str,
    revision: str,
    device: str,
    policy: str,
    include_stats: bool,
    train: np.ndarray,
    probe,
    epochs: int,
    teacher_weight: float,
) -> dict[str, object]:
    teacher = MOMENTTeacherAdapter.from_pretrained(
        model_id=model_id,
        channel_names=tuple(synth.CHANNELS),
        sample_rate_hz=1.0,
        window_size=synth.K,
        device=device,
        local_files_only=False,
        context_policy=policy,
        include_instance_stats=include_stats,
    )
    student = CNNEncoder(seed=12, d=16, epochs=epochs, batch=64)
    started = time.perf_counter()
    student.fit_distilled(
        train,
        teacher,
        teacher.spec,
        noise_lag=1,
        groups=None,
        teacher_weight=teacher_weight,
        teacher_batch=8,
    )
    fit_seconds = time.perf_counter() - started
    return {
        "context_policy": policy,
        "include_instance_stats": include_stats,
        "teacher_feature_dim": teacher.spec.feature_dim,
        "fit_seconds": fit_seconds,
        "student": _score(student, probe),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-id", default="AutonLab/MOMENT-1-small")
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--teacher-weight", type=float, default=0.25)
    args = parser.parse_args()
    if args.epochs <= 0 or args.teacher_weight < 0:
        raise ValueError("epochs must be positive and teacher weight non-negative")

    train = synth.generate(384, seed=901, maint_frac=1.0, anomalies=False).windows
    probe = synth.make_probe(seed=902, T=384, maint_frac=1.0)
    baseline = CNNEncoder(seed=12, d=16, epochs=args.epochs, batch=64)
    baseline.fit(train, noise_lag=1, groups=None)

    variants = [
        _run_variant(
            args.model_id, args.revision, args.device, "edge", False,
            train, probe, args.epochs, args.teacher_weight,
        ),
        _run_variant(
            args.model_id, args.revision, args.device, "linear", False,
            train, probe, args.epochs, args.teacher_weight,
        ),
        _run_variant(
            args.model_id, args.revision, args.device, "linear", True,
            train, probe, args.epochs, args.teacher_weight,
        ),
    ]
    result = {
        "role": "controlled synthetic context-policy study; not a held-out benchmark",
        "teacher": {
            "model_id": args.model_id,
            "revision": args.revision,
            "license": "MIT (verify upstream revision before deployment)",
            "native_context": 512,
        },
        "data": {
            "generator": "uengram.synth",
            "train_seed": 901,
            "probe_seed": 902,
            "train_windows": int(len(train)),
            "probe_windows": int(len(probe.windows)),
            "positive_windows": int((probe.anomaly > 0).sum()),
            "window_size": synth.K,
            "channels": list(synth.CHANNELS),
        },
        "student_contract": {
            "dimension": 16,
            "epochs": args.epochs,
            "teacher_weight": args.teacher_weight,
        },
        "baseline": _score(baseline, probe),
        "variants": variants,
        "interpretation": (
            "Use this matrix to select a context policy; it does not establish "
            "generalization to unseen machines or fault families."
        ),
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    main()

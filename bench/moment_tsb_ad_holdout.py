#!/usr/bin/env python3
"""Evaluate a frozen MOMENT-to-CNN protocol on unseen TSB-AD-U series.

The archive is supplied by the caller and is never copied into the repository.
The selected series and all split rules are written to the result artifact.
No labels are used to choose a teacher policy or student configuration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from uengram.encoder_cnn import CNNEncoder
from uengram.metrics import auroc, average_precision
from uengram.moment_teacher import MOMENTTeacherAdapter


SOURCE_URL = "https://www.thedatum.org/datasets/TSB-AD-U.zip"
MODEL_REVISION = "411e288267f82cce86296dbe4d6c8bc533cc162f"
DEFAULT_MEMBERS = (
    "003_NAB_id_3_WebService_tr_1362_1st_1462.csv",
    "186_SMD_id_9_Facility_tr_5925_1st_17580.csv",
    "400_UCR_id_98_Environment_tr_2046_1st_5391.csv",
    "549_SMAP_id_19_Sensor_tr_1998_1st_2098.csv",
    "715_YAHOO_id_165_Synthetic_tr_500_1st_658.csv",
    "830_Exathlon_id_21_Facility_tr_10766_1st_12590.csv",
)


@dataclass(frozen=True)
class Series:
    name: str
    values: np.ndarray
    labels: np.ndarray
    first_anomaly: int


def _read_series(archive: zipfile.ZipFile, member: str) -> Series:
    path = f"TSB-AD-U/{member}"
    with archive.open(path) as raw:
        rows = list(csv.DictReader(line.decode("utf-8") for line in raw))
    if not rows or "Data" not in rows[0] or "Label" not in rows[0]:
        raise ValueError(f"unexpected TSB-AD columns in {member}")
    values = np.asarray([float(row["Data"]) for row in rows], dtype=np.float32)
    labels = np.asarray([int(float(row["Label"])) for row in rows], dtype=np.int8)
    positives = np.flatnonzero(labels > 0)
    if not np.isfinite(values).all() or not len(positives):
        raise ValueError(f"{member} must be finite and contain an anomaly")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError(f"{member} has labels outside {{0, 1}}")
    return Series(member, values, labels, int(positives[0]))


def _windows(series: Series, K: int, stride: int, train_fraction: float):
    if series.first_anomaly <= K or not 0 < train_fraction < 1:
        raise ValueError(f"insufficient healthy prefix or invalid split for {series.name}")
    fit_cutoff = int(series.first_anomaly * train_fraction)
    healthy = series.values[:series.first_anomaly]
    mean = float(healthy.mean())
    scale = max(float(healthy.std()), 1e-6)
    values = (series.values - mean) / scale
    train, test, labels = [], [], []
    for start in range(0, len(values) - K + 1, stride):
        end = start + K
        window = values[start:end].reshape(1, K)
        label = int(series.labels[start:end].any())
        if end <= series.first_anomaly and start < fit_cutoff:
            train.append(window)
        elif start >= fit_cutoff:
            test.append(window)
            labels.append(label)
    if len(train) < 8 or not test or not any(labels) or all(labels):
        raise ValueError(f"split for {series.name} lacks train/test class coverage")
    return np.asarray(train, dtype=np.float32), np.asarray(test, dtype=np.float32), np.asarray(labels, dtype=bool)


def load_dataset(archive_path: Path, members: tuple[str, ...], K: int, stride: int,
                 train_fraction: float):
    train_parts, test_parts, label_parts, group_parts = [], [], [], []
    with zipfile.ZipFile(archive_path) as archive:
        for member in members:
            series = _read_series(archive, member)
            train, test, labels = _windows(series, K, stride, train_fraction)
            train_parts.append(train)
            test_parts.append(test)
            label_parts.append(labels)
            group_parts.extend([member] * len(test))
    train_windows = np.concatenate(train_parts, axis=0)
    test_windows = np.concatenate(test_parts, axis=0)
    return train_windows, test_windows, np.concatenate(label_parts), np.asarray(group_parts)


def _metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(auroc(scores, labels)),
        "average_precision": float(average_precision(scores, labels)),
        "windows": int(labels.size),
        "positive_windows": int(labels.sum()),
    }


def _score(encoder: CNNEncoder, test: np.ndarray, labels: np.ndarray,
           groups: np.ndarray, members: tuple[str, ...]) -> dict:
    scores = np.linalg.norm(encoder.encode(test), axis=1)
    return {
        "overall": _metrics(scores, labels),
        "per_series": {
            member: _metrics(scores[groups == member], labels[groups == member])
            for member in members
        },
    }


def _variant(model_id: str, revision: str, policy: str, include_stats: bool,
             train: np.ndarray, test: np.ndarray, labels: np.ndarray,
             groups: np.ndarray, members: tuple[str, ...], epochs: int,
             teacher_weight: float, device: str) -> dict:
    teacher = MOMENTTeacherAdapter.from_pretrained(
        model_id=model_id,
        channel_names=("value",),
        sample_rate_hz=1.0,
        window_size=train.shape[2],
        device=device,
        local_files_only=False,
        revision=revision,
        context_policy=policy,
        include_instance_stats=include_stats,
    )
    student = CNNEncoder(seed=12, d=16, epochs=epochs, batch=64)
    started = time.perf_counter()
    student.fit_distilled(
        train, teacher, teacher.spec, noise_lag=1,
        groups=np.zeros(len(train), dtype=np.int64),
        teacher_weight=teacher_weight, teacher_batch=8,
    )
    return {
        "context_policy": policy,
        "include_instance_stats": include_stats,
        "teacher_feature_dim": teacher.spec.feature_dim,
        "fit_seconds": time.perf_counter() - started,
        "scores": _score(student, test, labels, groups, members),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-id", default="AutonLab/MOMENT-1-small")
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--teacher-weight", type=float, default=0.25)
    args = parser.parse_args()
    if args.K <= 1 or args.stride <= 0 or args.epochs <= 0 or args.teacher_weight < 0:
        raise ValueError("K, stride, epochs, and teacher_weight are invalid")

    members = DEFAULT_MEMBERS
    train, test, labels, groups = load_dataset(
        args.archive, members, args.K, args.stride, args.train_fraction
    )
    baseline = CNNEncoder(seed=12, d=16, epochs=args.epochs, batch=64)
    started = time.perf_counter()
    baseline.fit(train, noise_lag=1, groups=np.zeros(len(train), dtype=np.int64))
    baseline_result = _score(baseline, test, labels, groups, members)
    baseline_result["fit_seconds"] = time.perf_counter() - started

    variants = [
        _variant(args.model_id, args.revision, "edge", False, train, test, labels,
                 groups, members, args.epochs, args.teacher_weight, args.device),
        _variant(args.model_id, args.revision, "linear", True, train, test, labels,
                 groups, members, args.epochs, args.teacher_weight, args.device),
    ]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    result = {
        "role": "frozen cross-domain TSB-AD-U holdout; no test-label model selection",
        "source": {
            "url": SOURCE_URL,
            "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
            "members": list(members),
            "license_note": "TSB-AD curation is Apache-2.0; original series licenses must be checked individually",
        },
        "protocol": {
            "window_size": args.K,
            "stride": args.stride,
            "healthy_train_fraction": args.train_fraction,
            "normalization": "per-series healthy-prefix mean and standard deviation",
            "student_dimension": 16,
            "epochs": args.epochs,
            "teacher_weight": args.teacher_weight,
            "git_commit": commit,
        },
        "teacher": {
            "model_id": args.model_id,
            "revision": args.revision,
            "native_context": 512,
        },
        "dataset": {
            "train_windows": int(len(train)),
            "test_windows": int(len(test)),
            "positive_windows": int(labels.sum()),
            "series": {
                member: {
                    "test_windows": int((groups == member).sum()),
                    "positive_windows": int(labels[groups == member].sum()),
                }
                for member in members
            },
        },
        "baseline": baseline_result,
        "variants": variants,
        "interpretation": "Development holdout across unrelated TSB-AD domains; not a universal accuracy claim.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

"""Evaluate two independent teachers on a CWRU asset-disjoint subset.

The train set contains healthy recordings 97--99. Recording 100 is an
unseen healthy load. Fault groups are held out by damaged-bearing identity.
Raw files are downloaded outside the repository and are never committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample

from uengram.chronos_teacher import Chronos2TeacherAdapter
from uengram.moment_teacher import MOMENTTeacherAdapter
from bench.mfpt_two_teacher_gate import (
    CHRONOS_REVISION,
    MOMENT_REVISION,
    _fit_and_score,
    _parameter_bytes,
    _parameter_count,
)

SOURCE_URL = "https://engineering.case.edu/sites/default/files/{file}.mat"
MANIFEST_URL = "https://raw.githubusercontent.com/VictorBauler/BearingDatasets/main/src/bearing_datasets/datasets/cwru/files.csv"
SAMPLE_RATE = 12_000.0
WINDOW = 512
STEP = 256

# The numeric file is the public CWRU recording identifier. Asset IDs group
# the four operating loads that use the same damaged bearing.
FILE_META = {
    97: ("normal", "healthy_0", "healthy", 0),
    98: ("normal", "healthy_1", "healthy", 1),
    99: ("normal", "healthy_2", "healthy", 2),
    100: ("normal", "healthy_3", "healthy", 3),
    105: ("fault", "IR007", "inner_race", 0),
    106: ("fault", "IR007", "inner_race", 1),
    107: ("fault", "IR007", "inner_race", 2),
    108: ("fault", "IR007", "inner_race", 3),
    118: ("fault", "B007", "rolling_element", 0),
    119: ("fault", "B007", "rolling_element", 1),
    120: ("fault", "B007", "rolling_element", 2),
    121: ("fault", "B007", "rolling_element", 3),
    130: ("fault", "OR007_6", "outer_race", 0),
    131: ("fault", "OR007_6", "outer_race", 1),
    132: ("fault", "OR007_6", "outer_race", 2),
    133: ("fault", "OR007_6", "outer_race", 3),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _de_time(path: Path, file_id: int) -> np.ndarray:
    data = loadmat(path)
    key = f"X{file_id:03d}_DE_time"
    if key not in data:
        raise KeyError(f"{path.name} has no drive-end channel {key}")
    return np.asarray(data[key]).reshape(-1).astype(np.float64)


def _windows(signal: np.ndarray, source_rate: float, limit: int) -> np.ndarray:
    target = max(WINDOW, round(len(signal) * SAMPLE_RATE / source_rate))
    signal = resample(signal, target).astype(np.float32)
    starts = np.arange(0, len(signal) - WINDOW + 1, STEP)
    if len(starts) > limit:
        starts = np.linspace(0, len(starts) - 1, limit, dtype=int)
    return np.stack([signal[start : start + WINDOW] for start in starts])[:, None, :]


def load_cwru(root: Path, limit: int) -> list[dict]:
    records = []
    for file_id, (kind, asset, fault, load_hp) in FILE_META.items():
        path = root / f"{file_id}.mat"
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}; download the pinned subset first")
        source_rate = 48_000.0 if file_id < 105 else 12_000.0
        records.append({
            "name": str(file_id),
            "source_file": path.name,
            "fault_type": kind,
            "field_data": False,
            "asset_id": asset,
            "fault_class": fault,
            "load_hp": load_hp,
            "sha256": _sha256(path),
            "windows": _windows(_de_time(path, file_id), source_rate, limit),
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--windows-per-recording", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    if args.windows_per_recording <= 0 or args.epochs <= 0:
        raise ValueError("windows-per-recording and epochs must be positive")

    records = load_cwru(args.raw_root, args.windows_per_recording)
    train_names = {"97", "98", "99"}
    train_records = [r for r in records if r["name"] in train_names]
    test_records = [r for r in records if r["name"] not in train_names]
    train_raw = np.concatenate([r["windows"] for r in train_records])
    test_raw = np.concatenate([r["windows"] for r in test_records])
    mean = train_raw.mean(axis=(0, 2), keepdims=True)
    std = np.maximum(train_raw.std(axis=(0, 2), keepdims=True), 1e-6)
    train = ((train_raw - mean) / std).astype(np.float32)
    test = ((test_raw - mean) / std).astype(np.float32)
    labels = np.concatenate([
        np.full(len(r["windows"]), r["fault_type"] != "normal") for r in test_records
    ])

    started = time.perf_counter()
    moment = MOMENTTeacherAdapter.from_pretrained(
        model_id="AutonLab/MOMENT-1-small", channel_names=("vibration",),
        sample_rate_hz=SAMPLE_RATE, window_size=WINDOW, device="cpu",
        revision=MOMENT_REVISION, context_policy="linear", include_instance_stats=True,
    )
    moment_load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    chronos = Chronos2TeacherAdapter.from_pretrained(
        model_id="amazon/chronos-2", channel_names=("vibration",),
        sample_rate_hz=SAMPLE_RATE, window_size=WINDOW, device="cpu",
        revision=CHRONOS_REVISION, prediction_length=8, batch_size=16,
    )
    chronos_load_seconds = time.perf_counter() - started
    teacher_cost = {}
    for name, teacher in (("moment", moment), ("chronos", chronos)):
        t0 = time.perf_counter()
        output = teacher.encode(train[: min(16, len(train))])
        teacher_cost[name] = {
            "feature_dim": int(output.shape[1]),
            "probe_ms_per_window": float(1000 * (time.perf_counter() - t0) / len(output)),
        }

    results = {
        "baseline_cnn": _fit_and_score("baseline_cnn", train, test, labels, test_records, epochs=args.epochs),
        "moment_student": _fit_and_score("moment_student", train, test, labels, test_records, teacher=moment, epochs=args.epochs),
        "chronos_student": _fit_and_score("chronos_student", train, test, labels, test_records, teacher=chronos, epochs=args.epochs),
        "two_teacher_student": _fit_and_score(
            "two_teacher_student", train, test, labels, test_records,
            teachers=[(moment, moment.spec), (chronos, chronos.spec)], epochs=args.epochs,
        ),
    }
    report = {
        "role": "untouched CWRU asset-disjoint fault holdout",
        "source": {"file_url": SOURCE_URL, "manifest_url": MANIFEST_URL,
                   "license_note": "Manifest license is unspecified; verify permission before commercial use."},
        "protocol": {
            "train_recordings": sorted(train_names),
            "test_recordings": [r["name"] for r in test_records],
            "held_out_asset_ids": sorted({r["asset_id"] for r in test_records if r["fault_type"] != "normal"}),
            "normal_test_recording": "100",
            "normalization": "training-normal mean and standard deviation only",
            "sample_rate_hz": SAMPLE_RATE, "window_size": WINDOW, "stride": STEP,
            "windows_per_recording_cap": args.windows_per_recording, "epochs": args.epochs,
            "threshold": "99th percentile of training-normal latent distance",
            "warning_time": "not applicable: CWRU files are static labelled recordings",
        },
        "dataset": {"recordings": [{k: v for k, v in r.items() if k != "windows"} | {"windows": len(r["windows"])} for r in records]},
        "teachers": {
            "moment": {"model_id": "AutonLab/MOMENT-1-small", "revision": MOMENT_REVISION,
                       "license": "MIT", "load_seconds": moment_load_seconds,
                       "parameter_count": _parameter_count(moment.model), "parameter_bytes": _parameter_bytes(moment.model), **teacher_cost["moment"]},
            "chronos": {"model_id": "amazon/chronos-2", "revision": CHRONOS_REVISION,
                        "license": "Apache-2.0", "load_seconds": chronos_load_seconds,
                        "parameter_count": _parameter_count(chronos.pipeline.model), "parameter_bytes": _parameter_bytes(chronos.pipeline.model), **teacher_cost["chronos"]},
        },
        "results": results,
        "resources": {"process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024},
        "limitations": [
            "CWRU has explicit damaged-bearing groups, but the healthy files do not expose physical asset IDs.",
            "The subset uses one fault size and one test-rig channel; it is not fleet-scale evidence.",
            "There are no independently adjudicated fault-onset timestamps, so warning time is not claimed.",
            "Teacher runtime is offline cost; only the student CNN is intended for edge deployment.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()

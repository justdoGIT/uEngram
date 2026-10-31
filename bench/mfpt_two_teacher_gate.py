"""Evaluate two independent time-series teachers on an untouched MFPT holdout.

The gate holds out the three real-world MFPT recordings from the test-rig
baseline recordings. This is asset and domain disjoint at recording level,
not a claim of fleet-scale field validation.
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

from uengram.encoder_cnn import CNNEncoder
from uengram.metrics import auroc, average_precision
from uengram.moment_teacher import MOMENTTeacherAdapter
from uengram.chronos_teacher import Chronos2TeacherAdapter

SOURCE_URL = (
    "https://web.archive.org/web/20250505045128id_/"
    "https://www.mfpt.org/wp-content/uploads/2020/02/"
    "MFPT-Fault-Data-Sets-20200227T131140Z-001.zip"
)
ARCHIVE_SHA256 = "cd46b9d95545c095899b2b9f6462ce60cdbfeae87a743625dd312e6a4df6e6fa"
MOMENT_REVISION = "411e288267f82cce86296dbe4d6c8bc533cc162f"
CHRONOS_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
SAMPLE_RATE = 2048.0
WINDOW = 512


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _folder_kind(name: str) -> tuple[str, bool] | None:
    if name.startswith("1 - "):
        return "normal", False
    if name.startswith(("2 - ", "3 - ", "4 - ")):
        return "fault", False
    if name.startswith("6 - "):
        return "fault", True
    return None


def _windows(signal: np.ndarray, sample_rate: float, limit: int) -> np.ndarray:
    target = max(WINDOW, round(len(signal) * SAMPLE_RATE / sample_rate))
    signal = resample(np.asarray(signal, dtype=np.float64), target).astype(np.float32)
    step = max(1, WINDOW // 4)
    starts = np.arange(0, len(signal) - WINDOW + 1, step)
    if len(starts) > limit:
        starts = np.linspace(0, len(starts) - 1, limit, dtype=int)
    return np.stack([signal[start : start + WINDOW] for start in starts])[:, None, :]


def load_mfpt(raw_root: Path, limit: int) -> list[dict]:
    root = raw_root / "MFPT Fault Data Sets"
    records = []
    for folder in sorted(root.iterdir()):
        kind = _folder_kind(folder.name)
        if kind is None:
            continue
        fault_type, field_data = kind
        for path in sorted(folder.glob("*.mat")):
            bearing = loadmat(path, squeeze_me=True, struct_as_record=False)["bearing"]
            signal = np.asarray(bearing.gs).reshape(-1)
            windows = _windows(signal, float(bearing.sr), limit)
            records.append({
                "name": path.stem,
                "source_file": f"{folder.name}/{path.name}",
                "fault_type": fault_type,
                "field_data": field_data,
                "windows": windows,
            })
    if len(records) != 23:
        raise ValueError(f"expected 23 MFPT recordings, found {len(records)}")
    return records



def _parameter_count(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))

def _parameter_bytes(model) -> int:
    return int(sum(p.numel() * p.element_size() for p in model.parameters()))


def _edge_bytes(student: CNNEncoder) -> int:
    return int(sum(p.numel() * p.element_size() for module in (student.conv, student.fc) for p in module.parameters()))

def _score(student: CNNEncoder, train: np.ndarray, test: np.ndarray, labels: np.ndarray,
          records: list[dict]) -> dict:
    train_z = student.encode(train)
    started = time.perf_counter()
    test_z = student.encode(test)
    encode_seconds = time.perf_counter() - started
    center = train_z.mean(axis=0)
    scale = np.maximum(train_z.std(axis=0), 1e-6)
    train_score = np.mean(((train_z - center) / scale) ** 2, axis=1)
    test_score = np.mean(((test_z - center) / scale) ** 2, axis=1)
    threshold = float(np.quantile(train_score, 0.99))
    file_scores = [
        float(test_score[offset : offset + len(record["windows"])].max())
        for offset, record in _offsets(records)
    ]
    file_labels = np.array([record["fault_type"] != "normal" for record in records])
    field = np.array([record["field_data"] for record in records])
    rig_fault = file_labels & ~field
    normal = ~file_labels
    flagged = np.asarray(file_scores) >= threshold
    return {
        "window_auroc": float(auroc(test_score, labels)),
        "window_average_precision": float(average_precision(test_score, labels)),
        "file_auroc": float(auroc(np.asarray(file_scores), file_labels)),
        "file_average_precision": float(average_precision(np.asarray(file_scores), file_labels)),
        "threshold_train_p99": threshold,
        "positive_file_recall": float(flagged[file_labels].mean()),
        "test_rig_fault_file_recall": float(flagged[rig_fault].mean()) if rig_fault.any() else None,
        "field_file_recall": float(flagged[field].mean()) if field.any() else None,
        "normal_file_false_alarm_rate": float(flagged[normal].mean()) if normal.any() else None,
        "normal_files_flagged": int(flagged[normal].sum()),
        "normal_files": int(normal.sum()),
        "test_windows": int(len(test)),
        "encode_test_ms_per_window": float(1000 * encode_seconds / len(test)),
    }


def _offsets(records: list[dict]):
    offset = 0
    for record in records:
        yield offset, record
        offset += len(record["windows"])


def _fit_and_score(name: str, train: np.ndarray, test: np.ndarray, labels: np.ndarray,
                   records: list[dict], teacher=None, teachers=None, epochs: int = 5) -> dict:
    student = CNNEncoder(seed=12, d=16, epochs=epochs, batch=32)
    started = time.perf_counter()
    if teachers is not None:
        student.fit_distilled_multi(
            train, teachers, noise_lag=1, groups=None, teacher_weight=0.25, teacher_batch=16
        )
    elif teacher is not None:
        student.fit_distilled(
            train, teacher, teacher.spec, noise_lag=1, groups=None,
            teacher_weight=0.25, teacher_batch=16,
        )
    else:
        student.fit(train, noise_lag=1, groups=None)
    result = _score(student, train, test, labels, records)
    result.update({
        "fit_seconds": time.perf_counter() - started,
        "edge_model_bytes": _edge_bytes(student),
        "training_teacher_head_bytes": int(
            sum(p.numel() * p.element_size() for p in student.teacher_head.parameters())
            if student.teacher_head is not None else 0
        ),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--windows-per-recording", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    if args.windows_per_recording <= 0 or args.epochs <= 0:
        raise ValueError("windows-per-recording and epochs must be positive")
    if _sha256(args.archive) != ARCHIVE_SHA256:
        raise ValueError("MFPT archive checksum does not match the pinned source")

    records = load_mfpt(args.raw_root, args.windows_per_recording)
    train_names = {"baseline_1", "baseline_2"}
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
        "role": "untouched MFPT asset/domain holdout; recording-level research gate",
        "protocol": {
            "train_recordings": [r["name"] for r in train_records],
            "test_recordings": [r["name"] for r in test_records],
            "normalization": "training-baseline mean and standard deviation only",
            "sample_rate_hz": SAMPLE_RATE,
            "window_size": WINDOW,
            "stride": WINDOW // 4,
            "windows_per_recording_cap": args.windows_per_recording,
            "epochs": args.epochs,
            "threshold": "99th percentile of training-normal latent distance",
            "warning_time": "not applicable: MFPT recordings are static fault experiments, not onset-labelled run-to-failure streams",
        },
        "dataset": {
            "license_note": "MFPT manifest says unspecified; distributed free for research. Verify before commercial use.",
            "recordings": [
                {k: v for k, v in r.items() if k != "windows"} | {"windows": len(r["windows"])}
                for r in records
            ],
        },
        "teachers": {
            "moment": {
                "model_id": "AutonLab/MOMENT-1-small",
                "revision": MOMENT_REVISION,
                "license": "MIT",
                "load_seconds": moment_load_seconds,
                "parameter_count": _parameter_count(moment.model),
                "parameter_bytes": _parameter_bytes(moment.model),
                **teacher_cost["moment"],
            },
            "chronos": {
                "model_id": "amazon/chronos-2",
                "revision": CHRONOS_REVISION,
                "license": "Apache-2.0",
                "load_seconds": chronos_load_seconds,
                "parameter_count": _parameter_count(chronos.pipeline.model),
                "parameter_bytes": _parameter_bytes(chronos.pipeline.model),
                **teacher_cost["chronos"],
            },
        },
        "results": results,
        "resources": {"process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024},
        "limitations": [
            "MFPT has 23 recordings but only three real-world recordings; this is not fleet-scale evidence.",
            "The holdout is recording- and domain-disjoint; physical bearing identity is not fully documented for every test-rig recording.",
            "There are no independently adjudicated fault-onset timestamps, so warning time is not claimed.",
            "Teacher runtime is offline cost; only the student CNN is intended for edge deployment.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: row for name, row in results.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()

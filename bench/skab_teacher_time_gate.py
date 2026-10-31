"""Evaluate compact and teacher-distilled CNNs on SKAB with time metrics.

SKAB is a real 1 Hz water-loop testbed with one healthy commissioning stream
and 34 independently labelled fault experiments. It is a wall-clock timing
gate, not a multi-asset fleet gate: all experiments use one physical testbed.
Raw data is downloaded outside git by ``uengram.skab.fetch``.
"""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

from bench.mfpt_two_teacher_gate import (
    CHRONOS_REVISION,
    MOMENT_REVISION,
    _edge_bytes,
    _parameter_bytes,
    _parameter_count,
)
from uengram.chronos_teacher import Chronos2TeacherAdapter
from uengram.encoder_cnn import CNNEncoder
from uengram.metrics import auroc, average_precision
from uengram.moment_teacher import MOMENTTeacherAdapter
from uengram import skab

WINDOW = 10
SAMPLE_RATE_HZ = 1.0
PERSISTENCE = 3


def _sustained(flags: np.ndarray, persistence: int = PERSISTENCE) -> np.ndarray:
    """Return starts of complete persistent alarm runs."""
    flags = np.asarray(flags, dtype=bool)
    starts = np.zeros(len(flags), dtype=bool)
    if len(flags) < persistence:
        return starts
    for i in range(len(flags) - persistence + 1):
        starts[i] = bool(np.all(flags[i : i + persistence]))
        if i > 0 and flags[i - 1]:
            starts[i] = False
    return starts


def _episodes(flags: np.ndarray, times: np.ndarray) -> list[np.datetime64]:
    """Return one timestamp for each complete persistent alarm episode."""
    return [times[i] for i in np.flatnonzero(_sustained(flags))]


def _prepare(root: Path, sensor_indices: list[int]):
    if not sensor_indices or any(i < 0 or i >= len(skab.CHANNELS) for i in sensor_indices):
        raise ValueError("sensor indices must select at least one valid SKAB channel")
    channel_names = tuple(skab.CHANNELS[i] for i in sensor_indices)
    stream = skab.load(root, fit_frac=0.3, K=WINDOW)
    windows, starts, ends, groups, splits, experiments = skab.windows(
        stream["t"], stream["X"], stream["split"], stream["experiment"], WINDOW
    )
    windows = windows[:, sensor_indices, :]
    fit = splits == "fit"
    cal = splits == "cal"
    test = splits == "test"
    train = windows[fit]
    calibration = windows[cal]
    records = []
    for experiment in range(len(skab.FAULT_FILES)):
        indices = np.flatnonzero(test & (experiments == experiment))
        if len(indices) == 0:
            raise ValueError(f"experiment {experiment} has no windows")
        onset = stream["onset"][experiment]
        offset = stream["offset"][experiment]
        labels = (ends[indices] >= onset) & (starts[indices] <= offset)
        records.append({
            "name": skab.FAULT_FILES[experiment],
            "experiment": experiment,
            "indices": indices,
            "windows": windows[indices],
            "starts": starts[indices],
            "ends": ends[indices],
            "labels": labels,
            "onset": onset,
            "offset": offset,
        })
    return train, calibration, records, channel_names


def _normalise(train: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = train.mean(axis=(0, 2), keepdims=True)
    std = np.maximum(train.std(axis=(0, 2), keepdims=True), 1e-6)
    return tuple(((array - mean) / std).astype(np.float32) for array in (train, *arrays))


def _score_sequence(student: CNNEncoder, windows: np.ndarray) -> np.ndarray:
    latent = student.encode(windows)
    center = student._fit_score_center
    scale = student._fit_score_scale
    return np.mean(((latent - center) / scale) ** 2, axis=1)


def _fit_student(name: str, train: np.ndarray, teacher=None, teachers=None, epochs: int = 5):
    student = CNNEncoder(seed=12, d=16, epochs=epochs, batch=32)
    started = time.perf_counter()
    if teachers is not None:
        student.fit_distilled_multi(
            train, teachers, noise_lag=1, groups=None, teacher_weight=0.25,
            teacher_batch=16,
        )
    elif teacher is not None:
        student.fit_distilled(
            train, teacher, teacher.spec, noise_lag=1, groups=None,
            teacher_weight=0.25, teacher_batch=16,
        )
    else:
        student.fit(train, noise_lag=1, groups=None)
    train_latent = student.encode(train)
    student._fit_score_center = train_latent.mean(axis=0)
    student._fit_score_scale = np.maximum(train_latent.std(axis=0), 1e-6)
    return student, time.perf_counter() - started


def _event_metrics(student: CNNEncoder, train: np.ndarray, calibration: np.ndarray,
                   records: list[dict]) -> dict:
    train_score = _score_sequence(student, train)
    calibration_score = _score_sequence(student, calibration)
    threshold = float(np.quantile(train_score, 0.99))
    cal_flags = calibration_score >= threshold
    cal_times = np.datetime64("1970-01-01T00:00:00", "s") + np.arange(
        len(calibration), dtype="timedelta64[s]"
    )
    cal_episodes = _episodes(cal_flags, cal_times)
    cal_hours = len(calibration) * WINDOW / 3600.0

    test_scores, test_labels = [], []
    event_rows = []
    for record in records:
        scores = _score_sequence(student, record["windows"])
        flags = scores >= threshold
        test_scores.append(scores)
        test_labels.append(record["labels"])
        starts = _sustained(flags)
        alarm_positions = np.flatnonzero(starts)
        valid_positions = alarm_positions[
            record["starts"][alarm_positions] <= record["offset"]
        ]
        first = record["starts"][valid_positions[0]] if len(valid_positions) else None
        lead = ((record["onset"] - first) / np.timedelta64(1, "s")) if first is not None else None
        detected = bool(len(valid_positions) and np.any(
            (record["starts"][valid_positions] >= record["onset"])
            & (record["starts"][valid_positions] <= record["offset"])
        ))
        warned = bool(len(valid_positions))
        event_rows.append({
            "experiment": record["name"],
            "detected": detected,
            "warned_before_offset": warned,
            "first_alarm": None if first is None else str(first),
            "lead_seconds": None if lead is None else float(lead),
            "alarm_episodes": len(_episodes(flags, record["starts"])),
        })
    scores = np.concatenate(test_scores)
    labels = np.concatenate(test_labels)
    leads = [row["lead_seconds"] for row in event_rows if row["lead_seconds"] is not None]
    return {
        "window_auroc": float(auroc(scores, labels)),
        "window_average_precision": float(average_precision(scores, labels)),
        "event_recall": float(np.mean([row["detected"] for row in event_rows])),
        "events_detected": int(sum(row["detected"] for row in event_rows)),
        "event_warning_recall": float(np.mean([row["warned_before_offset"] for row in event_rows])),
        "events_warned_before_offset": int(sum(row["warned_before_offset"] for row in event_rows)),
        "median_warning_seconds": None if not leads else float(np.median(leads)),
        "calibration_false_alarm_episodes": len(cal_episodes),
        "calibration_hours": float(cal_hours),
        "false_alarms_per_hour": float(len(cal_episodes) / cal_hours) if cal_hours else None,
        "threshold_train_p99": threshold,
        "events": event_rows,
    }


def _evaluate(name: str, train: np.ndarray, calibration: np.ndarray,
              records: list[dict], teacher=None, teachers=None, epochs: int = 5) -> dict:
    student, fit_seconds = _fit_student(name, train, teacher, teachers, epochs)
    test_windows = sum(len(record["windows"]) for record in records)
    started = time.perf_counter()
    result = _event_metrics(student, train, calibration, records)
    encode_seconds = time.perf_counter() - started
    result.update({
        "fit_seconds": float(fit_seconds),
        "edge_model_bytes": _edge_bytes(student),
        "encode_test_ms_per_window": float(1000 * encode_seconds / test_windows),
        "training_teacher_head_bytes": int(sum(
            p.numel() * p.element_size() for p in student.teacher_head.parameters()
        ) if student.teacher_head is not None else 0),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/skab"))
    parser.add_argument("--out", type=Path, default=Path("results/skab_teacher_time_gate.json"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--sensor-indices", default="0",
        help="comma-separated SKAB channel indices; default 0 bounds teacher runtime",
    )
    args = parser.parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    sensor_indices = [int(value) for value in args.sensor_indices.split(",") if value.strip()]
    train, calibration, records, channel_names = _prepare(args.root, sensor_indices)
    train, calibration = _normalise(train, calibration)
    started = time.perf_counter()
    moment = MOMENTTeacherAdapter.from_pretrained(
        model_id="AutonLab/MOMENT-1-small", channel_names=channel_names,
        sample_rate_hz=SAMPLE_RATE_HZ, window_size=WINDOW, device="cpu",
        revision=MOMENT_REVISION, context_policy="linear", include_instance_stats=True,
    )
    moment_load = time.perf_counter() - started
    started = time.perf_counter()
    chronos = Chronos2TeacherAdapter.from_pretrained(
        model_id="amazon/chronos-2", channel_names=channel_names,
        sample_rate_hz=SAMPLE_RATE_HZ, window_size=WINDOW, device="cpu",
        revision=CHRONOS_REVISION, prediction_length=8, batch_size=16,
    )
    chronos_load = time.perf_counter() - started
    probe = train[: min(8, len(train))]
    teacher_cost = {}
    for label, teacher in (("moment", moment), ("chronos", chronos)):
        started = time.perf_counter()
        encoded = teacher.encode(probe)
        teacher_cost[label] = {
            "feature_dim": int(encoded.shape[1]),
            "probe_ms_per_window": float(1000 * (time.perf_counter() - started) / len(encoded)),
        }
    results = {
        "baseline_cnn": _evaluate("baseline_cnn", train, calibration, records, epochs=args.epochs),
        "moment_student": _evaluate("moment_student", train, calibration, records, teacher=moment, epochs=args.epochs),
        "chronos_student": _evaluate("chronos_student", train, calibration, records, teacher=chronos, epochs=args.epochs),
        "two_teacher_student": _evaluate(
            "two_teacher_student", train, calibration, records,
            teachers=[(moment, moment.spec), (chronos, chronos.spec)], epochs=args.epochs,
        ),
    }
    report = {
        "role": "SKAB wall-clock teacher-transfer gate; one testbed, 34 labelled experiments",
        "source": {
            "dataset": "Skoltech Anomaly Benchmark v0.9",
            "commit": skab.COMMIT,
            "license": "GPL-3.0; research use only unless compatible licensing is confirmed",
            "protocol": "docs/validation-skab-protocol.md",
        },
        "protocol": {
            "channels": list(channel_names), "sample_rate_hz": SAMPLE_RATE_HZ,
            "window_size": WINDOW, "persistence_windows": PERSISTENCE,
            "false_alarm_unit": "persistent episodes per healthy calibration hour",
            "warning_unit": "seconds before labelled anomaly onset",
        },
        "dataset": {"healthy_fit_windows": len(train), "healthy_calibration_windows": len(calibration),
                    "fault_experiments": len(records),
                    "fault_windows": sum(len(r["windows"]) for r in records)},
        "teachers": {
            "moment": {"model_id": "AutonLab/MOMENT-1-small", "revision": MOMENT_REVISION,
                       "license": "MIT", "load_seconds": moment_load,
                       "parameter_count": _parameter_count(moment.model),
                       "parameter_bytes": _parameter_bytes(moment.model), **teacher_cost["moment"]},
            "chronos": {"model_id": "amazon/chronos-2", "revision": CHRONOS_REVISION,
                        "license": "Apache-2.0", "load_seconds": chronos_load,
                        "parameter_count": _parameter_count(chronos.pipeline.model),
                        "parameter_bytes": _parameter_bytes(chronos.pipeline.model), **teacher_cost["chronos"]},
        },
        "results": results,
        "resources": {"process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024},
        "limitations": [
            "All 35 files come from one water-loop testbed; this is not a multi-asset fleet result.",
            "Faults are injected experiments, not naturally occurring maintenance failures.",
            "The healthy calibration stream is short, so false-alarms-per-hour is noisy.",
            "Teachers are offline training components; only the compact CNN is edge-deployable.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: {k: value for k, value in row.items() if k != "events"}
                      for key, row in results.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()

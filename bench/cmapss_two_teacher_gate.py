"""Run an asset-disjoint two-teacher lifecycle gate on NASA CMAPSS FD001.

Units 1--40 provide only their first healthy commissioning windows for training.
Units 41--50 are untouched complete run-to-failure test engines. CMAPSS is a
simulation benchmark, so this is lifecycle evidence, not field validation.
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
    _parameter_bytes,
    _parameter_count,
)
from uengram import cmapss
from uengram.chronos_teacher import Chronos2TeacherAdapter
from uengram.encoder_cnn import CNNEncoder
from uengram.metrics import auroc, average_precision
from uengram.moment_teacher import MOMENTTeacherAdapter

K = 16
TRAIN_UNITS = set(range(1, 41))
TEST_UNITS = set(range(41, 51))
TRAIN_WINDOWS_PER_UNIT = 30
CALIBRATION_WINDOWS_PER_UNIT = 30
EOL_HORIZON = 20
PERSISTENCE = 3

def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _first_windows(windows: np.ndarray, units: np.ndarray, limit: int) -> np.ndarray:
    keep = np.concatenate([np.flatnonzero(units == unit)[:limit] for unit in sorted(set(units))])
    return keep


def _load_split(
    root: Path, train_windows_per_unit: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_data = cmapss.load(str(root / "train_FD001.txt"))
    train_data = train_data[np.isin(train_data[:, 0], list(TRAIN_UNITS | TEST_UNITS))]
    windows, units, relative_rul = cmapss.windows(train_data, K)
    train_mask = np.isin(units, list(TRAIN_UNITS))
    test_mask = np.isin(units, list(TEST_UNITS))
    train_idx = _first_windows(
        windows[train_mask], units[train_mask], train_windows_per_unit
    )
    train_windows = windows[train_mask][train_idx]
    train_units = units[train_mask][train_idx]
    test_windows = windows[test_mask]
    test_units = units[test_mask]
    test_rul = relative_rul[test_mask]
    return train_windows, train_units, test_windows, test_units, test_rul


def _score_latents(
    train_z: np.ndarray, test_z: np.ndarray, calibration_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    center = train_z.mean(axis=0)
    scale = np.maximum(train_z.std(axis=0), 1e-6)
    train_score = np.mean(((train_z - center) / scale) ** 2, axis=1)
    test_score = np.mean(((test_z - center) / scale) ** 2, axis=1)
    threshold = float(np.quantile(train_score[calibration_idx], 0.99))
    return train_score, test_score, threshold


def _sustained_starts(flags: np.ndarray) -> list[int]:
    starts = np.flatnonzero(flags & np.r_[True, ~flags[:-1]])
    return [
        int(start)
        for start in starts
        if start + PERSISTENCE <= len(flags)
        and np.all(flags[start : start + PERSISTENCE])
    ]


def _count_episodes(scores: np.ndarray, units: np.ndarray, threshold: float) -> int:
    flags = scores >= threshold
    return sum(
        len(_sustained_starts(flags[np.flatnonzero(units == unit)]))
        for unit in sorted(set(units))
    )


def _alarm_metrics(
    scores: np.ndarray, units: np.ndarray, rul: np.ndarray, threshold: float
) -> dict:
    labels = rul <= EOL_HORIZON
    flagged = scores >= threshold
    healthy_cycles = int(np.sum(rul > EOL_HORIZON))
    detected_any = 0
    detected_by_horizon = 0
    pre_eol_alarms = 0
    warning_cycles: list[float] = []
    for unit in sorted(set(units)):
        idx = np.flatnonzero(units == unit)
        unit_flags = flagged[idx]
        unit_rul = rul[idx]
        sustained = _sustained_starts(unit_flags)
        if sustained:
            detected_any += 1
            detected_by_horizon += int(any(unit_rul[start] <= EOL_HORIZON for start in sustained))
            pre_eol_alarms += sum(
                unit_rul[start] > EOL_HORIZON for start in sustained
            )
            first = sustained[0]
            if unit_rul[first] > EOL_HORIZON:
                warning_cycles.append(float(unit_rul[first] - EOL_HORIZON))
    events_total = len(set(units))
    return {
        "window_auroc": float(auroc(scores, labels)),
        "window_average_precision": float(average_precision(scores, labels)),
        "event_recall_before_failure": float(detected_any / events_total),
        "event_recall_at_eol_horizon": float(detected_by_horizon / events_total),
        "events_detected_before_failure": detected_any,
        "events_detected_by_horizon": detected_by_horizon,
        "events_total": events_total,
        "missed_events": events_total - detected_any,
        "pre_eol_alarm_episodes": pre_eol_alarms,
        "healthy_cycles": healthy_cycles,
        "pre_eol_alarms_per_100_healthy_cycles": float(
            100 * pre_eol_alarms / max(healthy_cycles, 1)
        ),
        "warning_cycles_before_eol_horizon_mean": (
            float(np.mean(warning_cycles)) if warning_cycles else None
        ),
        "warning_cycles_before_eol_horizon_median": (
            float(np.median(warning_cycles)) if warning_cycles else None
        ),
        "threshold_train_calibration_p99": threshold,
        "test_windows": int(len(scores)),
        "test_units": events_total,
    }


def _fit_and_score(
    name: str, train: np.ndarray, train_units: np.ndarray, test: np.ndarray,
    test_units: np.ndarray, test_rul: np.ndarray, epochs: int,
    teacher=None, teachers=None,
) -> dict:
    student = CNNEncoder(seed=12, d=16, epochs=epochs, batch=64)
    started = time.perf_counter()
    if teachers is not None:
        student.fit_distilled_multi(
            train, teachers, noise_lag=K, groups=train_units,
            teacher_weight=0.25, teacher_batch=128,
        )
    elif teacher is not None:
        student.fit_distilled(
            train, teacher, teacher.spec, noise_lag=K, groups=train_units,
            teacher_weight=0.25, teacher_batch=128,
        )
    else:
        student.fit(train, noise_lag=K, groups=train_units)
    fit_seconds = time.perf_counter() - started
    train_z = student.encode(train)
    started = time.perf_counter()
    test_z = student.encode(test)
    encode_ms = 1000 * (time.perf_counter() - started) / len(test)
    calibration_idx = np.concatenate([
        np.flatnonzero(train_units == unit)[:CALIBRATION_WINDOWS_PER_UNIT]
        for unit in sorted(TRAIN_UNITS)
    ])
    train_scores, scores, threshold = _score_latents(
        train_z, test_z, calibration_idx
    )
    result = _alarm_metrics(scores, test_units, test_rul, threshold)
    calibration_episodes = _count_episodes(
        train_scores[calibration_idx],
        train_units[calibration_idx],
        threshold,
    )
    result.update({
        "calibration_false_alarm_episodes": calibration_episodes,
        "calibration_healthy_windows": int(len(calibration_idx)),
        "calibration_false_alarms_per_100_healthy_windows": float(
            100 * calibration_episodes / max(len(calibration_idx), 1)
        ),
    })
    result.update({
        "fit_seconds": fit_seconds,
        "encode_test_ms_per_window": encode_ms,
        "edge_model_bytes": int(sum(p.numel() * p.element_size() for module in (student.conv, student.fc) for p in module.parameters())),
        "training_teacher_head_bytes": int(sum(p.numel() * p.element_size() for p in student.teacher_head.parameters())) if student.teacher_head is not None else 0,
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/cmapss"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--train-windows-per-unit", type=int, default=TRAIN_WINDOWS_PER_UNIT
    )
    parser.add_argument(
        "--sensor-indices", default="0",
        help="comma-separated retained CMAPSS sensor indices; default uses one sensor",
    )
    args = parser.parse_args()
    try:
        sensor_indices = tuple(int(item) for item in args.sensor_indices.split(","))
    except ValueError as exc:
        raise ValueError("sensor-indices must be comma-separated integers") from exc
    if not sensor_indices or min(sensor_indices) < 0:
        raise ValueError("sensor-indices must contain non-negative indices")
    if args.epochs <= 0 or args.train_windows_per_unit < CALIBRATION_WINDOWS_PER_UNIT:
        raise ValueError(
            "epochs must be positive and train-windows-per-unit must cover calibration"
        )
    train, train_units, test, test_units, test_rul = _load_split(
        args.root, args.train_windows_per_unit
    )
    if max(sensor_indices) >= train.shape[1]:
        raise ValueError("sensor index exceeds the retained CMAPSS sensor count")
    train = train[:, sensor_indices, :]
    test = test[:, sensor_indices, :]
    mean = train.mean(axis=(0, 2), keepdims=True)
    std = np.maximum(train.std(axis=(0, 2), keepdims=True), 1e-6)
    train = ((train - mean) / std).astype(np.float32)
    test = ((test - mean) / std).astype(np.float32)

    started = time.perf_counter()
    moment = MOMENTTeacherAdapter.from_pretrained(
        model_id="AutonLab/MOMENT-1-small",
        channel_names=tuple(f"s{i}" for i in sensor_indices),
        sample_rate_hz=1.0, window_size=K, device="cpu", revision=MOMENT_REVISION,
        context_policy="linear", include_instance_stats=True,
    )
    moment_load = time.perf_counter() - started
    started = time.perf_counter()
    chronos = Chronos2TeacherAdapter.from_pretrained(
        model_id="amazon/chronos-2",
        channel_names=tuple(f"s{i}" for i in sensor_indices),
        sample_rate_hz=1.0, window_size=K, device="cpu", revision=CHRONOS_REVISION,
        prediction_length=4, batch_size=128,
    )
    chronos_load = time.perf_counter() - started
    teacher_cost = {}
    for name, teacher in (("moment", moment), ("chronos", chronos)):
        started = time.perf_counter()
        output = teacher.encode(train[: min(16, len(train))])
        teacher_cost[name] = {"feature_dim": int(output.shape[1]), "probe_ms_per_window": 1000 * (time.perf_counter() - started) / len(output)}

    results = {
        "baseline_cnn": _fit_and_score("baseline_cnn", train, train_units, test, test_units, test_rul, args.epochs),
        "moment_student": _fit_and_score("moment_student", train, train_units, test, test_units, test_rul, args.epochs, teacher=moment),
        "chronos_student": _fit_and_score("chronos_student", train, train_units, test, test_units, test_rul, args.epochs, teacher=chronos),
        "two_teacher_student": _fit_and_score("two_teacher_student", train, train_units, test, test_units, test_rul, args.epochs, teachers=[(moment, moment.spec), (chronos, chronos.spec)]),
    }
    report = {
        "role": "CMAPSS FD001 unit-disjoint complete run-to-failure lifecycle gate",
        "dataset": "NASA CMAPSS FD001; simulated turbofan degradation, not field evidence",
        "protocol": {
            "train_units": [1, 40], "test_units": [41, 50],
            "train_windows_per_unit": args.train_windows_per_unit,
            "calibration_windows_per_unit": CALIBRATION_WINDOWS_PER_UNIT,
            "sensor_indices": list(sensor_indices),
            "window_size": K, "normalization": "training commissioning mean and standard deviation only",
            "eol_horizon_cycles": EOL_HORIZON, "persistence_windows": PERSISTENCE,
            "false_alarm_unit": "episodes per 100 healthy cycles; CMAPSS has no wall-clock hour",
            "warning_time": "cycles before RUL <= 20; complete run-to-failure labels available for held-out units",
        },
        "teachers": {
            "moment": {"model_id": "AutonLab/MOMENT-1-small", "revision": MOMENT_REVISION, "license": "MIT", "load_seconds": moment_load, "parameter_count": _parameter_count(moment.model), "parameter_bytes": _parameter_bytes(moment.model), **teacher_cost["moment"]},
            "chronos": {"model_id": "amazon/chronos-2", "revision": CHRONOS_REVISION, "license": "Apache-2.0", "load_seconds": chronos_load, "parameter_count": _parameter_count(chronos.pipeline.model), "parameter_bytes": _parameter_bytes(chronos.pipeline.model), **teacher_cost["chronos"]},
        },
        "results": results,
        "resources": {"process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024},
        "limitations": [
            "CMAPSS is simulated and contains one operating condition and one fault mode.",
            "The fixed fold has 40 training and 10 test engine units; repeated folds are future work.",
            "The reported false-alarm rate uses cycles rather than hours because CMAPSS has no wall-clock sampling interval.",
            "The large teachers are training-only; only the compact CNN is intended for edge deployment.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    print(json.dumps(results, indent=2, default=_json_default), flush=True)


if __name__ == "__main__":
    main()

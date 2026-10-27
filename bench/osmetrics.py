"""Loader and causal windows for the public OS-Metrics dataset.

The raw archive stays outside the repository.  This adapter only reads the
six aligned metric CSV files and the three event-log CSV files from a local
extraction directory.  It is deliberately separate from the preregistered
field protocol: OS-Metrics is an exploratory operational-data gate, not a
fresh industrial asset cohort.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SOURCE_URL = "https://zenodo.org/records/17981352"
ARCHIVE_URL = (
    "https://zenodo.org/api/records/17981352/files/OSMetrics.zip/content"
)
ARCHIVE_MD5 = "c1c57a31c99b1be3c05316be8fa1c00a"
SAMPLE_SECONDS = 15
METRIC_FILES = (
    "cpu_usage.csv",
    "memory_usage.csv",
    "memory_used.csv",
    "network_response_time_a.csv",
    "network_response_time_b.csv",
    "network_response_time_diff.csv",
)
EVENT_FILES = (
    "event_logs_cpu.csv",
    "event_logs_memory.csv",
    "event_logs_network.csv",
)


@dataclass(frozen=True)
class Event:
    event_id: str
    metric: str
    start_point: int
    end_point: int
    mode: str
    intensity: float


@dataclass(frozen=True)
class OSMetricsData:
    timestamps: np.ndarray
    values: np.ndarray
    metric_names: tuple[str, ...]
    events: tuple[Event, ...]


def _read_metric(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or tuple(rows[0]) != ("Time", "Value"):
        raise ValueError(f"unexpected metric schema: {path}")
    timestamps = [row["Time"] for row in rows]
    values = np.asarray([float(row["Value"]) for row in rows], dtype=np.float64)
    return timestamps, values


def _read_events(path: Path, metric: str) -> list[Event]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"id", "start_point", "end_point", "mode", "intensity"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"unexpected event schema: {path}")
    return [
        Event(
            event_id=row["id"],
            metric=metric,
            start_point=int(row["start_point"]),
            end_point=int(row["end_point"]),
            mode=row["mode"],
            intensity=float(row["intensity"]),
        )
        for row in rows
    ]


def load(root: str | Path) -> OSMetricsData:
    """Load aligned metrics and causally annotated event intervals."""
    root = Path(root)
    timestamps: list[str] | None = None
    columns: list[np.ndarray] = []
    names: list[str] = []
    for filename in METRIC_FILES:
        current_timestamps, values = _read_metric(root / filename)
        if timestamps is None:
            timestamps = current_timestamps
        elif current_timestamps != timestamps:
            raise ValueError(f"timestamps are not aligned: {filename}")
        columns.append(values)
        names.append(filename.removesuffix(".csv"))
    if timestamps is None:
        raise ValueError("no metric files found")
    events: list[Event] = []
    for filename, metric in zip(EVENT_FILES, ("cpu", "memory", "network")):
        events.extend(_read_events(root / "event_logs" / filename, metric))
    events.sort(key=lambda event: (event.start_point, event.end_point, event.event_id))
    return OSMetricsData(
        timestamps=np.asarray(timestamps, dtype="datetime64[s]"),
        values=np.column_stack(columns),
        metric_names=tuple(names),
        events=tuple(events),
    )


def fit_fill_values(data: OSMetricsData, end_point: int) -> np.ndarray:
    """Estimate invalid-value replacements using only the commissioning prefix."""
    if not 1 <= end_point <= len(data.values):
        raise ValueError("end_point must select a non-empty prefix")
    prefix = data.values[:end_point]
    valid = np.isfinite(prefix) & (prefix > -90000.0)
    medians = np.empty(prefix.shape[1], dtype=np.float64)
    for column in range(prefix.shape[1]):
        observed = prefix[valid[:, column], column]
        if len(observed) == 0:
            raise ValueError(f"metric {column} has no commissioning values")
        medians[column] = np.median(observed)
    return medians


def clean_values(data: OSMetricsData, fill_values: np.ndarray) -> np.ndarray:
    """Replace published sentinel values without looking beyond fit-derived values."""
    values = np.asarray(data.values, dtype=np.float64).copy()
    fill_values = np.asarray(fill_values, dtype=np.float64)
    if fill_values.shape != (values.shape[1],) or not np.isfinite(fill_values).all():
        raise ValueError("fill_values must contain one finite value per metric")
    invalid = ~np.isfinite(values) | (values <= -90000.0)
    for column in range(values.shape[1]):
        values[invalid[:, column], column] = fill_values[column]
    return values


def causal_windows(values: np.ndarray, length: int = 16, stride: int = 1):
    """Return windows ending at each sampled point; no future samples are used."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("values must be a finite matrix [time, channels]")
    if not isinstance(length, (int, np.integer)) or length < 2:
        raise ValueError("length must be at least two")
    if not isinstance(stride, (int, np.integer)) or stride < 1:
        raise ValueError("stride must be positive")
    starts = np.arange(0, len(x) - length + 1, stride, dtype=np.int64)
    windows = np.stack([x[start : start + length].T for start in starts])
    ends = starts + length - 1
    return windows, starts, ends


def event_overlap_mask(starts: np.ndarray, ends: np.ndarray, events: tuple[Event, ...]):
    """Mark a window when any annotated event interval overlaps it."""
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    mask = np.zeros(len(starts), dtype=bool)
    for event in events:
        mask |= (starts <= event.end_point) & (ends >= event.start_point)
    return mask


def event_window_mask(starts: np.ndarray, ends: np.ndarray, event: Event):
    """Return windows whose intervals overlap one event."""
    return (np.asarray(starts) <= event.end_point) & (np.asarray(ends) >= event.start_point)

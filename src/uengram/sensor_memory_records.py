"""Validation and EdgeMemory Lite encoding for sensor lifecycle events."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np

from .store import IngestResult
from .sensor_memory_types import SensorWindow


_SCHEMA_VERSION = 1
_MAX_STRUCTURED_BYTES = 1024


@dataclass(frozen=True)
class PendingSensorRecord:
    content: str
    structured_value: str
    source_id: str
    observed_at_ms: int
    idempotency_key: str
    keywords: tuple[str, ...]
    event_type: str
    retention_class: int
    raw_window_ref: str | None


def validate_sensor_window(window: SensorWindow, dimension: int) -> np.ndarray:
    for label, value, limit in (
        ("asset_id", window.asset_id, 48),
        ("subsystem_id", window.subsystem_id, 48),
        ("operating_mode", window.operating_mode, 48),
        ("feature_schema", window.feature_schema, 48),
    ):
        if not isinstance(value, str) or not value or "\0" in value:
            raise ValueError(f"{label} must be non-empty text without NUL")
        if len(value.encode("utf-8")) > limit:
            raise ValueError(f"{label} exceeds {limit} UTF-8 bytes")
    if window.raw_window_ref is not None:
        ref = window.raw_window_ref
        if not isinstance(ref, str) or not ref:
            raise ValueError("raw_window_ref must be non-empty text")
        if "\0" in ref or '"' in ref or "\\" in ref or len(ref.encode("utf-8")) > 192:
            raise ValueError("raw_window_ref must be JSON-safe and fit 192 UTF-8 bytes")

    for label, value in (
        ("step_index", window.step_index),
        ("observed_at_ms", window.observed_at_ms),
        ("raw_idx", window.raw_idx),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{label} must be an integer")
        if value < 0 or value > (1 << 63) - 1:
            raise ValueError(f"{label} must fit a non-negative signed 64-bit integer")
    if window.observed_at_ms == 0:
        raise ValueError("observed_at_ms must be positive")
    vector = np.asarray(window.features, dtype=np.float64)
    if vector.shape != (dimension,) or not np.isfinite(vector).all():
        raise ValueError(f"features must be finite with shape ({dimension},)")
    if (
        isinstance(window.retention_class, bool)
        or not isinstance(window.retention_class, Integral)
        or window.retention_class not in (1, 2, 3, 4)
    ):
        raise ValueError("retention_class must be an EdgeMemory class from 1 to 4")
    if window.source_event_id is not None:
        event_id = window.source_event_id
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("source_event_id must be non-empty text")
        if (
            "\0" in event_id or '"' in event_id or "\\" in event_id
            or len(event_id.encode("utf-8")) > 96
        ):
            raise ValueError("source_event_id must be JSON-safe and fit 96 UTF-8 bytes")
    if window.severity is not None:
        if isinstance(window.severity, bool) or window.severity_source is None:
            raise ValueError("severity needs a finite value and an explicit severity_source")
        try:
            valid_severity = math.isfinite(float(window.severity))
        except (TypeError, ValueError, OverflowError):
            valid_severity = False
        if not valid_severity:
            raise ValueError("severity must be a finite number")
    elif window.severity_source is not None:
        raise ValueError("severity_source requires severity")
    if window.severity_source is not None and (
        not isinstance(window.severity_source, str)
        or not window.severity_source
        or "\0" in window.severity_source
        or '"' in window.severity_source
        or "\\" in window.severity_source
        or len(window.severity_source.encode("utf-8")) > 32
    ):
        raise ValueError("severity_source must be JSON-safe and fit 32 UTF-8 bytes")
    return vector


def make_sensor_record(
    window: SensorWindow,
    decision: IngestResult,
    vector: np.ndarray,
    event_type: str,
    lifecycle_events: Sequence[tuple],
) -> PendingSensorRecord:
    source_event = window.source_event_id or hashlib.sha256(
        (
            f"{window.asset_id}\0{window.subsystem_id}\0{window.feature_schema}\0"
            f"{window.step_index}\0{decision.item_id}\0{event_type}\0"
        ).encode()
        + np.asarray(vector, dtype="<f8").tobytes()
    ).hexdigest()
    idempotency_key = "uengram:v1:" + hashlib.sha256(
        f"{window.asset_id}\0{window.subsystem_id}\0{source_event}\0"
        f"{decision.item_id}\0{event_type}".encode()
    ).hexdigest()
    score, health = float(decision.score), float(decision.health)
    metadata = {
        "schema": _SCHEMA_VERSION,
        "asset": window.asset_id,
        "subsystem": window.subsystem_id,
        "mode": window.operating_mode,
        "feature_schema": window.feature_schema,
        "step": int(window.step_index),
        "item_id": int(decision.item_id),
        "event_type": event_type,
        "action": decision.action,
        "novelty": score if math.isfinite(score) else None,
        "health": health if math.isfinite(health) else None,
        "flagged": decision.flagged,
        "severity": window.severity,
        "severity_source": window.severity_source,
        "feature_dimension": int(vector.size),
        "raw_window_ref": window.raw_window_ref,
        "source_event_id": source_event,
        "lifecycle_events": [list(event) for event in lifecycle_events],
    }
    structured = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
    if len(structured.encode("utf-8")) > _MAX_STRUCTURED_BYTES:
        raise ValueError("sensor event metadata exceeds EdgeMemory's 1024-byte value limit")
    source_id = "sensor:" + hashlib.sha256(
        f"{window.asset_id}\0{window.subsystem_id}".encode()
    ).hexdigest()[:24]
    content = (
        f"uEngram {event_type}: asset={window.asset_id}; "
        f"subsystem={window.subsystem_id}; mode={window.operating_mode}; "
        f"step={window.step_index}; item={decision.item_id}"
    )
    keywords = (
        "uengram_sensor",
        f"event_{event_type}",
        _tag("asset", window.asset_id),
        _tag("subsystem", window.subsystem_id),
        _tag("mode", window.operating_mode),
        f"item_{decision.item_id}",
    )
    return PendingSensorRecord(
        content=content,
        structured_value=structured,
        source_id=source_id,
        observed_at_ms=int(window.observed_at_ms),
        idempotency_key=idempotency_key,
        keywords=keywords,
        event_type=event_type,
        retention_class=4 if window.raw_window_ref is not None else int(window.retention_class),
        raw_window_ref=window.raw_window_ref,
    )


def _tag(label: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
    return f"{label}_{digest}"

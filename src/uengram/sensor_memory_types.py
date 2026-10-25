"""Public data contracts for the uEngram Sensor-State pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .store import IngestResult


class EdgeMemoryPort(Protocol):
    """Subset of the EdgeMemory Lite Python API used by the adapter."""

    def remember(self, content: str, **kwargs: object) -> object: ...

    def recall(self, **kwargs: object) -> object: ...

    def forget(self, memory_id: int) -> int: ...


class RawEvidencePins(Protocol):
    """Durable, idempotent pinning interface for a host time-series store.

    `pin` prevents eviction of an immutable raw window until `unpin` succeeds.
    The owner key is stable across retries and identifies one EdgeMemory record.
    """

    def pin(self, reference: str, owner_key: str) -> None: ...

    def unpin(self, reference: str, owner_key: str) -> None: ...


@dataclass(frozen=True)
class SensorWindow:
    asset_id: str
    subsystem_id: str
    operating_mode: str
    step_index: int
    observed_at_ms: int
    raw_idx: int
    features: np.ndarray
    feature_schema: str
    raw_window_ref: str | None = None
    source_event_id: str | None = None
    severity: float | None = None
    severity_source: str | None = None
    retention_class: int = 2


@dataclass(frozen=True)
class PatternRecall:
    item_id: int
    mean: np.ndarray
    variance: np.ndarray
    mass: float
    last_step: int
    lifecycle: str


@dataclass(frozen=True)
class SensorIngestReceipt:
    step_index: int
    decision: IngestResult
    memory_id: int | None
    idempotency_key: str | None
    raw_window_ref: str | None
    persisted: bool


@dataclass(frozen=True)
class SensorRecall:
    patterns: tuple[PatternRecall, ...]
    historical_events: tuple[object, ...]

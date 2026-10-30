"""Host-owned exact vector sidecar for EdgeMemory Lite lifecycle events.

Lite has no vector API. This index persists the compact vector snapshots already
stored in uEngram event metadata and scans them exactly under strict identity
filters. It is not an ANN implementation and must not be described as one.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Protocol

import numpy as np

from .retrieval import RankedEvent, RetrievalWeights, rank_events
from .sensor_memory_records import PendingSensorRecord

_SCHEMA_VERSION = 1


class SensorVectorIndex(Protocol):
    """Host-side vector index synchronized after an idempotent vendor write."""

    def upsert(self, memory_id: int, record: PendingSensorRecord) -> None: ...

    def forget(self, memory_id: int) -> None: ...

    def rank(
        self,
        query_vector: np.ndarray,
        *,
        now_ms: int,
        asset_id: str,
        subsystem_id: str,
        operating_mode: str | None = None,
        feature_schema: str | None = None,
        limit: int = 16,
        weights: RetrievalWeights | None = None,
    ) -> tuple[RankedEvent, ...]: ...


class FileSensorVectorIndex:
    """Fsync-backed, single-writer exact scan over persisted sensor event vectors."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def upsert(self, memory_id: int, record: PendingSensorRecord) -> None:
        if isinstance(memory_id, bool) or not isinstance(memory_id, int) or memory_id < 0:
            raise ValueError("memory_id must be a non-negative integer")
        events = list(self._read())
        entry = {
            "memory_id": memory_id,
            "idempotency_key": record.idempotency_key,
            "content": record.content,
            "structured_value": record.structured_value,
        }
        for existing in events:
            if existing["idempotency_key"] != record.idempotency_key:
                continue
            if existing != entry:
                raise ValueError("idempotency key already belongs to another vector event")
            return
        events.append(entry)
        self._write(events)

    def forget(self, memory_id: int) -> None:
        events = list(self._read())
        retained = [event for event in events if event["memory_id"] != memory_id]
        if len(retained) != len(events):
            self._write(retained)

    def rank(
        self,
        query_vector: np.ndarray,
        *,
        now_ms: int,
        asset_id: str,
        subsystem_id: str,
        operating_mode: str | None = None,
        feature_schema: str | None = None,
        limit: int = 16,
        weights: RetrievalWeights | None = None,
    ) -> tuple[RankedEvent, ...]:
        return tuple(
            rank_events(
                self._read(),
                query_vector,
                now_ms,
                weights,
                asset_id=asset_id,
                subsystem_id=subsystem_id,
                operating_mode=operating_mode,
                feature_schema=feature_schema,
                limit=limit,
            )
        )

    def _read(self) -> tuple[dict[str, object], ...]:
        if not self.path.exists():
            return ()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("schema") != _SCHEMA_VERSION:
                raise ValueError("unsupported schema")
            entries = payload["events"]
            if not isinstance(entries, list):
                raise TypeError("events must be a list")
            return tuple(self._entry(entry) for entry in entries)
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid sensor vector index at {self.path}") from exc

    def _write(self, events: list[dict[str, object]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"schema": _SCHEMA_VERSION, "events": events},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp"
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _entry(entry: object) -> dict[str, object]:
        if not isinstance(entry, dict):
            raise TypeError("event must be an object")
        required_text = ("idempotency_key", "content", "structured_value")
        if any(not isinstance(entry.get(key), str) for key in required_text):
            raise TypeError("event text fields must be strings")
        memory_id = entry.get("memory_id")
        if isinstance(memory_id, bool) or not isinstance(memory_id, int) or memory_id < 0:
            raise ValueError("event memory_id must be a non-negative integer")
        return {
            "memory_id": memory_id,
            "idempotency_key": entry["idempotency_key"],
            "content": entry["content"],
            "structured_value": entry["structured_value"],
        }

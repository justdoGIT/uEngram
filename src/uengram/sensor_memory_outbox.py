"""Crash-safe local outbox for idempotent sensor lifecycle delivery.

The outbox is deliberately independent of EdgeMemory. It persists a completed
uEngram decision before Lite is called, then removes it only after Lite accepts
the stable idempotency key. One application process must own each file.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .sensor_memory_records import PendingSensorRecord
from .store import IngestResult

_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PendingSensorDelivery:
    """A replayable lifecycle event and the receipt context it needs."""

    record: PendingSensorRecord
    decision: IngestResult
    step_index: int


class SensorEventOutbox(Protocol):
    """Durable queue used by :class:`SensorStateMemory` before vendor delivery."""

    def enqueue(self, delivery: PendingSensorDelivery) -> None: ...

    def pending(self) -> tuple[PendingSensorDelivery, ...]: ...

    def acknowledge(self, idempotency_key: str) -> None: ...


class FileSensorEventOutbox:
    """Atomically rewritten, fsync-backed outbox for one writer process.

    The file has no locking protocol. Give each concurrently running pipeline its
    own file or put process coordination around this class. A malformed file is
    rejected rather than silently discarded because it may hold undelivered events.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def enqueue(self, delivery: PendingSensorDelivery) -> None:
        deliveries = list(self.pending())
        for existing in deliveries:
            if existing.record.idempotency_key != delivery.record.idempotency_key:
                continue
            if existing != delivery:
                raise ValueError("idempotency key already belongs to another sensor delivery")
            return
        deliveries.append(delivery)
        self._write(deliveries)

    def pending(self) -> tuple[PendingSensorDelivery, ...]:
        if not self.path.exists():
            return ()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("schema") != _SCHEMA_VERSION:
                raise ValueError("unsupported schema")
            entries = payload["deliveries"]
            if not isinstance(entries, list):
                raise TypeError("deliveries must be a list")
            return tuple(self._decode(entry) for entry in entries)
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid sensor event outbox at {self.path}") from exc

    def acknowledge(self, idempotency_key: str) -> None:
        deliveries = list(self.pending())
        retained = [
            delivery
            for delivery in deliveries
            if delivery.record.idempotency_key != idempotency_key
        ]
        if len(retained) != len(deliveries):
            self._write(retained)

    def _write(self, deliveries: list[PendingSensorDelivery]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "schema": _SCHEMA_VERSION,
                "deliveries": [self._encode(delivery) for delivery in deliveries],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
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
    def _encode(delivery: PendingSensorDelivery) -> dict[str, object]:
        record = delivery.record
        decision = delivery.decision
        return {
            "record": {
                "content": record.content,
                "structured_value": record.structured_value,
                "source_id": record.source_id,
                "observed_at_ms": record.observed_at_ms,
                "idempotency_key": record.idempotency_key,
                "keywords": list(record.keywords),
                "event_type": record.event_type,
                "retention_class": record.retention_class,
                "raw_window_ref": record.raw_window_ref,
                "raw_window_checksum": record.raw_window_checksum,
                "raw_window_expires_at_ms": record.raw_window_expires_at_ms,
            },
            "decision": {
                "item_id": decision.item_id,
                "score": FileSensorEventOutbox._optional_number(decision.score),
                "flagged": decision.flagged,
                "action": decision.action,
                "health": FileSensorEventOutbox._optional_number(decision.health),
            },
            "step_index": delivery.step_index,
        }

    @staticmethod
    def _decode(entry: object) -> PendingSensorDelivery:
        if not isinstance(entry, dict):
            raise TypeError("delivery must be an object")
        record_data = entry["record"]
        decision_data = entry["decision"]
        if not isinstance(record_data, dict) or not isinstance(decision_data, dict):
            raise TypeError("delivery fields must be objects")
        record = PendingSensorRecord(
            content=FileSensorEventOutbox._text(record_data, "content"),
            structured_value=FileSensorEventOutbox._text(record_data, "structured_value"),
            source_id=FileSensorEventOutbox._text(record_data, "source_id"),
            observed_at_ms=FileSensorEventOutbox._integer(record_data, "observed_at_ms"),
            idempotency_key=FileSensorEventOutbox._text(record_data, "idempotency_key"),
            keywords=FileSensorEventOutbox._keywords(record_data),
            event_type=FileSensorEventOutbox._text(record_data, "event_type"),
            retention_class=FileSensorEventOutbox._integer(record_data, "retention_class"),
            raw_window_ref=FileSensorEventOutbox._optional_text(record_data, "raw_window_ref"),
            raw_window_checksum=FileSensorEventOutbox._optional_text(record_data, "raw_window_checksum"),
            raw_window_expires_at_ms=FileSensorEventOutbox._optional_integer(
                record_data, "raw_window_expires_at_ms"
            ),
        )
        flagged = decision_data["flagged"]
        if flagged is not None and not isinstance(flagged, bool):
            raise ValueError("flagged must be bool or null")
        decision = IngestResult(
            item_id=FileSensorEventOutbox._integer(decision_data, "item_id"),
            score=FileSensorEventOutbox._number_or_nan(decision_data, "score"),
            flagged=flagged,
            action=FileSensorEventOutbox._text(decision_data, "action"),
            health=FileSensorEventOutbox._number_or_nan(decision_data, "health"),
        )
        return PendingSensorDelivery(
            record=record,
            decision=decision,
            step_index=FileSensorEventOutbox._integer(entry, "step_index"),
        )

    @staticmethod
    def _text(data: dict[str, object], key: str) -> str:
        value = data[key]
        if not isinstance(value, str):
            raise TypeError(f"{key} must be text")
        return value

    @staticmethod
    def _optional_text(data: dict[str, object], key: str) -> str | None:
        value = data[key]
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{key} must be text or null")
        return value

    @staticmethod
    def _integer(data: dict[str, object], key: str) -> int:
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{key} must be an integer")
        return value

    @staticmethod
    def _optional_integer(data: dict[str, object], key: str) -> int | None:
        value = data[key]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f"{key} must be an integer or null")
        return value

    @staticmethod
    def _keywords(data: dict[str, object]) -> tuple[str, ...]:
        values = data["keywords"]
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError("keywords must be a list of text")
        return tuple(values)

    @staticmethod
    def _optional_number(value: object) -> float | None:
        number = float(value)
        return number if math.isfinite(number) else None

    @staticmethod
    def _number_or_nan(data: dict[str, object], key: str) -> float:
        value = data[key]
        if value is None:
            return float("nan")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{key} must be a number or null")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{key} must be finite")
        return number

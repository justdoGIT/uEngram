"""SQLite system of record for uEngram recovery and Lite projection."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

import numpy as np

from .confirmed_adaptation import ConfirmedLatentAdapter
from .engine import Engram
from .retrieval import RankedEvent, RetrievalWeights, rank_events
from .sensor_capsule_checkpoint import dump_checkpoint, restore_checkpoint
from .sensor_memory_outbox import PendingSensorDelivery
from .sensor_memory_records import PendingSensorRecord
from .sensor_memory_types import SensorWindow


@dataclass(frozen=True)
class CapsuleState:
    checkpoint_step: int
    last_step: int
    last_alarm: bool | None


class SensorMemoryCapsule:
    """Single-writer SQLite WAL, checkpoint, Lite-outbox and vector sidecar."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self._schema()

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def record_window(self, window: SensorWindow) -> None:
        payload = json.dumps({**asdict(window), "features": np.asarray(window.features, dtype=float).tolist()},
                             separators=(",", ":"))
        with self.db:
            prior = self.db.execute("SELECT payload FROM input_wal WHERE step=?", (window.step_index,)).fetchone()
            if prior is not None and prior[0] != payload:
                raise ValueError("step_index already belongs to another input window")
            self.db.execute("INSERT OR IGNORE INTO input_wal(step,payload) VALUES (?,?)", (window.step_index, payload))

    def checkpoint(
        self, engine: Engram, *, last_step: int, last_alarm: bool | None,
        adapter: ConfirmedLatentAdapter | None = None, delivery: PendingSensorDelivery | None = None,
    ) -> None:
        blob = dump_checkpoint(engine, adapter)
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO checkpoint(id,step,last_step,last_alarm,payload) VALUES (1,?,?,?,?)",
                            (last_step, last_step, None if last_alarm is None else int(last_alarm), blob))
            if delivery is not None:
                self._enqueue(delivery)

    def restore(self, engine: Engram, adapter: ConfirmedLatentAdapter | None = None) -> CapsuleState | None:
        row = self.db.execute("SELECT step,last_step,last_alarm,payload FROM checkpoint WHERE id=1").fetchone()
        if row is None:
            return None
        restore_checkpoint(engine, adapter, row[3])
        return CapsuleState(int(row[0]), int(row[1]), None if row[2] is None else bool(row[2]))

    def uncheckpointed_windows(self) -> tuple[SensorWindow, ...]:
        row = self.db.execute("SELECT step FROM checkpoint WHERE id=1").fetchone()
        after = -1 if row is None else int(row[0])
        return tuple(self._window(payload) for _, payload in self.db.execute(
            "SELECT step,payload FROM input_wal WHERE step>? ORDER BY step", (after,)
        ))

    def enqueue(self, delivery: PendingSensorDelivery) -> None:
        with self.db:
            self._enqueue(delivery)

    def pending(self) -> tuple[PendingSensorDelivery, ...]:
        return tuple(self._delivery(payload) for payload, in self.db.execute(
            "SELECT payload FROM lite_outbox ORDER BY rowid"
        ))

    def acknowledge(self, idempotency_key: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM lite_outbox WHERE event_id=?", (idempotency_key,))

    def upsert(self, memory_id: int, record: PendingSensorRecord) -> None:
        with self.db:
            self.db.execute("INSERT INTO vectors(event_id,memory_id,content,structured_value) VALUES (?,?,?,?) "
                            "ON CONFLICT(event_id) DO UPDATE SET memory_id=excluded.memory_id",
                            (record.idempotency_key, memory_id, record.content, record.structured_value))

    def forget(self, memory_id: int) -> None:
        with self.db:
            self.db.execute("DELETE FROM vectors WHERE memory_id=?", (memory_id,))

    def rank(self, query_vector: np.ndarray, *, now_ms: int, asset_id: str, subsystem_id: str,
             operating_mode: str | None = None, feature_schema: str | None = None,
             limit: int = 16, weights: RetrievalWeights | None = None) -> tuple[RankedEvent, ...]:
        rows = tuple({"memory_id": row[0], "content": row[1], "structured_value": row[2]}
                     for row in self.db.execute("SELECT memory_id,content,structured_value FROM vectors"))
        return tuple(rank_events(rows, query_vector, now_ms, weights, asset_id=asset_id,
                                 subsystem_id=subsystem_id, operating_mode=operating_mode,
                                 feature_schema=feature_schema, limit=limit))

    def request_pin(self, event_id: str, reference: str, checksum: str, expiry_ms: int | None) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO pin_intent(event_id,reference,checksum,expiry_ms,state) VALUES (?,?,?,?,?)",
                            (event_id, reference, checksum, expiry_ms, "requested"))

    def mark_pin(self, event_id: str, state: str) -> None:
        if state not in {"pinned", "released"}:
            raise ValueError("pin state must be pinned or released")
        with self.db:
            self.db.execute("UPDATE pin_intent SET state=? WHERE event_id=?", (state, event_id))

    def _enqueue(self, delivery: PendingSensorDelivery) -> None:
        payload = self._encode_delivery(delivery)
        row = self.db.execute("SELECT payload FROM lite_outbox WHERE event_id=?", (delivery.record.idempotency_key,)).fetchone()
        if row is not None and row[0] != payload:
            raise ValueError("idempotency key already belongs to another delivery")
        self.db.execute("INSERT OR IGNORE INTO lite_outbox(event_id,payload) VALUES (?,?)",
                        (delivery.record.idempotency_key, payload))

    def _schema(self) -> None:
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS input_wal(step INTEGER PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS checkpoint(id INTEGER PRIMARY KEY CHECK(id=1), step INTEGER NOT NULL,
                last_step INTEGER NOT NULL, last_alarm INTEGER, payload BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS lite_outbox(event_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS vectors(event_id TEXT PRIMARY KEY, memory_id INTEGER NOT NULL,
                content TEXT NOT NULL, structured_value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS pin_intent(event_id TEXT PRIMARY KEY, reference TEXT NOT NULL,
                checksum TEXT NOT NULL, expiry_ms INTEGER, state TEXT NOT NULL);
        """)

    @staticmethod
    def _encode_delivery(delivery: PendingSensorDelivery) -> str:
        return json.dumps({"record": asdict(delivery.record), "decision": asdict(delivery.decision),
                           "step": delivery.step_index}, separators=(",", ":"), allow_nan=True)

    @staticmethod
    def _delivery(payload: str) -> PendingSensorDelivery:
        from .store import IngestResult
        data = json.loads(payload)
        return PendingSensorDelivery(PendingSensorRecord(**data["record"]), IngestResult(**data["decision"]), int(data["step"]))

    @staticmethod
    def _window(payload: str) -> SensorWindow:
        data = json.loads(payload)
        data["features"] = np.asarray(data["features"], dtype=float)
        return SensorWindow(**data)

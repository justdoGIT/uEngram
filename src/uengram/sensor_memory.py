"""Compose uEngram's sensor lifecycle with EdgeMemory Lite persistence.

The EdgeMemory object is injected, so this module does not import CFFI or the
optional AutoGen integration. Lite persists lifecycle-event metadata as text;
raw sensor windows and embeddings remain in the host's evidence store.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral

import numpy as np

from .confirmed_adaptation import ConfirmedLatentAdapter
from .engine import Engram
from .retrieval import RetrievalWeights, rank_events
from .sensor_memory_outbox import PendingSensorDelivery, SensorEventOutbox
from .sensor_memory_records import (
    PendingSensorRecord,
    _tag,
    identity_key,
    make_sensor_record,
    validate_sensor_window,
)
from .sensor_memory_types import (
    EdgeMemoryPort,
    PatternRecall,
    RawEvidencePins,
    SensorIngestReceipt,
    SensorRecall,
    SensorWindow,
)
from .sensor_vector_index import SensorVectorIndex
from .store import IngestResult

_EDGE_EVENT = 2
_EDGE_SOURCE_SENSOR = 2
_EDGE_TRUST_OBSERVED = 3
_EDGE_PRIVACY_LOCAL_PRIVATE = 1
_EDGE_KIND_MASK_EVENT = 1 << (_EDGE_EVENT - 1)
_SCHEMA_VERSION = 1
_PERSIST_ACTIONS = frozenset({"new_candidate", "promote", "step_change"})



class SensorMemoryPersistenceError(RuntimeError):
    """Persistence failed after the in-memory detector consumed the window.

    Retry `persist_pending(error.pending)`; do not ingest that window again.
    """

    def __init__(
        self,
        pending: PendingSensorRecord,
        decision: IngestResult,
        cause: Exception,
    ) -> None:
        super().__init__(f"sensor memory persistence failed: {cause}")
        self.pending = pending
        self.decision = decision
        self.__cause__ = cause

class SensorStateMemory:
    """Sensor window → uEngram lifecycle → durable EdgeMemory event pipeline.

    One instance owns one uEngram engine and one EdgeMemory scope. Input windows
    must arrive once, in strictly increasing `step_index` order. `step_index`
    is a sample counter for uEngram's M-of-N timers; wall-clock milliseconds
    belong in `observed_at_ms` and must never be passed as the engine step.

    With an `adapter` (confirmed_adaptation.ConfirmedLatentAdapter), each identity key
    (asset, subsystem, mode, schema) keeps a bounded latent offset that is subtracted
    before scoring. It is updated only from windows the operator marked
    `operator_confirmed_normal` while no alarm was open and the key already had a
    learned normal state, so gradual wear nobody confirmed is never absorbed.
    """

    def __init__(
        self,
        engine: Engram,
        memory: EdgeMemoryPort,
        *,
        raw_evidence: RawEvidencePins | None = None,
        outbox: SensorEventOutbox | None = None,
        vector_index: SensorVectorIndex | None = None,
        last_step: int = -1,
        last_alarm: bool | None = None,
        adapter: ConfirmedLatentAdapter | None = None,
    ) -> None:
        if engine.events:
            raise ValueError("drain pre-existing engine events before attaching the adapter")
        if adapter is not None and adapter.d != engine.d:
            raise ValueError("adapter dimension must match the engine's")
        self.adapter = adapter
        self.engine = engine
        self.memory = memory
        self.raw_evidence = raw_evidence
        self.outbox = outbox
        self.vector_index = vector_index
        if isinstance(last_step, bool) or not isinstance(last_step, Integral) or last_step < -1:
            raise ValueError("last_step must be an integer greater than or equal to -1")
        if last_alarm is not None and not isinstance(last_alarm, bool):
            raise ValueError("last_alarm must be bool or None")
        self._last_step = int(last_step)
        self._last_alarm = last_alarm
        self._known_pins: dict[int, tuple[str, str, int | None]] = {}

    def ingest(self, window: SensorWindow) -> SensorIngestReceipt:
        vector = validate_sensor_window(window, self.engine.d)
        if window.step_index <= self._last_step:
            raise ValueError(
                "step_index must increase strictly; reorder or deduplicate input "
                "before calling uEngram"
            )
        if window.raw_window_ref is not None and self.raw_evidence is None:
            raise ValueError("raw_window_ref requires a RawEvidencePins implementation")

        key = identity_key(window)
        scored = vector if self.adapter is None else self.adapter.apply(vector[None, :], key)[0]
        decision = self.engine.ingest(scored, window.step_index, window.raw_idx, mode_key=key)
        if self.adapter is not None:
            self.adapter.update(
                vector[None, :], key,
                operator_confirmed_normal=window.operator_confirmed_normal,
                alarm_closed=decision.flagged is not True,
                commissioned=math.isfinite(decision.score),
            )
        self._last_step = window.step_index
        lifecycle_events = tuple(self.engine.drain_events())
        alarm_transition = False
        if decision.flagged is not None:
            alarm_transition = (
                decision.flagged is True
                if self._last_alarm is None
                else decision.flagged != self._last_alarm
            )
            self._last_alarm = decision.flagged
        if not self._should_persist(decision, alarm_transition, lifecycle_events):
            return SensorIngestReceipt(
                window.step_index, decision, None, None, None, False
            )

        event_type = (
            "alarm_open" if alarm_transition and decision.flagged
            else "alarm_clear" if alarm_transition
            else decision.action if decision.action in _PERSIST_ACTIONS
            else "lifecycle_update"
        )
        pending = make_sensor_record(
            window, decision, vector, event_type, lifecycle_events
        )
        try:
            return self._deliver(pending, decision, window.step_index)
        except Exception as exc:
            raise SensorMemoryPersistenceError(pending, decision, exc) from exc

    def persist_pending(
        self, pending: PendingSensorRecord, decision: IngestResult, step_index: int
    ) -> SensorIngestReceipt:
        """Retry storage without feeding the already-consumed window twice."""
        try:
            return self._deliver(pending, decision, step_index)
        except Exception as exc:
            raise SensorMemoryPersistenceError(pending, decision, exc) from exc

    def replay_outbox(self) -> tuple[SensorIngestReceipt, ...]:
        """Deliver pending records after restart without ingesting their windows again."""
        if self.outbox is None:
            return ()
        receipts = []
        for delivery in self.outbox.pending():
            try:
                receipt = self._persist(
                    delivery.record, delivery.decision, delivery.step_index
                )
                self.outbox.acknowledge(delivery.record.idempotency_key)
            except Exception as exc:
                raise SensorMemoryPersistenceError(
                    delivery.record, delivery.decision, exc
                ) from exc
            receipts.append(receipt)
        return tuple(receipts)

    def forget(self, receipt: SensorIngestReceipt) -> int:
        """Forget an event, then idempotently release its raw-window pin."""
        if receipt.memory_id is None:
            return 0
        owner = receipt.idempotency_key
        reference = receipt.raw_window_ref
        if reference is not None:
            if self.raw_evidence is None or owner is None:
                raise RuntimeError("receipt lacks the raw-evidence pin owner")
            known = self._known_pins.get(receipt.memory_id)
            if known is not None and known[:2] != (reference, owner):
                raise ValueError("receipt does not match the pinned memory")
        deleted = self.memory.forget(receipt.memory_id)
        if deleted and self.vector_index is not None:
            self.vector_index.forget(receipt.memory_id)
        if reference is not None:
            self.raw_evidence.unpin(reference, owner)
            if self._known_pins.pop(receipt.memory_id, None) is not None:
                self.engine.unprotect(receipt.decision.item_id)
        return deleted

    def expired_pins(self, now_ms: int) -> tuple[int, ...]:
        """Memory ids whose raw-window pin has passed its expiration, ascending by id.

        For a host janitor to act on by calling ``forget``; this adapter never unpins
        on its own initiative. A pin with no ``raw_window_ttl_ms`` never expires."""
        if isinstance(now_ms, bool) or not isinstance(now_ms, Integral) or now_ms < 0:
            raise ValueError("now_ms must be a non-negative integer")
        return tuple(sorted(
            memory_id for memory_id, (_, _, expires_at) in self._known_pins.items()
            if expires_at is not None and now_ms >= expires_at
        ))

    def recall(
        self,
        query_vector: np.ndarray,
        *,
        t_now: int,
        k: int = 8,
        mode: str = "current",
        asset_id: str | None = None,
        subsystem_id: str | None = None,
        operating_mode: str | None = None,
        historical_keywords: Sequence[str] = (),
        max_history: int = 16,
        now_ms: int | None = None,
        retrieval_weights: RetrievalWeights | None = None,
    ) -> SensorRecall:
        """Return exact uEngram pattern matches plus Lite keyword event matches."""
        for label, value in (("t_now", t_now), ("k", k), ("max_history", max_history)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        if mode not in ("current", "history"):
            raise ValueError("mode must be 'current' or 'history'")
        if now_ms is not None:
            if isinstance(now_ms, bool) or not isinstance(now_ms, Integral) or now_ms < 0:
                raise ValueError("now_ms must be a non-negative integer")
            if not asset_id or not subsystem_id:
                raise ValueError("ranked history requires both asset_id and subsystem_id")
        query = np.asarray(query_vector, dtype=np.float64)
        if query.shape != (self.engine.d,) or not np.isfinite(query).all():
            raise ValueError(f"query_vector must be finite with shape ({self.engine.d},)")
        ids = self.engine.recall(query, t_now, k, mode)
        view = self.engine.items()
        positions = {int(item_id): index for index, item_id in enumerate(view.ids)}
        patterns = tuple(
            PatternRecall(
                item_id=item_id,
                mean=view.mu[positions[item_id]].copy(),
                variance=view.var[positions[item_id]].copy(),
                mass=float(view.mass[positions[item_id]]),
                last_step=int(view.t_last[positions[item_id]]),
                lifecycle=self._lifecycle(int(self.engine.kind[
                    np.flatnonzero(self.engine.item_id == item_id)[0]
                ])),
            )
            for item_id in ids
            if item_id in positions
        )
        terms = tuple(dict.fromkeys(str(term) for term in historical_keywords if term))
        if terms or asset_id is not None or subsystem_id is not None:
            if not asset_id or not subsystem_id:
                raise ValueError("historical recall requires both asset_id and subsystem_id")
            required = [_tag("asset", asset_id), _tag("subsystem", subsystem_id)]
            if operating_mode is not None:
                if not operating_mode:
                    raise ValueError("operating_mode cannot be empty")
                required.append(_tag("mode", operating_mode))
            terms = tuple(dict.fromkeys((*required, *terms)))
            if len(terms) > 32 or any(len(term.encode("utf-8")) > 128 for term in terms):
                raise ValueError("historical query exceeds EdgeMemory's keyword limits")
        if not terms or max_history == 0:
            return SensorRecall(patterns, ())
        response = self.memory.recall(
            keywords=terms,
            kind_mask=_EDGE_KIND_MASK_EVENT,
            max_results=max_history,
        )
        items = tuple(getattr(response, "items", ()))
        ranked = ()
        if now_ms is not None:
            if self.vector_index is not None:
                ranked = self.vector_index.rank(
                    query, now_ms=now_ms, asset_id=asset_id, subsystem_id=subsystem_id,
                    operating_mode=operating_mode, limit=max_history,
                    weights=retrieval_weights,
                )
            elif items:
                ranked = tuple(rank_events(
                    items, query, now_ms, retrieval_weights,
                    asset_id=asset_id, subsystem_id=subsystem_id, operating_mode=operating_mode,
                    limit=max_history,
                ))
        return SensorRecall(patterns, items, ranked)

    def _deliver(
        self, pending: PendingSensorRecord, decision: IngestResult, step_index: int
    ) -> SensorIngestReceipt:
        if self.outbox is not None:
            self.outbox.enqueue(PendingSensorDelivery(pending, decision, step_index))
        receipt = self._persist(pending, decision, step_index)
        if self.outbox is not None:
            self.outbox.acknowledge(pending.idempotency_key)
        return receipt

    def _persist(
        self, pending: PendingSensorRecord, decision: IngestResult, step_index: int
    ) -> SensorIngestReceipt:
        pinned = False
        if pending.raw_window_ref is not None:
            assert self.raw_evidence is not None
            self.raw_evidence.pin(pending.raw_window_ref, pending.idempotency_key,
                                  pending.raw_window_checksum, pending.raw_window_expires_at_ms)
            pinned = True
        try:
            result = self.memory.remember(
                pending.content,

                kind=_EDGE_EVENT,
                structured_value=pending.structured_value,
                source_class=_EDGE_SOURCE_SENSOR,
                source_id=pending.source_id,
                trust_class=_EDGE_TRUST_OBSERVED,
                privacy_class=_EDGE_PRIVACY_LOCAL_PRIVATE,
                observed_at_ms=pending.observed_at_ms,
                idempotency_key=pending.idempotency_key,
                keywords=pending.keywords,
                retention_class=pending.retention_class,
                policy_version=_SCHEMA_VERSION,
            )
        except Exception:
            if pinned:
                assert self.raw_evidence is not None
                self.raw_evidence.unpin(pending.raw_window_ref, pending.idempotency_key)
            raise
        memory_id = int(result.memory_id)
        if self.vector_index is not None:
            self.vector_index.upsert(memory_id, pending)
        if pinned:
            already_known = memory_id in self._known_pins
            self._known_pins[memory_id] = (
                pending.raw_window_ref, pending.idempotency_key, pending.raw_window_expires_at_ms,
            )
            if not already_known:
                self.engine.protect(decision.item_id)
        return SensorIngestReceipt(
            step_index, decision, memory_id, pending.idempotency_key,
            pending.raw_window_ref, True,
        )


    @staticmethod
    def _should_persist(
        decision: IngestResult,
        alarm_transition: bool,
        lifecycle_events: tuple[tuple, ...],
    ) -> bool:
        return bool(
            decision.action in _PERSIST_ACTIONS
            or alarm_transition
            or lifecycle_events
        )

    @staticmethod
    def _lifecycle(kind: int) -> str:
        return {1: "candidate", 2: "episodic", 3: "semantic", 4: "superseded"}.get(
            kind, "unknown"
        )


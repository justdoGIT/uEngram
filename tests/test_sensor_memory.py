from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from uengram import SensorMemoryPersistenceError, SensorStateMemory, SensorWindow
from uengram.engine import Engram, EngramConfig


class MemoryBackend:
    """Small contract implementation with the same observable Lite semantics."""

    def __init__(self):
        self.rows = {}
        self.fail_once = False

    def remember(self, content, **kwargs):
        if self.fail_once:
            self.fail_once = False
            raise OSError("simulated storage outage")
        key = kwargs["idempotency_key"]
        for row in self.rows.values():
            if row["idempotency_key"] == key:
                return SimpleNamespace(memory_id=row["memory_id"])
        memory_id = len(self.rows) + 1
        self.rows[memory_id] = {
            "memory_id": memory_id,
            "content": content,
            "keywords": tuple(kwargs["keywords"]),
            "retention_class": kwargs["retention_class"],
            **kwargs,
        }
        return SimpleNamespace(memory_id=memory_id)

    def recall(self, *, keywords, kind_mask, max_results):
        requested = set(keywords)
        rows = [row for row in self.rows.values() if requested.intersection(row["keywords"])]
        rows.sort(key=lambda row: (-len(requested.intersection(row["keywords"])), -row["memory_id"]))
        return SimpleNamespace(items=tuple(rows[:max_results]))

    def forget(self, memory_id):
        return int(self.rows.pop(memory_id, None) is not None)


class PinStore:
    def __init__(self):
        self.pins = set()
        self.details = {}

    def pin(self, reference, owner_key, checksum, expires_at_ms):
        self.pins.add((reference, owner_key))
        self.details[(reference, owner_key)] = (checksum, expires_at_ms)

    def unpin(self, reference, owner_key):
        self.pins.discard((reference, owner_key))
        self.details.pop((reference, owner_key), None)


def make_pipeline(store=None, pins=None):
    config = EngramConfig(
        B_sem=4,
        B_epi=4,
        m_promote=5,
        n_promote=8,
        use_step_detect=False,
        alarm_threshold=1.0,
        consolidate_every=10_000,
    )
    engine = Engram(config, 2, np.ones(2), raw_window_bytes=64)
    return SensorStateMemory(engine, store or MemoryBackend(), raw_evidence=pins)


def window(step, values, **overrides):
    fields = {
        "asset_id": "pump-17",
        "subsystem_id": "bearing",
        "operating_mode": "steady",
        "step_index": step,
        "observed_at_ms": 1_800_000_000_000 + step * 1000,
        "raw_idx": step,
        "features": np.asarray(values, dtype=float),
        "feature_schema": "bearing-v1",
    }
    fields.update(overrides)
    return SensorWindow(**fields)


def test_lifecycle_and_alarm_edges_are_persisted_and_recalled():
    store = MemoryBackend()
    pipeline = make_pipeline(store)
    baseline = [pipeline.ingest(window(step, [0, 0])) for step in range(5)]
    opened = pipeline.ingest(window(5, [10, 10]))
    repeated = pipeline.ingest(window(6, [10, 10]))
    cleared = pipeline.ingest(window(7, [0, 0]))

    assert [item.decision.action for item in baseline][0] == "new_candidate"
    assert baseline[-1].decision.action == "promote"
    assert json.loads(store.rows[opened.memory_id]["structured_value"])["event_type"] == "alarm_open"
    assert repeated.persisted is False
    assert repeated.decision.flagged is True
    assert json.loads(store.rows[cleared.memory_id]["structured_value"])["event_type"] == "alarm_clear"
    assert len(store.rows) == 4

    recalled = pipeline.recall(
        np.zeros(2), t_now=7, asset_id="pump-17", subsystem_id="bearing",
        max_history=8,
    )
    assert {item["event_type"] for item in map(
        lambda row: json.loads(row["structured_value"]), recalled.historical_events
    )} == {"new_candidate", "promote", "alarm_open", "alarm_clear"}
    assert any(pattern.lifecycle == "semantic" for pattern in recalled.patterns)


def test_raw_evidence_is_pinned_critical_and_released_on_forget():
    store, pins = MemoryBackend(), PinStore()
    pipeline = make_pipeline(store, pins)
    receipt = pipeline.ingest(window(
        0, [0, 0], raw_window_ref="ring://pump-17/0-4", raw_window_checksum="sha256:deadbeef",
    ))

    assert (receipt.raw_window_ref, receipt.idempotency_key) in pins.pins
    assert pins.details[(receipt.raw_window_ref, receipt.idempotency_key)] == ("sha256:deadbeef", None)
    assert store.rows[receipt.memory_id]["retention_class"] == 4
    assert pipeline.forget(receipt) == 1
    assert not pins.pins


def test_raw_window_ref_without_a_checksum_is_rejected():
    pipeline = make_pipeline(None, PinStore())
    with pytest.raises(ValueError, match="raw_window_checksum is required"):
        pipeline.ingest(window(0, [0, 0], raw_window_ref="ring://pump-17/0-4"))


def test_pin_expiration_is_tracked_and_queryable():
    store, pins = MemoryBackend(), PinStore()
    pipeline = make_pipeline(store, pins)
    base_ms = 1_800_000_000_000
    receipt = pipeline.ingest(window(
        0, [0, 0], raw_window_ref="ring://pump-17/0-4", raw_window_checksum="sha256:deadbeef",
        raw_window_ttl_ms=5_000,
    ))

    expected_expiry = base_ms + 5_000
    assert pins.details[(receipt.raw_window_ref, receipt.idempotency_key)] == ("sha256:deadbeef", expected_expiry)
    assert pipeline.expired_pins(expected_expiry - 1) == ()
    assert pipeline.expired_pins(expected_expiry) == (receipt.memory_id,)
    assert pipeline.forget(receipt) == 1
    assert pipeline.expired_pins(expected_expiry) == ()


def test_pin_without_a_ttl_never_expires():
    store, pins = MemoryBackend(), PinStore()
    pipeline = make_pipeline(store, pins)
    receipt = pipeline.ingest(window(
        0, [0, 0], raw_window_ref="ring://pump-17/0-4", raw_window_checksum="sha256:deadbeef",
    ))
    assert pipeline.expired_pins(10 ** 15) == ()


def test_persistence_retry_does_not_ingest_the_sensor_window_twice():
    store = MemoryBackend()
    store.fail_once = True
    pipeline = make_pipeline(store)

    with pytest.raises(SensorMemoryPersistenceError) as captured:
        pipeline.ingest(window(0, [0, 0]))
    assert len(pipeline.engine.items().ids) == 1
    assert pipeline.engine.items().mass[0] == 1

    receipt = pipeline.persist_pending(
        captured.value.pending, captured.value.decision, step_index=0
    )
    assert receipt.persisted
    assert len(store.rows) == 1
    assert pipeline.engine.items().mass[0] == 1


def test_bad_or_replayed_window_is_rejected_before_engine_state_changes():
    pipeline = make_pipeline()
    with pytest.raises(ValueError, match="retention_class must be an EdgeMemory class"):
        pipeline.ingest(window(0, [0, 0], retention_class=2.0))
    with pytest.raises(ValueError, match="observed_at_ms must be positive"):
        pipeline.ingest(window(0, [0, 0], observed_at_ms=0))
    assert len(pipeline.engine.items().ids) == 0

    pipeline.ingest(window(0, [0, 0]))
    with pytest.raises(ValueError, match="step_index must increase strictly"):
        pipeline.ingest(window(0, [0, 0]))
    assert pipeline.engine.items().mass[0] == 1


def _pressure_pipeline(pins=None):
    config = EngramConfig(B_sem=4, B_epi=2, use_step_detect=False, alarm_threshold=1.0,
                          consolidate_every=10_000)
    engine = Engram(config, 2, np.ones(2), raw_window_bytes=64)
    return SensorStateMemory(engine, MemoryBackend(), raw_evidence=pins)


def _first_item_survives(pipeline, **first_window):
    """Ingest one outlier, then two further distant outliers into a 2-slot episodic bank."""
    first = pipeline.ingest(window(0, [3, 0], **first_window))
    pipeline.ingest(window(1, [100, 0]))
    pipeline.ingest(window(2, [200, 0]))
    return first, first.decision.item_id in set(pipeline.engine.items().ids.tolist())


def test_pinned_raw_evidence_shields_its_slot_from_eviction_pressure():
    _, survives_unpinned = _first_item_survives(_pressure_pipeline())
    assert not survives_unpinned  # control: without a pin, the oldest candidate is the victim

    pins = PinStore()
    pipeline = _pressure_pipeline(pins)
    receipt, survives = _first_item_survives(
        pipeline, raw_window_ref="ring://pump-17/0-4", raw_window_checksum="sha256:deadbeef")
    assert survives
    assert receipt.decision.item_id in pipeline.engine.protected


def test_forgetting_the_pin_makes_the_slot_evictable_again():
    pins = PinStore()
    pipeline = _pressure_pipeline(pins)
    receipt = pipeline.ingest(window(
        0, [3, 0], raw_window_ref="ring://pump-17/0-4", raw_window_checksum="sha256:deadbeef"))
    assert pipeline.forget(receipt) == 1
    assert receipt.decision.item_id not in pipeline.engine.protected
    pipeline.ingest(window(1, [100, 0]))
    pipeline.ingest(window(2, [200, 0]))
    assert receipt.decision.item_id not in set(pipeline.engine.items().ids.tolist())

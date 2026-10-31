from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from uengram import (
    FileSensorEventOutbox,
    SensorMemoryPersistenceError,
    SensorStateMemory,
    SensorWindow,
)
from uengram.engine import Engram, EngramConfig


class MemoryBackend:
    def __init__(self, *, fail_remember: bool = False):
        self.rows = {}
        self.fail_remember = fail_remember

    def remember(self, content, **kwargs):
        if self.fail_remember:
            raise OSError("simulated vendor outage")
        key = kwargs["idempotency_key"]
        for row in self.rows.values():
            if row["idempotency_key"] == key:
                return SimpleNamespace(memory_id=row["memory_id"])
        memory_id = len(self.rows) + 1
        self.rows[memory_id] = {"memory_id": memory_id, "content": content, **kwargs}
        return SimpleNamespace(memory_id=memory_id)

    def recall(self, **_kwargs):
        return SimpleNamespace(items=())

    def forget(self, memory_id):
        return int(self.rows.pop(memory_id, None) is not None)


def make_pipeline(memory, outbox):
    config = EngramConfig(
        B_sem=4,
        B_epi=4,
        m_promote=5,
        n_promote=8,
        use_step_detect=False,
        alarm_threshold=1.0,
        consolidate_every=10_000,
    )
    return SensorStateMemory(
        Engram(config, 2, np.ones(2), raw_window_bytes=64), memory, outbox=outbox
    )


def window(step):
    return SensorWindow(
        asset_id="pump-17",
        subsystem_id="bearing",
        operating_mode="steady",
        step_index=step,
        observed_at_ms=1_800_000_000_000 + step * 1000,
        raw_idx=step,
        features=np.zeros(2),
        feature_schema="bearing-v1",
    )


def test_file_outbox_replays_after_a_vendor_outage_without_reingesting(tmp_path):
    path = tmp_path / "sensor-outbox.json"
    unavailable = MemoryBackend(fail_remember=True)
    pipeline = make_pipeline(unavailable, FileSensorEventOutbox(path))

    with pytest.raises(SensorMemoryPersistenceError) as captured:
        pipeline.ingest(window(0))

    assert len(pipeline.engine.items().ids) == 1
    assert [entry.record.idempotency_key for entry in FileSensorEventOutbox(path).pending()] == [
        captured.value.pending.idempotency_key
    ]

    available = MemoryBackend()
    restarted = make_pipeline(available, FileSensorEventOutbox(path))
    receipts = restarted.replay_outbox()

    assert [receipt.idempotency_key for receipt in receipts] == [
        captured.value.pending.idempotency_key
    ]
    assert len(available.rows) == 1
    assert len(restarted.engine.items().ids) == 0
    assert FileSensorEventOutbox(path).pending() == ()


def test_outbox_replays_idempotently_after_vendor_commit_before_local_ack(tmp_path):
    path = tmp_path / "sensor-outbox.json"

    class AckFailureOutbox(FileSensorEventOutbox):
        def __init__(self, path):
            super().__init__(path)
            self.fail_once = True

        def acknowledge(self, idempotency_key):
            if self.fail_once:
                self.fail_once = False
                raise OSError("simulated local acknowledgement failure")
            super().acknowledge(idempotency_key)

    memory = MemoryBackend()
    pipeline = make_pipeline(memory, AckFailureOutbox(path))

    with pytest.raises(SensorMemoryPersistenceError):
        pipeline.ingest(window(0))

    assert len(memory.rows) == 1
    assert len(FileSensorEventOutbox(path).pending()) == 1

    restarted = make_pipeline(memory, FileSensorEventOutbox(path))
    receipts = restarted.replay_outbox()

    assert [receipt.memory_id for receipt in receipts] == [1]
    assert len(memory.rows) == 1
    assert FileSensorEventOutbox(path).pending() == ()

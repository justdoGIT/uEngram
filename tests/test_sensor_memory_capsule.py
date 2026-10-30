from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from uengram import (
    CapsuleSensorStateMemory,
    SensorMemoryCapsule,
    SensorStateMemory,
    SensorWindow,
)
from uengram.engine import Engram, EngramConfig


class Memory:
    def __init__(self):
        self.rows = {}

    def remember(self, content, **kwargs):
        key = kwargs["idempotency_key"]
        for row in self.rows.values():
            if row["idempotency_key"] == key:
                return SimpleNamespace(memory_id=row["memory_id"])
        memory_id = len(self.rows) + 1
        self.rows[memory_id] = {"memory_id": memory_id, "content": content, **kwargs}
        return SimpleNamespace(memory_id=memory_id)

    def recall(self, **_kwargs):
        return SimpleNamespace(items=tuple(self.rows.values()))

    def forget(self, memory_id):
        return int(self.rows.pop(memory_id, None) is not None)


def config():
    return EngramConfig(B_sem=4, B_epi=4, m_promote=5, n_promote=8,
                        use_step_detect=False, alarm_threshold=1.0, consolidate_every=10_000)


def window(step, values):
    return SensorWindow("pump-17", "bearing", "steady", step, 1_800_000_000_000 + step * 1000,
                        step, np.asarray(values, dtype=float), "bearing-v1")


def test_capsule_restores_engine_and_exposes_only_uncheckpointed_input(tmp_path):
    capsule = SensorMemoryCapsule(tmp_path / "capsule.db")
    engine = Engram(config(), 2, np.ones(2), 64)
    memory = Memory()
    pipeline = SensorStateMemory(engine, memory, outbox=capsule, vector_index=capsule)

    for step, values in enumerate(([0, 0], [10, 0])):
        item = window(step, values)
        capsule.record_window(item)
        receipt = pipeline.ingest(item)
        capsule.checkpoint(engine, last_step=step, last_alarm=receipt.decision.flagged)

    capsule.record_window(window(2, [20, 0]))
    restored = Engram(config(), 2, np.ones(2), 64)
    state = capsule.restore(restored)

    assert state is not None and state.last_step == 1
    np.testing.assert_array_equal(restored.item_id, engine.item_id)
    np.testing.assert_allclose(restored.mu, engine.mu)
    assert [item.step_index for item in capsule.uncheckpointed_windows()] == [2]
    ranked = capsule.rank(np.zeros(2), now_ms=window(1, [0, 0]).observed_at_ms,
                          asset_id="pump-17", subsystem_id="bearing")
    assert ranked and ranked[0].event["memory_id"] == 1
    capsule.request_pin("event-1", "ring://pump/0", "sha256:x", None)
    capsule.mark_pin("event-1", "pinned")
    capsule.close()

def test_capsule_runner_replays_a_wal_suffix_after_interrupted_ingest(tmp_path):
    capsule = SensorMemoryCapsule(tmp_path / "capsule.db")
    interrupted = window(0, [0, 0])
    capsule.record_window(interrupted)  # Simulate the durable write just before process loss.

    runner = CapsuleSensorStateMemory(capsule, Engram(config(), 2, np.ones(2), 64), Memory())
    recovered = runner.recover()

    assert [receipt.step_index for receipt in recovered] == [0]
    assert runner.pipeline.engine.items().mass[0] == 1
    assert capsule.uncheckpointed_windows() == ()
    capsule.close()


from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from uengram import FileSensorVectorIndex, SensorStateMemory, SensorWindow
from uengram.engine import Engram, EngramConfig


class LatestOnlyMemory:
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
        latest = max(self.rows, default=None)
        items = () if latest is None else (self.rows[latest],)
        return SimpleNamespace(items=items)

    def forget(self, memory_id):
        return int(self.rows.pop(memory_id, None) is not None)


def make_pipeline(memory, index):
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
        Engram(config, 2, np.ones(2), raw_window_bytes=64),
        memory,
        vector_index=index,
    )


def window(step, values):
    return SensorWindow(
        asset_id="pump-17",
        subsystem_id="bearing",
        operating_mode="steady",
        step_index=step,
        observed_at_ms=1_800_000_000_000 + step * 1000,
        raw_idx=step,
        features=np.asarray(values, dtype=float),
        feature_schema="bearing-v1",
    )


def test_file_vector_index_recovers_an_older_exact_match_lite_omits(tmp_path):
    memory = LatestOnlyMemory()
    path = tmp_path / "sensor-vectors.json"
    pipeline = make_pipeline(memory, FileSensorVectorIndex(path))
    pipeline.ingest(window(0, [0, 0]))
    pipeline.ingest(window(1, [10, 0]))
    pipeline.ingest(window(2, [20, 0]))

    recall = pipeline.recall(
        np.zeros(2),
        t_now=2,
        now_ms=window(2, [0, 0]).observed_at_ms,
        asset_id="pump-17",
        subsystem_id="bearing",
    )

    assert len(recall.historical_events) == 1  # Lite returns its keyword-limited candidate set.
    assert len(recall.ranked_history) == 3  # The sidecar scans every persisted lifecycle vector.
    assert recall.ranked_history[0].event["memory_id"] == 1

    restarted = FileSensorVectorIndex(path)
    ranked = restarted.rank(
        np.zeros(2),
        now_ms=window(2, [0, 0]).observed_at_ms,
        asset_id="pump-17",
        subsystem_id="bearing",
    )
    assert ranked[0].event["memory_id"] == 1
    assert {result.event["memory_id"] for result in ranked[1:]} == {2, 3}

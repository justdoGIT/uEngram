"""Predeclared A/B/C recovery experiment for the Sensor Memory Capsule.

Run: ``python -m bench.capsule_resilience --out ~/.local/state/uengram/capsule.json``.
The result measures systems recovery only; it is not a fault-detection benchmark.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from uengram import (
    CapsuleSensorStateMemory,
    FileSensorEventOutbox,
    FileSensorVectorIndex,
    SensorMemoryCapsule,
    SensorStateMemory,
    SensorWindow,
)
from uengram.engine import Engram, EngramConfig


class LatestOnlyMemory:
    """Lite-like idempotent event store whose keyword recall returns one row."""

    def __init__(self) -> None:
        self.rows: dict[int, dict] = {}

    def remember(self, content: str, **kwargs: object) -> object:
        key = kwargs["idempotency_key"]
        for row in self.rows.values():
            if row["idempotency_key"] == key:
                return SimpleNamespace(memory_id=row["memory_id"])
        memory_id = len(self.rows) + 1
        self.rows[memory_id] = {"memory_id": memory_id, "content": content, **kwargs}
        return SimpleNamespace(memory_id=memory_id)

    def recall(self, **_kwargs: object) -> object:
        latest = max(self.rows, default=None)
        return SimpleNamespace(items=() if latest is None else (self.rows[latest],))

    def forget(self, memory_id: int) -> int:
        return int(self.rows.pop(memory_id, None) is not None)


def _config() -> EngramConfig:
    return EngramConfig(B_sem=8, B_epi=8, m_promote=5, n_promote=8,
                        use_step_detect=False, alarm_threshold=1.0, consolidate_every=10_000)


def _engine() -> Engram:
    return Engram(_config(), 2, np.ones(2), raw_window_bytes=64)


def _windows() -> tuple[SensorWindow, ...]:
    return tuple(SensorWindow("capsule-asset", "bearing", "steady", step,
                               1_900_000_000_000 + step * 1000, step,
                               np.asarray([float(step) * 10, 0.0]), "capsule-v1")
                 for step in range(4))


def _signature(engine: Engram) -> tuple[bytes, bytes, bytes]:
    return engine.kind.tobytes(), engine.item_id.tobytes(), engine.mu.tobytes()


def _reference(windows: tuple[SensorWindow, ...]) -> tuple[bytes, bytes, bytes]:
    pipeline = SensorStateMemory(_engine(), LatestOnlyMemory())
    for window in windows:
        pipeline.ingest(window)
    return _signature(pipeline.engine)


def _run_lite_or_sidecar(condition: str, windows: tuple[SensorWindow, ...], root: Path) -> dict:
    memory = LatestOnlyMemory()
    kwargs = {}
    if condition == "sidecar":
        kwargs = {"outbox": FileSensorEventOutbox(root / "outbox.json"),
                  "vector_index": FileSensorVectorIndex(root / "vectors.json")}
    pipeline = SensorStateMemory(_engine(), memory, **kwargs)
    for window in windows[:2]:
        pipeline.ingest(window)
    recovery_start = time.perf_counter_ns()
    pipeline = SensorStateMemory(_engine(), memory, **kwargs)  # process restart loses detector state
    for window in windows[2:]:
        pipeline.ingest(window)
    recovery_ns = time.perf_counter_ns() - recovery_start
    ranked = pipeline.recall(np.zeros(2), t_now=3, now_ms=windows[-1].observed_at_ms,
                             asset_id="capsule-asset", subsystem_id="bearing").ranked_history
    return {"engine": pipeline.engine, "memory": memory, "recovery_ns": recovery_ns,
            "ranked_count": len(ranked), "storage_bytes": _files_bytes(root)}


def _run_capsule(windows: tuple[SensorWindow, ...], root: Path, crash: str) -> dict:
    memory = LatestOnlyMemory()
    path = root / "capsule.db"
    runner = CapsuleSensorStateMemory(SensorMemoryCapsule(path), _engine(), memory)
    for window in windows[:2]:
        runner.ingest(window)
    recovery_start = time.perf_counter_ns()
    if crash == "after_wal":
        runner.capsule.record_window(windows[2])
    elif crash == "after_ingest":
        runner.capsule.record_window(windows[2])
        runner.pipeline.ingest(windows[2])
    elif crash == "after_checkpoint":
        runner.ingest(windows[2])
    else:
        raise ValueError("crash must be after_wal, after_ingest, or after_checkpoint")
    runner.capsule.close()
    runner = CapsuleSensorStateMemory(SensorMemoryCapsule(path), _engine(), memory)
    runner.recover()
    runner.ingest(windows[3])
    recovery_ns = time.perf_counter_ns() - recovery_start
    ranked = runner.pipeline.recall(np.zeros(2), t_now=3, now_ms=windows[-1].observed_at_ms,
                                    asset_id="capsule-asset", subsystem_id="bearing").ranked_history
    runner.capsule.close()
    return {"engine": runner.pipeline.engine, "memory": memory, "recovery_ns": recovery_ns,
            "ranked_count": len(ranked), "storage_bytes": _files_bytes(root)}


def run(crash: str = "after_ingest") -> dict:
    """Return deterministic A/B/C systems metrics for one crash boundary."""
    windows, reference = _windows(), _reference(_windows())
    conditions: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="uengram-capsule-") as temporary:
        root = Path(temporary)
        for condition in ("lite", "sidecar", "capsule"):
            result = (_run_capsule(windows, root / condition, crash) if condition == "capsule"
                      else _run_lite_or_sidecar(condition, windows, root / condition))
            conditions[condition] = {
                "state_equivalent_to_uninterrupted": _signature(result["engine"]) == reference,
                "vendor_event_count": len(result["memory"].rows),
                "ranked_history_count": result["ranked_count"],
                "recovery_ns": result["recovery_ns"],
                "local_storage_bytes": result["storage_bytes"],
            }
    return {"protocol": "capsule-resilience-v1", "crash_point": crash,
            "conditions": conditions,
            "interpretation": "systems recovery and retrieval completeness only; not detector accuracy"}


def _files_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.glob("**/*") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crash", choices=("after_wal", "after_ingest", "after_checkpoint"),
                        default="after_ingest")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.crash)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

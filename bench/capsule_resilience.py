"""Predeclared A/B/C recovery experiment for the Sensor Memory Capsule.

Run: ``python -m bench.capsule_resilience --out ~/.local/state/uengram/capsule.json``.
The result measures systems recovery only; it is not a fault-detection benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import socket
import sys
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

    def records(self) -> dict[str, str]:
        """Every stored lifecycle record as {identity: content}; identity is the idempotency key."""
        return {row["idempotency_key"]: row["content"] for row in self.rows.values()}


def _new_memory(vendor: str, path: Path):
    """The vendor store: the in-process stub, or the real EdgeMemory Lite shared library."""
    if vendor == "stub":
        return LatestOnlyMemory()
    if vendor == "real":
        from bench.capsule_real_lite import (
            RealLiteMemory,  # needs the vendor binding and cffi
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        return RealLiteMemory(path)
    raise ValueError("vendor must be stub or real")


def _restart_memory(memory):
    """A vendor restart: persistent stores reopen their storage; the in-process stub just persists."""
    return memory.reopen() if hasattr(memory, "reopen") else memory


def _close_memory(memory) -> None:
    if hasattr(memory, "close"):
        memory.close()


def _config() -> EngramConfig:
    return EngramConfig(B_sem=8, B_epi=8, m_promote=5, n_promote=8,
                        use_step_detect=False, alarm_threshold=1.0, consolidate_every=10_000)


def _engine(dimension: int) -> Engram:
    return Engram(_config(), dimension, np.ones(dimension), raw_window_bytes=64)


def _fixture_windows(window_count: int) -> tuple[SensorWindow, ...]:
    return tuple(SensorWindow("capsule-asset", "bearing", "steady", step,
                              1_900_000_000_000 + step * 1000, step,
                              np.asarray([float(step) * 10, 0.0]), "capsule-v1")
                 for step in range(window_count))


def _smd_windows(machine: str, window_count: int) -> tuple[SensorWindow, ...]:
    from bench.stream_smd import smd_stream

    stream = smd_stream(machine)
    indices = np.flatnonzero(stream.test)[:window_count]
    if len(indices) != window_count:
        raise ValueError(f"{machine} has fewer than {window_count} test windows")
    return tuple(SensorWindow(machine, "server", str(stream.state[index]), step,
                              1_900_000_000_000 + step * 60_000, int(index),
                              np.nan_to_num(stream.features[index], nan=0.0,
                                            posinf=0.0, neginf=0.0),
                              "smd-window-mean-v1", source_event_id=f"{machine}:{index}")
                 for step, index in enumerate(indices))


def _windows(stream: str, machine: str, window_count: int) -> tuple[SensorWindow, ...]:
    if window_count < 4:
        raise ValueError("window_count must be at least four")
    if stream == "fixture":
        return _fixture_windows(window_count)
    if stream == "smd":
        return _smd_windows(machine, window_count)
    raise ValueError("stream must be fixture or smd")


def _input_info(stream: str, machine: str, windows: tuple[SensorWindow, ...]) -> dict:
    digest = hashlib.sha256()
    for window in windows:
        digest.update(np.ascontiguousarray(window.features, dtype=np.float64).tobytes())
    return {"stream": stream, "machine": None if stream == "fixture" else machine,
            "window_count": len(windows), "feature_dimension": len(windows[0].features),
            "sha256": digest.hexdigest()}


def _signature(engine: Engram) -> tuple[bytes, ...]:
    return (engine.mu.tobytes(), engine.S.tobytes(), engine.ckpt_mu.tobytes(), engine.n.tobytes(),
            engine.kind.tobytes(), engine.t_created.tobytes(), engine.t_last.tobytes(),
            engine.surpr.tobytes(), engine.item_id.tobytes(), engine.superseded_by.tobytes(),
            engine.mode_key.tobytes(), engine.ewma_e.tobytes(), engine.ref_mu.tobytes(),
            engine.ref_var.tobytes(), engine.hits.tobytes())


def _reference(windows: tuple[SensorWindow, ...], memory=None) -> tuple[tuple[bytes, ...], object]:
    memory = LatestOnlyMemory() if memory is None else memory
    pipeline = SensorStateMemory(_engine(len(windows[0].features)), memory)
    for window in windows:
        pipeline.ingest(window)
    return _signature(pipeline.engine), memory


def _event_keys(memory) -> frozenset[str]:
    return frozenset(memory.records())


def _alarm_open_keys(memory) -> frozenset[str]:
    return frozenset(key for key, content in memory.records().items() if " alarm_open:" in content)


def _query(pipeline: SensorStateMemory, windows: tuple[SensorWindow, ...]):
    last = windows[-1]
    return pipeline.recall(np.zeros(len(last.features)), t_now=last.step_index,
                           now_ms=last.observed_at_ms, asset_id=last.asset_id,
                           subsystem_id=last.subsystem_id,
                           operating_mode=last.operating_mode).ranked_history


def _run_lite_or_sidecar(condition: str, windows: tuple[SensorWindow, ...], root: Path, cut: int,
                         vendor: str = "stub") -> dict:
    memory = _new_memory(vendor, root.with_name(root.name + "-vendor"))
    kwargs = {}
    if condition == "sidecar":
        kwargs = {"outbox": FileSensorEventOutbox(root / "outbox.json"),
                  "vector_index": FileSensorVectorIndex(root / "vectors.json")}
    pipeline = SensorStateMemory(_engine(len(windows[0].features)), memory, **kwargs)
    for window in windows[:cut]:
        pipeline.ingest(window)
    recovery_start = time.perf_counter_ns()
    memory = _restart_memory(memory)
    pipeline = SensorStateMemory(_engine(len(windows[0].features)), memory, **kwargs)
    for window in windows[cut:]:
        pipeline.ingest(window)
    recovery_ns = time.perf_counter_ns() - recovery_start
    return {"engine": pipeline.engine, "memory": memory, "recovery_ns": recovery_ns,
            "ranked_count": len(_query(pipeline, windows)), "storage_bytes": _files_bytes(root)}


def _run_capsule(windows: tuple[SensorWindow, ...], root: Path, crash: str, cut: int,
                 vendor: str = "stub") -> dict:
    memory = _new_memory(vendor, root.with_name(root.name + "-vendor"))
    path = root / "capsule.db"
    runner = CapsuleSensorStateMemory(SensorMemoryCapsule(path), _engine(len(windows[0].features)), memory)
    for window in windows[:cut]:
        runner.ingest(window)
    recovery_start = time.perf_counter_ns()
    interrupted = windows[cut]
    if crash == "after_wal":
        runner.capsule.record_window(interrupted)
    elif crash == "after_ingest":
        runner.capsule.record_window(interrupted)
        runner.pipeline.ingest(interrupted)
    elif crash == "after_checkpoint":
        runner.ingest(interrupted)
    else:
        raise ValueError("crash must be after_wal, after_ingest, or after_checkpoint")
    runner.capsule.close()
    memory = _restart_memory(memory)
    runner = CapsuleSensorStateMemory(SensorMemoryCapsule(path), _engine(len(windows[0].features)), memory)
    runner.recover()
    for window in windows[cut + 1:]:
        runner.ingest(window)
    recovery_ns = time.perf_counter_ns() - recovery_start
    ranked = _query(runner.pipeline, windows)
    runner.capsule.close()
    return {"engine": runner.pipeline.engine, "memory": memory, "recovery_ns": recovery_ns,
            "ranked_count": len(ranked), "storage_bytes": _files_bytes(root)}


def _proc_field(path: str, key: str) -> str | None:
    """First ``key`` value from a Linux /proc file, or None where unavailable."""
    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith(key):
                return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def storage_of(path: str | os.PathLike[str]) -> dict:
    """The filesystem backing ``path`` (longest matching mount in /proc/mounts), best effort.

    fsync is nearly free on tmpfs, so a timing taken there says nothing about durable-commit cost.
    """
    unknown = {"fstype": None, "source": None, "rotational": None}
    try:
        target = str(Path(path).resolve())
        best = None
        for line in Path("/proc/mounts").read_text().splitlines():
            source, mount, fstype = line.split()[:3]
            mount = mount.replace("\\040", " ")
            if (target == mount or target.startswith(mount.rstrip("/") + "/")) and (
                    best is None or len(mount) > len(best[1])):
                best = (source, mount, fstype)
    except OSError:
        return unknown
    if best is None:
        return unknown
    match = re.fullmatch(r"/dev/(?:(nvme\d+n\d+|mmcblk\d+)p\d+|([a-z]+)\d*)", best[0])
    device = None if match is None else (match.group(1) or match.group(2))
    try:
        rotational = None if device is None else bool(int(Path(f"/sys/block/{device}/queue/rotational")
                                                          .read_text().strip()))
    except (OSError, ValueError):
        rotational = None
    return {"fstype": best[2], "source": best[0], "rotational": rotational}


def host_provenance(target_label: str | None) -> dict:
    """Capture host facts; a label identifies, but does not certify, target hardware."""
    mem_kib = _proc_field("/proc/meminfo", "MemTotal")
    return {"target_label": target_label, "hostname": socket.gethostname(),
            "platform": platform.platform(), "machine": platform.machine(),
            "cpu_model": _proc_field("/proc/cpuinfo", "model name"),
            "memory_mib": None if mem_kib is None else int(mem_kib.split()[0]) // 1024,
            "storage": storage_of(tempfile.gettempdir()),
            "python": sys.version.split()[0], "numpy": np.__version__, "cpu_count": os.cpu_count()}


def evaluate(windows: tuple[SensorWindow, ...], crash: str, restart_at: int | None = None,
             vendor: str = "stub") -> dict:
    """Run the three arms over ``windows``, restarting at index ``restart_at`` (default midpoint).

    Returns ``conditions`` (the v3 artifact fields), the ``reference`` record count, and
    ``alarm_open`` counts, which are kept out of the v3 artifact schema. ``vendor`` selects the
    event store: ``stub`` (in-process) or ``real`` (the EdgeMemory Lite shared library).
    """
    cut = len(windows) // 2 if restart_at is None else restart_at
    if not 1 <= cut <= len(windows) - 1:
        raise ValueError("restart_at must be between 1 and len(windows) - 1")
    conditions: dict[str, dict] = {}
    missing_alarm: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="uengram-capsule-") as temporary:
        root = Path(temporary)
        reference, reference_memory = _reference(windows, _new_memory(vendor, root / "reference-vendor"))
        reference_keys = _event_keys(reference_memory)
        reference_alarms = _alarm_open_keys(reference_memory)
        _close_memory(reference_memory)
        for condition in ("lite", "sidecar", "capsule"):
            result = (_run_capsule(windows, root / condition, crash, cut, vendor) if condition == "capsule"
                      else _run_lite_or_sidecar(condition, windows, root / condition, cut, vendor))
            keys = _event_keys(result["memory"])
            missing_alarm[condition] = len(reference_alarms - _alarm_open_keys(result["memory"]))
            _close_memory(result["memory"])
            conditions[condition] = {
                "state_equivalent_to_uninterrupted": _signature(result["engine"]) == reference,
                "event_keys_match_reference": keys == reference_keys,
                "missing_event_count": len(reference_keys - keys),
                "extra_event_count": len(keys - reference_keys),
                "vendor_event_count": len(keys),
                "ranked_history_count": result["ranked_count"],
                "recovery_ns": result["recovery_ns"],
                "local_storage_bytes": result["storage_bytes"],
            }
    return {"conditions": conditions, "reference_event_count": len(reference_keys),
            "reference_alarm_open_count": len(reference_alarms), "missing_alarm_open": missing_alarm}


def run(
    crash: str = "after_ingest", *, stream: str = "fixture", machine: str = "machine-1-1",
    window_count: int = 4, target_label: str | None = None, require_target_label: bool = False,
    vendor: str = "stub",
) -> dict:
    """Return A/B/C recovery metrics for an exact fixture or SMD replay prefix."""
    if require_target_label and not target_label:
        raise ValueError("target label is required for a target-hardware result")
    windows = _windows(stream, machine, window_count)
    outcome = evaluate(windows, crash, vendor=vendor)
    return {"protocol": "capsule-resilience-v3", "crash_point": crash, "vendor": vendor,
            "input": _input_info(stream, machine, windows), "host": host_provenance(target_label),
            "reference": {"vendor_event_count": outcome["reference_event_count"]},
            "conditions": outcome["conditions"],
            "interpretation": "systems recovery and retrieval completeness only; not detector accuracy"}


def _files_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.glob("**/*") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crash", choices=("after_wal", "after_ingest", "after_checkpoint"),
                        default="after_ingest")
    parser.add_argument("--stream", choices=("fixture", "smd"), default="fixture")
    parser.add_argument("--machine", default="machine-1-1", help="SMD machine identifier")
    parser.add_argument("--windows", type=int, default=4, help="replay-prefix length, at least four")
    parser.add_argument("--vendor", choices=("stub", "real"), default="stub",
                        help="event store: in-process stub, or the real EdgeMemory Lite library")
    parser.add_argument("--target-label", help="explicit target hardware identifier")
    parser.add_argument("--require-target-label", action="store_true",
                        help="refuse to create an unnamed target-hardware artifact")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.crash, stream=args.stream, machine=args.machine, window_count=args.windows,
                 target_label=args.target_label, require_target_label=args.require_target_label,
                 vendor=args.vendor)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

"""Cost of durability: checkpoint cadence and SQLite sync mode against a memory-only run.

python -m bench.capsule_cost_study --label msi --out results/capsule_cost/msi.json

Per (synchronous, checkpoint_every) configuration it measures per-window ingest latency, the
on-disk size, and the worst-case recovery (a crash leaving ``checkpoint_every - 1`` consumed
windows beyond the last checkpoint, then open + restore + replay). Repeats are interleaved
across configurations so slow drift in the disk does not favour one of them. Latency is
dominated by fsync cost, so it is storage-specific; run it on each target. The vendor store is
the in-process stub, isolating the capsule's own cost. SMD labels are never read.

A relaxed setting is only claimed safe for the process-death model tested elsewhere (SIGKILL):
SQLite ``synchronous=NORMAL`` weakens durability against power loss, which this study cannot test.
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from bench import capsule_resilience as cr
from uengram import CapsuleSensorStateMemory, SensorMemoryCapsule, SensorStateMemory

SYNCS = ("FULL", "NORMAL")
CADENCES = (1, 4, 16, 64)


def _stats(latencies_ns: list[int]) -> dict:
    ms = np.asarray(latencies_ns, dtype=float) / 1e6
    return {"mean_ms": float(ms.mean()), "p50_ms": float(np.percentile(ms, 50)),
            "p99_ms": float(np.percentile(ms, 99)), "total_ms": float(ms.sum())}


def memory_only(windows) -> dict:
    pipeline = SensorStateMemory(cr._engine(len(windows[0].features)), cr.LatestOnlyMemory())
    latencies = []
    for window in windows:
        start = time.perf_counter_ns()
        pipeline.ingest(window)
        latencies.append(time.perf_counter_ns() - start)
    return _stats(latencies)


def worst_case_replay_count(window_count: int, cadence: int) -> int:
    """Largest prefix length that leaves exactly ``cadence - 1`` windows past the last checkpoint."""
    return ((window_count - (cadence - 1)) // cadence) * cadence + (cadence - 1)


def capsule(windows, sync: str, cadence: int) -> dict:
    dimension = len(windows[0].features)
    with tempfile.TemporaryDirectory(prefix="uengram-cost-") as tmp:
        root = Path(tmp)
        memory = cr.LatestOnlyMemory()
        store = SensorMemoryCapsule(root / "capsule.db", synchronous=sync)
        runner = CapsuleSensorStateMemory(store, cr._engine(dimension), memory, checkpoint_every=cadence)
        latencies = []
        for window in windows:
            start = time.perf_counter_ns()
            runner.ingest(window)
            latencies.append(time.perf_counter_ns() - start)
        store.close()
        size = cr._files_bytes(root)
        # Worst-case recovery on a fresh store: crash with cadence - 1 windows past the last checkpoint.
        crash_root = root / "crash"
        crash_root.mkdir()
        count = worst_case_replay_count(len(windows), cadence)
        crashed = SensorMemoryCapsule(crash_root / "capsule.db", synchronous=sync)
        victim = CapsuleSensorStateMemory(crashed, cr._engine(dimension), cr.LatestOnlyMemory(),
                                          checkpoint_every=cadence)
        for window in windows[:count]:
            victim.ingest(window)
        crashed.close()
        start = time.perf_counter_ns()
        reopened = SensorMemoryCapsule(crash_root / "capsule.db", synchronous=sync)
        survivor = CapsuleSensorStateMemory(reopened, cr._engine(dimension), cr.LatestOnlyMemory(),
                                            checkpoint_every=cadence)
        replayed = len(survivor.recover())
        recovery_ms = (time.perf_counter_ns() - start) / 1e6
        reopened.close()
    return {**_stats(latencies), "disk_bytes": size, "recovery_ms": recovery_ms,
            "replayed_windows": replayed, "max_replay_windows": cadence - 1}


def _median(runs: list[dict], key: str) -> float:
    return statistics.median(run[key] for run in runs)


def study(windows, repeats: int) -> dict:
    """Interleaved repeats; a discarded warm-up pass precedes the measured ones."""
    configs = [(sync, cadence) for sync in SYNCS for cadence in CADENCES]
    memory_only(windows)
    for sync, cadence in configs:
        capsule(windows, sync, cadence)
    baseline: list[dict] = []
    measured: dict[tuple[str, int], list[dict]] = {config: [] for config in configs}
    for _ in range(repeats):
        baseline.append(memory_only(windows))
        for sync, cadence in configs:
            measured[(sync, cadence)].append(capsule(windows, sync, cadence))
    rows = []
    for (sync, cadence), runs in measured.items():
        if any(run["replayed_windows"] != run["max_replay_windows"] for run in runs):
            raise RuntimeError(f"{sync}/{cadence}: worst-case replay did not replay cadence - 1 windows")
        rows.append({"synchronous": sync, "checkpoint_every": cadence,
                     **{key: _median(runs, key) for key in ("mean_ms", "p50_ms", "p99_ms", "total_ms",
                                                            "disk_bytes", "recovery_ms")},
                     "max_replay_windows": cadence - 1})
    return {"memory_only": {key: _median(baseline, key) for key in ("mean_ms", "p50_ms", "p99_ms", "total_ms")},
            "configs": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", default="machine-1-1")
    parser.add_argument("--windows", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--label", required=True, help="target label for the host that runs this")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    windows = cr._smd_windows(args.machine, args.windows)
    doc = {"protocol": "capsule-cost-study-v1", "input": cr._input_info("smd", args.machine, windows),
           "host": cr.host_provenance(args.label), "repeats": args.repeats, **study(windows, args.repeats),
           "interpretation": "per-window ingest cost and worst-case replay; storage-specific; "
                             "NORMAL's power-loss durability is not tested here"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    base = doc["memory_only"]["mean_ms"]
    print(f"{args.label}: memory-only {base:.3f} ms/window")
    for row in doc["configs"]:
        print(f"  {row['synchronous']:<6} every {row['checkpoint_every']:>2}: {row['mean_ms']:.3f} ms/window "
              f"(p99 {row['p99_ms']:.2f}), recovery {row['recovery_ms']:.1f} ms, "
              f"{row['disk_bytes']} B", flush=True)


if __name__ == "__main__":
    main()

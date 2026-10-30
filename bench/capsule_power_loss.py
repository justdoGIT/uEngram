"""Power-loss crash-consistency test for the Sensor Memory Capsule.

python -m bench.capsule_power_loss --out results/capsule_power_loss.json

SIGKILL (``bench.capsule_kill_test``) proves recovery after a killed *process*, but the OS page
cache survives a kill and the kernel typically still flushes it to the real device in the
background, so a SIGKILL test never actually attacks the window between a write and its fsync.
This tool does: it records every file operation a real ingest run performs under an LD_PRELOAD
shim (``bench/crash_shim.c``), then replays that recorded log (``bench.capsule_crash_model``) to
build on-disk images a power cut could legally leave at any point -- unsynced writes partially
applied, a deletion's directory entry never made durable -- and points the real capsule code at
each one. See ``capsule_crash_model`` for the exact model and its stated limits (whole-syscall
writes, no sub-write tearing, no out-of-order block-device completion).

Two checks per crash image, against an uninterrupted-replay reference computed for every prefix
length of the window set:

  capsule_consistent: opening and recovering the capsule succeeds (no exception, SQLite
    ``integrity_check`` passes) and the recovered engine state exactly equals *some* prefix-length
    uninterrupted replay -- not merely "close", and not trusted from the recovered last_step alone.
  vendor_consistent: after draining the outbox, the vendor store's lifecycle records exactly equal
    that same prefix's uninterrupted record set. The capsule and the vendor store are not one
    transaction (documented limitation); this measures how often a crash actually splits them.

SMD labels are never read. This is a systems-recovery result, not a detection-accuracy one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from bench import capsule_crash_model as cm
from bench import capsule_resilience as cr
from bench.capsule_kill_test import SqliteLiteMemory, save_windows
from uengram import CapsuleSensorStateMemory, SensorMemoryCapsule

SHIM_SOURCE = Path(__file__).with_name("crash_shim.c")


def compiled_shim(cache_dir: Path) -> Path:
    """Build (or reuse) the LD_PRELOAD shim, keyed on its source hash."""
    digest = hashlib.sha256(SHIM_SOURCE.read_bytes()).hexdigest()[:16]
    so_path = cache_dir / f"libcrashshim-{digest}.so"
    if not so_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["gcc", "-shared", "-fPIC", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-o", str(so_path), str(SHIM_SOURCE), "-ldl", "-lpthread"], check=True)
    return so_path


def record_trace(shim: Path, root: Path, windows_path: Path, synchronous: str,
                 checkpoint_every: int, control: str = "none") -> Path:
    """Run one ingest to completion under the shim; return the recorded op-log path."""
    log_path = root / "trace.log"
    env = {**os.environ, "LD_PRELOAD": str(shim), "UENGRAM_CC_ROOT": str(root),
           "UENGRAM_CC_LOG": str(log_path)}
    proc = subprocess.run([sys.executable, "-m", "bench.capsule_power_loss", "record",
                           str(root), str(windows_path), synchronous, str(checkpoint_every),
                           control],
                          env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or "DONE" not in proc.stdout:
        raise RuntimeError(f"trace recording failed: {proc.stderr[-2000:]}")
    return log_path


def _record_child(root: Path, windows_path: Path, synchronous: str, checkpoint_every: int,
                  control: str) -> None:
    import ctypes

    from bench.capsule_kill_test import load_windows

    mark = ctypes.CDLL(None).uengram_cc_mark  # exported by the preloaded shim
    windows = load_windows(windows_path)
    capsule = SensorMemoryCapsule(root / "capsule.db", synchronous=synchronous)
    if control != "none":  # negative controls only: the capsule API refuses these settings
        capsule.db.execute("PRAGMA synchronous=OFF")
    if control == "no-journal":
        capsule.db.execute("PRAGMA journal_mode=OFF")
    vendor = SqliteLiteMemory(root / "vendor.db")
    runner = CapsuleSensorStateMemory(capsule, cr._engine(len(windows[0].features)), vendor,
                                      checkpoint_every=checkpoint_every)
    for n, window in enumerate(windows, 1):
        runner.ingest(window)
        mark(f"ingested {n}".encode())  # the ingest returned: this window is acknowledged
    capsule.close()
    print("DONE", flush=True)


def _reference_table(windows: tuple) -> tuple[list, list]:
    """Engine signature and vendor key set after every prefix length 0..len(windows)."""
    memory = cr.LatestOnlyMemory()
    pipeline_engine = cr._engine(len(windows[0].features))
    from uengram import SensorStateMemory

    pipeline = SensorStateMemory(pipeline_engine, memory)
    signatures, keysets = [cr._signature(pipeline_engine)], [frozenset()]
    for window in windows:
        pipeline.ingest(window)
        signatures.append(cr._signature(pipeline.engine))
        keysets.append(cr._event_keys(memory))
    return signatures, keysets


def _crash_boundaries(ops: list, interesting_paths: tuple[str, ...], spread: int,
                      cap: int | None = None) -> list[int]:
    """Evenly spaced indices across the trace, plus every index right after a sync of interest.

    ``cap`` thins the result to at most that many indices by even subsampling, so long traces stay
    tractable; with no cap every post-sync index is kept.
    """
    n = len(ops)
    even = {round(i * n / spread) for i in range(1, spread)}
    after_sync = {i + 1 for i, op in enumerate(ops)
                  if op.kind in (cm.OP_SYNC, cm.OP_DIRSYNC) and
                  any(op.path.endswith(p) or op.path == "" for p in interesting_paths)}
    chosen = sorted((even | after_sync) & set(range(1, n)))
    if cap is not None and len(chosen) > cap:
        chosen = sorted({chosen[i * len(chosen) // cap] for i in range(cap)})
    return chosen


def _evaluate_image(image: dict, root: Path, crash_dir: Path, windows: tuple, synchronous: str,
                    checkpoint_every: int, signatures: list, keysets: list, acknowledged: int) -> dict:
    cm.materialize(image, root, crash_dir)
    try:
        engine = cr._engine(len(windows[0].features))
        capsule = SensorMemoryCapsule(crash_dir / "capsule.db", synchronous=synchronous)
        vendor = SqliteLiteMemory(crash_dir / "vendor.db")
        runner = CapsuleSensorStateMemory(capsule, engine, vendor, checkpoint_every=checkpoint_every)
        runner.recover()
        runner.pipeline.replay_outbox()
        integrity = capsule.db.execute("PRAGMA integrity_check").fetchone()[0]
        actual_sig = cr._signature(engine)
        matches = [k for k, sig in enumerate(signatures) if sig == actual_sig]
        k = matches[-1] if matches else None
        capsule_ok = integrity == "ok" and k is not None
        at_recovery = frozenset(vendor.records())
        vendor_ok = capsule_ok and at_recovery == keysets[k]
        split = None
        if capsule_ok and not vendor_ok:
            split = ("vendor_ahead" if keysets[k] <= at_recovery
                     else "vendor_behind" if at_recovery <= keysets[k] else "vendor_diverged")
        continued = final_vendor_ok = False
        if capsule_ok:
            for window in windows[k:]:
                runner.ingest(window)
            continued = cr._signature(engine) == signatures[-1]
            final_vendor_ok = frozenset(vendor.records()) == keysets[-1]
        capsule.close()
        return {"capsule_consistent": capsule_ok, "vendor_consistent": vendor_ok, "vendor_split": split,
                "effective_prefix": k, "integrity": integrity, "continues_to_completion": continued,
                "final_vendor_consistent": final_vendor_ok, "acknowledged": acknowledged,
                "acknowledged_lost": max(0, acknowledged - k) if capsule_ok else None, "error": None}
    except Exception as exc:  # the crash image itself is adversarial input under test
        return {"capsule_consistent": False, "vendor_consistent": False, "vendor_split": None,
                "effective_prefix": None, "integrity": None, "continues_to_completion": False,
                "final_vendor_consistent": False, "acknowledged": acknowledged,
                "acknowledged_lost": None, "error": f"{type(exc).__name__}: {exc}"}


def run(machine: str, window_count: int, synchronous: str, checkpoint_every: int, *,
        spread: int, combos_per_point: int, cache_dir: Path, control: str = "none",
        max_boundaries: int | None = None) -> dict:
    windows = cr._smd_windows(machine, window_count)
    signatures, keysets = _reference_table(windows)
    shim = compiled_shim(cache_dir)
    results = []
    with tempfile.TemporaryDirectory(prefix="uengram-powerloss-") as tmp:
        root = Path(tmp) / "root"
        root.mkdir()
        windows_path = Path(tmp) / "windows.npz"
        save_windows(windows_path, windows)
        log_path = record_trace(shim, root, windows_path, synchronous, checkpoint_every, control)
        ops = cm.read_log(log_path)
        boundaries = _crash_boundaries(ops, ("capsule.db", "vendor.db"), spread, max_boundaries)
        for index in boundaries:
            images = cm.crash_images(ops, index, max_combos=combos_per_point)
            for n, image in enumerate(images):
                with tempfile.TemporaryDirectory(prefix="uengram-powerloss-image-") as crash_tmp:
                    outcome = _evaluate_image(image, root, Path(crash_tmp), windows, synchronous,
                                              checkpoint_every, signatures, keysets,
                                              cm.acknowledged_count(ops, index))
                results.append({"crash_index": index, "image": n, "total_ops": len(ops),
                                "window_count": len(windows), **outcome})
    return {
        "protocol": "capsule-power-loss-v2", "machine": machine, "window_count": window_count,
        "synchronous": synchronous if control == "none" else f"{control} (negative control)",
        "control": control, "checkpoint_every": checkpoint_every,
        "host": cr.host_provenance(None),
        "summary": summarize(results),
        "results": results,
        "limits": "whole-syscall-write granularity, no sub-write tearing, no block-device "
                  "reordering below the page cache; fsync credits a file's own create; "
                  "SMD labels never read",
    }


def summarize(results: list[dict]) -> dict:
    n = len(results)
    corrupt = [r for r in results if not r["capsule_consistent"]]
    return {
        "trials": n,
        "capsule_consistent": sum(r["capsule_consistent"] for r in results),
        "vendor_consistent_at_recovery": sum(r["vendor_consistent"] for r in results),
        "vendor_split_at_recovery": _histogram([r["vendor_split"] for r in results]),
        "final_vendor_consistent": sum(r["final_vendor_consistent"] for r in results),
        "capsule_inconsistent": len(corrupt),
        "capsule_inconsistent_examples": [
            {"crash_index": r["crash_index"], "image": r["image"], "error": r["error"]}
            for r in corrupt[:5]],
        "continues_to_completion_of_consistent": sum(
            r["continues_to_completion"] for r in results if r["capsule_consistent"]),
        "images_losing_acknowledged_windows": sum(
            1 for r in results if r["acknowledged_lost"]),
        "acknowledged_windows_lost_max": max(
            (r["acknowledged_lost"] for r in results if r["acknowledged_lost"] is not None), default=0),
        "acknowledged_windows_lost_distribution": _histogram(
            [r["acknowledged_lost"] for r in results if r["acknowledged_lost"] is not None]),
    }


def _histogram(values: list) -> dict:
    out: dict[str, int] = {}
    for v in values:
        if v is None:
            continue
        out[str(v)] = out.get(str(v), 0) + 1
    return out


def main() -> None:
    argv = sys.argv[1:]
    if argv[:1] == ["record"]:
        _record_child(Path(argv[1]), Path(argv[2]), argv[3], int(argv[4]), argv[5])
        return
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--machine", default="machine-1-1")
    parser.add_argument("--windows", type=int, default=48)
    parser.add_argument("--sync", choices=("FULL", "NORMAL"), default="FULL")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--spread", type=int, default=40, help="evenly spaced crash points")
    parser.add_argument("--combos-per-point", type=int, default=6)
    parser.add_argument("--max-boundaries", type=int, default=200,
                        help="cap on crash points (evenly thinned); keeps long traces tractable")
    parser.add_argument("--control", choices=("none", "sync-off", "no-journal"), default="none",
                        help="negative control: sync-off disables fsync; no-journal also disables "
                             "the journal, so a torn multi-page update can corrupt the database")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(os.path.expanduser("~/.cache/uengram-crash-shim")))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.machine, args.windows, args.sync, args.checkpoint_every, spread=args.spread,
                 combos_per_point=args.combos_per_point, cache_dir=args.cache_dir,
                 control=args.control, max_boundaries=args.max_boundaries)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

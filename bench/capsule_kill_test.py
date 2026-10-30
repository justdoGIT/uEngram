"""Real process-death test: SIGKILL a capsule-backed ingest loop, restart, and compare.

python -m bench.capsule_kill_test --trials 150 --out results/capsule_kill_test.json
python -m bench.capsule_kill_test --vendor real ...   # the real EdgeMemory Lite shared library

A child process ingests an SMD replay prefix through ``CapsuleSensorStateMemory`` against a
durable vendor store. The parent SIGKILLs it at a random moment, starts a second child that
recovers, drains the outbox and finishes the stream from the recovered cursor, then checks
(1) vendor lifecycle records equal an uninterrupted run, (2) no duplicate vendor rows,
(3) the checkpointed engine arrays equal an uninterrupted run, (4) SQLite integrity.

SIGKILL stops the process but the operating system still flushes its page cache, so this does
NOT test power loss or torn writes. SMD labels are never read.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bench import capsule_resilience as cr
from uengram import CapsuleSensorStateMemory, SensorMemoryCapsule, SensorWindow

PROTOCOL = "capsule-kill-test-v2"
PAUSE_ENV = "UENGRAM_VENDOR_POST_COMMIT_MS"


class SqliteLiteMemory:
    """Durable, idempotent vendor stand-in that survives a killed process."""

    def __init__(self, path: Path) -> None:
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS memories(id INTEGER PRIMARY KEY AUTOINCREMENT, "
                        "key TEXT UNIQUE NOT NULL, payload TEXT NOT NULL)")
        self._pause_s = float(os.environ.get(PAUSE_ENV, "0")) / 1000

    def remember(self, content: str, **kwargs: object) -> object:
        key = str(kwargs["idempotency_key"])
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO memories(key, payload) VALUES (?, ?)",
                            (key, json.dumps({"content": content, **kwargs})))
        if self._pause_s:  # fault injection: widen "vendor committed, checkpoint still pending"
            time.sleep(self._pause_s)
        (memory_id,) = self.db.execute("SELECT id FROM memories WHERE key=?", (key,)).fetchone()
        return SimpleNamespace(memory_id=memory_id)

    def recall(self, **_kwargs: object) -> object:
        row = self.db.execute("SELECT id, payload FROM memories ORDER BY id DESC LIMIT 1").fetchone()
        return SimpleNamespace(items=() if row is None else ({"memory_id": row[0], **json.loads(row[1])},))

    def forget(self, memory_id: int) -> int:
        with self.db:
            return self.db.execute("DELETE FROM memories WHERE id=?", (memory_id,)).rowcount

    def records(self) -> dict[str, str]:
        return {key: json.loads(payload)["content"]
                for key, payload in self.db.execute("SELECT key, payload FROM memories")}

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def close(self) -> None:
        self.db.close()


def open_vendor(vendor: str, directory: Path):
    """The durable vendor store inside a trial directory."""
    if vendor == "stub":
        return SqliteLiteMemory(directory / "vendor.db")
    from bench.capsule_real_lite import RealLiteMemory

    return RealLiteMemory(directory / "vendor-real")


def vendor_count(vendor: str, store) -> int:
    return store.count() if vendor == "stub" else store.memory.get_usage().scope_memory_count


def vendor_identities(vendor: str, store) -> frozenset[str]:
    """Idempotency keys for the stub; record contents for the real store, which hides its keys."""
    return frozenset(store.records())


def expected_identities(vendor: str, reference_memory) -> frozenset[str]:
    records = reference_memory.records()
    return frozenset(records) if vendor == "stub" else frozenset(records.values())


def peek_records(vendor: str, directory: Path) -> dict[str, str]:
    """Vendor records at the moment of death, read without letting a store recover in place."""
    if vendor == "stub":
        store = SqliteLiteMemory(directory / "vendor.db")
    else:
        scratch = directory / "peek"
        scratch.mkdir()
        for stored in directory.glob("vendor-real.*"):  # the store is a .dbs file plus a .log
            shutil.copy2(stored, scratch / stored.name)
        from bench.capsule_real_lite import RealLiteMemory

        store = RealLiteMemory(scratch / "vendor-real")
    try:
        return store.records()
    finally:
        store.close()


def save_windows(path: Path, windows: tuple[SensorWindow, ...]) -> None:
    np.savez(path, features=np.stack([w.features for w in windows]),
             raw_idx=np.array([w.raw_idx for w in windows]), machine=windows[0].asset_id)


def load_windows(path: Path) -> tuple[SensorWindow, ...]:
    with np.load(path) as data:
        machine, features, raw = str(data["machine"]), data["features"], data["raw_idx"]
    return tuple(SensorWindow(machine, "server", "0", step, 1_900_000_000_000 + step * 60_000,
                              int(raw[step]), features[step], "smd-window-mean-v1",
                              source_event_id=f"{machine}:{int(raw[step])}")
                 for step in range(len(features)))


SYNC_ENV, CADENCE_ENV = "UENGRAM_CAPSULE_SYNC", "UENGRAM_CAPSULE_CKPT_EVERY"


def _open(directory: Path, dimension: int, vendor: str):
    """Open the capsule and vendor with the configuration the parent exported to every process."""
    store = open_vendor(vendor, directory)
    capsule = SensorMemoryCapsule(directory / "capsule.db", synchronous=os.environ.get(SYNC_ENV, "FULL"))
    runner = CapsuleSensorStateMemory(capsule, cr._engine(dimension), store,
                                      checkpoint_every=int(os.environ.get(CADENCE_ENV, "1")))
    return runner, store


def child(mode: str, directory: Path, windows_path: Path, vendor: str) -> None:
    windows = load_windows(windows_path)
    runner, _ = _open(directory, len(windows[0].features), vendor)
    if mode == "finish":
        runner.recover()
        runner.pipeline.replay_outbox()
    print("READY", flush=True)
    for window in windows[runner.pipeline.last_step + 1:]:
        runner.ingest(window)
    print("DONE", flush=True)


def _spawn(mode: str, directory: Path, windows_path: Path, vendor: str,
           pause_ms: float = 0.0) -> subprocess.Popen:
    env = {**os.environ, PAUSE_ENV: str(pause_ms)}
    return subprocess.Popen([sys.executable, "-m", "bench.capsule_kill_test", "child", mode,
                             str(directory), str(windows_path), vendor],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)


def uninterrupted_seconds(windows_path: Path, vendor: str, pause_ms: float = 0.0) -> float:
    """Wall time of the ingest loop alone, used to range the random kill delay."""
    with tempfile.TemporaryDirectory(prefix="uengram-kill-cal-") as tmp:
        proc = _spawn("run", Path(tmp), windows_path, vendor, pause_ms)
        assert proc.stdout.readline().strip() == "READY", proc.stderr.read()
        start = time.perf_counter()
        proc.wait()
        return time.perf_counter() - start


def _event_step(content: str) -> int:
    return int(content.split("step=")[1].split(";")[0])


def trial(windows: tuple[SensorWindow, ...], windows_path: Path, reference: tuple, delay_s: float,
          vendor: str, pause_ms: float = 0.0) -> dict:
    expected_signature, expected = reference
    with tempfile.TemporaryDirectory(prefix="uengram-kill-") as tmp:
        directory = Path(tmp)
        victim = _spawn("run", directory, windows_path, vendor, pause_ms)
        assert victim.stdout.readline().strip() == "READY", victim.stderr.read()
        time.sleep(delay_s)
        finished = victim.poll() is not None
        victim.kill()  # SIGKILL
        victim.wait()
        peek = sqlite3.connect(directory / "capsule.db")
        checkpoint = peek.execute("SELECT last_step FROM checkpoint WHERE id=1").fetchone()
        logged = peek.execute("SELECT COUNT(*) FROM input_wal").fetchone()[0]
        peek.close()
        vendor_steps = [_event_step(c) for c in peek_records(vendor, directory).values()]
        survivor = _spawn("finish", directory, windows_path, vendor)
        out, err = survivor.communicate(timeout=120)
        result = {"delay_ms": round(delay_s * 1000, 3), "finished_before_kill": finished,
                  "checkpoint_step_at_kill": -1 if checkpoint is None else checkpoint[0],
                  "logged_windows_at_kill": logged,
                  "vendor_event_step_at_kill": max(vendor_steps, default=-1),
                  "recovery_exit_code": survivor.returncode, "recovery_ok": "DONE" in out}
        if survivor.returncode != 0:
            result["stderr_tail"] = err.strip().splitlines()[-3:]
            return {**result, "keys_match": False, "no_duplicates": False,
                    "state_equivalent": False, "integrity_ok": False}
        runner, store = _open(directory, len(windows[0].features), vendor)
        runner.recover()  # verify by recovering from the final on-disk state, replaying any log suffix
        runner.pipeline.replay_outbox()
        identities, count = vendor_identities(vendor, store), vendor_count(vendor, store)
        integrity = runner.capsule.db.execute("PRAGMA integrity_check").fetchone()[0]
        equivalent = (cr._signature(runner.pipeline.engine) == expected_signature
                      and runner.pipeline.last_step == windows[-1].step_index)
        store.close()
        runner.capsule.close()
        return {**result, "keys_match": identities == expected, "no_duplicates": count == len(identities),
                "state_equivalent": equivalent, "integrity_ok": integrity == "ok"}


def summarize(trials: list[dict]) -> dict:
    return {"trials": len(trials),
            "killed_mid_stream": sum(not t["finished_before_kill"] for t in trials),
            "killed_with_logged_but_uncheckpointed_input": sum(
                t["logged_windows_at_kill"] > t["checkpoint_step_at_kill"] + 1 for t in trials),
            "killed_before_first_window": sum(t["logged_windows_at_kill"] == 0 for t in trials),
            "killed_with_vendor_ahead_of_checkpoint": sum(
                t["vendor_event_step_at_kill"] > t["checkpoint_step_at_kill"] for t in trials),
            "recovery_failed": sum(not t["recovery_ok"] for t in trials),
            "keys_match": sum(t["keys_match"] for t in trials),
            "no_duplicates": sum(t["no_duplicates"] for t in trials),
            "state_equivalent": sum(t["state_equivalent"] for t in trials),
            "integrity_ok": sum(t["integrity_ok"] for t in trials)}


def main() -> None:
    argv = sys.argv[1:]
    if argv[:1] == ["child"]:
        child(argv[1], Path(argv[2]), Path(argv[3]), argv[4])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machines", default="machine-1-1,machine-2-1,machine-3-1")
    parser.add_argument("--windows", type=int, default=128)
    parser.add_argument("--trials", type=int, default=150, help="total, split evenly over machines")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vendor", choices=("stub", "real"), default="stub",
                        help="durable vendor store: SQLite stand-in, or the real EdgeMemory Lite library")
    parser.add_argument("--vendor-post-commit-ms", type=float, default=0.0,
                        help="fault injection: pause the victim's vendor store after each commit")
    parser.add_argument("--sync", choices=("FULL", "NORMAL"), default="FULL",
                        help="capsule SQLite synchronous mode")
    parser.add_argument("--checkpoint-every", type=int, default=1,
                        help="checkpoint after every N ingested windows")
    parser.add_argument("--out", type=Path, default=Path("results/capsule_kill_test.json"))
    args = parser.parse_args()
    os.environ[SYNC_ENV], os.environ[CADENCE_ENV] = args.sync, str(args.checkpoint_every)  # children inherit
    rng, machines, trials = random.Random(args.seed), args.machines.split(","), []
    per_machine = max(1, args.trials // len(machines))
    for machine in machines:
        windows = cr._smd_windows(machine, args.windows)
        reference_signature, reference_memory = cr._reference(windows)
        reference = (reference_signature, expected_identities(args.vendor, reference_memory))
        with tempfile.TemporaryDirectory(prefix="uengram-kill-windows-") as tmp:
            windows_path = Path(tmp) / "windows.npz"
            save_windows(windows_path, windows)
            loop = uninterrupted_seconds(windows_path, args.vendor, args.vendor_post_commit_ms)
            print(f"{machine}: uninterrupted loop {loop * 1000:.0f} ms, reference "
                  f"{len(reference[1])} records", flush=True)
            for n in range(per_machine):
                result = {"machine": machine, **trial(windows, windows_path, reference,
                                                      rng.uniform(0, loop * 1.1), args.vendor,
                                                      args.vendor_post_commit_ms)}
                trials.append(result)
                if (n + 1) % 10 == 0:
                    print(f"  {machine} {n + 1}/{per_machine} {summarize(trials)}", flush=True)
    doc = {"protocol": PROTOCOL, "params": {"machines": machines, "windows": args.windows,
                                            "trials": len(trials), "seed": args.seed,
                                            "vendor": args.vendor, "sync": args.sync,
                                            "checkpoint_every": args.checkpoint_every,
                                            "vendor_post_commit_ms": args.vendor_post_commit_ms},
           "host": cr.host_provenance(None), "summary": summarize(trials), "trials": trials,
           "limits": "SIGKILL only; the OS page cache survives, so power loss and torn writes are untested"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(json.dumps(doc["summary"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

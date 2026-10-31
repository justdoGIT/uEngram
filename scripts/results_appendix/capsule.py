"""Sensor memory capsule tables, built from the committed result files."""
from __future__ import annotations

import json
from pathlib import Path

ARMS = ("lite", "sidecar", "capsule")
KILL_FILES = (
    ("capsule_kill_test.json", "stub store, natural timing"),
    ("capsule_kill_test_vendor_pause.json", "stub store, 40 ms pause after vendor commit"),
    ("capsule_kill_test_real.json", "real EdgeMemory Lite, natural timing"),
    ("capsule_kill_test_real_vendor_pause.json", "real EdgeMemory Lite, 40 ms pause after commit"),
    ("capsule_kill_test_relaxed_normal_1.json", "stub store, NORMAL sync, checkpoint every window"),
    ("capsule_kill_test_relaxed_full_16.json", "stub store, FULL sync, checkpoint every 16 windows"),
    ("capsule_kill_test_relaxed_normal_16.json", "stub store, NORMAL sync, checkpoint every 16 windows"),
)


def _load(repo: Path, name: str) -> dict:
    return json.loads((repo / "results" / name).read_text())


def _storage(host: dict) -> str:
    storage = host.get("storage")
    if not storage:
        return "not recorded"
    kind = "HDD" if storage.get("rotational") else ""
    return f"{storage['fstype']}{' ' + kind if kind else ''}".strip()


def _values(cells: list[dict], key: str) -> str:
    return "/".join(str(v) for v in sorted({n for c in cells for n in c[key]}))


def _resilience(repo: Path, name: str, title: str, outcome: str, note: str) -> dict:
    summary = _load(repo, name)
    rows = []
    for host, entry in summary["hosts"].items():
        for arm in ARMS:
            cells = [crash[arm] for crash in entry["crashes"].values()]
            runs = sum(c["runs"] for c in cells)
            rows.append([
                host, _storage(entry["host"]), arm,
                f"{sum(c['state_equivalent_runs'] for c in cells)}/{runs}",
                f"{sum(c['event_keys_match_reference_runs'] for c in cells)}/{runs}",
                _values(cells, "missing_event_counts"), _values(cells, "extra_event_counts"),
                _values(cells, "ranked_history_counts"),
                f"{entry['crashes']['after_ingest'][arm]['recovery_ms']['median']:.1f}",
            ])
    inp = summary["input"]
    prefix = (f"SMD {inp['machine']}, first {inp['window_count']} test windows, restart at the midpoint; "
              f"3 crash boundaries per host. The uninterrupted reference emits "
              f"{summary['reference_event_count']} lifecycle records. ")
    return {"title": title, "outcome": outcome, "source": f"results/{name}",
            "columns": ["host", "temp storage", "arm", "state equals uninterrupted",
                        "event keys equal reference", "missing records", "extra records",
                        "ranked history", "restart to completion ms (after_ingest)"],
            "rows": rows, "note": prefix + note}


def _sweep(repo: Path) -> dict:
    rows, parts = [], []
    for name, vendor in (("capsule_restart_sweep.json", "stub"), ("capsule_restart_sweep_real.json", "real")):
        doc = _load(repo, name)
        summary, params = doc["summary"], doc["params"]
        parts.append(f"{vendor}: {summary['machines']} machines x {params['positions']} restart positions, "
                     f"{params['windows']}-window prefix")
        for arm in ARMS:
            a = summary["arms"][arm]
            rows.append([vendor, arm, a["cells"], f"{a['state_equivalent']}/{a['cells']}",
                         f"{a['keys_match']}/{a['cells']}",
                         f"{a['alarm_lost']}/{a['cells_where_reference_alarms']}",
                         f"{a['cells_with_missing_events']}/{a['cells']}",
                         f"{a['cells_with_extra_events']}/{a['cells']}"])
    return {"title": "Restart position and machine sweep",
            "outcome": "positive", "source": "results/capsule_restart_sweep.json, results/capsule_restart_sweep_real.json",
            "columns": ["store", "arm", "cells", "state equals uninterrupted", "event keys equal reference",
                        "alarm-open lost (cells where reference alarms)", "cells missing records",
                        "cells with extra records"],
            "rows": rows,
            "note": "; ".join(parts) + ". A cell is one machine at one restart position. Alarm records come "
                    "from the harness's fixed alarm_threshold=1.0, not the calibrated alarm policy of the "
                    "detection experiments, so this measures divergence from an uninterrupted run, not missed "
                    "faults. The stub store runs on tmpfs; the real store is the EdgeMemory Lite library."}


def _warmup(repo: Path) -> dict:
    doc = _load(repo, "capsule_warmup_ablation.json")
    rows = [[arm["warmup_windows"], arm["first_flagged_step"],
             ", ".join(f"{k} x{v}" for k, v in sorted(arm["records_after_restart"].items())) or "none",
             "yes" if arm["keys_equal_reference_after_restart"] else "no"] for arm in doc["arms"]]
    ref = doc["reference"]
    return {"title": "Warm-up ablation of a stateless restart", "outcome": "mixed",
            "source": "results/capsule_warmup_ablation.json",
            "columns": ["pre-restart windows replayed", "first alarm step", "records emitted after restart",
                        "equals reference records"],
            "rows": rows,
            "note": f"{doc['input']['machine']}, restart at window {doc['restart_at']}; the uninterrupted "
                    f"reference first alarms at step {ref['first_flagged_step']} and emits "
                    f"{ref['records_after_restart']} after the restart. Replaying enough history restores the "
                    "reference records, which supports (does not prove) the explanation that a cold restart "
                    "re-baselines on already-drifted data. One machine, one restart position."}


def _kill(repo: Path) -> dict:
    rows = []
    for name, label in KILL_FILES:
        doc = _load(repo, name)
        s = doc["summary"]
        rows.append([label, s["trials"], s["killed_mid_stream"], s["killed_with_logged_but_uncheckpointed_input"],
                     s["killed_with_vendor_ahead_of_checkpoint"], f"{s['keys_match']}/{s['trials']}",
                     f"{s['no_duplicates']}/{s['trials']}", f"{s['state_equivalent']}/{s['trials']}",
                     f"{s['integrity_ok']}/{s['trials']}"])
    storage = _storage(_load(repo, KILL_FILES[0][0])["host"])
    return {"title": "SIGKILL recovery campaigns", "outcome": "positive",
            "source": "results/capsule_kill_test*.json",
            "columns": ["campaign", "trials", "killed mid-stream", "input logged, not checkpointed",
                        "vendor ahead of checkpoint", "records equal reference", "no duplicates",
                        "state equals uninterrupted", "SQLite integrity ok"],
            "rows": rows,
            "note": f"Child process SIGKILLed at a random moment, then a second process recovers and finishes "
                    f"the stream; 3 SMD machines, 128-window prefix, temp storage {storage}. Trials that finished "
                    "before the kill landed are included in the totals. SIGKILL leaves the OS page cache intact, "
                    "so this does NOT test power loss or torn writes. The relaxed rows show only that those "
                    "configurations survive process death; NORMAL sync weakens durability against power loss."}


def _cost(repo: Path) -> dict:
    rows = []
    for host in ("msi", "lenovo", "hp", "asus"):
        for storage in ("disk", "tmpfs"):
            doc = _load(repo, f"capsule_cost/{host}-{storage}.json")
            base = doc["memory_only"]["mean_ms"]
            for cfg in doc["configs"]:
                rows.append([host, _storage(doc["host"]), doc["input"]["window_count"], cfg["synchronous"],
                             cfg["checkpoint_every"], f"{base:.2f}", f"{cfg['mean_ms']:.2f}",
                             f"{cfg['p99_ms']:.1f}", f"{cfg['recovery_ms']:.0f}", cfg["max_replay_windows"]])
    return {"title": "Cost of durability: sync mode and checkpoint cadence", "outcome": "mixed",
            "source": "results/capsule_cost/*.json",
            "columns": ["host", "temp storage", "windows", "sync", "checkpoint every", "memory-only ms/window",
                        "capsule ms/window", "p99 ms", "worst-case recovery ms", "max replayed windows"],
            "rows": rows,
            "note": "Median over interleaved repeats (5 on msi at 256 windows; 3 on the other hosts at 64 windows). "
                    "On tmpfs fsync is nearly free, so FULL equals NORMAL and the cost is CPU work in "
                    "checkpoint serialization. On disk, FULL costs one fsync per window even at long "
                    "cadences (the input log commits every window); NORMAL avoids that but weakens durability "
                    "against power loss, which is not tested. hp's overlay storage varied roughly 6x between "
                    "runs, so its figures are indicative only. msi is an interactive workstation."}


def _power_loss(repo: Path) -> dict:
    rows = []
    for path in sorted((repo / "results" / "capsule_power_loss").glob("*.json")):
        doc = json.loads(path.read_text())
        summary = doc["summary"]
        control = doc.get("control", "none")
        sync = doc["synchronous"]
        rows.append([
            doc["machine"], sync, doc["checkpoint_every"], doc["window_count"],
            control, summary["trials"],
            f"{summary['capsule_consistent']}/{summary['trials']}",
            summary.get("images_losing_acknowledged_windows", 0),
            summary.get("acknowledged_windows_lost_max", 0),
            f"{summary['final_vendor_consistent']}/{summary['trials']}",
        ])
    if not rows:
        raise ValueError("capsule power-loss campaign has no result files")
    return {
        "title": "Power-cut crash-image recovery",
        "outcome": "mixed",
        "source": "results/capsule_power_loss/*.json",
        "columns": ["machine", "sync", "checkpoint every", "windows", "control", "trials",
                    "capsule consistent", "images losing acknowledged windows",
                    "max acknowledged windows lost", "final vendor consistent"],
        "rows": rows,
        "note": "The LD_PRELOAD trace is replayed into crash images at whole-syscall write "
                "boundaries; the model has no sub-write tearing or block-device reordering. "
                "FULL and NORMAL are safe only for the modeled image boundaries. NORMAL can "
                "lose acknowledged windows, while disabling the journal is a negative control "
                "that exposes database corruption. This is a model result, not hardware power-loss evidence.",
    }


def build(repo: Path) -> list[dict]:
    return [
        _resilience(repo, "capsule_resilience_summary.json",
                    "Capsule restart recovery against Lite-only and file-sidecar persistence (tmpfs)", "mixed",
                    "Three crash boundaries x 5 repeats per host, stub event store. Storage was not recorded in "
                    "these artifacts; /tmp was tmpfs on all four hosts, so "
                    "the timings exclude fsync cost and measure CPU work. Without detector state the restarted "
                    "arms re-emit two records and miss the alarm-open record. The restart is a reopen inside "
                    "one process, and the hosts are x86 desktops, so this is not a power-loss or embedded result."),
        _resilience(repo, "capsule_resilience_disk_summary.json",
                    "Capsule restart recovery with temp storage on each host's disk", "mixed",
                    "Same experiment with TMPDIR on the host's disk (5 repeats on msi, 3 on each other host). "
                    "fsync is real here, so the capsule's restart-to-completion time "
                    "is far above the memory-only arms. hp's overlay storage varied roughly 6x between runs."),
        _resilience(repo, "capsule_resilience_real_summary.json",
                    "Capsule restart recovery against the real EdgeMemory Lite library", "mixed",
                    "msi only, real vendor store (records identified by content, since recall hides the "
                    "idempotency key), temp storage not recorded (the default tmpfs). The real store returns "
                    "every matching record up to its limit, so the Lite-only 'ranked history' of the stub "
                    "(newest record only) does not apply here."),
        _sweep(repo), _warmup(repo), _kill(repo), _cost(repo), _power_loss(repo),
    ]

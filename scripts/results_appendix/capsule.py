"""Sensor memory capsule restart-recovery table, built from the aggregated resilience summary."""
from __future__ import annotations

import json
from pathlib import Path

ARMS = ("lite", "sidecar", "capsule")


def _values(cells: list[dict], key: str) -> str:
    return "/".join(str(v) for v in sorted({n for c in cells for n in c[key]}))


def build(repo: Path) -> list[dict]:
    summary = json.loads((repo / "results" / "capsule_resilience_summary.json").read_text())
    rows = []
    for host, entry in summary["hosts"].items():
        for arm in ARMS:
            cells = [crash[arm] for crash in entry["crashes"].values()]
            runs = sum(c["runs"] for c in cells)
            after_ingest = entry["crashes"]["after_ingest"][arm]
            rows.append([
                host, arm,
                f"{sum(c['state_equivalent_runs'] for c in cells)}/{runs}",
                f"{sum(c['event_keys_match_reference_runs'] for c in cells)}/{runs}",
                _values(cells, "missing_event_counts"), _values(cells, "extra_event_counts"),
                _values(cells, "ranked_history_counts"),
                f"{after_ingest['recovery_ms']['median']:.1f}",
            ])
    inp = summary["input"]
    note = (f"SMD {inp['machine']}, first {inp['window_count']} test windows, restart at the midpoint; "
            f"3 crash boundaries x 5 repeats per host. The uninterrupted reference emits "
            f"{summary['reference_event_count']} lifecycle records. Without detector state the restarted "
            "arms re-emit two records and miss the alarm-open record; the capsule matches the reference "
            "exactly but costs roughly five to six times the restart-to-completion time (median ms at the "
            "after_ingest boundary, noisy on the interactive workstation). The restart is a reopen inside "
            "one process, the vendor store is a Lite-like stand-in, and the hosts are x86 desktops, so "
            "this is not a power-loss, vendor, or embedded-hardware result.")
    return [{
        "title": "Capsule restart recovery against Lite-only and file-sidecar persistence",
        "outcome": "mixed", "source": "results/capsule_resilience_summary.json",
        "columns": ["host", "arm", "state equals uninterrupted", "event keys equal reference",
                    "missing records", "extra records", "ranked history", "restart to completion ms"],
        "rows": rows, "note": note,
    }]

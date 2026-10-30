"""Aggregate per-host capsule resilience artifacts into one auditable summary.

python -m bench.capsule_resilience_summary --root results/capsule_resilience \
    --out results/capsule_resilience_summary.json

Layout: ``<root>/<target-label>/<crash>-<n>.json``. Artifacts must share one input digest;
a mixed input set is a structural error, not something to average over. Correctness
results are counted, never dropped: a failing run lowers ``state_equivalent_runs``.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

CONDITIONS = ("lite", "sidecar", "capsule")
PROTOCOL = "capsule-resilience-v3"


def _load(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*/*.json")):
        doc = json.loads(path.read_text())
        if doc.get("protocol") != PROTOCOL:
            raise ValueError(f"{path}: expected protocol {PROTOCOL}")
        if doc["host"].get("target_label") != path.parent.name:
            raise ValueError(f"{path}: target label does not match its directory")
        rows.append(doc)
    if not rows:
        raise ValueError(f"no artifacts under {root}")
    return rows


def _condition(runs: list[dict]) -> dict:
    ms = [run["recovery_ns"] / 1e6 for run in runs]
    return {
        "runs": len(runs),
        "state_equivalent_runs": sum(run["state_equivalent_to_uninterrupted"] for run in runs),
        "event_keys_match_reference_runs": sum(run["event_keys_match_reference"] for run in runs),
        "missing_event_counts": sorted({run["missing_event_count"] for run in runs}),
        "extra_event_counts": sorted({run["extra_event_count"] for run in runs}),
        "vendor_event_counts": sorted({run["vendor_event_count"] for run in runs}),
        "ranked_history_counts": sorted({run["ranked_history_count"] for run in runs}),
        "recovery_ms": {"median": statistics.median(ms), "min": min(ms), "max": max(ms)},
        "local_storage_bytes_median": statistics.median(run["local_storage_bytes"] for run in runs),
    }


def summarize(root: Path) -> dict:
    rows = _load(root)
    inputs = {json.dumps(row["input"], sort_keys=True) for row in rows}
    if len(inputs) != 1:
        raise ValueError(f"artifacts do not share one input stream: {len(inputs)} distinct inputs")
    references = {row["reference"]["vendor_event_count"] for row in rows}
    if len(references) != 1:
        raise ValueError(f"artifacts disagree on the uninterrupted reference: {sorted(references)}")
    hosts: dict[str, dict] = {}
    for label in sorted({row["host"]["target_label"] for row in rows}):
        mine = [row for row in rows if row["host"]["target_label"] == label]
        facts = {json.dumps({k: row["host"][k] for k in
                             ("hostname", "platform", "machine", "cpu_model", "memory_mib",
                              "python", "numpy", "cpu_count")},
                            sort_keys=True) for row in mine}
        if len(facts) != 1:
            raise ValueError(f"{label}: host metadata differs between artifacts")
        crashes = {}
        for crash in sorted({row["crash_point"] for row in mine}):
            same = [row for row in mine if row["crash_point"] == crash]
            crashes[crash] = {c: _condition([row["conditions"][c] for row in same]) for c in CONDITIONS}
        hosts[label] = {"host": json.loads(next(iter(facts))), "crashes": crashes}
    capsule = [c["capsule"] for h in hosts.values() for c in h["crashes"].values()]
    others = [c[name] for h in hosts.values() for c in h["crashes"].values() for name in ("lite", "sidecar")]
    return {
        "protocol": PROTOCOL, "input": json.loads(next(iter(inputs))), "hosts": hosts,
        "reference_event_count": next(iter(references)),
        "checks": {
            "artifacts": len(rows),
            "capsule_state_equivalent": all(c["state_equivalent_runs"] == c["runs"] for c in capsule),
            "capsule_event_keys_match_reference": all(
                c["event_keys_match_reference_runs"] == c["runs"] for c in capsule),
            "lite_or_sidecar_state_equivalent_runs": sum(c["state_equivalent_runs"] for c in others),
            "lite_or_sidecar_event_keys_match_reference_runs": sum(
                c["event_keys_match_reference_runs"] for c in others),
            "lite_or_sidecar_runs": sum(c["runs"] for c in others),
        },
        "interpretation": "systems recovery only; timings are host-local, not embedded-device latency",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/capsule_resilience"))
    parser.add_argument("--out", type=Path, default=Path("results/capsule_resilience_summary.json"))
    args = parser.parse_args()
    summary = summarize(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["checks"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Sweep the restart position over many SMD machines for the three persistence arms.

python -m bench.capsule_restart_sweep --out results/capsule_restart_sweep.json

For each machine the first ``--windows`` test windows are replayed with a restart injected at
``--positions`` evenly spaced indices. Per (machine, position) and arm the sweep records whether
engine state and lifecycle-record keys match an uninterrupted run, and whether any
alarm-open record of the reference is missing. SMD labels are never read; this is a
systems-recovery sweep, not a detection benchmark.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from bench import capsule_resilience as cr
from uengram import smd

ARMS = ("lite", "sidecar", "capsule")
PROTOCOL = "capsule-restart-sweep-v1"


def restart_positions(window_count: int, positions: int) -> list[int]:
    """Distinct, evenly spaced restart indices strictly inside the prefix."""
    grid = np.linspace(1, window_count - 1, positions + 2)[1:-1]
    return sorted({round(float(x)) for x in grid})


def _machine(task: tuple[str, int, int, str, str]) -> tuple[str, dict]:
    machine, window_count, positions, crash, vendor = task
    windows = cr._smd_windows(machine, window_count)
    rows, reference = [], None
    for cut in restart_positions(window_count, positions):
        outcome = cr.evaluate(windows, crash, restart_at=cut, vendor=vendor)
        reference = {"events": outcome["reference_event_count"],
                     "alarm_open": outcome["reference_alarm_open_count"]}
        rows.append({"restart_at": cut, **{arm: {
            "state_equivalent": outcome["conditions"][arm]["state_equivalent_to_uninterrupted"],
            "keys_match": outcome["conditions"][arm]["event_keys_match_reference"],
            "missing_events": outcome["conditions"][arm]["missing_event_count"],
            "extra_events": outcome["conditions"][arm]["extra_event_count"],
            "missing_alarm_open": outcome["missing_alarm_open"][arm],
        } for arm in ARMS}})
    return machine, {"input": cr._input_info("smd", machine, windows), "reference": reference,
                     "positions": rows}


def summarize(per_machine: dict) -> dict:
    """Pool (machine, position) cells per arm; alarm loss is judged where the reference alarms."""
    summary = {"machines": len(per_machine),
               "machines_with_reference_alarm": sum(m["reference"]["alarm_open"] > 0
                                                   for m in per_machine.values()),
               "arms": {}}
    for arm in ARMS:
        cells = [(m["reference"], p[arm]) for m in per_machine.values() for p in m["positions"]]
        alarming = [(ref, cell) for ref, cell in cells if ref["alarm_open"] > 0]
        summary["arms"][arm] = {
            "cells": len(cells),
            "state_equivalent": sum(cell["state_equivalent"] for _, cell in cells),
            "keys_match": sum(cell["keys_match"] for _, cell in cells),
            "cells_where_reference_alarms": len(alarming),
            "alarm_lost": sum(cell["missing_alarm_open"] > 0 for _, cell in alarming),
            "cells_with_missing_events": sum(cell["missing_events"] > 0 for _, cell in cells),
            "cells_with_extra_events": sum(cell["extra_events"] > 0 for _, cell in cells),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machines", default="all", help="comma-separated SMD machines, or all")
    parser.add_argument("--windows", type=int, default=256)
    parser.add_argument("--positions", type=int, default=16)
    parser.add_argument("--crash", choices=("after_wal", "after_ingest", "after_checkpoint"),
                        default="after_ingest")
    parser.add_argument("--vendor", choices=("stub", "real"), default="stub",
                        help="event store: in-process stub, or the real EdgeMemory Lite library")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--out", type=Path, default=Path("results/capsule_restart_sweep.json"))
    args = parser.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    machines = smd.MACHINES if args.machines == "all" else tuple(args.machines.split(","))
    tasks = [(m, args.windows, args.positions, args.crash, args.vendor) for m in machines]
    per_machine = {}
    with mp.get_context("spawn").Pool(min(args.jobs, len(tasks))) as pool:
        for done, (machine, result) in enumerate(pool.imap(_machine, tasks), 1):  # ordered
            per_machine[machine] = result
            print(f"[{done}/{len(tasks)}] {machine} reference={result['reference']}", flush=True)
    doc = {"protocol": PROTOCOL,
           "params": {"windows": args.windows, "positions": args.positions, "crash": args.crash,
                      "vendor": args.vendor},
           "host": cr.host_provenance(None), "summary": summarize(per_machine),
           "per_machine": per_machine,
           "interpretation": "systems recovery only; SMD labels never read; not detector accuracy"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(json.dumps(doc["summary"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

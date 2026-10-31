"""Why does a restarted engine lose the alarm? Warm-up ablation of a stateless restart.

python -m bench.capsule_warmup_ablation --out results/capsule_warmup_ablation.json

Hypothesis from score traces: a restarted engine re-baselines on whatever it sees first, so a
drift that began before the restart is learned as normal. If so, replaying enough pre-restart
windows into a fresh engine should bring the alarm back, with a dose-response in the replay
length. The reference is an uninterrupted run; warm-up events are discarded, so the
records counted are only those emitted after the restart. SMD labels are never read.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from bench import capsule_resilience as cr
from uengram import SensorStateMemory
from uengram.sensor_memory_types import SensorWindow


def _kind(row: dict) -> str:
    return row["content"].split(":")[0].removeprefix("uEngram ")


def _trace(windows: tuple[SensorWindow, ...], pipeline: SensorStateMemory) -> list[dict]:
    rows = []
    for window in windows:
        result = pipeline.ingest(window)
        score = result.decision.score
        rows.append({"step": window.step_index, "flagged": result.decision.flagged,
                     "score": None if math.isnan(score) else round(float(score), 3)})
    return rows


def warmup_ablation(windows: tuple[SensorWindow, ...], restart_at: int, warmups: list[int]) -> dict:
    """Records emitted after ``restart_at`` for a fresh engine warmed by the last K windows."""
    dim = len(windows[0].features)
    ref_memory = cr.LatestOnlyMemory()
    ref_pipeline = SensorStateMemory(cr._engine(dim), ref_memory)
    reference_trace = _trace(windows, ref_pipeline)
    after = {row["idempotency_key"]: _kind(row) for row in ref_memory.rows.values()
             if int(row["content"].split("step=")[1].split(";")[0]) >= restart_at}
    out = {"restart_at": restart_at,
           "reference": {"records_after_restart": dict(Counter(after.values())),
                         "first_flagged_step": next((r["step"] for r in reference_trace
                                                     if r["flagged"] and r["step"] >= restart_at), None)},
           "arms": []}
    for k in warmups:
        pipeline = SensorStateMemory(cr._engine(dim), cr.LatestOnlyMemory())
        for window in windows[max(0, restart_at - k):restart_at]:
            pipeline.ingest(window)
        memory = cr.LatestOnlyMemory()
        pipeline.memory = memory  # discard warm-up events: count only post-restart records
        trace = _trace(windows[restart_at:], pipeline)
        keys = {row["idempotency_key"] for row in memory.rows.values()}
        out["arms"].append({
            "warmup_windows": k,
            "first_flagged_step": next((r["step"] for r in trace if r["flagged"]), None),
            "records_after_restart": dict(Counter(_kind(row) for row in memory.rows.values())),
            "keys_equal_reference_after_restart": keys == set(after),
        })
    out["score_trace"] = {"reference": [r for r in reference_trace if restart_at <= r["step"] < restart_at + 14],
                          "restarted_cold": _trace_cold(windows, restart_at, dim)}
    return out


def _trace_cold(windows: tuple[SensorWindow, ...], restart_at: int, dim: int) -> list[dict]:
    pipeline = SensorStateMemory(cr._engine(dim), cr.LatestOnlyMemory())
    return _trace(windows[restart_at:restart_at + 14], pipeline)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", default="machine-1-1")
    parser.add_argument("--windows", type=int, default=64)
    parser.add_argument("--restart-at", type=int, default=32)
    parser.add_argument("--warmups", default="0,2,4,8,16,24,32")
    parser.add_argument("--out", type=Path, default=Path("results/capsule_warmup_ablation.json"))
    args = parser.parse_args()
    windows = cr._windows("smd", args.machine, args.windows)
    doc = {"protocol": "capsule-warmup-ablation-v1", "input": cr._input_info("smd", args.machine, windows),
           **warmup_ablation(windows, args.restart_at, [int(k) for k in args.warmups.split(",")]),
           "interpretation": "mechanism probe on one machine and one restart position; not detector accuracy"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    for arm in doc["arms"]:
        print(f"warm-up {arm['warmup_windows']:>2}: first flagged {arm['first_flagged_step']}, "
              f"records {arm['records_after_restart']}, keys equal reference: "
              f"{arm['keys_equal_reference_after_restart']}", flush=True)


if __name__ == "__main__":
    main()

"""Competitive-admission ablation: full competition vs. accept-everything.

python -m bench.admission_ablation --out results/admission_ablation.json

Patches ``episodic_admission_ratio`` (ratio 1.0 = full competition, the default; ratio 0.0 =
accept any arrival into a full bank) and runs the synthetic lifecycle (DSP encoder, method
uengram) at each budget, seeds 0-4. Also traces one seed per budget to count how often the
ratio-comparison branch in ``make_episodic_room`` actually fires and rejects an arrival.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np

import bench.common as common
import bench.run_synth as rs
from uengram import lifecycle

_orig_make_store = common.make_store
_orig_make_room = lifecycle.make_episodic_room
METRICS = ("event_recall", "write_reduction_pct", "ap")


def _same(a: float, b: float) -> bool:
    """Equal within tolerance; two NaNs (e.g. AP undefined at a tiny budget) count as equal."""
    return (a != a and b != b) or abs(a - b) < 1e-9


def _seeds(spec: str) -> list[int]:
    lo, _, hi = spec.partition("-")
    return list(range(int(lo), int(hi) + 1)) if hi else [int(lo)]


def _run(budget: int, ratio: float, seeds: str) -> dict:
    """Run in this process: ``run_synth.main`` uses a spawn Pool, whose workers re-import the
    module and would silently ignore the patched ``make_store``."""
    def patched_store(method, b, d, sigma0, raw_bytes, seed, over, infl, *a, **k):
        over = dict(over or {})
        over["episodic_admission_ratio"] = ratio
        return _orig_make_store(method, b, d, sigma0, raw_bytes, seed, over, infl, *a, **k)

    rs.make_store = patched_store
    try:
        rows = [line for seed in _seeds(seeds)
                for line in rs.run_one((seed, "dsp", 20000, ["uengram"], [budget], "ablation", 0.01, 4000))]
    finally:
        rs.make_store = _orig_make_store
    return {key: st.mean(row["metrics"][key] for row in rows) for key in METRICS}


def _trace(budget: int, ratio: float) -> dict:
    counts = {"epi_branch": 0, "cand_branch": 0, "rejected": 0, "capacity_skip": 0}

    def counted(eng, t, incoming_surprisal=None):
        group = np.flatnonzero((eng.kind == lifecycle.CAND) | (eng.kind == lifecycle.EPI))
        if len(group) >= eng.cfg.B_epi:
            epi = np.flatnonzero(eng.kind == lifecycle.EPI)
            if len(epi):
                counts["epi_branch"] += 1
                importance = lifecycle._episodic_importance(eng, epi, t)
                threshold = eng.cfg.episodic_admission_ratio * importance[int(np.argmin(importance))]
                if incoming_surprisal is not None and incoming_surprisal <= threshold:
                    counts["rejected"] += 1
            else:
                counts["cand_branch"] += 1
        else:
            counts["capacity_skip"] += 1
        return _orig_make_room(eng, t, incoming_surprisal)

    def patched_store(method, b, d, sigma0, raw_bytes, seed, over, infl, *a, **k):
        over = dict(over or {})
        over["episodic_admission_ratio"] = ratio
        return _orig_make_store(method, b, d, sigma0, raw_bytes, seed, over, infl, *a, **k)

    lifecycle.make_episodic_room = counted
    rs.make_store = patched_store
    task = (0, "dsp", 20000, ["uengram"], [budget], f"trace-B{budget}", 0.01, 4000)
    rs.run_one(task)
    lifecycle.make_episodic_room = _orig_make_room
    rs.make_store = _orig_make_store
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budgets", default="2,4,8,16,256")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--out", default="results/admission_ablation.json")
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    rows = []
    for budget in budgets:
        on = _run(budget, 1.0, args.seeds)
        off = _run(budget, 0.0, args.seeds)
        trace_on = _trace(budget, 1.0)
        row = {"budget": budget, "competitive_admission": on, "accept_all": off,
               "trace_ratio_1_0_one_seed": trace_on,
               "metrics_identical": all(_same(on[k], off[k]) for k in METRICS)}
        rows.append(row)
        print(f"budget={budget} identical={row['metrics_identical']} "
              f"trace={trace_on} on={on} off={off}", flush=True)

    out = {"seeds": args.seeds, "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True, allow_nan=True) + "\n")


if __name__ == "__main__":
    main()

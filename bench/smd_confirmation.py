"""SMD confirmation: frozen alarm policies (selected on MetroPT-3) on 28 unseen server machines.

python -m bench.smd_confirmation --out results/smd_confirmation.json

Per machine: uengram (DSP d=16 and d=36), half-space trees and MemStream (DSP d=16, seeds 0-4),
and uengram with one DSP feature group dropped (as in bench.feature_group_ablation). The alarm
configurations are the frozen ones in bench.run_v2.FROZEN; nothing is tuned on SMD. Machines run in
a spawn pool (--jobs); the feature-group patch is applied inside each worker. BLAS and torch are
limited to one thread in every mode: MemStream's AP changes in the third decimal with numpy's BLAS
thread count, so this keeps the artifact independent of --jobs. A failing configuration is
recorded as an error, not dropped. The summary pools events and clean hours over machines.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics as st
from pathlib import Path

# Must precede the numpy/torch imports below; spawn workers inherit it. Also avoids N workers x N threads.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

from bench import feature_group_ablation as fga
from bench.alarm_eval import apply_policy
from bench.run_v2 import FROZEN
from bench.stream_smd import smd_stream
from bench.v2_scoring import score_stream
from uengram import encoders, smd

BASELINE_SEEDS = (0, 1, 2, 3, 4)
GROUP_RUNS = ("no_mean", "no_log_std", "no_band_energy")


def _run(stream, method: str, enc: str, seed: int) -> dict:
    try:
        m, d, _ = apply_policy(stream, score_stream(stream, method, enc, seed), FROZEN[method])
    except Exception as exc:  # recorded in the artifact; one failing configuration must not hide the rest
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"events_detected": int(m["events_detected"]), "n_events": len(stream.events),
            "auroc": float(m["auroc"]), "ap": float(m["ap"]),
            "false_alarm_episodes": int(d["false_alarm_episodes"]),
            "clean_hours": float(d["clean_observed_hours"])}


def _single_thread_torch() -> None:
    """MemStream trains a torch autoencoder; multi-threaded reductions change float summation order,
    so a thread count other than 1 would make its numbers depend on --jobs."""
    import torch
    torch.set_num_threads(1)


def _machine(machine: str) -> dict:
    _single_thread_torch()
    stream = smd_stream(machine)
    out = {"uengram_d16": _run(stream, "uengram", "dsp", 0), "uengram_d36": _run(stream, "uengram", "dsp_d36", 0)}
    for method in ("hst", "memstream"):
        out[method] = [_run(stream, method, "dsp", s) for s in BASELINE_SEEDS]
    for label in GROUP_RUNS:
        encoders.band_features = fga._patched(fga.GROUPS[label])
        try:
            out[f"uengram_{label}"] = _run(stream, "uengram", "dsp", 0)
        finally:
            encoders.band_features = fga._orig
    return out


def _pool(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    hours = sum(r["clean_hours"] for r in ok)
    return {"machines_ok": len(ok), "machines_failed": len(rows) - len(ok),
            "events_detected": sum(r["events_detected"] for r in ok), "events": sum(r["n_events"] for r in ok),
            "false_alarms_per_week": 168.0 * sum(r["false_alarm_episodes"] for r in ok) / hours if hours else None,
            "mean_auroc": st.mean(r["auroc"] for r in ok) if ok else None,
            "mean_ap": st.mean(r["ap"] for r in ok) if ok else None}


def summarize(per_machine: dict) -> dict:
    keys = [k for k in next(iter(per_machine.values())) if k not in ("hst", "memstream")]
    summary = {k: _pool([m[k] for m in per_machine.values()]) for k in keys}
    for method in ("hst", "memstream"):
        # one pooled row per seed, then the mean and range over seeds
        seeds = [_pool([m[method][i] for m in per_machine.values()]) for i in range(len(BASELINE_SEEDS))]
        summary[method] = {key: [s[key] for s in seeds] for key in ("events_detected", "false_alarms_per_week", "mean_auroc")}
        summary[method]["events"] = seeds[0]["events"]
        summary[method]["machines_failed"] = max(s["machines_failed"] for s in seeds)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/smd_confirmation.json")
    parser.add_argument("--machines", default="all")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                        help="worker processes; each is limited to one BLAS/torch thread")
    args = parser.parse_args()
    machines = smd.MACHINES if args.machines == "all" else tuple(args.machines.split(","))
    per_machine = {}
    if args.jobs > 1:
        with mp.get_context("spawn").Pool(min(args.jobs, len(machines))) as pool:
            results = pool.imap(_machine, machines)  # ordered, so the artifact is independent of --jobs
            for i, (machine, result) in enumerate(zip(machines, results), 1):
                per_machine[machine] = result
                r = result["uengram_d16"]
                print(f"[{i}/{len(machines)}] {machine} uengram_d16 {r.get('events_detected', r)}/{r.get('n_events', '')}", flush=True)
    else:
        for i, machine in enumerate(machines, 1):
            per_machine[machine] = _machine(machine)
            r = per_machine[machine]["uengram_d16"]
            print(f"[{i}/{len(machines)}] {machine} uengram_d16 {r.get('events_detected', r)}/{r.get('n_events', '')}", flush=True)
    out = {"protocol": "frozen FROZEN policies from bench.run_v2; train 70/30 fit/cal, test scored; K=10; 1-minute grid",
           "machines": list(machines), "per_machine": per_machine, "summary": summarize(per_machine)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out["summary"], indent=1), flush=True)


if __name__ == "__main__":
    main()

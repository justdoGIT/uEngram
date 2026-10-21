"""Synthetic benchmark: python -m bench.run_synth [--quick] ..."""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import numpy as np

from bench.common import ALL_METHODS, dump, make_store, parse_seeds, run_stream
from bench.evaluate import evaluate_run
from bench.provenance import Provenance
from uengram import synth
from uengram.encoders import DSPEncoder
from uengram.metrics import median_heuristic

RAW_BYTES = synth.C * synth.K * 4
SWEEP_METHODS = ("fifo", "reservoir", "uengram")


def make_encoder(name: str, seed: int):
    if name == "dsp":
        return DSPEncoder()
    from uengram.encoder_cnn import CNNEncoder

    return CNNEncoder(seed=seed)


def run_one(task):
    seed, enc_name, T, methods, budgets, tag = task
    train = synth.generate(8000, seed + 10_000)
    enc = make_encoder(enc_name, seed).fit(train.windows, 1, None)
    stream = synth.generate(T, seed)
    probe = synth.make_probe(seed, T)
    Z, PZ = enc.encode(stream.windows), enc.encode(probe.windows)
    h = median_heuristic(Z[-2000:])
    lines = []
    for B in budgets:
        for method in methods:
            t0 = time.time()
            store = make_store(method, B, enc.d, enc.sigma0, RAW_BYTES, seed)
            prov = Provenance()

            def record(t, res, store=store, prov=prov):
                if res.item_id != -1:
                    prov.record(res.item_id, int(stream.anomaly[t]), int(stream.regime[t]),
                                int(stream.era[t]), int(stream.event_id[t]))
                prov.apply(store.drain_events())

            oracle = (stream.anomaly > 0) * 0.8 + 0.1 if method == "reference_oracle" else None
            scores, flags, lat = run_stream(store, Z, record, oracle)
            m = evaluate_run(store, prov, stream, Z, PZ, probe, scores, flags, h, lat)
            lines.append({"dataset": "synth", "seed": seed, "encoder": enc_name,
                          "method": method, "budget": B, "metrics": m})
            print(f"[{tag}] seed={seed} enc={enc_name} method={method} B={B} "
                  f"done in {time.time() - t0:.1f}s", flush=True)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--T", type=int, default=20000)
    ap.add_argument("--encoders", default="dsp,cnn")
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", default="results/synth_runs.jsonl")
    ap.add_argument("--sweep-budgets", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.T, a.seeds, a.encoders = 3000, "0", "dsp"
    methods = tuple(a.methods.split(","))
    budgets = (a.budget,)
    tag = "run_synth"
    if a.sweep_budgets:
        methods, budgets, tag = SWEEP_METHODS, tuple(int(b) for b in a.sweep_budgets.split(",")), "sweep"
    tasks = [(s, e, a.T, methods, budgets, tag) for s in parse_seeds(a.seeds) for e in a.encoders.split(",")]
    open(a.out, "w").close()
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        for lines in pool.imap_unordered(run_one, tasks):
            dump(a.out, lines)


if __name__ == "__main__":
    main()

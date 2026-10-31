"""Synthetic benchmark: python -m bench.run_synth [--quick] ..."""
from __future__ import annotations

import argparse
import multiprocessing as mp
import re
import time

from bench.common import (ALL_METHODS, calibrate_alarm, degradation, dump, make_store,
                          measure_step_inflation, parse_seeds, run_stream)
from bench.evaluate import evaluate_run
from bench.provenance import Provenance
from uengram import synth
from uengram.encoders import DSPEncoder
from uengram.metrics import median_heuristic

RAW_BYTES = synth.C * synth.K * 4
SWEEP_METHODS = ("fifo", "reservoir", "uengram")
TEMP_CH, RPM_CH = synth.CHANNELS.index("coolant_temp"), synth.CHANNELS.index("rpm")


def make_encoder(name: str, seed: int):
    if name == "dsp":
        return DSPEncoder()
    wide = re.fullmatch(r"dsp_d(\d+)(?:_g(\d+))?", name)  # _g<L>: DSPEncoder gate_lag
    if wide:
        return DSPEncoder(d=int(wide[1]), gate_lag=int(wide[2] or 0))
    if name == "dsp_slope":
        from uengram.encoder_slope import DSPSlopeEncoder

        return DSPSlopeEncoder()
    if name == "ctx":
        from uengram.encoder_ctx import ContextEncoder

        return ContextEncoder(lag_channels=(TEMP_CH,), drive_channel=RPM_CH)
    if name == "cnn2":
        from uengram.encoder_cnn2 import CNN2Encoder

        return CNN2Encoder(seed=seed)
    from uengram.encoder_cnn import CNNEncoder

    return CNNEncoder(seed=seed)


def run_one(task):
    seed, enc_name, T, methods, budgets, tag, alarm_alpha, cal_T = task
    train = synth.generate(8000, seed + 10_000)
    enc = make_encoder(enc_name, seed).fit(train.windows, 1, None)
    infl, infl_quality = measure_step_inflation(enc.encode(train.windows))
    stream = synth.generate(T, seed)
    probe = synth.make_probe(seed, T)
    # fault-free commissioning run, used only to calibrate uEngram's alarm threshold
    cal = synth.generate(cal_T, seed + 30_000, maint_frac=1.0, anomalies=False)
    Z, PZ, Zcal = enc.encode(stream.windows), enc.encode(probe.windows, sequential=False), enc.encode(cal.windows)
    h = median_heuristic(Z[-2000:])
    t_m = int(0.6 * T)
    lines = []
    for B in budgets:
        for method in methods:
            t0 = time.time()
            over, extra = {}, {}
            if method.startswith("uengram"):
                thr = calibrate_alarm(method, B, enc.d, enc.sigma0, RAW_BYTES, seed, Zcal, alarm_alpha,
                                      step_inflation=infl)
                over, extra = {"alarm_threshold": thr}, {"alarm_threshold": thr}
            store = make_store(method, B, enc.d, enc.sigma0, RAW_BYTES, seed, over, infl)
            prov = Provenance()
            sup_t: list[int] = []
            active_pre: set[int] = set()  # items that received windows just before maintenance
            maint_sup: list[int] = []  # times at which one of those items was superseded

            def record(t, res, store=store, prov=prov, sup_t=sup_t, active_pre=active_pre,
                       maint_sup=maint_sup):
                if res.item_id != -1:
                    prov.record(res.item_id, int(stream.anomaly[t]), int(stream.regime[t]),
                                int(stream.era[t]), int(stream.event_id[t]))
                    if t_m - 200 <= t < t_m:
                        active_pre.add(res.item_id)
                evs = store.drain_events()
                for e in evs:
                    if e[0] == "supersede":
                        sup_t.append(t)
                        if t >= t_m and e[1] in active_pre:
                            maint_sup.append(t - t_m)
                prov.apply(evs)

            oracle = (stream.anomaly > 0) * 0.8 + 0.1 if method == "reference_oracle" else None
            scores, flags, lat = run_stream(store, Z, record, oracle)
            m = evaluate_run(store, prov, stream, Z, PZ, probe, scores, flags, h, lat)
            m.update(extra, n_supersede=len(sup_t),
                     supersede_delay_maint=float(min(maint_sup)) if maint_sup else float("nan"))
            lines.append({"dataset": "synth", "seed": seed, "encoder": enc_name,
                          "method": method, "budget": B, "metrics": m, **degradation(enc, infl_quality)})
            print(f"[{tag}] seed={seed} enc={enc_name} method={method} B={B} "
                  f"done in {time.time() - t0:.1f}s", flush=True)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--T", type=int, default=20000)
    ap.add_argument("--encoders", default="dsp,ctx,cnn")
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", default="results/synth_runs.jsonl")
    ap.add_argument("--sweep-budgets", default=None)
    ap.add_argument("--alarm-alpha", type=float, default=0.01)
    ap.add_argument("--cal-T", type=int, default=4000)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.T, a.seeds, a.encoders = 3000, "0", "dsp"
    methods = tuple(a.methods.split(","))
    budgets = (a.budget,)
    tag = "run_synth"
    if a.sweep_budgets:
        methods, budgets, tag = SWEEP_METHODS, tuple(int(b) for b in a.sweep_budgets.split(",")), "sweep"
    tasks = [(s, e, a.T, methods, budgets, tag, a.alarm_alpha, a.cal_T)
             for s in parse_seeds(a.seeds) for e in a.encoders.split(",")]
    open(a.out, "w").close()
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        for lines in pool.imap_unordered(run_one, tasks):
            dump(a.out, lines)


if __name__ == "__main__":
    main()

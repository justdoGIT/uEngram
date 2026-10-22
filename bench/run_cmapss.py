"""CMAPSS FD001 benchmark: python -m bench.run_cmapss [--quick] ..."""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import numpy as np

from bench.common import dump, make_store, measure_step_inflation, parse_seeds, run_stream
from bench.evaluate import footprint, latency, mmd_metrics
from bench.provenance import Provenance
from bench.run_synth import make_encoder
from uengram import cmapss
from uengram.health import HealthRUL
from uengram.metrics import auroc, average_precision, median_heuristic

METHODS = "naive,fifo,reservoir,reference_labelfree,uengram,uengram_frozen"
CFG_OVER = {"t_dormant": 400}
K, C = 16, len(cmapss.SENSORS)
RAW_BYTES = C * K * 4
EOL, HEALTHY = 20, 75
COMMISSION = 30  # windows per training unit in the commissioning stream


def rul_predict(store, prov, Zt, T) -> np.ndarray:
    it = store.items()
    mass = dict(zip(it.ids.tolist(), it.mass.tolist()))
    preds = []
    for q in Zt:
        num = den = 0.0
        for i in store.recall(q, T - 1, 5, "history"):
            r = prov.mean_rul(i)
            if r is not None:
                num += mass[i] * r
                den += mass[i]
        preds.append(num / den if den else np.nan)
    return np.array(preds)


def run_one(task):
    seed, enc_name, methods, units_hi = task
    train = cmapss.load("data/cmapss/train_FD001.txt")
    test = cmapss.load("data/cmapss/test_FD001.txt")
    rul_true = np.loadtxt("data/cmapss/RUL_FD001.txt")
    Wtr, Utr, _ = cmapss.windows(train[train[:, 0] <= 50], K)
    enc = make_encoder(enc_name, seed).fit(Wtr, K, Utr)
    Ztr = enc.encode(Wtr)
    infl = measure_step_inflation(Ztr, Utr)
    # commissioning stream: the first COMMISSION windows of every training unit,
    # assumed healthy without labels; uEngram freezes its normal states as the
    # healthy reference for the health score
    Zc = Ztr[np.concatenate([np.flatnonzero(Utr == u)[:COMMISSION] for u in np.unique(Utr)])]
    Ws, Us, Rs = cmapss.windows(train[(train[:, 0] > 50) & (train[:, 0] <= units_hi)], K)
    Wt, _, _ = cmapss.windows(test, K, last_only=True)
    Z, Zt = enc.encode(Ws), enc.encode(Wt, sequential=False)
    T = len(Z)
    h = median_heuristic(Z[-2000:])
    pos, neg = Rs <= EOL, Rs > HEALTHY
    keep = pos | neg
    stream_units = np.unique(Us)
    lines = []
    for method in methods:
        t0 = time.time()
        store = make_store(method, 256, enc.d, enc.sigma0, RAW_BYTES, seed, CFG_OVER, infl)
        if method.startswith("uengram"):
            ref = make_store(method, 256, enc.d, enc.sigma0, RAW_BYTES, seed, CFG_OVER, infl)
            for t, z in enumerate(Zc):
                ref.ingest(z, t, t)
            store.set_reference(*ref.reference())
        prov = Provenance()
        health = np.full(T, np.nan)

        def record(t, res, store=store, prov=prov, health=health):
            if res.item_id != -1:
                prov.record(res.item_id, rul=float(Rs[t]), unit=int(Us[t]), eol=bool(Rs[t] <= EOL))
            prov.apply(store.drain_events())
            health[t] = res.health

        scores, _, lat = run_stream(store, Z, record)
        m = footprint(store, T)
        m["auroc"] = auroc(scores[keep], pos[keep])
        m["ap"] = average_precision(scores[keep], pos[keep])
        has_health = not np.isnan(health).all()
        m["health_auroc"] = auroc(health[keep], pos[keep]) if has_health else float("nan")
        m["health_ap"] = average_precision(health[keep], pos[keep]) if has_health else float("nan")
        pred = rul_predict(store, prov, Zt, T)
        ok = ~np.isnan(pred)
        err = pred[ok] - rul_true[ok]
        m["rul_rmse"] = float(np.sqrt(np.mean(err ** 2))) if ok.any() else float("nan")
        m["rul_mae"] = float(np.mean(np.abs(err))) if ok.any() else float("nan")
        # RUL from the health score: a table fitted on the labelled stream windows
        # (the labels the lookup above uses), applied to the test units' last windows
        if has_health:
            err_h = HealthRUL().fit(health, Rs).predict(store.health(Zt)) - rul_true
            m["rul_rmse_health"] = float(np.sqrt(np.mean(err_h ** 2)))
            m["rul_mae_health"] = float(np.mean(np.abs(err_h)))
        else:
            m["rul_rmse_health"] = m["rul_mae_health"] = float("nan")
        live_eol = set()
        for i in store.items().ids.tolist():
            if i in prov.items:
                live_eol |= prov.items[i].eol_units
        m["eol_retention"] = float(np.mean([u in live_eol for u in stream_units]))
        m.update(mmd_metrics(store, Z, h))
        m.update(latency(lat))
        lines.append({"dataset": "cmapss", "seed": seed, "encoder": enc_name,
                      "method": method, "budget": 256, "metrics": m})
        print(f"[run_cmapss] seed={seed} enc={enc_name} method={method} "
              f"done in {time.time() - t0:.1f}s", flush=True)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--encoders", default="dsp,cnn")
    ap.add_argument("--methods", default=METHODS)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", default="results/cmapss_runs.jsonl")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    units_hi = 100
    if a.quick:
        a.seeds, a.encoders, units_hi = "0", "dsp", 60
    cmapss.fetch()
    methods = tuple(a.methods.split(","))
    tasks = [(s, e, methods, units_hi) for s in parse_seeds(a.seeds) for e in a.encoders.split(",")]
    open(a.out, "w").close()
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        for lines in pool.imap_unordered(run_one, tasks):
            dump(a.out, lines)


if __name__ == "__main__":
    main()

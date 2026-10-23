"""MetroPT-3 benchmark: python -m bench.run_metropt [--quick] ...

Real air-compressor stream with four reported air leaks. Protocol in
docs/EXPLAINED.md section 16. All methods score the same encoded windows;
uEngram additionally reports the health score from its frozen reference.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import numpy as np

from bench.common import CAL_WARMUP, dump, make_store, parse_seeds
from bench.evaluate import footprint, latency
from bench.metropt_calibration import calibrate_steps
from bench.metropt_eval import evaluate
from bench.provenance import Provenance
from bench.run_synth import make_encoder
from uengram import metropt
from uengram.engine import EngramConfig
from uengram.stats import conformal_threshold
from uengram.stream_baselines import HSTDetector, MemStream

METHODS = ("naive", "fifo", "reservoir", "reference_labelfree", "uengram",
           "uengram_frozen", "hst", "memstream")
B, D, K = 256, 16, 60
RAW_BYTES = len(metropt.ANALOG) * K * 4
ALPHA, MEMSTREAM_EPOCHS = 0.01, 5000
PROTOCOL = "metropt-v1-commissioning-innovations"


def _score(model, Z, label, *, raw_indices=None, failure_ids=None):
    """Prequential scoring; ground truth is recorded only after each ingest."""
    scores, health = np.empty(len(Z)), np.full(len(Z), np.nan)
    lat = np.empty(len(Z), dtype=np.int64)
    provenance = Provenance()
    last_progress = time.monotonic()
    for i, z in enumerate(Z):
        raw = i if raw_indices is None else int(raw_indices[i])
        t0 = time.perf_counter_ns()
        result = model.ingest(z, i, raw)
        lat[i] = time.perf_counter_ns() - t0
        scores[i], health[i] = result.score, result.health
        if failure_ids is not None:
            if result.item_id >= 0:
                provenance.record(result.item_id, event_id=int(failure_ids[i]))
            provenance.apply(model.drain_events())
        if time.monotonic() - last_progress >= 30:
            print(f"{label} windows {i + 1}/{len(Z)}", flush=True)
            last_progress = time.monotonic()
    return scores, health, lat, provenance


def run_one(task):
    seed, enc_name, methods = task
    label = f"[run_metropt] seed={seed} enc={enc_name}"
    print(f"{label} loading windows and fitting encoder", flush=True)
    t, X, cols = metropt.load(metropt.fetch())
    W, S, E, G = metropt.windows(t, X, cols, K)
    fit = E < np.datetime64(metropt.FIT_END)
    cal = (S >= np.datetime64(metropt.FIT_END)) & (E < np.datetime64(metropt.TRAIN_END))
    test = S >= np.datetime64(metropt.TRAIN_END)
    enc = make_encoder(enc_name, seed).fit(W[fit], 1, G[fit])
    Z = enc.encode(W)
    Zfit, Zcal, Ztest = Z[fit], Z[cal], Z[test]
    tst = np.flatnonzero(test)
    fid = metropt.failure_id(S[test], E[test])
    negative = metropt.clean_negative(S[test], E[test])
    dataset = {"fit_windows": int(fit.sum()), "calibration_windows": int(cal.sum()),
               "test_windows": int(test.sum()), "sample_count": len(t),
               "first_timestamp": str(t[0]), "last_timestamp": str(t[-1]),
               "features": list(metropt.ANALOG), "window_samples": K}
    print(f"{label} fit={len(Zfit)} calibration={len(Zcal)} test={len(Ztest)}", flush=True)

    lines = []
    for method in methods:
        begin = time.monotonic()
        stage = f"{label} method={method}"
        print(f"{stage} calibrating", flush=True)
        calibration = {}
        if method in ("hst", "memstream"):
            if method == "hst":
                cal_model = HSTDetector(Zfit, seed)
                warmup = cal_model.warmup_count
            else:
                def progress(epoch, total):
                    if epoch % 1000 == 0:
                        print(f"{stage} training AE {epoch}/{total}", flush=True)

                base = MemStream(Zfit, seed, epochs=MEMSTREAM_EPOCHS, progress_callback=progress)
                cal_model = base.clone(beta=-float("inf"))
                warmup = 0
            cal_scores, _, _, _ = _score(cal_model, Zcal, stage + " calibration")
            thr = conformal_threshold(cal_scores[max(CAL_WARMUP, warmup):], ALPHA)
            model = HSTDetector(Zfit, seed) if method == "hst" else base.clone(beta=thr)
            detector = True
        else:
            infl = None
            if method.startswith("uengram"):
                cfg = EngramConfig(B_sem=B - B // 4, B_epi=B // 4,
                                   beta_min=0.0 if method == "uengram_frozen" else .01)
                infl, calibration = calibrate_steps(Zfit, G[fit], cfg, enc.sigma0, RAW_BYTES)
                print(f"{stage} warmed innovations={calibration['warm_samples']}", flush=True)
            cal_model = make_store(method, B, D, enc.sigma0, RAW_BYTES, seed, None, infl)
            cal_scores, _, _, _ = _score(cal_model, Zcal, stage + " calibration")
            thr = conformal_threshold(cal_scores[CAL_WARMUP:], ALPHA)
            model = make_store(method, B, D, enc.sigma0, RAW_BYTES, seed,
                               {"alarm_threshold": thr}, infl)
            if method.startswith("uengram"):
                reference = make_store(method, B, D, enc.sigma0, RAW_BYTES, seed, None, infl)
                _score(reference, Zfit, stage + " reference")
                model.set_reference(*reference.reference())
            detector, warmup = False, 0
        print(f"{stage} scoring test stream", flush=True)
        scores, health, lat, provenance = _score(
            model, Ztest, stage + " test", raw_indices=tst,
            failure_ids=None if detector else fid)
        m, details = evaluate(scores, health, S[test], E[test], fid, negative, thr, warmup)
        if detector:
            m.update({k: float("nan") for k in ("items_end", "index_bytes", "pinned_raw_bytes",
                                               "writes", "bytes_written", "write_reduction_pct",
                                               "failures_retained")})
            if method == "memstream":
                details["detector_memory_bytes"] = model.memory_bytes
                details["detector_parameter_bytes"] = sum(
                    p.numel() * p.element_size() for p in model.encoder.parameters())
                details["memory_updates"] = model.count
        else:
            m.update(footprint(model, len(Ztest)))
            retained = set()
            for iid in model.items().ids.tolist():
                if iid in provenance.items:
                    retained.update(provenance.items[iid].events)
            m["failures_retained"] = float(len(retained))
            for failure in details["failures"]:
                failure["retained_provenance"] = failure["failure_id"] in retained
        m.update(latency(lat))
        calibration.update({"alarm_alpha": ALPHA, "alarm_warmup": max(CAL_WARMUP, warmup),
                            "test_warmup": warmup})
        lines.append({"dataset": "metropt", "seed": seed, "encoder": enc_name,
                      "method": method, "budget": B, "protocol": PROTOCOL,
                      "calibration": calibration, "data": dataset, "details": details, "metrics": m})
        print(f"{stage} AUROC={m['auroc']:.3f} AP={m['ap']:.3f} "
              f"detected={m['failures_detected']:.0f}/4 FA/week={m['false_alarm_per_week']:.2f} "
              f"retained={m['failures_retained']} done in {time.monotonic() - begin:.1f}s", flush=True)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--encoders", default="dsp,cnn")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", default="results/metropt_runs.jsonl")
    ap.add_argument("--quick", action="store_true", help="full-length DSP stream, seed 0 only")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.encoders = "0", "dsp"
    methods, encoders = tuple(a.methods.split(",")), a.encoders.split(",")
    if not set(methods) <= set(METHODS) or not set(encoders) <= {"dsp", "cnn", "cnn2"}:
        ap.error("unknown method or encoder")
    metropt.fetch()  # fetch once, before workers can race to create the cache
    tasks = [(s, e, methods) for s in parse_seeds(a.seeds) for e in encoders]
    with open(a.out, "w"):
        pass
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        for lines in pool.imap_unordered(run_one, tasks):
            dump(a.out, lines)


if __name__ == "__main__":
    main()

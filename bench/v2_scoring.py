"""Per-window scores for protocol v2: one memory runs continuously through
commissioning, calibration and test (docs/EXPLAINED.md section 17).

v1 restarted every method from an empty state at the start of the test
stream. v2 deploys like a monitored machine: the memory learns the
commissioning period, the healthy reference is frozen at its end, alarm
thresholds come from the following calibration period, and monitoring
continues from that state. Scores do not depend on the alarm policy, so they
are cached and every policy is evaluated on identical scores.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from bench.alarm_eval import ACTIONS
from bench.encoding_cache import cached_encoding, input_hash
from bench.common import make_store
from bench.evaluate import footprint, latency
from bench.metropt_calibration import calibrate_steps
from bench.provenance import Provenance
from uengram.engine import EngramConfig
from uengram.stats import conformal_threshold
from uengram.stream_baselines import HSTDetector, MemStream

B, D = 256, 16
ALPHA = 0.01  # MemStream memory-update gate, as in v1
MEMSTREAM_EPOCHS = 5000


def make_v2_encoder(name: str, seed: int):
    if name == "dsp":
        from uengram.encoders import DSPEncoder
        return DSPEncoder()
    if name in ("cnn2", "cnn2s"):
        from uengram.encoder_cnn2 import CNN2Encoder
        return CNN2Encoder(seed=seed)
    if name == "tab":
        from uengram.encoder_tab import TabularEncoder
        return TabularEncoder()
    raise ValueError(f"unknown encoder {name}")


def encode(stream, enc_name: str, seed: int):
    enc = make_v2_encoder(enc_name, seed)
    fit = stream.fit
    if enc_name == "cnn2s":
        enc.fit(stream.W[fit], 1, stream.segment[fit], cond=stream.cond[fit])
        return enc, enc.encode(stream.W, cond=stream.cond)
    enc.fit(stream.W[fit], 1, stream.segment[fit])
    return enc, enc.encode(stream.W)


def _progress(label):
    last = [time.monotonic()]

    def tick(i, n):
        if time.monotonic() - last[0] >= 30:
            print(f"{label} windows {i + 1}/{n}", flush=True)
            last[0] = time.monotonic()
    return tick


def score_stream(stream, method: str, enc_name: str, seed: int, label: str = "", encoding_dir=None) -> dict:
    """Per-window novelty, health, memory action and run facts for one method."""
    enc, Z = (cached_encoding(stream, enc_name, seed, encoding_dir) if encoding_dir else
              encode(stream, enc_name, seed))
    return score_embeddings(stream, method, enc_name, seed, enc, Z, label,
                            record_encoding=encoding_dir is not None)


def score_embeddings(stream, method, enc_name, seed, enc, Z, label="", *, record_encoding=True):
    """Reuse the continuous scoring loop with explicitly supplied embeddings.

    This entry point permits normalization ablations without changing the
    frozen v2 encoder factory or any existing detector mathematics.
    """
    n = len(Z)
    nov, health = np.full(n, np.nan), np.full(n, np.nan)
    act = np.zeros(n, np.int8)
    lat = []
    order = np.flatnonzero(stream.fit | stream.cal | stream.test)
    tick = _progress(f"{label} {method}")
    meta = {"method": method, "encoder": enc_name, "seed": seed, "stream": stream.name}
    meta["input_hash"] = input_hash(stream, enc_name)
    reference_mu, reference_var = np.empty((0, D)), np.empty((0, D))
    if record_encoding:
        meta["encoding"] = enc.encoding_meta
        meta["encoding_path"] = enc.encoding_path
    if method == "uengram":
        cfg = EngramConfig(B_sem=B - B // 4, B_epi=B // 4)
        infl, cal_info = calibrate_steps(Z[stream.fit], stream.segment[stream.fit], cfg,
                                         enc.sigma0, stream.raw_bytes,
                                         progress=_progress(f"{label} commissioning"))
        model = make_store("uengram", B, D, enc.sigma0, stream.raw_bytes, seed, None, infl)
        last_fit = int(np.flatnonzero(stream.fit)[-1])
        eid, prov = stream.event_id(), Provenance()
        first_test = int(np.flatnonzero(stream.test)[0])
        writes_before = bytes_before = 0
        for t, i in enumerate(order):
            if i == first_test:
                writes_before, bytes_before = model.ledger.writes, model.ledger.bytes_written
            t0 = time.perf_counter_ns()
            r = model.ingest(Z[i], t, int(i))
            if stream.test[i]:
                lat.append(time.perf_counter_ns() - t0)
                if r.item_id >= 0:
                    prov.record(r.item_id, event_id=int(eid[i]))
            prov.apply(model.drain_events())
            nov[i], health[i], act[i] = r.score, r.health, ACTIONS.index(r.action)
            if i == last_fit:
                model.set_reference(*model.reference())
            tick(t, len(order))
        retained = set()
        for iid in model.items().ids.tolist():
            if iid in prov.items:
                retained |= prov.items[iid].events
        meta.update(footprint(model, int(stream.test.sum())))
        meta["commissioning_calibration_writes"] = writes_before
        meta["total_writes"] = meta["writes"]
        meta["writes"] -= writes_before
        meta["bytes_written"] -= bytes_before
        meta["write_reduction_pct"] = 100 * (1 - meta["writes"] / int(stream.test.sum()))
        reference_mu, reference_var = model.ref_mu.copy(), model.ref_var.copy()
        meta["events_retained"] = sorted(int(e) for e in retained)
        meta["step_calibration"] = cal_info
        meta["steps_in_test"] = int((act[stream.test] == ACTIONS.index("step_change")).sum())
    elif method == "hst":
        model = HSTDetector(Z[stream.fit], seed)
        for t, i in enumerate(order):
            t0 = time.perf_counter_ns()
            nov[i] = model.ingest(Z[i], t, int(i)).score
            if stream.test[i]:
                lat.append(time.perf_counter_ns() - t0)
            tick(t, len(order))
        act[order] = ACTIONS.index("score")
    elif method == "memstream":
        base = MemStream(Z[stream.fit], seed, epochs=MEMSTREAM_EPOCHS)
        probe = base.clone(beta=-float("inf"))
        cal_idx = np.flatnonzero(stream.cal)
        beta = conformal_threshold(np.array([probe.ingest(Z[i], 0, 0).score for i in cal_idx]), ALPHA)
        model = base.clone(beta=beta)
        rest = np.flatnonzero(stream.cal | stream.test)
        for t, i in enumerate(rest):
            t0 = time.perf_counter_ns()
            nov[i] = model.ingest(Z[i], t, int(i)).score
            if stream.test[i]:
                lat.append(time.perf_counter_ns() - t0)
            tick(t, len(rest))
        act[rest] = ACTIONS.index("score")
        meta["memstream_beta"] = beta
        meta["detector_memory_bytes"] = model.memory_bytes
    else:
        raise ValueError(f"unknown method {method}")
    meta.update(latency(lat))
    return {"novelty": nov, "health": health, "action": act, "meta": meta,
            "reference_mu": reference_mu, "reference_var": reference_var}


def cached_scores(stream, method, enc_name, seed, scores_dir, label="", rescore=False) -> dict:
    os.makedirs(scores_dir, exist_ok=True)
    path = os.path.join(scores_dir, f"{stream.name}_{method}_{enc_name}_{seed}_final.npz")
    if os.path.exists(path) and not rescore:
        z = np.load(path)
        if json.loads(str(z["meta"]))["input_hash"] != input_hash(stream, enc_name):
            raise ValueError(f"stale score cache: {path}; choose a new scores directory")
        return {"novelty": z["novelty"], "health": z["health"], "action": z["action"],
                "meta": json.loads(str(z["meta"])),
                "reference_mu": z["reference_mu"], "reference_var": z["reference_var"]}
    sc = score_stream(stream, method, enc_name, seed, label, os.path.join(scores_dir, "encoding"))
    np.savez_compressed(path, novelty=sc["novelty"], health=sc["health"], action=sc["action"],
                        reference_mu=sc["reference_mu"], reference_var=sc["reference_var"],
                        meta=json.dumps(sc["meta"]))
    return sc

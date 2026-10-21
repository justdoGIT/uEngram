"""Metrics for one (store, stream) run."""
from __future__ import annotations

import numpy as np

from uengram.metrics import auroc, average_precision, gauss_mixture_mmd2

NAN = float("nan")
WINDOW = 2000


def footprint(store, T: int) -> dict:
    return {
        "items_end": int(len(store.items().ids)),
        "index_bytes": int(store.index_bytes()),
        "pinned_raw_bytes": int(store.pinned_raw_bytes()),
        "writes": int(store.ledger.writes),
        "bytes_written": int(store.ledger.bytes_written),
        "write_reduction_pct": float(100.0 * (1.0 - store.ledger.writes / T)),
    }


def mmd_metrics(store, Z: np.ndarray, h: float) -> dict:
    """Mixture MMD^2 of live items vs recent / whole-history embeddings."""
    if not getattr(store, "in_ref_space", True):
        return {"mmd2_current": NAN, "mmd2_history": NAN}
    T = Z.shape[0]
    it = store.items()
    out = {}
    recent = Z[-WINDOW:]
    cur = it.t_last >= T - WINDOW
    if cur.any():
        out["mmd2_current"] = gauss_mixture_mmd2(
            it.mass[cur], it.mu[cur], it.var[cur], np.ones(len(recent)), recent, np.zeros_like(recent), h)
    else:
        out["mmd2_current"] = NAN
    n = min(WINDOW, T)
    sub = Z[np.random.default_rng(0).choice(T, n, replace=False)]
    out["mmd2_history"] = gauss_mixture_mmd2(
        it.mass, it.mu, it.var, np.ones(n), sub, np.zeros_like(sub), h) if len(it.ids) else NAN
    return out


def latency(latencies_ns) -> dict:
    us = np.asarray(latencies_ns, dtype=np.float64) / 1e3
    return {"lat_p50_us": float(np.percentile(us, 50)), "lat_p99_us": float(np.percentile(us, 99))}


def _retention(store, prov, stream) -> dict:
    live = {int(i) for i in store.items().ids}
    kept_events = set()
    for i in live:
        it = prov.items.get(i)
        if it is not None:
            kept_events |= it.events
    ev_all = np.unique(stream.event_id[stream.event_id >= 0])
    seiz = np.unique(stream.event_id[stream.anomaly == 4])
    kept = lambda ids: float(np.mean([e in kept_events for e in ids])) if len(ids) else NAN  # noqa: E731
    types = {prov.majority_anomaly(i) for i in live}
    return {
        "event_recall": kept(ev_all),
        "seizure_recall": kept(seiz),
        "type_recall": float(np.mean([k in types for k in (1, 2, 3, 4)])),
    }


def _detection(stream, scores, flags, start=200) -> dict:
    y = stream.anomaly[start:] > 0
    s = np.asarray(scores)[start:]
    out = {"auroc": auroc(s, y), "ap": average_precision(s, y)}
    keys = ("precision", "recall", "f1", "false_flag_rate_era0", "false_flag_rate_era1")
    if flags is None:
        return {**out, **{k: NAN for k in keys}}
    f = np.asarray(flags, dtype=bool)[start:]
    tp = int(np.sum(f & y))
    prec = tp / max(int(f.sum()), 1)
    rec = tp / max(int(y.sum()), 1)
    era = stream.era[start:]
    nom = ~y
    out.update(
        precision=float(prec), recall=float(rec),
        f1=float(2 * prec * rec / (prec + rec)) if prec + rec > 0 else 0.0,
        false_flag_rate_era0=float(f[nom & (era == 0)].mean()) if (nom & (era == 0)).any() else NAN,
        false_flag_rate_era1=float(f[nom & (era == 1)].mean()) if (nom & (era == 1)).any() else NAN,
    )
    return out


def _retrieval(store, prov, probe_z, probe, T) -> dict:
    stale, regacc, typeprec = [], [], []
    for z, a, r in zip(probe_z, probe.anomaly, probe.regime):
        if a == 0:
            top = store.recall(z, T - 1, 1, "current")
            re = prov.majority_regime_era(top[0]) if top else None
            stale.append(re is not None and re[1] == 0)
            regacc.append(re == (int(r), 1))
        else:
            top = store.recall(z, T - 1, 1, "history")
            typeprec.append(bool(top) and prov.majority_anomaly(top[0]) == int(a))
    return {"stale_at1": float(np.mean(stale)), "regime_acc_at1": float(np.mean(regacc)),
            "type_prec_at1": float(np.mean(typeprec))}


def evaluate_run(store, prov, stream, Z, probe_z, probe, scores, flags, h, latencies) -> dict:
    T = len(stream.anomaly)
    m = footprint(store, T)
    m.update(_retention(store, prov, stream))
    m.update(_detection(stream, scores, flags))
    m.update(_retrieval(store, prov, probe_z, probe, T))
    m.update(mmd_metrics(store, Z, h))
    m.update(latency(latencies))
    return m

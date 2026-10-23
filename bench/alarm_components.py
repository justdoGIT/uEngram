"""How each alarm component behaves, by physical operating state (v8 diagnosis).

python -m bench.alarm_components  ->  results/v8_component_profile.json

Per stream, with the actual alarm (frozen v2 AlarmConfig) and the frozen scores:
  states       eligible windows per operating state in commissioning, calibration, test
  rates        per window class (clean, 24 h before a report, event) and state: own
               reference, novelty and health above setpoint, new memory state, low
               pressure, oil-level departure
  clustering   two-state Markov fit of the counted exceedances on clean windows and the
               false triggers per week each persistence model implies (independent vs
               Markov), against the observed openings
  correlation  phi of each component with sustained load, on clean and event windows
  separability AUROC of novelty and health, clean vs event and clean vs pre-report,
               within each state (pooled AUROC mixes in the state)
  physics      healthy fill time to the 8.2 bar cut-in; oil-level departure by state and
               oil temperature
  v5_hold      the v5 selected rule with reset vs three-valued hold (and staleness), plus
               the monthly error share of its dominant channel on judged clean windows
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bench.alarm_definition import CUT_IN_BAR, Definition, fill_windows
from bench.alarm_eval import ACTIONS, apply_policy
from bench.alarm_math import arl0, markov_fit
from bench.alarm_sensitivity import BASELINE
from bench.error_gated_alarm import dataset_errors
from bench.modality_alarm import ModalityConfig, calibrate, channel_values, evaluate, three_valued_condition
from bench.streams import load_stream
from uengram import metropt
from uengram.metrics import auroc

STREAMS = ("metropt3", "metropt2_jan", "metropt2_apr")
H = np.timedelta64(1, "h")
OUT = Path("results/v8_component_profile.json")


def window_class(stream, test) -> np.ndarray:
    end, ev = stream.end[test], stream.event_id()[test]
    pre = np.zeros(len(test), bool)
    for a, _ in stream.events:
        fs = np.datetime64(a, "s")
        pre |= (end >= fs - 24 * H) & (end < fs) & (ev < 0)
    clean = stream.clean_negative()[test]
    return np.where(ev >= 0, "event", np.where(pre, "pre24h", np.where(clean, "clean", "other")))


def _rate(x, m):
    return round(float(x[m].mean()), 4) if m.any() else None


def profile(d: Definition) -> dict:
    s, t, sc = d.stream, d.test, d.scores
    m0, det, pol = apply_policy(s, sc, BASELINE)
    v, st, cls = det["valid"], s.state[t], window_class(s, t)
    thr = np.array([pol.thresholds(int(x)) for x in st])
    comp = {"own_reference": np.array([int(x) in pol.baselines for x in st]),
            "novelty_above": sc["novelty"][t] > thr[:, 0], "health_above": sc["health"][t] > thr[:, 1],
            "new_memory_state": np.array([ACTIONS[a] == "new_candidate" for a in sc["action"][t]]),
            "low_pressure": d.sources["dataset"][0][:, 0], "oil_departed": d.sources["dataset"][0][:, 1]}
    out = {"states": {split: [int((getattr(s, split) & s.monitorable & (s.state == k)).sum()) for k in (0, 1, 2)]
                      for split in ("fit", "cal", "test")}}
    out["rates"] = {f"{c}/state{k}": {"windows": int((v & (cls == c) & (st == k)).sum()),
                                      **{n: _rate(x, v & (cls == c) & (st == k)) for n, x in comp.items()}}
                    for c in ("clean", "pre24h", "event") for k in (0, 1, 2) if (v & (cls == c) & (st == k)).any()}
    clean = v & (cls == "clean")
    idx = np.flatnonzero(clean)
    contig = np.r_[False, (np.diff(idx) == 1) & (s.segment[t][idx][1:] == s.segment[t][idx][:-1])]
    weeks = float((s.end[t][clean] - s.start[t][clean]).astype(np.int64).sum()) / (7 * 86400)
    fit = markov_fit(pol.exceeded[idx], contig)
    out["clustering"] = {"counted_exceedances": fit, "clean_weeks": weeks,
                         "observed_openings_per_week": int((det["opened"] & clean).sum()) / weeks}
    if 0 < fit["a"] < 1 and fit["b"] < 1 and fit["p"] > 0:
        out["clustering"]["predicted_triggers_per_week_k3"] = {
            model: {"independent": len(idx) / arl0(model, 3, fit["p"], fit["p"]) / weeks,
                    "markov": len(idx) / arl0(model, 3, fit["a"], fit["b"]) / weeks}
            for model in ("counter", "consecutive", "m_of_n", "ewma")}
    out["phi_with_sustained_load"] = {}
    for c in ("clean", "event"):
        m = v & (cls == c)
        X = np.column_stack([st == 2] + [comp[n] for n in comp if n != "own_reference"])[m].astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            R = np.corrcoef(X.T)
        out["phi_with_sustained_load"][c] = {n: (round(float(R[0, j + 1]), 3) if np.isfinite(R[0, j + 1]) else None)
                                             for j, n in enumerate(n for n in comp if n != "own_reference")}
    out["separability"] = {}

    def separation(x, negative, positive):
        return auroc(np.r_[x[negative], x[positive]], np.r_[np.zeros(negative.sum()), np.ones(positive.sum())])
    for k in (0, 1, 2):
        cl, ev, pr = clean & (st == k), v & (cls == "event") & (st == k), v & (cls == "pre24h") & (st == k)
        if ev.sum() < 5 or cl.sum() < 5:
            continue
        out["separability"][f"state{k}"] = {
            name: {"clean_vs_event": separation(sc[name][t], cl, ev),
                   "clean_vs_pre24h": separation(sc[name][t], cl, pr) if pr.sum() >= 5 else None}
            for name in ("novelty", "health")}
    temp = s.features[t][:, list(s.feature_names).index("oil_temp_c")]
    oil = comp["oil_departed"]
    out["physics"] = {"fill_windows": fill_windows(s),
                      "oil_departed_by_state": [_rate(oil, v & (st == k)) for k in (0, 1, 2)],
                      "oil_temp_median_departed_vs_not": [float(np.median(temp[v & oil])) if (v & oil).any() else None,
                                                         float(np.median(temp[v & ~oil]))],
                      "event_low_pressure": event_low_pressure(s)}
    return out


def event_low_pressure(stream) -> list[dict]:
    """Per event, the dataset low-pressure windows (all windows, eligible or not) and, for
    the first, how long its logging segment had run and whether TP3 had reached the 8.2 bar
    cut-in earlier in that segment (pressure lost after it was built) or not (never built)."""
    E, names = dataset_errors(stream)
    low = E[:, names.index("lps")] | E[:, names.index("tp3_below_7bar")]
    tp3_max = stream.W[:, list(metropt.ANALOG).index("TP3")].max(axis=1)
    seg, event = stream.segment, stream.event_id()
    first_of_segment = np.r_[0, np.flatnonzero(seg[1:] != seg[:-1]) + 1]
    out = []
    for e in range(len(stream.events)):
        idx = np.flatnonzero(low & (event == e))
        rec = {"event": e, "windows": int(len(idx)),
               "eligible_windows": int((stream.monitorable[idx]).sum()) if len(idx) else 0}
        if len(idx):
            i = int(idx[0])
            a = int(first_of_segment[first_of_segment <= i][-1])
            rec["first_minutes_into_segment"] = float((stream.end[i] - stream.start[a]).astype(np.int64)) / 60
            rec["cut_in_reached_before_first"] = bool((tp3_max[a:i] >= CUT_IN_BAR).any())
            rec["cut_in_reached_in_segment_until_first"] = bool((tp3_max[a:i + 1] >= CUT_IN_BAR).any())
        out.append(rec)
    return out


def v5_hold(stream) -> dict:
    sel = json.loads(Path("results/v5_modality_selection.json").read_text())["selected"]["config"]
    cfg = ModalityConfig(**(sel | {"channels": tuple(sel["channels"])}))
    rows = {}
    variants = (("reset", {}), ("hold", {"three_valued": True}),
                ("hold_stale36", {"three_valued": True, "stale": 36}))
    for label, kw in variants:
        m, _ = evaluate(stream, cfg, **kw)
        rows[label] = {k: m[k] for k in ("events_detected", "false_alarm_per_week", "alarm_time_share_clean")}
    t = np.flatnonzero(stream.test)
    values = channel_values(stream, cfg)
    cond, decided, errors, names = three_valued_condition(stream, t, cfg, calibrate(stream, cfg, values), values)
    clean = stream.clean_negative()[t] & stream.monitorable[t] & decided
    j = int(np.argmax(errors[clean].mean(axis=0)))
    month = stream.start[t].astype("datetime64[M]")
    rows["judged_clean_condition_true"] = float(cond[clean].mean())
    rows["dominant_channel"] = names[j]
    rows["dominant_monthly_error_share"] = {str(mo): float(errors[clean & (month == mo), j].mean())
                                            for mo in np.unique(month[clean])}
    return rows


def main():
    out = {}
    for name in STREAMS:
        d = Definition(load_stream(name))
        out[name] = profile(d) | {"v5_hold": v5_hold(d.stream)}
        print(name, json.dumps(out[name]["clustering"].get("predicted_triggers_per_week_k3", {}).get("counter")),
              "observed", round(out[name]["clustering"]["observed_openings_per_week"], 3), flush=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()

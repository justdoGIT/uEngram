"""Protocol v2 confirmation runs (docs/EXPLAINED.md section 17).

python -m bench.run_v2 --streams metropt2 --methods uengram:dsp,hst:dsp,memstream:dsp \
    --seeds 0-4 --out results/v2_metropt2.jsonl
python -m bench.run_v2 --streams care --methods uengram:tab,hst:tab,memstream:tab \
    --seeds 0 --out results/v2_care.jsonl

Every run scores the stream once (continuous memory, bench.v2_scoring) and
evaluates two alarm policies on identical scores: the v1 per-window rule and
the method's v2 configuration frozen from the MetroPT-3 development sweep
(bench.alarm_dev). Nothing in this file may change after the first
confirmation run.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os

import numpy as np

from bench.alarm_eval import apply_policy
from bench.common import dump, parse_seeds
from bench.streams import load_stream
from bench.freeze_frames import archive_frames
from bench.v2_scoring import cached_scores
from uengram.alarm import AlarmConfig

PROTOCOL = "v2-continuous-alarm-policy"
V1_RULE = AlarmConfig(k_novelty=1, k_health=0, per_state=False, new_state_alert=False)
# Selected on MetroPT-3 by bench.alarm_dev (selection rule in its docstring).
FROZEN = {
    "uengram": AlarmConfig(alpha=0.003, k_novelty=3, k_health=0, per_state=True,
                           new_state_alert=True, aci_gamma=0.0, health_needs_baseline=True),
    "hst": AlarmConfig(alpha=0.01, k_novelty=2, k_health=0, per_state=False,
                       new_state_alert=False, aci_gamma=0.005),
    "memstream": AlarmConfig(alpha=0.01, k_novelty=3, k_health=0, per_state=False,
                             new_state_alert=False, aci_gamma=0.005),
}
SCORES = os.path.expanduser("~/.local/state/uengram/v2scores")
MAX_FRAMES = 50  # JSON preview only; the separate archive contains every episode
EVIDENCE = os.path.expanduser("~/.local/state/uengram/evidence")


def stream_names(spec: str) -> list[str]:
    names = []
    for part in spec.split(","):
        if part == "metropt2":
            names += ["metropt2_jan", "metropt2_apr"]
        elif part == "care":
            from bench.stream_care import care_stream_names
            names += care_stream_names()
        else:
            names.append(part)
    return names


def _care_events(stream, d) -> dict:
    from bench.care_score import evaluate_event

    test = np.flatnonzero(stream.test)
    normal = stream.metric_mask[test]
    eid = stream.event_id()[test]
    label = stream.extra["care_label"]
    if label == "anomaly":
        in_event = eid == 0
        end_pos = int(np.flatnonzero(in_event)[-1])
    else:
        in_event = np.zeros(len(test), bool)
        stamp = stream.end[test]  # exact supplied SCADA row times
        end_pos = int(np.flatnonzero(stamp <= np.datetime64(stream.extra["care_event_end"]))[-1])
    rec = evaluate_event(d["active"], normal, in_event, end_pos, label)
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in rec.items()}


def _run_task(task):
    name, method, enc, seed, scores_dir = task
    label = f"[run_v2] {name} {method}/{enc} seed={seed}"
    print(f"{label} scoring", flush=True)
    stream = load_stream(name)
    sc = cached_scores(stream, method, enc, seed, scores_dir, label)
    lines = []
    for policy, cfg in (("v1_rule", V1_RULE), ("v2", FROZEN[method])):
        m, d, pol = apply_policy(stream, sc, cfg)
        details = {k: v for k, v in d.items() if k not in ("active", "opened")}
        details["stream_metadata"] = {k: stream.extra[k] for k in ("farm", "asset", "care_label")
                                      if k in stream.extra}
        details["episodes"] = pol.episodes[:MAX_FRAMES]
        latent = np.load(sc["meta"]["encoding_path"])["Z"] if method == "uengram" else None
        details["evidence"] = archive_frames(
            stream, pol.episodes, os.path.join(EVIDENCE, f"{name}_{method}_{enc}_{seed}_{policy}.npz"),
            sc["meta"], (sc["reference_mu"], sc["reference_var"]) if method == "uengram" else None,
            latent)
        kinds = [a["kind"] for a in pol.alerts]
        details["alerts"] = {k: kinds.count(k) for k in set(kinds)}
        if "care_label" in stream.extra:
            details["care_event"] = _care_events(stream, d)
        if method == "uengram":
            m["events_retained"] = float(len(sc["meta"]["events_retained"]))
            for k in ("items_end", "writes", "write_reduction_pct", "index_bytes", "pinned_raw_bytes"):
                m[k] = float(sc["meta"][k])
        m["lat_p99_us"] = float(sc["meta"]["lat_p99_us"])
        lines.append({"protocol": PROTOCOL, "stream": name, "method": method, "encoder": enc,
                      "seed": seed, "policy": policy, "config": cfg.__dict__, "metrics": m,
                      "details": details, "run": sc["meta"]})
        print(f"{label} {policy}: detected={m['events_detected']:.0f}/{len(stream.events)} "
              f"FA/week={m['false_alarm_per_week']:.2f} alerts/week={m['alerts_per_week']:.1f}", flush=True)
    if method == "uengram" and enc == "dsp" and seed == 0 and "parked_share" in stream.extra:
        from bench.leak_duty import leak_duty
        m, d = leak_duty(stream)
        d["evidence"] = archive_frames(
            stream, d["episodes"], os.path.join(EVIDENCE, f"{name}_leak_duty.npz"),
            {"method": "leak_duty", "alpha": 0.01, "persistence": 3, "min_parked_share": 0.9})
        lines.append({"protocol": PROTOCOL, "stream": name, "method": "leak_duty", "encoder": "none",
                      "seed": 0, "policy": "secondary", "metrics": m, "details": d})
        print(f"{label} leak_duty: detected={m['events_detected']:.0f}/{len(stream.events)} "
              f"FA/week={m['false_alarm_per_week']:.2f}", flush=True)
    return lines


def run_task(task):
    try:
        return _run_task(task)
    except (ValueError, np.linalg.LinAlgError) as error:
        name, method, enc, seed, _ = task
        reason = f"{type(error).__name__}: {error}"
        print(f"[run_v2] FAILED {name} {method}/{enc} seed={seed}: {reason}", flush=True)
        return [{"protocol": PROTOCOL, "stream": name, "method": method, "encoder": enc,
                 "seed": seed, "policy": "failed", "status": "invalid_calibration", "error": reason}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", required=True, help="metropt2, care, or stream names")
    ap.add_argument("--methods", default="uengram:dsp,hst:dsp,memstream:dsp")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--scores-dir", default=SCORES)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    methods = [m.split(":") for m in a.methods.split(",")]
    tasks = [(n, m, e, s, a.scores_dir) for n in stream_names(a.streams) for m, e in methods
             for s in parse_seeds(a.seeds) if not (m == "uengram" and e in ("dsp", "tab") and s > 0)]
    print(f"[run_v2] {len(tasks)} runs -> {a.out}", flush=True)
    with open(a.out, "w"):
        pass
    failed = 0
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        for lines in pool.imap_unordered(run_task, tasks):
            dump(a.out, lines)
            failed += sum(r.get("status") == "invalid_calibration" for r in lines)
    if failed:
        raise SystemExit(f"{failed} runs failed calibration; diagnostics are in {a.out}")


if __name__ == "__main__":
    main()

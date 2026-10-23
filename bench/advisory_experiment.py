"""Paired second-stage advisory experiment over the frozen normal baseline.

    --backend laya           local open-weight Laya (ConvAI Innovations), never labelled Jev
    --backend openrouter-jev hosted TypeSafe Jev via OpenRouter (bench.jev_experiment)

Laya: `--prereg` writes results/v3_laya_preregistration.json from trace inputs only (no model
answers, no labels); a run refuses to start unless the code, rule and baseline still match it.
Interpretations, on the SAME whole test timeline:
    conservative_preserve_baseline = baseline alarm | fresh Laya review elevation
    advisory_assessment_only       = fresh Laya review elevation | (baseline alarm while the
                                     latest assessment is unavailable)  # outage never silences
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path

import numpy as np

from bench import advisory_eval as ev
from bench import laya_advisory as la
from bench.jev_advisory import PROMPT_HASH, QUESTIONS
from bench.metropt_eval import episodes
from bench.streams import load_stream

PROTOCOL = "v3-laya-fixed-candidate-triage"
CODE_FILES = ("bench/laya_advisory.py", "bench/advisory_eval.py", "bench/advisory_experiment.py")
CAVEATS = [
    "Laya (convaiinnovations/laya, English base, ModernBERT-large 421M) is a different model "
    "from TypeSafe Jev; nothing here is a Jev result.",
    "Vendor typed-decisions benchmark: base English Laya 0.362 vs 0.766 fine-tuned; this "
    "checkpoint is the base model.",
    "Compressor telemetry summaries are out-of-domain for Laya; probabilities are not claimed "
    "calibrated, and the checkpoint warns that some temperatures are invalid.",
    "Brier/ECE use coarse window labels (event-interval overlap vs clean negatives) only.",
    "All three recordings were consumed earlier; results are exploratory, not confirmatory.",
]


def rule() -> dict:
    return {"candidate_states": {"unresolved": 2, "alarm": 3},
            "schedule": "assess on candidate entry, then at most once per observed hour while "
                        "decision_state stays in {unresolved, alarm}; clean in-reference windows "
                        "are never assessed (they appear only as previous-window context)",
            "reassess_seconds": ev.REASSESS_SECONDS, "fresh_seconds": ev.FRESH_SECONDS,
            "elevate": "review.choice == 'review' AND P(review) >= 0.80 AND TypeSafe Choice "
                       "confidence (p_max-1/n)/(1-1/n) of the review probabilities >= 0.50",
            "review_probability": la.REVIEW_PROBABILITY, "review_confidence": la.REVIEW_CONFIDENCE,
            "laya_confidence_fields": "Laya `confidence` and `answer_confidence` are recorded "
                                      "descriptively and never used for decisions",
            "unavailable": "exception/invalid/truncated answer => explicit unavailable; never "
                           "elevates and never suppresses a baseline alarm",
            "interpretations": {"conservative_preserve_baseline": "baseline | fresh elevation",
                                "advisory_assessment_only": "fresh elevation | (baseline & "
                                                            "latest assessment unavailable)"}}


def frozen_spec(freeze_path) -> dict:
    code = {f: ev.sha256_file(f) for f in CODE_FILES}
    return {"protocol": PROTOCOL, "backend": la.BACKEND, "model_id": la.MODEL_ID,
            "revision": la.REVISION, "prompt_hash": PROMPT_HASH, "questions": QUESTIONS,
            "state_format": la.STATE_FORMAT, "rule": rule(), "code_sha256": code,
            "baseline_freeze": {"path": str(freeze_path), "sha256": ev.sha256_file(freeze_path)}}


def _log(fh, msg):
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n"); fh.flush()


def _streams(freeze):
    for name, record in freeze["selected"].items():
        z = ev.load_trace(record)
        due = ev.assessment_schedule(z["decision_state"], z["end"])
        yield name, record, z, la.due_states(z, due)


def preregister(freeze_path, out, log):
    freeze = json.loads(Path(freeze_path).read_text())
    agent, prov = la.load_agent()
    states, per_stream = [], {}
    for name, record, z, due in _streams(freeze):
        states += list(due.values())
        per_stream[name] = {"trace_sha256": record["trace"]["sha256"], "assessments": len(due),
                            "candidate_windows": int(np.isin(z["decision_state"], ev.CANDIDATE_STATES).sum())}
    tokens = la.budget(agent, states)
    if tokens["truncated_requests"]:
        raise SystemExit(f"token budget exceeded; shorten the state format: {tokens}")
    example = next(iter(la.due_states(*_example()).values()))
    doc = {**frozen_spec(freeze_path), "artifact": "preregistration",
           "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "recorded_before_model_answers_on_machine_cases": True, "model": prov,
           "input_exclusions": {"never_sent": ["report labels", "event/fault dates or IDs",
                                               "stream/file/asset names", "absolute timestamps",
                                               "future or post-decision windows"],
                                "warning_channels": list(la.BANNED_FEATURES),
                                "allowed_features": list(la.FEATURES),
                                "time": "hours since the flagged period began; window minutes"},
           "token_budget": tokens, "per_stream": per_stream, "synthetic_example_state": example,
           "metrics": ["events detected", "false alarms/week", "clean alarm share",
                       "notification episodes/week", "window balanced accuracy/F1",
                       "per-event lead", "coverage", "latency p50/p95",
                       "Brier/ECE of P(condition=fault_consistent) and P(review) vs coarse labels"],
           "caveats": CAVEATS}
    Path(out).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    _log(log, f"preregistration written {out} sha256={ev.sha256_file(out)} tokens={tokens}")


def _example():
    """Synthetic non-machine trace used only to show the state format in the preregistration."""
    names = np.asarray(list(la.FEATURES) + ["lps_share"])
    n = 3
    z = {"decision_state": np.array([1, 2, 2]), "state": np.array([1, 1, 1]),
         "start": np.array([0, 600, 1200]), "end": np.array([600, 1200, 1800]),
         "novelty": np.array([1.0, 3.0, 2.5]), "novelty_threshold": np.full(n, 2.0),
         "health": np.array([0.5, 0.9, 1.1]), "health_threshold": np.full(n, 1.0),
         "features": np.tile(np.arange(len(names), dtype=float), (n, 1)), "feature_names": names,
         "baseline_counts": json.dumps({"1": {"novelty_threshold_supported": True,
                                              "monitorable_windows": 500}}),
         "config": json.dumps({"per_state": True})}
    return z, ev.assessment_schedule(z["decision_state"], z["end"])


def run_assessments(agent, prov, due, cache_dir, log, label, every=30.0):
    """Assess every scheduled state once; cached answers are reused, never recomputed."""
    results, fresh, last = {}, 0, time.monotonic()
    for k, (i, state) in enumerate(due.items(), 1):
        path = cache_dir / f"{la.cache_key(prov, state)}.json"
        if path.exists():
            results[i] = json.loads(path.read_text())
        else:
            res = la.assess(agent, state)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(res))
            os.replace(tmp, path)
            results[i], fresh = res, fresh + 1
        if time.monotonic() - last >= every:
            _log(log, f"[{label}] {k}/{len(due)} assessed ({fresh} new model calls)")
            last = time.monotonic()
    return results


def _pct(values, q):
    return float(np.percentile(values, q)) if values else None


def stream_row(name, record, z, results, prov, prereg_sha):
    stream = load_stream(name)
    test = np.flatnonzero(stream.test)
    if len(test) != len(z["end"]) or not np.array_equal(test, z["test_input_index"]):
        raise ValueError(f"test timeline mismatch for {name}")
    base = np.asarray(z["active"], bool)
    outcomes = {i: (r["elevated"] if r["available"] else None) for i, r in results.items()}
    elevated, unavailable = ev.fresh_timeline(z["decision_state"], z["end"], outcomes)
    timelines = ev.interpretations(base, elevated, unavailable)
    metrics, details = {}, {}
    for key, active in timelines.items():
        metrics[key], details[key] = ev.paired_metrics(stream, z, test, active)
    y = ev.coarse_labels(stream, z, test)
    ok = [i for i, r in results.items() if r["available"]]
    lab = [i for i in ok if y[i] >= 0]
    calib = {q: ev.calibration([results[i]["answers"][q]["probabilities"][opt] for i in lab],
                               [y[i] for i in lab])
             for q, opt in (("condition", "fault_consistent"), ("review", "review"))}
    valid = ev.metric_valid(stream, z, test)
    end = np.asarray(z["end"], dtype="datetime64[s]")
    weeks = float((z["end"][valid] - z["start"][valid]).sum()) / (7 * 86400)
    lat = [r["elapsed_ms"] for r in results.values() if r["available"]]
    errors = {}
    for r in results.values():
        if not r["available"]:
            errors[r["error"].split(":")[0]] = errors.get(r["error"].split(":")[0], 0) + 1
    mask = np.zeros(len(end), bool); mask[list(results)] = True
    return {"protocol": PROTOCOL, "backend": la.BACKEND, "model_id": la.MODEL_ID,
            "revision": prov["revision"], "laya_version": prov["laya_version"],
            "prompt_hash": PROMPT_HASH, "state_format": la.STATE_FORMAT,
            "preregistration_sha256": prereg_sha, "stream": name, "method": record["method"],
            "encoder": record["encoder"], "variant": record["variant"], "seed": record["seed"],
            "baseline_trace": record["trace"],
            "baseline_reproduction": ev.baseline_reproduction(record["metrics"], metrics["baseline"]),
            "metrics": metrics, "events": {k: d["events"] for k, d in details.items()},
            "window_confusion": {k: d["window_confusion"] for k, d in details.items()},
            "burden": {"candidate_windows": int(np.isin(z["decision_state"], ev.CANDIDATE_STATES).sum()),
                       "assessments": len(results), "assessments_per_week": len(results) / weeks,
                       "assessment_episodes_per_week": episodes(end[mask & valid]) / weeks,
                       "elevated_windows": int(elevated.sum()), "observed_weeks": weeks},
            "coverage": len(ok) / len(results) if results else None,
            "unavailable": len(results) - len(ok), "unavailable_errors": errors,
            "truncated_requests": sum(1 for r in results.values() if (r.get("error") or "").startswith("ValueError: truncated")),
            "elevations": sum(1 for i in ok if results[i]["elevated"]),
            "review_choice_counts": {c: sum(results[i]["answers"]["review"]["choice"] == c for i in ok)
                                     for c in QUESTIONS["review"]["criteria"]},
            "latency_ms": {"p50": _pct(lat, 50), "p95": _pct(lat, 95), "n": len(lat)},
            "calibration_vs_coarse_labels": calib, "caveats": CAVEATS}


def run(freeze_path, prereg_path, out, summary, cache_dir, log):
    prereg = json.loads(Path(prereg_path).read_text())
    spec = frozen_spec(freeze_path)
    drift = [k for k in spec if prereg.get(k) != spec[k]]
    if drift:
        raise SystemExit(f"code/rule/baseline changed after preregistration: {drift}")
    agent, prov = la.load_agent(prereg["model"]["weights_sha256"])
    if prov["weights_sha256"] != prereg["model"]["weights_sha256"]:
        raise SystemExit("loaded weights differ from the preregistered digests")
    prereg_sha = ev.sha256_file(prereg_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, record, z, due in _streams(json.loads(Path(freeze_path).read_text())):
        tokens = la.budget(agent, due.values())
        if tokens["truncated_requests"]:
            raise SystemExit(f"{name}: token budget exceeded {tokens}")
        _log(log, f"[{name}] {len(due)} assessments; max sequence tokens {tokens['max_sequence_tokens']}")
        results = run_assessments(agent, prov, due, cache_dir, log, name)
        row = stream_row(name, record, z, results, prov, prereg_sha)
        row["token_budget"] = tokens
        rows.append(row)
        _log(log, f"[{name}] done coverage={row['coverage']} elevations={row['elevations']}")
    Path(out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    keys = ("events_detected", "false_alarm_per_week", "alarm_time_share_clean",
            "notification_episodes_per_week_all", "window_balanced_accuracy", "window_f1", "lead_hours")
    Path(summary).write_text(json.dumps({
        "protocol": PROTOCOL, "backend": la.BACKEND, "model_id": la.MODEL_ID,
        "preregistration_sha256": prereg_sha, "results": str(out), "caveats": CAVEATS,
        "streams": {r["stream"]: {"coverage": r["coverage"], "elevations": r["elevations"],
                                  "latency_ms": r["latency_ms"], "burden": r["burden"],
                                  "truncated_requests": r["truncated_requests"],
                                  "baseline_reproduced": all(v["match"] for v in r["baseline_reproduction"].values()),
                                  **{k: {m: r["metrics"][k].get(m) for m in keys} for k in r["metrics"]},
                                  "calibration_vs_coarse_labels": r["calibration_vs_coarse_labels"]}
                    for r in rows}}, indent=2, sort_keys=True) + "\n")
    _log(log, f"wrote {out} and {summary}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=("laya", "openrouter-jev"), required=True)
    p.add_argument("--freeze", default="results/v3_normal_freeze.json")
    p.add_argument("--prereg", action="store_true", help="laya: write the preregistration only")
    p.add_argument("--prereg-path", default="results/v3_laya_preregistration.json")
    p.add_argument("--out", default=None)
    p.add_argument("--summary", default="results/v3_laya_summary.json")
    p.add_argument("--cache", default=None)
    p.add_argument("--log", default=str(Path.home() / ".local/state/uengram/laya_run.log"))
    p.add_argument("--normal-frozen", action="store_true", help="openrouter-jev: acknowledge NORMAL_FROZEN")
    a = p.parse_args()
    if a.backend == "openrouter-jev":
        from bench import jev_experiment
        cache = Path(a.cache or "~/.local/state/uengram/jev_cache").expanduser()
        rows = jev_experiment.run(a.freeze, a.out or "results/v3_jev.jsonl", cache, a.normal_frozen)
        print(json.dumps({"rows": len(rows)}))
        return
    Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    with open(a.log, "a") as log:
        if a.prereg:
            preregister(a.freeze, a.prereg_path, log)
        else:
            cache = Path(a.cache or "~/.local/state/uengram/laya_cache").expanduser()
            run(a.freeze, a.prereg_path, a.out or "results/v3_laya.jsonl", a.summary, cache, log)


if __name__ == "__main__":
    main()

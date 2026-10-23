"""Paired second-stage Jev assessment over the preregistered normal freeze."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from bench.alarm_eval import evaluate_alarms
from bench.jev_advisory import (MODEL, PROMPT_HASH, PROMPT_VERSION, assess, cache_key,
                               causal_state)
from bench.streams import load_stream

PROTOCOL = "v3-jev-fixed-candidate-triage"
PROBABILITY_THRESHOLD = 0.8
CONFIDENCE_THRESHOLD = 0.5
REASSESS_SECONDS = 3600


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _load_trace(record):
    meta = record["trace"]
    path = Path(meta["path"])
    if path.stat().st_size != meta["bytes"] or sha(path) != meta["sha256"]:
        raise ValueError(f"baseline trace integrity mismatch: {path}")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _json(z, key):
    return json.loads(str(z[key]))


def _candidate_summary(z, i):
    state = int(z["decision_state"][i])
    counts = _json(z, "baseline_counts")
    cfg = _json(z, "config")
    st = int(z["state"][i]) if cfg.get("per_state") else -1
    support = counts.get(str(st), counts.get(st, {}))
    features = z["features"][i]
    names = [str(x) for x in z["feature_names"]]
    excluded = {"lps_share", "lps", "oil_level", "pressure_switch"}
    physical = {name: (None if not np.isfinite(value) else float(value))
                for name, value in zip(names, features)
                if name.lower().replace(" ", "_") not in excluded}
    def num(key):
        x = float(z[key][i]); return x if np.isfinite(x) else None
    return {"novelty": num("novelty"), "health": num("health"),
            "novelty_threshold": num("novelty_threshold"), "health_threshold": num("health_threshold"),
            "monitorable": bool(z["monitorable"][i]), "active": bool(z["active"][i]),
            "decision_state": state, "support": support,
            "baseline_scores": {"novelty": num("novelty"), "health": num("health")},
            "missingness": {"novelty_missing": num("novelty") is None, "health_missing": num("health") is None},
            "physical": physical}


def _opened(active):
    return active & ~np.r_[False, active[:-1]]


def _metrics(stream, z, active, opened, test):
    valid = np.asarray(z["monitorable"], bool)
    event_id = stream.event_id()[test]
    negative = stream.clean_negative()[test]
    if stream.metric_mask is not None:
        valid &= stream.metric_mask[test]
        negative &= stream.metric_mask[test]
    events = [(a, b) for a, b in stream.events]
    start = np.asarray(z["start"], dtype="datetime64[s]")
    end = np.asarray(z["end"], dtype="datetime64[s]")
    return evaluate_alarms(start, end, event_id, negative, valid, events, active, opened,
                           z["novelty"], z["health"])[0]


def assessment_schedule(decision_state, end):
    """Candidate entry and hourly cadence, indexed only by observed time."""
    due = np.zeros(len(end), dtype=bool)
    last = None
    for i, (state, timestamp) in enumerate(zip(decision_state, end)):
        if int(state) not in (2, 3):
            last = None
            continue
        now = float(timestamp)
        if last is None or now - last >= REASSESS_SECONDS:
            due[i] = True
            last = now
    return due


def advisory_elevation(answer):
    return (answer.get("choice") == "review" and
            answer.get("probabilities", {}).get("review", 0) >= PROBABILITY_THRESHOLD and
            answer.get("confidence", 0) >= CONFIDENCE_THRESHOLD)


def elevation_timeline(decision_state, end, assessment_answers):
    """Hold a successful review decision only while the latest assessment is fresh."""
    elevated = np.zeros(len(end), dtype=bool)
    latest_time, latest_review = None, False
    for i, (state, timestamp) in enumerate(zip(decision_state, end)):
        if int(state) not in (2, 3):
            latest_time, latest_review = None, False
            continue
        now = float(timestamp)
        if i in assessment_answers:
            latest_time, latest_review = now, advisory_elevation(assessment_answers[i])
        elevated[i] = (latest_review and latest_time is not None and
                       now - latest_time <= REASSESS_SECONDS)
    return elevated


def run(freeze_path, out_path, cache_dir, ack_normal_frozen):
    if not ack_normal_frozen:
        raise RuntimeError("machine-data requests require explicit NORMAL_FROZEN acknowledgement")
    freeze_path = Path(freeze_path)
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("schema") != 1 or not freeze.get("selected"):
        raise ValueError("unsupported or empty frozen normal schema")
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    output, failures = [], 0
    for stream_name, record in freeze["selected"].items():
        times, stream_failures = [], 0
        z = _load_trace(record)
        stream = load_stream(stream_name)
        test = np.flatnonzero(stream.test)
        if len(test) != len(z["start"]):
            raise ValueError(f"test timeline mismatch for {stream_name}")
        baseline = np.asarray(z["active"], bool)
        assessments = {}
        due_mask = assessment_schedule(z["decision_state"], z["end"])
        previous = None
        t0 = float(z["start"][0]) if len(baseline) else 0
        for i in range(len(baseline)):
            if int(z["decision_state"][i]) not in (2, 3):
                previous = None
                continue
            current = _candidate_summary(z, i)
            now = float(z["end"][i])
            if due_mask[i]:
                if previous is None and i > 0 and int(z["decision_state"][i - 1]) == 1:
                    previous = _candidate_summary(z, int(i - 1))
                state = causal_state(current, previous, max(0, (now - t0) / 3600))
                key = cache_key(state, record["trace"]["sha256"], record["trace"]["sha256"])
                cache = cache_dir / f"{key}.json"
                if cache.exists():
                    result = json.loads(cache.read_text())
                else:
                    result_obj = assess(state)
                    result = {"available": result_obj.available, "answers": result_obj.answers,
                              "model": result_obj.model, "usage": result_obj.usage,
                              "elapsed_ms": result_obj.elapsed_ms, "error": result_obj.error}
                    tmp = cache.with_suffix(".tmp")
                    tmp.write_text(json.dumps(result, separators=(",", ":")))
                    os.replace(tmp, cache)
                if result.get("available"):
                    times.append(float(result.get("elapsed_ms", 0)))
                else:
                    failures += 1
                    stream_failures += 1
                assessments[int(i)] = {"cache_key": key, **result}
            previous = current
        review = elevation_timeline(
            z["decision_state"], z["end"],
            {i: a["answers"]["review"] for i, a in assessments.items()
             if a.get("available") and a.get("answers")})
        advisory = baseline | review
        opened_base, opened_adv = _opened(baseline), _opened(advisory)
        metrics_base = _metrics(stream, z, baseline, opened_base, test)
        metrics_adv = _metrics(stream, z, advisory, opened_adv, test)
        assessments_path = cache_dir / f"{stream_name}_assessment_indices.json"
        assessments_path.write_text(json.dumps({str(i): a for i, a in assessments.items()}, separators=(",", ":")))
        output.append({"protocol": PROTOCOL, "stream": stream_name, "method": record["method"],
                       "encoder": record["encoder"], "variant": record["variant"], "seed": record["seed"],
                       "baseline_trace": record["trace"], "assessment_indices": str(assessments_path),
                       "baseline_metrics": metrics_base, "conservative_preserve_baseline_metrics": metrics_base,
                       "advisory_assessment_only_metrics": metrics_adv,
                       "candidate_windows": int(np.isin(z["decision_state"], (2, 3)).sum()),
                       "assessment_count": len(assessments),
                       "coverage": (sum(a.get("available", False) for a in assessments.values()) / len(assessments)
                                    if assessments else 0),
                       "unavailable_count": sum(not a.get("available", False) for a in assessments.values()),
                       "latency_ms": {"p50": float(np.percentile(times, 50)) if times else None,
                                      "p95": float(np.percentile(times, 95)) if times else None},
                       "model_requested": MODEL,
                       "model_returned": [a.get("model") for a in assessments.values() if a.get("model")],
                       "prompt_version": PROMPT_VERSION, "prompt_hash": PROMPT_HASH,
                       "fixed_thresholds": {"review_probability": PROBABILITY_THRESHOLD,
                                            "confidence": CONFIDENCE_THRESHOLD,
                                            "reassessment_seconds": REASSESS_SECONDS},
                       "request_failures": stream_failures})
    Path(out_path).write_text("\n".join(json.dumps(row) for row in output) + "\n")
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze", default="results/v3_normal_freeze.json")
    p.add_argument("--out", default="results/v3_jev.jsonl")
    p.add_argument("--cache", default="~/.local/state/uengram/jev_cache")
    p.add_argument("--normal-frozen", action="store_true", help="explicitly acknowledge parent NORMAL_FROZEN")
    a = p.parse_args()
    rows = run(a.freeze, a.out, Path(a.cache).expanduser(), a.normal_frozen)
    print(json.dumps({"rows": len(rows), "output": a.out, "cache": str(Path(a.cache).expanduser())}))


if __name__ == "__main__":
    main()

"""Paired second-stage Jev assessment over the preregistered normal freeze."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from bench.advisory_eval import REASSESS_SECONDS, assessment_schedule, fresh_timeline, interpretations
from bench.advisory_report import corrected_metrics, replay
from bench.jev_advisory import (MODEL, PROMPT_HASH, PROMPT_VERSION, assess, cache_key,
                               causal_state)
from bench.streams import load_stream

PROTOCOL = "v3-jev-fixed-candidate-triage"
PROBABILITY_THRESHOLD = 0.8
CONFIDENCE_THRESHOLD = 0.5


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


def advisory_elevation(answer):
    return (answer.get("choice") == "review" and
            answer.get("probabilities", {}).get("review", 0) >= PROBABILITY_THRESHOLD and
            answer.get("confidence", 0) >= CONFIDENCE_THRESHOLD)


def advisory_timelines(baseline, decision_state, end, assessments):
    """Apply successful, negative and unavailable assessments on the same timeline."""
    outcomes = {
        i: advisory_elevation(result["answers"]["review"]) if result["available"] else None
        for i, result in assessments.items()
    }
    elevated, unavailable = fresh_timeline(decision_state, end, outcomes)
    return interpretations(baseline, elevated, unavailable)


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
        timelines = advisory_timelines(baseline, z["decision_state"], z["end"], assessments)
        alerts, score_cache, _ = replay(stream, record, z)
        metrics = {name: corrected_metrics(stream, z, test, active, alerts)[0]
                   for name, active in timelines.items()}
        assessments_path = cache_dir / f"{stream_name}_assessment_indices.json"
        assessments_path.write_text(json.dumps({str(i): a for i, a in assessments.items()}, separators=(",", ":")))
        output.append({"protocol": PROTOCOL, "stream": stream_name, "method": record["method"],
                       "encoder": record["encoder"], "variant": record["variant"], "seed": record["seed"],
                       "baseline_trace": record["trace"], "assessment_indices": str(assessments_path),
                       "baseline_metrics": metrics["baseline"],
                       "conservative_preserve_baseline_metrics": metrics["conservative_preserve_baseline"],
                       "advisory_assessment_only_metrics": metrics["advisory_assessment_only"],
                       "notification_reconstruction_score_cache": score_cache,
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

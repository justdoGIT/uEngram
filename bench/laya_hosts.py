"""Finish the preregistered local Laya run on a second host (hp) with identical weights.

Answers come from two hosts running byte-identical English Laya checkpoint files:
`msi` (in-process, bench.advisory_experiment) and `hp` (laya-serve over the parent's SSH
forward). Every merged answer carries its host. Order of use:

    crosscheck  re-ask >=100 msi-answered states on hp (fixed seed) -> results/v3_laya_hp_crosscheck.json
    prereg      freeze results/v3_laya_hp_preregistration.json before any new machine case
    run         hp answers every scheduled state msi has not answered (resume-safe, sequential)
    build       merge msi + hp answers -> results/v3_laya.jsonl + results/v3_laya_summary.json

Rule, questions, state format, thresholds and metrics are the local preregistration's; the
files it hashed (bench/laya_advisory.py, advisory_eval.py, advisory_experiment.py) are reused
unchanged and re-verified by `build`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np

from bench import advisory_eval as ev
from bench import advisory_experiment as ax
from bench import laya_advisory as la
from bench import laya_http as lh
from bench.jev_advisory import PROMPT_HASH, QUESTIONS

ENDPOINT = "http://127.0.0.1:18001"
CODE_FILES = ("bench/laya_hosts.py", "bench/laya_http.py") + ax.CODE_FILES + ("bench/jev_advisory.py",)
SAMPLE_SIZE, SAMPLE_SEED = 120, 0
LOCAL_CACHE = Path("~/.local/state/uengram/laya_cache").expanduser()
HP_CACHE = Path("~/.local/state/uengram/laya_hp_cache").expanduser()


def hp_key(hp_prov: dict, state: str) -> str:
    return la.cache_key({"revision": hp_prov["revision"],
                         "laya_version": f"laya-http:hp:{hp_prov['laya_version']}"}, state)


def load_cached(path: Path, host: str):
    if not path.exists():
        return None
    rec = json.loads(path.read_text())
    if rec.get("host", host) != host:
        raise ValueError(f"{path} records host {rec.get('host')!r}, expected {host!r}")
    return {**rec, "host": host}


def merge(due: dict, local_prov: dict, hp_prov: dict, local_cache: Path, hp_cache: Path) -> dict:
    """index -> answer record labelled with its host; msi answers win where both exist."""
    out = {}
    for i, state in due.items():
        rec = load_cached(local_cache / f"{la.cache_key(local_prov, state)}.json", "msi")
        rec = rec or load_cached(hp_cache / f"{hp_key(hp_prov, state)}.json", "hp")
        if rec is None:
            raise ValueError(f"assessment {i} has no answer from any host")
        out[i] = rec
    return out


def compare_pair(a: dict, b: dict) -> dict:
    """Differences between two answers to the identical request."""
    pa, pb = a["answers"]["review"]["probabilities"], b["answers"]["review"]["probabilities"]
    return {"flip": a["elevated"] != b["elevated"],
            "d_review_probability": abs(pa["review"] - pb["review"]),
            "d_review_confidence": abs(la.typesafe_confidence(pa) - la.typesafe_confidence(pb)),
            "d_condition_fault": abs(a["answers"]["condition"]["probabilities"]["fault_consistent"]
                                     - b["answers"]["condition"]["probabilities"]["fault_consistent"]),
            "d_max_any": max(abs(a["answers"][q]["probabilities"][o] - b["answers"][q]["probabilities"][o])
                             for q in QUESTIONS for o in QUESTIONS[q]["criteria"])}


def summarize_pairs(pairs: list[dict]) -> dict:
    keys = ("d_review_probability", "d_review_confidence", "d_condition_fault", "d_max_any")
    return {"pairs": len(pairs), "decision_flips": sum(p["flip"] for p in pairs),
            **{k: {"max": max((p[k] for p in pairs), default=None),
                   "mean": float(np.mean([p[k] for p in pairs])) if pairs else None} for k in keys}}


def _setup(args):
    local = json.loads(Path(args.local_prereg).read_text())
    if local["baseline_freeze"]["sha256"] != ev.sha256_file(args.freeze):
        raise SystemExit("baseline freeze changed since the local preregistration")
    local_prov = {"revision": local["model"]["revision"], "laya_version": local["model"]["laya_version"]}
    streams = list(ax._streams(json.loads(Path(args.freeze).read_text())))
    return local, local_prov, streams


def _hp_prov(health: dict, versions: dict) -> dict:
    return {"revision": health["revisions"][lh.ROUTE], "laya_version": versions.get("laya", "unknown")}


def remote_facts(host: str, revision: str) -> dict:
    """Read-only on hp: checkpoint digests (HF_HOME=/tmp/laya-hp/hf) and package versions."""
    files = " ".join(lh.WEIGHT_FILES)
    cmd = (f"cd /tmp/laya-hp/hf/hub/models--convaiinnovations--laya/snapshots/{revision} && "
           f"for f in {files}; do printf '%s ' \"$f\"; sha256sum < \"$(readlink -f \"$f\")\" | cut -d' ' -f1; done; "
           "/tmp/laya-hp/.venv/bin/python -c \"import importlib.metadata as m; "
           "print('versions', m.version('laya'), m.version('torch'), m.version('transformers'))\"; "
           "tr '\\0' '\\n' < /proc/$(pgrep -f laya-serve | head -1)/environ | grep -E '^LAYA_(THREADS|MODELS|HOST|PORT)='")
    out = subprocess.run(["ssh", host, cmd], capture_output=True, text=True, timeout=180, check=True).stdout
    digests, versions, env = {}, {}, {}
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[0] == "versions":
            versions = dict(zip(("laya", "torch", "transformers"), parts[1:]))
        elif line.startswith("LAYA_"):
            k, v = line.split("=", 1); env[k] = v
        elif len(parts) == 2:
            digests[parts[0]] = parts[1]
    return {"weights_sha256": digests, "versions": versions, "serve_env": env}


def _hp_answer(client, hp_prov, state, cache: Path, host="hp") -> dict:
    path = cache / f"{hp_key(hp_prov, state)}.json"
    rec = load_cached(path, host)
    if rec is None:
        rec = {**lh.assess(client, state), "host": host}
        if lh.cacheable(rec):
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec))
            os.replace(tmp, path)
    return rec


def crosscheck(args, log):
    local, local_prov, streams = _setup(args)
    client = lh.HttpLaya(args.endpoint)
    health = client.health()
    remote = remote_facts(args.host, health["revisions"][lh.ROUTE])
    hp_prov = _hp_prov(health, remote["versions"])
    answered = [(name, i, s) for name, _, _, due in streams for i, s in due.items()
                if (LOCAL_CACHE / f"{la.cache_key(local_prov, s)}.json").exists()]
    sample = sorted(random.Random(SAMPLE_SEED).sample(answered, min(SAMPLE_SIZE, len(answered))))
    HP_CACHE.mkdir(parents=True, exist_ok=True)
    pairs = []
    for k, (name, i, state) in enumerate(sample, 1):
        a = load_cached(LOCAL_CACHE / f"{la.cache_key(local_prov, state)}.json", "msi")
        b = _hp_answer(client, hp_prov, state, HP_CACHE)
        if a["available"] and b["available"]:
            pairs.append({"stream": name, "index": int(i), **compare_pair(a, b)})
        if k % 20 == 0:
            ax._log(log, f"[crosscheck] {k}/{len(sample)}")
    end = client.health()
    doc = {"artifact": "cross-host check (msi vs hp, identical requests)", "seed": SAMPLE_SEED,
           "sample_rule": f"random.Random({SAMPLE_SEED}).sample of msi-answered scheduled states, n={len(sample)}",
           "streams_available_in_msi_cache": sorted({n for n, _, _ in answered}),
           "sampled_per_stream": {n: sum(s[0] == n for s in sample) for n in sorted({s[0] for s in sample})},
           "hp_unavailable": len(sample) - len(pairs), "summary": summarize_pairs(pairs), "pairs": pairs,
           "hp_health_start": health, "hp_health_end": end, "hp_remote": remote,
           "weights_identical_to_msi": remote["weights_sha256"] == local["model"]["weights_sha256"]}
    Path(args.crosscheck_path).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    ax._log(log, f"crosscheck {args.crosscheck_path}: {json.dumps(doc['summary'])}")


def frozen_spec(args) -> dict:
    return {"backend": "laya-http", "host": "hp", "route": lh.ROUTE, "prompt_hash": PROMPT_HASH,
            "questions": QUESTIONS, "state_format": la.STATE_FORMAT,
            "code_sha256": {f: ev.sha256_file(f) for f in CODE_FILES},
            "local_preregistration": {"path": args.local_prereg, "sha256": ev.sha256_file(args.local_prereg)},
            "baseline_freeze": {"path": args.freeze, "sha256": ev.sha256_file(args.freeze)},
            "transport": {"endpoint": args.endpoint, "timeout_s": lh.TIMEOUT, "retries": 1,
                          "concurrency": 1, "pace_s": lh.PACE_SECONDS}}


def preregister(args, log):
    local, _, _ = _setup(args)
    cross = json.loads(Path(args.crosscheck_path).read_text())
    health = lh.HttpLaya(args.endpoint).health()
    remote = remote_facts(args.host, health["revisions"][lh.ROUTE])
    doc = {**frozen_spec(args), "artifact": "preregistration-addendum (second host)",
           "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "recorded_before_new_machine_cases_on_hp": True,
           "hosts": {"msi": {"how": "in-process laya.load, torch 2 threads", "revision": local["model"]["revision"],
                             "weights_sha256": local["model"]["weights_sha256"]},
                     "hp": {"how": "laya-serve via parent SSH forward 127.0.0.1:18001 -> hp 127.0.0.1:8000",
                            "health_at_preregistration": health, "remote": remote}},
           "weights_identical": remote["weights_sha256"] == local["model"]["weights_sha256"],
           "statement": "Answers come from two hosts (msi local, hp) with byte-identical checkpoint "
                        "files; msi answers are kept wherever they exist and hp answers every remaining "
                        "scheduled state. The cross-host check bounds the host difference.",
           "crosscheck": {"path": args.crosscheck_path, "sha256": ev.sha256_file(args.crosscheck_path),
                          "summary": cross["summary"]},
           "rule": local["rule"], "per_request_requirements": {"routing.model": lh.ROUTE,
                                                               "usage.truncated": False,
                                                               "usage.state_tokens_dropped": 0},
           "abort_if": "/health english revision differs from the preregistered one at any check",
           "caveats": ax.CAVEATS}
    if cross["summary"]["decision_flips"]:
        raise SystemExit("cross-host decision flips > 0: not mixing hosts")
    Path(args.prereg_path).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    ax._log(log, f"hp addendum {args.prereg_path} sha256={ev.sha256_file(args.prereg_path)}")


def _check_drift(args):
    prereg = json.loads(Path(args.prereg_path).read_text())
    spec = frozen_spec(args)
    drift = [k for k in spec if prereg.get(k) != spec[k]]
    if drift:
        raise SystemExit(f"changed after the hp addendum: {drift}")
    return prereg


def run(args, log):
    prereg = _check_drift(args)
    _, local_prov, streams = _setup(args)
    client = lh.HttpLaya(args.endpoint)
    revision = prereg["hosts"]["hp"]["health_at_preregistration"]["revisions"][lh.ROUTE]
    hp_prov = {"revision": revision, "laya_version": prereg["hosts"]["hp"]["remote"]["versions"]["laya"]}
    health_log = []

    def check_health():
        try:
            h = client.health()
        except Exception as exc:  # noqa: BLE001 - endpoint down: stop instead of burning requests
            h = {"error": f"{type(exc).__name__}: {str(exc)[:200]}", "revisions": {}}
        health_log.append({"utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                           "english_revision": h["revisions"].get(lh.ROUTE), "error": h.get("error")})
        Path(args.health_log).write_text(json.dumps(health_log))
        if h["revisions"].get(lh.ROUTE) != revision:
            raise SystemExit(f"hp health failed or revision changed: {h}")

    check_health()
    for name, _, _, due in streams:
        todo = {i: s for i, s in due.items() if not (LOCAL_CACHE / f"{la.cache_key(local_prov, s)}.json").exists()}
        ax._log(log, f"[hp {name}] {len(todo)} of {len(due)} scheduled states not answered on msi")
        last, failed = time.monotonic(), 0
        for k, state in enumerate(todo.values(), 1):
            failed += not _hp_answer(client, hp_prov, state, HP_CACHE)["available"]
            if time.monotonic() - last >= 30:
                check_health()
                ax._log(log, f"[hp {name}] {k}/{len(todo)} ({failed} unavailable, {client.retried} retries)")
                last = time.monotonic()
    check_health()
    ax._log(log, f"hp run complete; requests={client.calls} retries={client.retried}")


def _tokenizer_agent():
    from huggingface_hub import snapshot_download
    from laya.agent import Agent, _load_tokenizer
    root = Path(snapshot_download(la.MODEL_ID, revision=la.REVISION, local_files_only=True,
                                  allow_patterns=["rl_agent_config.json", "tokenizer/*"]))
    cfg = json.loads((root / "rl_agent_config.json").read_text())
    return type("TokAgent", (), {"tok": _load_tokenizer(str(root / "tokenizer"), cfg), "cfg": cfg,
                                 "_to_internal": staticmethod(Agent._to_internal)})


def build(args, log):
    prereg = _check_drift(args)
    local = json.loads(Path(args.local_prereg).read_text())
    if [k for k in ax.frozen_spec(args.freeze) if local.get(k) != ax.frozen_spec(args.freeze)[k]]:
        raise SystemExit("local preregistered code/rule/baseline changed")
    _, local_prov, streams = _setup(args)
    hp = prereg["hosts"]["hp"]
    hp_prov = {"revision": hp["health_at_preregistration"]["revisions"][lh.ROUTE],
               "laya_version": hp["remote"]["versions"]["laya"]}
    prov = {**local_prov, "backend": la.BACKEND}
    tok_agent, rows = _tokenizer_agent(), []
    for name, record, z, due in streams:
        results = merge(due, local_prov, hp_prov, LOCAL_CACHE, HP_CACHE)
        row = ax.stream_row(name, record, z, results, prov, ev.sha256_file(args.local_prereg))
        tokens = la.budget(tok_agent, due.values())
        ok = [r for r in results.values() if r["available"]]
        row.update({"token_budget": tokens, "hp_addendum_sha256": ev.sha256_file(args.prereg_path),
                    "max_observed_input_tokens": max((r["usage"]["input_tokens"] for r in ok), default=None),
                    "hosts": {h: {"assessments": sum(r["host"] == h for r in results.values()),
                                  "unavailable": sum(r["host"] == h and not r["available"] for r in results.values()),
                                  "latency_ms": {"p50": ax._pct([r["elapsed_ms"] for r in ok if r["host"] == h], 50),
                                                 "p95": ax._pct([r["elapsed_ms"] for r in ok if r["host"] == h], 95)},
                                  "revision": local_prov["revision"] if h == "msi" else hp_prov["revision"]}
                              for h in ("msi", "hp")},
                    "assessment_hosts": {str(i): r["host"] for i, r in sorted(results.items())},
                    "crosscheck_summary": prereg["crosscheck"]["summary"]})
        rows.append(row)
        ax._log(log, f"[build {name}] coverage={row['coverage']} elevations={row['elevations']} "
                     f"hosts={ {h: v['assessments'] for h, v in row['hosts'].items()} }")
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    keys = ("events_detected", "false_alarm_per_week", "alarm_time_share_clean",
            "notification_episodes_per_week_all", "window_balanced_accuracy", "window_f1", "lead_hours")
    Path(args.summary).write_text(json.dumps({
        "protocol": ax.PROTOCOL, "backend": la.BACKEND, "model_id": la.MODEL_ID,
        "preregistration_sha256": ev.sha256_file(args.local_prereg),
        "hp_addendum_sha256": ev.sha256_file(args.prereg_path), "results": args.out,
        "caveats": ax.CAVEATS, "crosscheck_summary": prereg["crosscheck"]["summary"],
        "streams": {r["stream"]: {"coverage": r["coverage"], "elevations": r["elevations"], "hosts": r["hosts"],
                                  "burden": r["burden"], "truncated_requests": r["truncated_requests"],
                                  "max_sequence_tokens": r["token_budget"]["max_sequence_tokens"],
                                  "baseline_reproduced": all(v["match"] for v in r["baseline_reproduction"].values()),
                                  "review_choice_counts": r["review_choice_counts"],
                                  "events": r["events"],
                                  **{k: {m: r["metrics"][k].get(m) for m in keys} for k in r["metrics"]},
                                  "calibration_vs_coarse_labels": r["calibration_vs_coarse_labels"]}
                    for r in rows}}, indent=2, sort_keys=True) + "\n")
    ax._log(log, f"wrote {args.out} and {args.summary}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=("crosscheck", "prereg", "run", "build"))
    p.add_argument("--endpoint", default=ENDPOINT)
    p.add_argument("--host", default="hp", help="ssh alias for read-only provenance")
    p.add_argument("--freeze", default="results/v3_normal_freeze.json")
    p.add_argument("--local-prereg", default="results/v3_laya_preregistration.json")
    p.add_argument("--crosscheck-path", default="results/v3_laya_hp_crosscheck.json")
    p.add_argument("--prereg-path", default="results/v3_laya_hp_preregistration.json")
    p.add_argument("--health-log", default="results/v3_laya_hp_health.json")
    p.add_argument("--out", default="results/v3_laya.jsonl")
    p.add_argument("--summary", default="results/v3_laya_summary.json")
    p.add_argument("--log", default=str(Path.home() / ".local/state/uengram/laya_hp.log"))
    a = p.parse_args()
    Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    with open(a.log, "a") as log:
        {"crosscheck": crosscheck, "prereg": preregister, "run": run, "build": build}[a.step](a, log)


if __name__ == "__main__":
    main()
